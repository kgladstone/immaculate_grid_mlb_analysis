from __future__ import annotations

from pathlib import Path
import re
import ast
import textwrap

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from analytics.analysis import analyze_shame_index, grid_to_date
from config.constants import GRID_PLAYERS, GRID_PLAYERS_RESTRICTED, RULE5_FULL_BANS_CSV_PATH, PROMPTS_CSV_PATH


MOVING_AVERAGE_WINDOW = 28
WEEK_COUNT = 8
MAX_CUSTOM_RANGE_DAYS = 92
ROLLING_REPORT_DAYS = 90


def rolling_report_bounds(end_date=None) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp(end_date if end_date is not None else pd.Timestamp.now()).normalize()
    return end - pd.Timedelta(days=ROLLING_REPORT_DAYS - 1), end


def _month_bounds(report_month: str | pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    month_start = pd.Timestamp(report_month).to_period("M").start_time.normalize()
    month_end = month_start + pd.offsets.MonthEnd(0)
    return month_start, month_end.normalize()


def _date_bounds(start_date: str | pd.Timestamp, end_date: str | pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError("Report start date must be on or before the end date.")
    day_count = int((end - start).days) + 1
    if day_count > MAX_CUSTOM_RANGE_DAYS:
        raise ValueError(
            f"Custom report ranges are limited to {MAX_CUSTOM_RANGE_DAYS} days."
        )
    return start, end


def _is_complete_calendar_quarter(start: pd.Timestamp, end: pd.Timestamp) -> bool:
    quarter = start.to_period("Q")
    return start == quarter.start_time.normalize() and end == quarter.end_time.normalize()


def _is_complete_calendar_month(start: pd.Timestamp, end: pd.Timestamp) -> bool:
    return start == start.to_period("M").start_time.normalize() and end == start.to_period("M").end_time.normalize()


def format_report_title(start: pd.Timestamp, end: pd.Timestamp) -> str:
    """Return a deterministic title for a complete month or an arbitrary date range."""
    if _is_complete_calendar_quarter(start, end):
        return f"Immaculate Grid Quarterly Report | Q{start.quarter} {start.year} | {start:%b %d} - {end:%b %d}"
    if (end - start).days + 1 == ROLLING_REPORT_DAYS:
        return f"Immaculate Grid Quarterly Report | {start:%b %d, %Y} - {end:%b %d, %Y}"
    if _is_complete_calendar_month(start, end):
        return f"Immaculate Grid Analysis Monthly Report {start:%B %Y}"
    if start == end:
        period = f"{start:%B} {start.day}, {start.year}"
    elif start.year == end.year and start.month == end.month:
        period = f"{start:%B} {start.day}-{end.day}, {start.year}"
    elif start.year == end.year:
        period = f"{start:%B} {start.day}-{end:%B} {end.day}, {start.year}"
    else:
        period = f"{start:%B} {start.day}, {start.year}-{end:%B} {end.day}, {end.year}"
    return f"Immaculate Grid Analysis Report {period}"


def report_filename(start: pd.Timestamp, end: pd.Timestamp) -> str:
    if _is_complete_calendar_quarter(start, end):
        return f"immaculate_grid_quarterly_report_{start.year}_Q{start.quarter}.pdf"
    if (end - start).days + 1 == ROLLING_REPORT_DAYS:
        return f"immaculate_grid_quarterly_report_{end:%Y_%m_%d}.pdf"
    if _is_complete_calendar_month(start, end):
        return f"immaculate_grid_monthly_report_{start:%Y_%m}.pdf"
    return f"immaculate_grid_report_{start:%Y_%m_%d}_to_{end:%Y_%m_%d}.pdf"


def _prepare_results(texts_df: pd.DataFrame, report_end: pd.Timestamp) -> pd.DataFrame:
    required = {"grid_number", "name", "correct", "score", "date"}
    missing = required - set(texts_df.columns)
    if missing:
        raise ValueError(f"Monthly Snapshot results are missing columns: {sorted(missing)}")

    results = texts_df.copy()
    results["grid_number"] = pd.to_numeric(results["grid_number"], errors="coerce")
    results["correct"] = pd.to_numeric(results["correct"], errors="coerce")
    results["score"] = pd.to_numeric(results["score"], errors="coerce")
    results["date"] = pd.to_datetime(results["date"], errors="coerce").dt.normalize()
    results = results.dropna(subset=["grid_number", "name", "correct", "score", "date"])
    results = results[results["name"].isin(GRID_PLAYERS_RESTRICTED)]
    # Calendar reports follow the grid day, even when a result is shared later.
    results["date"] = results["grid_number"].map(lambda grid: pd.Timestamp(grid_to_date(int(grid))).normalize())
    results = results[results["date"] <= report_end]
    if results.empty:
        return results

    # Match analytics preprocessing: retain the highest score for duplicate person/grid rows.
    results = results.loc[results.groupby(["grid_number", "name"])["score"].idxmax()].copy()
    missed = 9 - results["correct"]
    results["avg_rarity_correct"] = np.where(
        results["correct"].eq(0),
        100.0,
        (results["score"] - (100 * missed)) / results["correct"],
    )
    return results.sort_values(["name", "date", "grid_number"])


def _moving_averages(results: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, group in results.groupby("name", sort=True):
        group = group.copy().sort_values(["date", "grid_number"])
        group["ma_score"] = group["score"].rolling(MOVING_AVERAGE_WINDOW).mean()
        group["ma_correct"] = group["correct"].rolling(MOVING_AVERAGE_WINDOW).mean()
        group["ma_rarity_correct"] = group["avg_rarity_correct"].rolling(MOVING_AVERAGE_WINDOW).mean()
        frames.append(group)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _immaculate_streaks(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize consecutive 9/9 grid IDs across all prepared history.

    Missing grids and non-9/9 results break a run. Current ends at each
    submitter's latest recorded grid; ties for longest use the latest run.
    """
    rows = []
    for name, group in results.groupby("name", sort=True):
        run = best = 0
        start = best_start = best_end = previous = None
        for row in group.sort_values("grid_number").itertuples(index=False):
            grid = int(row.grid_number)
            if row.correct == 9:
                if previous is None or grid != previous + 1 or run == 0:
                    run, start = 1, grid
                else:
                    run += 1
                if run >= best:
                    best, best_start, best_end = run, start, grid
            else:
                run = 0
            previous = grid
        dates = "-"
        if best:
            dates = (
                f"{pd.Timestamp(grid_to_date(best_start)):%m/%d/%y} - "
                f"{pd.Timestamp(grid_to_date(best_end)):%m/%d/%y}"
            )
        rows.append({"name": name, "longest": best, "longest dates": dates, "current": run})
    return pd.DataFrame(rows, columns=["name", "longest", "longest dates", "current"]).sort_values(
        ["longest", "name"], ascending=[False, True], ignore_index=True
    )


def _eight_week_starts(report_end: pd.Timestamp) -> pd.DatetimeIndex:
    final_week = report_end - pd.Timedelta(days=report_end.weekday())
    return pd.date_range(final_week - pd.Timedelta(weeks=WEEK_COUNT - 1), periods=WEEK_COUNT, freq="7D")


def _shame_incidents(
    images_df: pd.DataFrame,
    report_end: pd.Timestamp,
    report_start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if report_start is None:
        weeks = _eight_week_starts(report_end)
    else:
        first_week = report_start - pd.Timedelta(days=report_start.weekday())
        last_week = report_end - pd.Timedelta(days=report_end.weekday())
        weeks = pd.date_range(first_week, last_week, freq="7D")
    people = sorted(GRID_PLAYERS_RESTRICTED)
    if report_start is not None and not images_df.empty:
        images_df = images_df.copy()
        grid_dates = pd.to_numeric(images_df["grid_number"], errors="coerce").map(
            lambda grid: pd.Timestamp(grid_to_date(int(grid))) if pd.notna(grid) else pd.NaT
        )
        images_df = images_df[grid_dates.between(report_start, report_end)]
    shame = analyze_shame_index(images_df)
    columns = ["week", "submitter", "index", "players used"]
    if shame.empty:
        return pd.DataFrame(columns=columns)

    def _player_names(value: object, rule5: bool = False) -> list[str]:
        names = []
        for line in str(value or "").splitlines():
            name = re.sub(r"\s+\([^)]*\)\s*$", "", line).strip()
            if name:
                names.append(f"{name} [Banned]" if rule5 else f"{name} [Repeated]")
        return names

    shame = shame.copy()
    shame["week_start"] = pd.to_datetime(shame["week_start"], errors="coerce").dt.normalize()
    shame["shame_index"] = pd.to_numeric(shame["shame_index"], errors="coerce").fillna(0).astype(int)
    shame = shame[
        shame["week_start"].isin(weeks)
        & shame["submitter"].isin(people)
        & shame["shame_index"].gt(0)
    ].sort_values(["week_start", "submitter"], ascending=[False, True])

    rows = []
    for row in shame.itertuples(index=False):
        player_names = _player_names(row.repeated_players)
        player_names.extend(_player_names(row.rule5_banned_players, rule5=True))
        players_text = "; ".join(player_names) if player_names else "Unspecified"
        players_text = "\n".join(textwrap.wrap(players_text, width=85))
        rows.append(
            {
                "week": row.week_start.strftime("%b %-d"),
                "submitter": row.submitter,
                "index": row.shame_index,
                "players used": players_text,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _score_extremes(
    results: pd.DataFrame, report_start: pd.Timestamp, report_end: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["name", "score", "date", "grid_number"]
    period = results[results["date"].between(report_start, report_end)][columns].copy()
    period["date"] = period["date"].dt.strftime("%b %-d")
    best = period.nsmallest(5, ["score", "grid_number"]).reset_index(drop=True)
    worst = period.nlargest(5, ["score", "grid_number"]).reset_index(drop=True)
    return best, worst


def _recent_bans(report_end: pd.Timestamp, report_start: pd.Timestamp | None = None) -> pd.DataFrame:
    columns = ["player", "grid_number", "date", "response"]
    path = Path(RULE5_FULL_BANS_CSV_PATH)
    if not path.exists():
        return pd.DataFrame(columns=columns)
    bans = pd.read_csv(path)
    if bans.empty or not {"player", "grid_number", "response"}.issubset(bans.columns):
        return pd.DataFrame(columns=columns)
    bans = bans.copy()
    bans["grid_number"] = pd.to_numeric(bans["grid_number"], errors="coerce")
    bans = bans.dropna(subset=["grid_number", "player"])
    bans["grid_number"] = bans["grid_number"].astype(int)
    bans["date"] = bans["grid_number"].map(lambda value: pd.Timestamp(grid_to_date(value)).normalize())
    start = report_start if report_start is not None else _eight_week_starts(report_end)[0]
    bans = bans[bans["date"].between(start, report_end)]
    bans["date"] = bans["date"].dt.strftime("%b %-d")
    return bans[columns].sort_values(["grid_number", "player"], ascending=[False, True]).reset_index(drop=True)


def _normalize_player(value: object) -> str:
    """Match the normalization used by the existing bans-and-saves analysis."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", str(value).strip().lower()))


def _most_used_players(images_df: pd.DataFrame, report_end: pd.Timestamp) -> tuple[pd.DataFrame, int]:
    """Rank recorded player names by distinct cohort grids across all history."""
    columns = ["rank", "player", "grids", "share of grids"]
    if images_df.empty:
        return pd.DataFrame(columns=columns), 0
    images = images_df[images_df["submitter"].isin(GRID_PLAYERS_RESTRICTED)].copy()
    images["grid_number"] = pd.to_numeric(images["grid_number"], errors="coerce")
    images = images.dropna(subset=["grid_number"])
    images = images.drop_duplicates(["submitter", "grid_number"], keep="last")
    rows = []
    for row in images.itertuples(index=False):
        grid = int(row.grid_number)
        if pd.Timestamp(grid_to_date(grid)) > report_end or not isinstance(row.responses, dict):
            continue
        for value in row.responses.values():
            if not isinstance(value, str) or not value.strip():
                continue
            key = _normalize_player(value)
            if key:
                rows.append({"key": key, "player": re.sub(r"\s+", " ", value.strip()), "grid": grid})
    if not rows:
        return pd.DataFrame(columns=columns), 0
    cells = pd.DataFrame(rows)
    total = int(cells["grid"].nunique())
    counts = cells.groupby("key", as_index=False).agg(player=("player", "first"), grids=("grid", "nunique"))
    counts = counts.sort_values(["grids", "key"], ascending=[False, True]).reset_index(drop=True)
    counts["rank"] = counts["grids"].rank(method="min", ascending=False).astype(int)
    counts["share of grids"] = counts["grids"].map(lambda count: f"{count / total:.1%}")
    return counts[columns], total


def _intersection_prompt(value: object) -> str:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return re.sub(r"\s+", " ", value).strip() or "Unavailable"
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return " / ".join(re.sub(r"\s+", " ", str(part)).strip() for part in value)
    return "Unavailable"


def _save_metrics(
    images_df: pd.DataFrame,
    report_end: pd.Timestamp,
    report_start: pd.Timestamp | None = None,
    prompts_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build reporting-window save events and all-time player save totals.

    A save occurs when three members of the restricted four-person cohort use the
    same player in the same cell and the fourth submitted that grid but did not.
    Individual events cover the reporting window; player totals cover all history
    through report_end. Save significance still uses
    all history strictly before that grid: distinct prior grids featuring the MLB
    player divided by prior grids with at least one parsed cohort player name.
    """
    event_columns = [
        "player", "prompt", "saved by", "used instead", "grid", "date", "prior player grids",
        "prior grids", "save significance",
    ]
    player_columns = ["player", "saves", "last saved on", "save significance"]
    empty = (
        pd.DataFrame(columns=event_columns),
        pd.DataFrame(columns=player_columns),
    )
    if images_df.empty or not {"submitter", "grid_number", "responses"}.issubset(images_df.columns):
        return empty

    submissions = images_df.copy()
    submissions["grid_number"] = pd.to_numeric(submissions["grid_number"], errors="coerce")
    submissions = submissions.dropna(subset=["submitter", "grid_number"])
    submissions["grid_number"] = submissions["grid_number"].astype(int)
    # Grid IDs are the canonical date source throughout analytics preprocessing.
    submissions["date"] = submissions["grid_number"].map(
        lambda value: pd.Timestamp(grid_to_date(value)).normalize()
    )
    submissions = submissions[submissions["date"] <= report_end].copy()
    submissions = submissions.drop_duplicates(["submitter", "grid_number"], keep="last")
    if submissions.empty:
        return empty

    cell_rows = []
    for row in submissions.itertuples(index=False):
        responses = row.responses
        if not isinstance(responses, dict):
            continue
        for position, response in responses.items():
            normalized = _normalize_player(response)
            if not normalized:
                continue
            cell_rows.append(
                {
                    "submitter": row.submitter,
                    "grid_number": row.grid_number,
                    "date": row.date,
                    "position": position,
                    "player_key": normalized,
                    "player_display": re.sub(r"\s+", " ", str(response).strip()),
                }
            )
    cells = pd.DataFrame(cell_rows)
    if cells.empty:
        return empty
    cells = cells.drop_duplicates(["submitter", "grid_number", "position"], keep="last")

    # Pick a stable display spelling while retaining normalized matching semantics.
    display_names = (
        cells.groupby(["player_key", "player_display"], as_index=False)
        .size()
        .sort_values(["player_key", "size", "player_display"], ascending=[True, False, True])
        .drop_duplicates("player_key")
        .set_index("player_key")["player_display"]
        .to_dict()
    )
    cohort = set(GRID_PLAYERS_RESTRICTED)
    event_start = report_start if report_start is not None else _eight_week_starts(report_end)[0]
    restricted_submissions = submissions[submissions["submitter"].isin(cohort)]
    submitted_by_grid = restricted_submissions.groupby("grid_number")["submitter"].agg(set).to_dict()
    restricted_cells = cells[cells["submitter"].isin(cohort)]
    observed_grid_ids = set(restricted_cells["grid_number"])

    prompt_lookup = {}
    if prompts_df is not None and "grid_id" in prompts_df.columns:
        for record in prompts_df.to_dict("records"):
            grid = pd.to_numeric(record.get("grid_id"), errors="coerce")
            if pd.notna(grid):
                for position, value in record.items():
                    if position != "grid_id":
                        prompt_lookup[(int(grid), position)] = _intersection_prompt(value)
    raw_events = []
    for (grid_number, position, player_key), group in restricted_cells.groupby(
        ["grid_number", "position", "player_key"], sort=False
    ):
        users = set(group["submitter"])
        if len(users) != len(cohort) - 1 or submitted_by_grid.get(grid_number, set()) != cohort:
            continue
        saver = next(iter(cohort - users))
        event_date = pd.Timestamp(group["date"].min()).normalize()
        saver_cell = restricted_cells[
            restricted_cells["grid_number"].eq(grid_number)
            & restricted_cells["position"].eq(position)
            & restricted_cells["submitter"].eq(saver)
        ]
        used_instead = (
            str(saver_cell.iloc[-1]["player_display"])
            if not saver_cell.empty
            else "No answer"
        )
        prior_player_grids = int(
            restricted_cells.loc[
                restricted_cells["grid_number"].lt(grid_number)
                & restricted_cells["player_key"].eq(player_key),
                "grid_number",
            ].nunique()
        )
        # Use the same observed player-name population as the numerator.
        # Text-only days and screenshots without any parsed names do not count.
        prior_grids = sum(0 <= prior < grid_number for prior in observed_grid_ids)
        significance = prior_player_grids / prior_grids if prior_grids else 0.0
        raw_events.append(
            {
                "player": display_names[player_key],
                "prompt": prompt_lookup.get((int(grid_number), position), "Unavailable"),
                "saved by": saver,
                "used instead": used_instead,
                "grid": int(grid_number),
                "date": event_date,
                "prior player grids": prior_player_grids,
                "prior grids": prior_grids,
                "save significance": significance,
            }
        )

    events = pd.DataFrame(raw_events, columns=event_columns)
    if events.empty:
        players = pd.DataFrame(columns=player_columns)
    else:
        events = events.sort_values(
            ["save significance", "prior player grids", "grid", "player"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)
        latest = (
            events.sort_values(["grid", "player"])
            .groupby("player", as_index=False)
            .tail(1)
            .copy()
        )
        save_counts = events.groupby("player").size().rename("saves")
        latest["saves"] = latest["player"].map(save_counts).astype(int)
        latest["last saved on"] = latest.apply(
            lambda row: f"{int(row['grid'])} | {pd.Timestamp(row['date']):%b %-d %y}",
            axis=1,
        )
        latest["_significance_sort"] = latest["save significance"]
        latest["save significance"] = latest.apply(
            lambda row: (
                f"{int(row['prior player grids'])}/{int(row['prior grids'])} "
                f"({float(row['save significance']):.1%})"
            ),
            axis=1,
        )
        players = (
            latest.sort_values(["saves", "_significance_sort", "player"], ascending=[False, False, True])
            .reset_index(drop=True)
        )
        events = events[events["date"].between(event_start, report_end)].copy().reset_index(drop=True)
        if events.empty:
            return events[event_columns], players[player_columns]
        events["date"] = pd.to_datetime(events["date"]).dt.strftime("%b %-d %y")
        events["save significance"] = events.apply(
            lambda row: (
                f"{int(row['prior player grids'])}/{int(row['prior grids'])} "
                f"({float(row['save significance']):.1%})"
            ),
            axis=1,
        )
    return events[event_columns], players[player_columns]


def _style_axis(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=9.2, fontweight="bold", pad=5, color="#17243A")
    ax.set_ylabel(ylabel, fontsize=7, color="#526075")
    ax.tick_params(axis="both", labelsize=6.5, colors="#526075", length=2)
    ax.grid(axis="y", color="#DCE3ED", linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#BBC6D5")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))


def _draw_table(ax, title: str, frame: pd.DataFrame, col_widths=None, font_size: float = 6.2, fit_multiline: bool = False) -> None:
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=9.2, fontweight="bold", pad=5, color="#17243A")
    display = frame.copy()
    display.columns = [str(col).replace("grid_number", "grid").replace("name", "submitter").title() for col in display.columns]
    if display.empty:
        ax.text(0, 0.84, "None in this period", fontsize=7.5, color="#526075", va="top")
        return
    table = ax.table(
        cellText=display.astype(str).values,
        colLabels=display.columns,
        cellLoc="left",
        colLoc="left",
        colWidths=col_widths,
        bbox=[0, 0, 1, 0.92],
    )
    if fit_multiline:
        weights = [1.5] + [
            0.6 + max(str(value).count("\n") + 1 for value in row)
            for row in display.astype(str).values
        ]
        for (row, _), cell in table.get_celld().items():
            cell.set_height(weights[row] / sum(weights))
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#DCE3ED")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#E9EFF7")
            cell.set_text_props(weight="bold", color="#17243A")
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#F7F9FC")
            cell.set_text_props(color="#27364C")


def generate_monthly_snapshot_pdf(
    texts_df: pd.DataFrame,
    images_df: pd.DataFrame,
    output_path: str | Path,
    report_month: str | pd.Timestamp | None = None,
    *,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> Path:
    """Generate a rolling quarterly report, with optional legacy month/range modes."""
    if report_month is None and start_date is None:
        start_date, end_date = rolling_report_bounds(end_date)
    custom_range = start_date is not None or end_date is not None
    if custom_range:
        if start_date is None or end_date is None:
            raise ValueError("Both start_date and end_date are required for a custom report range.")
        report_start, report_end = _date_bounds(start_date, end_date)
    elif report_month is not None:
        report_start, report_end = _month_bounds(report_month)
    else:
        raise ValueError("Provide report_month or both start_date and end_date.")

    # Calendar-month mode retains the snapshot's original eight-week event context.
    event_start = report_start if custom_range else None
    period_label = "Selected Range" if custom_range else "Last 8 Weeks"
    score_period_label = "Selected Range" if custom_range else "Report Month"
    calendar_quarter = _is_complete_calendar_quarter(report_start, report_end)
    quarterly = calendar_quarter or (report_end - report_start).days + 1 >= ROLLING_REPORT_DAYS
    if quarterly:
        period_label = score_period_label = f"Q{report_start.quarter} {report_start.year}" if calendar_quarter else f"Last {(report_end - report_start).days + 1} Days"
    title = format_report_title(report_start, report_end)

    results = _prepare_results(texts_df, report_end)
    if results.empty:
        raise ValueError(f"No restricted-population results exist through {report_end:%B %Y}.")

    moving = _moving_averages(results)
    streaks = _immaculate_streaks(results)
    best, worst = _score_extremes(results, report_start, report_end)
    shame = _shame_incidents(images_df, report_end, event_start)
    bans = _recent_bans(report_end, event_start)
    prompts = pd.read_csv(PROMPTS_CSV_PATH) if PROMPTS_CSV_PATH.exists() else None
    save_events, saved_players = _save_metrics(images_df, report_end, event_start, prompts_df=prompts)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {name: GRID_PLAYERS.get(name, {}).get("color", "#2B6CB0") for name in GRID_PLAYERS_RESTRICTED}

    fig = plt.figure(figsize=(11, 8.5), facecolor="#F4F7FB")
    gs = fig.add_gridspec(
        4, 12,
        height_ratios=[0.38, 1.9, 2.3, 2.8],
        left=0.045, right=0.975, top=0.96, bottom=0.055,
        hspace=0.48, wspace=0.72,
    )

    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(
        0, 0.72,
        title,
        fontsize=19 if len(title) <= 65 else 16,
        fontweight="bold", color="#14233B", va="center",
    )
    first_date = results["date"].min()
    title_ax.text(
        0, 0.08,
        f"Restricted population  |  Trends: {first_date:%b %Y}-{report_end:%b %Y}  |  28-game moving averages",
        fontsize=7.8, color="#5A687B", va="center",
    )

    chart_specs = [
        ("ma_score", "Moving Average Grid Rarity", "Score (lower is better)"),
        ("ma_correct", "Moving Average Number Correct", "Correct (out of 9)"),
        ("ma_rarity_correct", "Moving Average Correct-Cell Rarity", "Rarity (lower is better)"),
    ]
    for index, (metric, title, ylabel) in enumerate(chart_specs):
        ax = fig.add_subplot(gs[1, index * 4:(index + 1) * 4])
        for name, group in moving.groupby("name", sort=True):
            group = group.dropna(subset=[metric])
            ax.plot(group["date"], group[metric], label=name, color=colors[name], linewidth=1.15)
        _style_axis(ax, title, ylabel)
        if index == 2:
            ax.legend(loc="best", fontsize=5.8, frameon=False, ncol=2, handlelength=1.4)

    shame_details = shame.copy()
    if quarterly:
        shame = shame.groupby("submitter", as_index=False).agg(
            incidents=("index", "sum"), weeks=("week", "nunique")
        ).set_index("submitter").reindex(sorted(results["name"].unique()), fill_value=0).reset_index()
        shame = shame.sort_values(["incidents", "submitter"], ascending=[False, True])
    shame_ax = fig.add_subplot(gs[2, 0:4])
    _draw_table(
        shame_ax,
        f"Shame Index - {period_label}",
        shame,
        col_widths=[0.45, 0.30, 0.25] if quarterly else [0.15, 0.19, 0.10, 0.56],
        font_size=7 if quarterly else 5.25,
    )

    scores_gs = gs[2, 4:8].subgridspec(2, 1, hspace=0.34)
    score_columns = ["name", "score", "date", "grid_number"]
    best_ax = fig.add_subplot(scores_gs[0, 0])
    _draw_table(best_ax, f"Top 5 Grid Scores - {score_period_label}", best[score_columns], [0.34, 0.18, 0.27, 0.18], 6.2)
    worst_ax = fig.add_subplot(scores_gs[1, 0])
    _draw_table(worst_ax, f"Bottom 5 Grid Scores - {score_period_label}", worst[score_columns], [0.34, 0.18, 0.27, 0.18], 6.2)

    records_gs = gs[2, 8:12].subgridspec(2, 1, height_ratios=[1.35, 1], hspace=0.65)
    streaks_ax = fig.add_subplot(records_gs[0, 0])
    _draw_table(
        streaks_ax, "Immaculate Streaks - All Time", streaks[["name", "longest", "longest dates"]],
        col_widths=[0.26, 0.20, 0.54], font_size=5.6,
    )
    streaks_ax.text(
        0, -0.10, "Consecutive grid IDs; missing grids break runs.",
        transform=streaks_ax.transAxes, fontsize=4.8, color="#526075", va="top",
    )

    bans_ax = fig.add_subplot(records_gs[1, 0])
    bans_display = bans[["date", "player", "grid_number", "response"]].copy()
    bans_display["response"] = bans_display["response"].map(
        lambda value: "\n".join(textwrap.wrap(str(value), width=31, max_lines=2, placeholder="..."))
    )
    _draw_table(
        bans_ax,
        f"Rule 5 Bans - {period_label}",
        bans_display[["date", "player", "grid_number"]] if quarterly else bans_display,
        col_widths=[0.22, 0.58, 0.20] if quarterly else [0.16, 0.25, 0.12, 0.47],
        font_size=5.25,
    )

    popular_saves_ax = fig.add_subplot(gs[3, :] if quarterly else gs[3, 0:6])
    popular_saves = save_events.head(8).copy()
    for column, width in [("prompt", 36), ("used instead", 19), ("player", 20)]:
        popular_saves[column] = popular_saves[column].map(
            lambda value: "\n".join(textwrap.wrap(str(value), width=width))
        )
    popular_saves["grid / date"] = popular_saves.apply(
        lambda row: f"{row['grid']} | {row['date']}", axis=1
    ) if not popular_saves.empty else pd.Series(dtype=str)
    _draw_table(
        popular_saves_ax,
        f"Most Popular Saves at Time of Save - {period_label}",
        popular_saves[["player", "prompt", "saved by", "used instead", "grid / date", "save significance"]],
        col_widths=[0.16, 0.29, 0.08, 0.16, 0.14, 0.17],
        font_size=6.4 if quarterly else 4.25, fit_multiline=True,
    )
    saved_players_display = saved_players.head(8).copy()
    if not quarterly:
        saved_players_ax = fig.add_subplot(gs[3, 6:12])
        _draw_table(saved_players_ax, "Players Saved Most - All Time",
                    saved_players_display, col_widths=[0.31, 0.11, 0.30, 0.28], font_size=5.15)

    fig.text(
        0.975, 0.018,
        f"Generated from local analytics data | Report period {report_start:%b %-d, %Y} to {report_end:%b %-d, %Y}",
        ha="right", fontsize=5.8, color="#6C788A",
    )

    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        if quarterly:
            leaders, observed_days = _most_used_players(images_df, report_end)
            leaders_fig = plt.figure(figsize=(11, 8.5), facecolor="#F4F7FB")
            leaders_fig.text(0.06, 0.94, "Players on the Most Grids | All Time", fontsize=18, weight="bold", color="#14233B")
            leaders_fig.text(0.06, 0.90, f"Through {report_end:%B %d, %Y} | {observed_days:,} grid days with parsed names | Top 20", fontsize=10, color="#526075")
            for column in range(2):
                leaders_ax = leaders_fig.add_axes([0.06 + column * 0.46, 0.43, 0.42, 0.42])
                _draw_table(leaders_ax, "", leaders.iloc[column * 10:(column + 1) * 10],
                            col_widths=[0.11, 0.49, 0.16, 0.24], font_size=8)
            saved_ax = leaders_fig.add_axes([0.06, 0.14, 0.88, 0.23])
            _draw_table(saved_ax, "Players Saved Most - All Time", saved_players_display,
                        col_widths=[0.31, 0.11, 0.30, 0.28], font_size=6.8)
            leaders_fig.text(0.06, 0.09, "One count per player name per grid, even if several people or cells use that name.\nShare = distinct grids featuring that name / grid days with at least one parsed name from the report group.\nAll available history is included; partial coverage counts. Shared names may combine different players.",
                             fontsize=7, color="#526075", va="top", linespacing=1.4)
            pdf.savefig(leaders_fig, facecolor=leaders_fig.get_facecolor())
            plt.close(leaders_fig)
        if quarterly and (not shame_details.empty or not bans_display.empty):
            # Keep the full 90-day weekly detail legible instead of squeezing it
            # into the summary dashboard. Every row is retained across pages.
            for offset in range(0, max(1, len(shame_details)), 16):
                last_page = offset + 16 >= len(shame_details)
                include_bans = last_page and not bans_display.empty
                detail_fig = plt.figure(figsize=(11, 8.5), facecolor="#F4F7FB")
                if include_bans:
                    detail_grid = detail_fig.add_gridspec(2, 1, height_ratios=[max(5, len(shame_details) - offset), max(4, len(bans_display))], hspace=0.3)
                    detail_ax = detail_fig.add_subplot(detail_grid[0])
                    bans_detail_ax = detail_fig.add_subplot(detail_grid[1])
                    _draw_table(bans_detail_ax, f"Rule 5 Bans - {period_label}", bans_display,
                                col_widths=[0.13, 0.22, 0.10, 0.55], font_size=7)
                else:
                    detail_ax = detail_fig.add_subplot(111)
                detail_fig.subplots_adjust(left=0.06, right=0.94, top=0.88, bottom=0.08)
                if include_bans:
                    detail_ax.set_position([0.06, 0.55, 0.88, 0.32])
                    bans_detail_ax.set_position([0.06, 0.08, 0.88, 0.34])
                _draw_table(
                    detail_ax,
                    f"Shame Index Details | {report_start:%b %d} - {report_end:%b %d, %Y}",
                    shame_details.iloc[offset:offset + 16],
                    col_widths=[0.12, 0.15, 0.08, 0.65], font_size=8, fit_multiline=True,
                )
                detail_fig.text(0.94, 0.025, f"{period_label} report | Detail page {offset // 16 + 1}", ha="right", fontsize=6, color="#526075")
                pdf.savefig(detail_fig, facecolor=detail_fig.get_facecolor())
                plt.close(detail_fig)
        if quarterly:
            definitions = [
                ("Reporting window", f"Scores, saves, bans, and Shame Index cover {report_start:%B %d, %Y} through {report_end:%B %d, %Y}, inclusive ({(report_end - report_start).days + 1} days). Trend charts and longest immaculate streaks use all available history through the end date."),
                ("What counts as a save?", "All four report participants must have submitted the grid. Three use the same player in the same cell; the fourth uses someone else or leaves it blank. The fourth participant gets the save. Saves are counted per player, cell, and grid."),
                ("Save significance", "Prior player grids / prior grids, expressed as a percentage. Prior player grids counts distinct earlier grid IDs where at least one report participant used that player in any cell. Multiple uses on one grid count once. Prior grids counts distinct earlier grid days with at least one parsed player name from a report participant. Text-only days, days without screenshots, and screenshots with no parsed names are excluded. Multiple submitters on one day count once; partial group coverage is sufficient. The save grid itself and all later grids are excluded."),
                ("How to read it", "Example: Tom Seaver appeared on 121 of 956 earlier grid days with parsed player names, giving 12.7% save significance. A higher percentage means a more historically common player was saved; it is not a probability or a measure of statistical significance. Coverage may be partial: a qualifying day need not have names for every participant. Missing screenshots can still lower the numerator."),
                ("Save tables", "Most Popular Saves ranks individual save events by significance at the time of the save. Players Saved Most counts all saves through the report end date; its Last Saved On and Save Significance describe that player's latest save through that date. Each summary table shows its top eight entries."),
                ("Shame Index", "Within each Monday-Sunday week, each use of a repeated player after the first adds one point. Each player used on a grid after their Rule 5 ban adds one more point. Weeks counts weeks with at least one point. Boundary weeks include only grids inside the report window."),
                ("Same-name exception: Frank Thomas", "Two uses on the same grid establish the two distinct Frank Thomases, so that pair adds no repeat point. Further uses in the week still count as repeats. Without a same-grid pair, uses on separate grids remain ambiguous and are still flagged. This exception does not remove Rule 5 ban checks."),
            ]
            definition_fig = plt.figure(figsize=(11, 8.5), facecolor="#F4F7FB")
            definition_fig.text(0.06, 0.94, "Appendix | Metric Definitions", fontsize=18, weight="bold", color="#14233B")
            y = 0.87
            for heading, body in definitions:
                definition_fig.text(0.06, y, heading, fontsize=10, weight="bold", color="#17243A", va="top")
                lines = textwrap.wrap(body, width=126)
                definition_fig.text(0.06, y - 0.027, "\n".join(lines), fontsize=9, color="#27364C", va="top", linespacing=1.35)
                y -= 0.027 + len(lines) * 0.020 + 0.025
            pdf.savefig(definition_fig, facecolor=definition_fig.get_facecolor())
            plt.close(definition_fig)
    plt.close(fig)
    return output_path
