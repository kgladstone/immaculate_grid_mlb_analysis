from __future__ import annotations

import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import pytest

from app.services.monthly_snapshot import (
    _date_bounds,
    rolling_report_bounds,
    _eight_week_starts,
    _immaculate_streaks,
    _prepare_results,
    _save_metrics,
    _shame_incidents,
    format_report_title,
    generate_monthly_snapshot_pdf,
    report_filename,
)
from scripts import generate_monthly_snapshot as monthly_snapshot_script


def _sample_results() -> pd.DataFrame:
    rows = []
    for day, date in enumerate(pd.date_range("2024-01-01", periods=62), start=1):
        for offset, name in enumerate(("Keith", "Sam", "Will", "Rachel", "Cliff")):
            correct = 9 - ((day + offset) % 3)
            rows.append(
                {
                    "grid_number": day,
                    "name": name,
                    "correct": correct,
                    "score": 40 + (offset * 30) + day,
                    "date": date,
                }
            )
    return pd.DataFrame(rows)


def test_cli_rebuilds_responses_from_derived_fuzzy_metadata(tmp_path, monkeypatch):
    path = tmp_path / "images_metadata.csv"
    pd.DataFrame(
        [
            {
                "submitter": "Sam",
                "date": "2026-07-04",
                "grid_number": 1188,
                "image_filename": "sam.jpg",
                "position": "middle_left",
                "response": "Curt Schilling",
            },
            {
                "submitter": "Sam",
                "date": "2026-07-04",
                "grid_number": 1188,
                "image_filename": "sam.jpg",
                "position": "middle_center",
                "response": None,
            },
        ]
    ).to_csv(path, index=False)
    monkeypatch.setattr(monthly_snapshot_script, "IMAGES_METADATA_CSV_PATH", path)

    images = monthly_snapshot_script._load_derived_images()

    assert images.loc[0, "responses"] == {
        "middle_left": "Curt Schilling",
        "middle_center": "",
    }


def test_prepare_results_uses_only_restricted_population_and_report_cutoff():
    prepared = _prepare_results(_sample_results(), pd.Timestamp("2024-02-29"))

    assert "Cliff" not in set(prepared["name"])
    assert prepared["date"].max() == pd.Timestamp("2024-02-29")
    assert prepared["avg_rarity_correct"].notna().all()


def test_immaculate_streaks_use_grid_order_break_on_gaps_and_count_corrected_results():
    # Submission dates/order need not match grid order. Two tied longest runs
    # are separated by a missing grid; the latest run supplies the date range.
    rows = [
        {"name": "Will", "grid_number": grid, "correct": correct}
        for grid, correct in [(1274, 8), (1273, 9), (1272, 9), (1270, 9), (1269, 9)]
    ]
    rows += [{"name": "Sam", "grid_number": 1274, "correct": 9}]
    rows += [{"name": "Rachel", "grid_number": 1274, "correct": 7}]
    result = _immaculate_streaks(pd.DataFrame(rows)).set_index("name")
    assert result.loc["Will", "longest"] == 2
    assert result.loc["Will", "current"] == 0
    assert result.loc["Will", "longest dates"] == "09/25/26 - 09/26/26"
    assert result.loc["Sam", "longest"] == result.loc["Sam", "current"] == 1
    assert result.loc["Rachel", "longest"] == result.loc["Rachel", "current"] == 0
    assert result.loc["Rachel", "longest dates"] == "-"


def test_immaculate_streaks_respect_prepared_history_cutoff():
    data = pd.DataFrame([
        {"name": "Keith", "grid_number": grid, "correct": 9, "score": 20, "date": date}
        for grid, date in [(1272, "2026-09-25"), (1273, "2026-09-26"), (1274, "2026-09-27")]
    ])
    result = _immaculate_streaks(_prepare_results(data, pd.Timestamp("2026-09-26")))
    assert result.iloc[0]["longest"] == result.iloc[0]["current"] == 2


def test_immaculate_streaks_empty_input():
    result = _immaculate_streaks(pd.DataFrame(columns=["name", "grid_number", "correct"]))
    assert result.empty
    assert list(result.columns) == ["name", "longest", "longest dates", "current"]


def test_report_title_and_filename_are_deterministic_for_months_and_ranges():
    month_start = pd.Timestamp("2026-08-01")
    month_end = pd.Timestamp("2026-08-31")
    range_start = pd.Timestamp("2026-06-08")
    range_end = pd.Timestamp("2026-07-31")

    assert format_report_title(month_start, month_end) == "Immaculate Grid Analysis Monthly Report August 2026"
    assert report_filename(month_start, month_end) == "immaculate_grid_monthly_report_2026_08.pdf"
    assert format_report_title(range_start, range_end) == "Immaculate Grid Analysis Report June 8-July 31, 2026"
    assert report_filename(range_start, range_end) == "immaculate_grid_report_2026_06_08_to_2026_07_31.pdf"


def test_custom_range_validation_preserves_single_page_limit():
    assert _date_bounds("2026-07-01", "2026-07-31") == (
        pd.Timestamp("2026-07-01"),
        pd.Timestamp("2026-07-31"),
    )
    with pytest.raises(ValueError, match="90 days"):
        _date_bounds("2026-06-01", "2026-09-01")


def test_eight_week_window_has_exactly_eight_monday_buckets():
    weeks = _eight_week_starts(pd.Timestamp("2026-07-31"))

    assert list(weeks) == list(pd.date_range("2026-06-08", periods=8, freq="7D"))


def test_shame_incidents_include_repeated_and_rule5_player_names(monkeypatch):
    shame = pd.DataFrame(
        [
            {
                "submitter": "Rachel",
                "week_start": "2026-07-20",
                "shame_index": 2,
                "repeated_players": "Tyler Clippard (1205, 1209)",
                "rule5_banned_players": "Alex Bregman (banned 1041; used 1208)",
            }
        ]
    )
    monkeypatch.setattr("app.services.monthly_snapshot.analyze_shame_index", lambda _: shame)

    incidents = _shame_incidents(pd.DataFrame(), pd.Timestamp("2026-07-31"))

    assert incidents.loc[0, "players used"] == "Tyler Clippard; Alex Bregman [R5]"
    assert incidents.loc[0, "index"] == 2


def test_save_metrics_use_only_history_before_each_save(monkeypatch):
    monkeypatch.setattr(
        "app.services.monthly_snapshot.GRID_PLAYERS_RESTRICTED",
        {name: {} for name in ("A", "B", "C", "D")},
    )
    rows = []
    for grid in (10, 15, 20, 30):
        for name in ("A", "B", "C", "D"):
            responses = {"top_left": f"Unique {name} {grid}"}
            if grid == 10 and name in ("A", "B"):
                responses["top_center"] = "Popular Player"
            if grid == 20 and name in ("A", "B", "C"):
                responses["top_left"] = "Popular Player"
            if grid == 30:
                responses["bottom_right"] = "Popular Player"
            rows.append(
                {
                    "submitter": name,
                    "grid_number": grid,
                    "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=grid),
                    "responses": responses,
                }
            )
    events, players = _save_metrics(pd.DataFrame(rows), pd.Timestamp("2023-05-31"))

    assert len(events) == 1
    assert events.loc[0, "player"] == "Popular Player"
    assert events.loc[0, "saved by"] == "D"
    assert events.loc[0, "used instead"] == "Unique D 20"
    assert events.loc[0, "prior player grids"] == 1
    assert events.loc[0, "prior grids"] == 2
    assert events.loc[0, "save significance"] == "1/2 (50.0%)"
    assert players.to_dict("records") == [
        {
            "player": "Popular Player",
            "saves": 1,
            "last saved on": "20 | Apr 22 23",
            "save significance": "1/2 (50.0%)",
        }
    ]


def test_save_metrics_require_the_saver_to_have_submitted_the_grid(monkeypatch):
    monkeypatch.setattr(
        "app.services.monthly_snapshot.GRID_PLAYERS_RESTRICTED",
        {name: {} for name in ("A", "B", "C", "D")},
    )
    images = pd.DataFrame(
        [
            {"submitter": name, "grid_number": 10, "responses": {"top_left": "Same Player"}}
            for name in ("A", "B", "C")
        ]
    )

    events, _ = _save_metrics(images, pd.Timestamp("2024-12-31"))

    assert events.empty


def test_save_metrics_limit_events_to_eight_weeks_but_keep_all_prior_usage(monkeypatch):
    monkeypatch.setattr(
        "app.services.monthly_snapshot.GRID_PLAYERS_RESTRICTED",
        {name: {} for name in ("A", "B", "C", "D")},
    )
    rows = []
    grid_dates = {
        10: pd.Timestamp("2026-01-01"),
        20: pd.Timestamp("2026-01-11"),
        100: pd.Timestamp("2026-04-01"),
        200: pd.Timestamp("2026-07-20"),
    }
    for grid, date in grid_dates.items():
        for name in ("A", "B", "C", "D"):
            response = f"Unique {name} {grid}"
            if grid == 10 and name in ("A", "B"):
                response = "Popular Player"
            if grid in (20, 200) and name in ("A", "B", "C"):
                response = "Popular Player"
            rows.append(
                {
                    "submitter": name,
                    "grid_number": grid,
                    "date": date,
                    "responses": {"top_left": response},
                }
            )

    events, players = _save_metrics(pd.DataFrame(rows), pd.Timestamp("2023-10-31"))

    assert events["grid"].tolist() == [200]
    assert events.loc[0, "used instead"] == "Unique D 200"
    assert events.loc[0, "prior player grids"] == 2
    assert events.loc[0, "prior grids"] == 3
    assert events.loc[0, "save significance"] == "2/3 (66.7%)"
    assert players.loc[0, "saves"] == 1


def test_monthly_snapshot_is_a_single_landscape_pdf(tmp_path):
    output = tmp_path / "snapshot.pdf"
    images = pd.DataFrame(columns=["submitter", "grid_number", "responses"])

    generate_monthly_snapshot_pdf(_sample_results(), images, output, "2024-02")

    pdf_bytes = output.read_bytes()
    assert pdf_bytes.startswith(b"%PDF")
    assert b"/MediaBox [ 0 0 792 612 ]" in pdf_bytes
    assert b"/Count 1" in pdf_bytes


def test_rolling_quarter_uses_exactly_90_inclusive_days():
    start, end = rolling_report_bounds("2026-09-28")
    assert start == pd.Timestamp("2026-07-01")
    assert end == pd.Timestamp("2026-09-28")
    assert _date_bounds(start, end) == (start, end)
    assert "Quarterly" in format_report_title(start, end)
    assert report_filename(start, end) == "immaculate_grid_quarterly_report_2026_09_28.pdf"
    leap_start, leap_end = rolling_report_bounds("2024-03-01")
    assert (leap_end - leap_start).days + 1 == 90


def test_shame_window_excludes_incidents_outside_exact_dates(monkeypatch):
    from analytics.analysis import grid_to_date
    images = pd.DataFrame({"grid_number": [1184, 1185, 1274, 1275]})
    seen = []
    def capture(frame):
        seen.extend(frame.grid_number.tolist())
        return pd.DataFrame()
    monkeypatch.setattr("app.services.monthly_snapshot.analyze_shame_index", capture)
    _shame_incidents(images, pd.Timestamp(grid_to_date(1274)), pd.Timestamp(grid_to_date(1185)))
    assert seen == [1185, 1274]



def test_save_denominator_requires_names_and_counts_partial_days_once(monkeypatch):
    monkeypatch.setattr("app.services.monthly_snapshot.GRID_PLAYERS_RESTRICTED", {n: {} for n in "ABCD"})
    images = pd.DataFrame([
        {"submitter": "A", "grid_number": 10, "responses": {"top_left": "Saved Player"}},
        {"submitter": "B", "grid_number": 10, "responses": {"top_left": "Other"}},
        {"submitter": "A", "grid_number": 12, "responses": {"top_left": "Other"}},
        {"submitter": "B", "grid_number": 15, "responses": {"top_left": "", "top_right": "  "}},
        {"submitter": "Outsider", "grid_number": 16, "responses": {"top_left": "Other"}},
        {"submitter": "A", "grid_number": 30, "responses": {"top_left": "Saved Player"}},
        *[{"submitter": n, "grid_number": 20, "responses": {"top_left": "Saved Player" if n != "D" else "Other"}} for n in "ABCD"],
    ])
    events, _ = _save_metrics(images, pd.Timestamp("2023-05-01"))
    assert events.iloc[0]["prior player grids"] == 1
    assert events.iloc[0]["prior grids"] == 2
    assert events.iloc[0]["save significance"] == "1/2 (50.0%)"
