from __future__ import annotations

import ast
import json
from pathlib import Path
from functools import lru_cache
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rapidfuzz import process
import streamlit as st

from data.io.mlb_reference import clean_name, load_mlb_player_names
from config.constants import MESSAGES_CSV_PATH


def _safe_lahman_table(table_name: str) -> pd.DataFrame:
    try:
        import pybaseball.lahman as lahman
    except ImportError:
        return pd.DataFrame()

    loader = getattr(lahman, table_name, None)
    if loader is None:
        return pd.DataFrame()
    try:
        df = loader()
    except Exception:
        return pd.DataFrame()
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _parse_responses(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            out = json.loads(value)
            return out if isinstance(out, dict) else {}
        except Exception:
            try:
                out = ast.literal_eval(value)
                return out if isinstance(out, dict) else {}
            except Exception:
                return {}
    return {}


def _normalize_name(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def _normalize_grid_number(value) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


@lru_cache(maxsize=1)
def _known_player_names() -> tuple[str, ...]:
    names = sorted(load_mlb_player_names())
    return tuple(names)


@lru_cache(maxsize=20000)
def _canonicalize_player_name_for_ui(raw_name: str) -> str:
    raw_text = str(raw_name).strip()

    name = clean_name(raw_text)
    if not name:
        return ""
    # Strip OCR-leading/trailing punctuation noise.
    name = re.sub(r"^[\.\-'\s]+", "", name)
    name = re.sub(r"[\.\-'\s]+$", "", name)
    # Strip obvious OCR singleton tokens at boundaries (e.g., "Z Mike Trout", "Barry Larkin Y").
    name = re.sub(r"^[A-Za-z]\s+", "", name)
    name = re.sub(r"\s+[A-Za-z]$", "", name)
    if not name:
        return ""

    known = _known_player_names()
    known_set = set(known)
    if name in known_set:
        return name

    # Common OCR artifact: leading stray character token (e.g., "Z Ian Happ" -> "Ian Happ")
    parts = name.split()
    if len(parts) >= 3 and len(parts[0]) == 1:
        trimmed = " ".join(parts[1:])
        if trimmed in known_set:
            return trimmed
        name = trimmed

    best = process.extractOne(name, known, score_cutoff=88)
    canonical = str(best[0]) if best else name
    return canonical


def _build_usage_df(images_df: pd.DataFrame) -> pd.DataFrame:
    if images_df.empty or "responses" not in images_df.columns or "submitter" not in images_df.columns:
        return pd.DataFrame(columns=["submitter", "player", "submission_date", "grid_number"])

    rows = []
    for _, row in images_df.iterrows():
        submitter = row.get("submitter")
        responses = _parse_responses(row.get("responses"))
        raw_date = row.get("date") if pd.notna(row.get("date")) else row.get("image_date")
        submission_date = pd.to_datetime(raw_date, errors="coerce")
        grid_number = row.get("grid_number")
        if not submitter or not responses:
            continue
        for val in responses.values():
            player = _canonicalize_player_name_for_ui(str(val).strip())
            if player:
                rows.append(
                    {
                        "submitter": str(submitter),
                        "player": player,
                        "submission_date": submission_date,
                        "grid_number": grid_number,
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["submitter", "player", "submission_date", "grid_number"])
    out = pd.DataFrame(rows)
    known_set = set(_known_player_names())
    # Only enforce canonical whitelist when we actually have a canonical list loaded.
    # If cache is not built yet, keep parsed rows so the UI is still usable.
    if known_set:
        out = out[out["player"].isin(known_set)].copy()
    out["player_norm"] = out["player"].map(_normalize_name)
    return out


@lru_cache(maxsize=1)
def _load_median_year_lookup(cache_dir: Path = Path("bin/baseball_cache")) -> dict[str, int]:
    cache_dir = cache_dir if isinstance(cache_dir, Path) else Path(cache_dir)
    appearances_path = cache_dir / "appearances.csv"
    if not appearances_path.exists():
        return {}
    try:
        appearances = pd.read_csv(appearances_path, usecols=["playerID", "yearID"], dtype=str)
    except Exception:
        return {}

    appearances = appearances.dropna(subset=["playerID", "yearID"]).copy()
    if appearances.empty:
        return {}
    appearances["playerID"] = appearances["playerID"].astype(str).str.strip()
    appearances["yearID"] = pd.to_numeric(appearances["yearID"], errors="coerce")
    appearances = appearances.dropna(subset=["yearID"]).copy()
    if appearances.empty:
        return {}
    appearances["yearID"] = appearances["yearID"].round()

    median_years = (
        appearances.groupby("playerID", as_index=False)["yearID"]
        .median()
    )
    if median_years.empty:
        return {}
    def _round_year(value):
        if pd.isna(value):
            return pd.NA
        try:
            return int(round(float(value)))
        except Exception:
            return pd.NA

    median_years["yearID"] = median_years["yearID"].map(_round_year).astype("Int64")
    median_by_player = {str(pid).strip(): int(year) for pid, year in median_years.set_index("playerID")["yearID"].dropna().items()}

    player_file = cache_dir / "People.csv"
    if not player_file.exists():
        return {}
    try:
        players = pd.read_csv(player_file, dtype=str)
    except Exception:
        return {}

    id_col = next(
        (col for col in players.columns if col.lower() in {"playerid", "key_bbref"}),
        None,
    )
    first_col = next(
        (col for col in players.columns if col.lower() in {"namefirst", "name_first"}),
        None,
    )
    last_col = next(
        (col for col in players.columns if col.lower() in {"namelast", "name_last"}),
        None,
    )
    if not id_col or not first_col or not last_col:
        return {}

    players = players[[id_col, first_col, last_col]].copy()
    players.columns = ["player_id", "name_first", "name_last"]
    players["player_id"] = players["player_id"].astype(str).str.strip()
    players["name_first"] = players["name_first"].fillna("").astype(str).str.strip()
    players["name_last"] = players["name_last"].fillna("").astype(str).str.strip()

    lookup: dict[str, int] = {}
    for _, row in players.iterrows():
        pid = row.get("player_id")
        if not pid:
            continue
        median_year = median_by_player.get(pid)
        if median_year is None:
            continue
        full_name = clean_name(f"{row.get('name_first', '')} {row.get('name_last', '')}")
        if not full_name:
            continue
        norm = _normalize_name(full_name)
        if norm:
            lookup[norm] = median_year
    return lookup


@lru_cache(maxsize=1)
def _load_career_war_lookup(cache_dir: Path = Path("bin/baseball_cache")) -> dict[str, float]:
    war_cache_path = cache_dir / "war.csv"
    if not war_cache_path.exists():
        return {}
    try:
        war_df = pd.read_csv(war_cache_path, dtype=str)
    except Exception:
        return {}
    if "player_norm" not in war_df.columns or "career_war" not in war_df.columns:
        return {}
    war_df["career_war"] = pd.to_numeric(war_df["career_war"], errors="coerce")
    war_df = war_df.dropna(subset=["player_norm", "career_war"])
    if war_df.empty:
        return {}
    war_df["player_norm"] = war_df["player_norm"].astype(str).map(_normalize_name)
    war_df = war_df[war_df["player_norm"] != ""]
    war_lookup = dict(zip(war_df["player_norm"], war_df["career_war"]))
    return war_lookup


@lru_cache(maxsize=1)
def _load_grid_rarity_scores() -> pd.DataFrame:
    path = Path(MESSAGES_CSV_PATH)
    if not path.exists():
        return pd.DataFrame(columns=["submitter", "grid_number", "rarity_score"])
    try:
        df = pd.read_csv(path, usecols=["name", "grid_number", "score"])
    except Exception:
        return pd.DataFrame(columns=["submitter", "grid_number", "rarity_score"])

    df = df.rename(columns={"name": "submitter"})
    df["submitter"] = df["submitter"].astype(str).str.strip()
    df["grid_number"] = df["grid_number"].map(_normalize_grid_number)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["submitter", "grid_number", "score"])
    if df.empty:
        return pd.DataFrame(columns=["submitter", "grid_number", "rarity_score"])

    # Lower score = rarer / better rarity outcome; keep each submitter-grid's best score.
    out = (
        df.groupby(["submitter", "grid_number"], as_index=False)["score"]
        .min()
        .rename(columns={"score": "rarity_score"})
    )
    return out


def _render_player_search_chart(usage_df: pd.DataFrame) -> None:
    try:
        import plotly.express as px
    except ImportError:
        st.error("Plotly is required for the interactive player trend chart. Install with: `pip install plotly`.")
        return

    st.markdown("### Player Search")
    player_query = st.text_input(
        "Type to filter player dropdown",
        value="",
        key="player_search_query",
        placeholder="e.g., griffey",
    ).strip().lower()

    all_players = sorted(usage_df["player"].dropna().astype(str).unique().tolist())
    if player_query:
        candidates = [p for p in all_players if player_query in p.lower()]
    else:
        candidates = all_players[:100]

    if not candidates:
        st.info("No matching player names.")
        return

    selected_player = st.selectbox(
        "Select player",
        options=candidates,
        key="player_search_select",
    )

    # Global usage summary for this player across all submitters.
    player_all_rows = usage_df[usage_df["player"] == selected_player].copy()
    total_uses = int(len(player_all_rows))
    st.metric("Total Uses", f"{total_uses:,}")
    by_submitter = (
        player_all_rows.groupby("submitter", as_index=False)
        .size()
        .rename(columns={"size": "uses"})
        .sort_values(["uses", "submitter"], ascending=[False, True])
    )
    if not by_submitter.empty:
        st.caption("Breakdown by submitter")
        st.dataframe(by_submitter, use_container_width=True, hide_index=True)

    player_rows = usage_df[
        (usage_df["player"] == selected_player) & usage_df["submission_date"].notna()
    ].copy()
    if player_rows.empty:
        st.info("No dated submissions found for this player.")
        return

    submitter_options = sorted(player_rows["submitter"].dropna().astype(str).unique().tolist())
    submitter_filter_key = "player_search_submitter_filter"
    submitter_filter_player_key = "player_search_submitter_filter_player"

    if (
        submitter_filter_key not in st.session_state
        or st.session_state.get(submitter_filter_player_key) != selected_player
    ):
        st.session_state[submitter_filter_key] = submitter_options
        st.session_state[submitter_filter_player_key] = selected_player

    selected_submitters = [
        s for s in st.session_state.get(submitter_filter_key, []) if s in submitter_options
    ]
    if not selected_submitters:
        selected_submitters = submitter_options

    player_rows = player_rows[player_rows["submitter"].isin(selected_submitters)].copy()
    if player_rows.empty:
        st.info("No data for selected submitter filter.")
        return

    trend = (
        player_rows.assign(month=player_rows["submission_date"].dt.to_period("M").dt.to_timestamp())
        .groupby(["month", "submitter"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["month", "submitter"])
    )

    fig = px.area(
        trend,
        x="month",
        y="count",
        color="submitter",
        title=f"Monthly Usage Over Time (Stacked Area): {selected_player}",
        labels={"month": "Month", "count": "Times Used"},
    )
    fig.update_xaxes(dtick="M1", tickformat="%Y-%m")
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    trend_table = trend.copy()
    trend_table["month"] = trend_table["month"].dt.strftime("%b %Y")
    st.dataframe(trend_table, use_container_width=True)
    st.multiselect(
        "Filter submitters (applies to chart and table)",
        options=submitter_options,
        default=selected_submitters,
        key=submitter_filter_key,
        help="This filter is intentionally placed at the bottom of this graph section.",
    )


def _render_mosaic_style_plot(usage_df: pd.DataFrame) -> None:
    try:
        import plotly.express as px
    except ImportError:
        st.error("Plotly is required for the interactive mosaic chart. Install with: `pip install plotly`.")
        return

    st.markdown("### Tableau-Style Mosaic (Single Submitter)")
    submitters = sorted(usage_df["submitter"].dropna().astype(str).unique().tolist())
    if not submitters:
        st.info("No submitters available for mosaic chart.")
        return

    selected_submitter = st.selectbox(
        "Submitter",
        options=submitters,
        key="mosaic_submitter",
    )
    top_n = st.slider("Top players to include", min_value=5, max_value=30, value=12, step=1, key="mosaic_top_n")

    user_df = usage_df[usage_df["submitter"] == selected_submitter].copy()
    if user_df.empty:
        st.info("No data available for mosaic-style chart.")
        return

    dated_df = user_df.dropna(subset=["submission_date"]).copy()
    if not dated_df.empty:
        min_date = dated_df["submission_date"].min().date()
        max_date = dated_df["submission_date"].max().date()
        selected_date_range = st.slider(
            "Submission date range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            key="mosaic_date_range",
        )
        start_date, end_date = selected_date_range
        user_df = user_df[
            user_df["submission_date"].dt.date.between(start_date, end_date, inclusive="both")
            | user_df["submission_date"].isna()
        ].copy()

    if user_df.empty:
        st.info("No player usages for the selected submitter/date range.")
        return

    counts = user_df["player"].value_counts()
    top_counts = counts.head(top_n).copy()

    total = int(top_counts.sum())
    mosaic_df = (
        top_counts.rename_axis("player")
        .reset_index(name="count")
        .assign(
            submitter=selected_submitter,
            pct=lambda d: 100.0 * d["count"] / max(total, 1),
        )
    )

    fig = px.treemap(
        mosaic_df,
        path=["submitter", "player"],
        values="count",
        color="pct",
        color_continuous_scale="Blues",
        custom_data=["count", "pct"],
    )
    fig.update_traces(
        texttemplate="%{label}<br>%{customdata[1]:.1f}%",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Uses: %{customdata[0]}<br>"
            "Share: %{customdata[1]:.2f}%<extra></extra>"
        ),
        root_color="lightgray",
    )
    fig.update_layout(
        margin=dict(t=40, l=10, r=10, b=10),
        coloraxis_colorbar_title="% Share",
        title=f"Player Usage Mosaic: {selected_submitter}",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def _render_median_year_histogram(usage_df: pd.DataFrame, median_lookup: dict[str, int]) -> None:
    st.markdown("### Median Year Histogram by User (Usage Weighted)")
    if not median_lookup:
        st.warning(
            "Median-year histograms rely on `bin/baseball_cache` (run `python src/scripts/build_baseball_cache.py`) and the refreshed responses in `bin/csv/images_metadata.csv`."
        )
        return

    merged_all = usage_df.assign(median_year=usage_df["player_norm"].map(median_lookup))
    matched = merged_all["median_year"].notna().sum()
    st.caption(f"Mapped median year for {matched:,}/{len(merged_all):,} player usages.")

    merged = merged_all.dropna(subset=["median_year"]).copy()
    if merged.empty:
        st.info("No player usages could be mapped to median year.")
        return
    merged["median_year"] = merged["median_year"].astype(int)

    submitters = sorted(merged["submitter"].unique().tolist())
    n = len(submitters)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, max(3.5 * rows, 4)))
    axes = np.array(axes).reshape(-1)

    bins = np.arange(1870, 2031, 5)
    year_ticks = np.arange(bins[0], bins[-1] + 1, 10)
    for i, submitter in enumerate(submitters):
        ax = axes[i]
        vals = merged.loc[merged["submitter"] == submitter, "median_year"]
        if vals.empty:
            ax.set_visible(False)
            continue
        weights = np.ones(len(vals)) * (100.0 / max(len(vals), 1))
        ax.hist(vals, bins=bins, weights=weights, color="#e76f51", alpha=0.85, edgecolor="white")
        ax.set_title(submitter)
        ax.set_xlabel("Median Year")
        ax.set_ylabel("Freq %")
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xticks(year_ticks)
        ax.tick_params(axis="x", rotation=45, labelsize=8)

    for extra_ax in axes[len(submitters):]:
        extra_ax.axis("off")

    fig.suptitle("Distribution of Players' Median Years by Submitter", y=1.01)
    fig.tight_layout()
    st.pyplot(fig)


def _render_median_year_histogram_unique_players(usage_df: pd.DataFrame, median_lookup: dict[str, int]) -> None:
    st.markdown("### Median Year Histogram by Submitter (Unique Players)")
    if not median_lookup:
        st.warning(
            "Median-year histograms rely on `bin/baseball_cache` (run `python src/scripts/build_baseball_cache.py`) and the refreshed responses in `bin/csv/images_metadata.csv`."
        )
        return

    merged_all = usage_df.assign(median_year=usage_df["player_norm"].map(median_lookup))
    unique_players = merged_all.drop_duplicates(subset=["submitter", "player_norm"]).copy()
    matched = unique_players["median_year"].notna().sum()
    st.caption(
        f"Mapped median year for {matched:,}/{len(unique_players):,} unique submitter-player pairs."
    )

    merged = unique_players.dropna(subset=["median_year"]).copy()
    if merged.empty:
        st.info("No unique players could be mapped to median year.")
        return
    merged["median_year"] = merged["median_year"].astype(int)

    submitters = sorted(merged["submitter"].unique().tolist())
    n = len(submitters)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, max(3.5 * rows, 4)))
    axes = np.array(axes).reshape(-1)

    bins = np.arange(1870, 2031, 5)
    year_ticks = np.arange(bins[0], bins[-1] + 1, 10)
    for i, submitter in enumerate(submitters):
        ax = axes[i]
        vals = merged.loc[merged["submitter"] == submitter, "median_year"]
        if vals.empty:
            ax.set_visible(False)
            continue
        weights = np.ones(len(vals)) * (100.0 / max(len(vals), 1))
        ax.hist(vals, bins=bins, weights=weights, color="#2a9d8f", alpha=0.85, edgecolor="white")
        ax.set_title(submitter)
        ax.set_xlabel("Median Year")
        ax.set_ylabel("Freq %")
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xticks(year_ticks)
        ax.tick_params(axis="x", rotation=45, labelsize=8)

    for extra_ax in axes[len(submitters):]:
        extra_ax.axis("off")

    fig.suptitle("Distribution of Unique Players' Median Years by Submitter", y=1.01)
    fig.tight_layout()
    st.pyplot(fig)


def _render_war_histogram(usage_df: pd.DataFrame, war_lookup: dict[str, float]) -> None:
    st.markdown("### Career WAR Distribution by Submitter (Usage Weighted)")
    if not war_lookup:
        st.warning(
            "Career WAR requires Lahman WAR tables and the local `bin/baseball_cache` (run `python src/scripts/build_baseball_cache.py` to refresh)."
        )
        return

    merged_all = usage_df.assign(career_war=usage_df["player_norm"].map(war_lookup))
    matched = merged_all["career_war"].notna().sum()
    st.caption(f"Mapped career WAR for {matched:,}/{len(merged_all):,} player usages.")

    merged = merged_all.dropna(subset=["career_war"]).copy()
    if merged.empty:
        st.info("No player usages could be mapped to career WAR.")
        return

    submitters = sorted(merged["submitter"].unique().tolist())
    n = len(submitters)
    cols = 2
    rows = int(np.ceil(n / cols)) or 1
    fig, axes = plt.subplots(rows, cols, figsize=(12, max(3.5 * rows, 4)))
    axes = np.array(axes).reshape(-1)
    bins = np.arange(-10, 106, 5)

    for i, submitter in enumerate(submitters):
        ax = axes[i]
        vals = merged.loc[merged["submitter"] == submitter, "career_war"]
        if vals.empty:
            ax.set_visible(False)
            continue
        weights = np.ones(len(vals)) * (100.0 / max(len(vals), 1))
        ax.hist(vals, bins=bins, weights=weights, color="#264653", alpha=0.85, edgecolor="white")
        ax.set_title(submitter)
        ax.set_xlabel("Career WAR")
        ax.set_ylabel("Freq %")

    for extra_ax in axes[len(submitters):]:
        extra_ax.set_visible(False)

    fig.suptitle("Usage Frequency by Career WAR (Percent)", y=1.01)
    fig.tight_layout()
    st.pyplot(fig)


def _render_war_vs_rarity_scatter(usage_df: pd.DataFrame, war_lookup: dict[str, float]) -> None:
    st.markdown("### Avg Career WAR vs Grid Rarity Score (One Scatter per Submitter)")
    if not war_lookup:
        st.warning(
            "Career WAR requires Lahman WAR tables and the local `bin/baseball_cache` (run `python src/scripts/build_baseball_cache.py` to refresh)."
        )
        return

    rarity_df = _load_grid_rarity_scores()
    if rarity_df.empty:
        st.warning(
            "Rarity scores require `bin/csv/text_message_responses.csv` (refresh Texts in Add/Update Data)."
        )
        return

    usage = usage_df.copy()
    usage["grid_number"] = usage["grid_number"].map(_normalize_grid_number)
    usage["career_war"] = usage["player_norm"].map(war_lookup)

    by_submitter_grid = (
        usage.groupby(["submitter", "grid_number"], as_index=False)
        .agg(
            avg_career_war=("career_war", "mean"),
            mapped_players=("career_war", lambda s: int(s.notna().sum())),
            total_players=("career_war", "size"),
        )
    )
    by_submitter_grid = by_submitter_grid.dropna(subset=["avg_career_war"]).copy()
    merged = by_submitter_grid.merge(
        rarity_df,
        on=["submitter", "grid_number"],
        how="inner",
        validate="1:1",
    )
    if merged.empty:
        st.info("No submitter-grid rows had both rarity score and mapped career WAR.")
        return

    max_mapped = int(merged["mapped_players"].max()) if not merged.empty else 0
    min_mapped = st.slider(
        "Minimum mapped players per submitter-grid",
        min_value=1,
        max_value=max(1, max_mapped),
        value=min(6, max(1, max_mapped)),
        step=1,
        key="war_rarity_min_mapped",
    )
    plot_df = merged[merged["mapped_players"] >= int(min_mapped)].copy()
    if plot_df.empty:
        st.info("No rows remain after the mapped-player filter.")
        return

    submitters = sorted(plot_df["submitter"].astype(str).unique().tolist())
    st.caption(
        f"Showing {len(plot_df):,} submitter-grid points across {len(submitters)} submitters "
        f"(min mapped players per point: {min_mapped})."
    )

    n = len(submitters)
    cols = 2
    rows = int(np.ceil(n / cols)) or 1
    fig, axes = plt.subplots(rows, cols, figsize=(12, max(4.0 * rows, 4)), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    for i, submitter in enumerate(submitters):
        ax = axes[i]
        sub = plot_df[plot_df["submitter"] == submitter].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        ax.scatter(sub["rarity_score"], sub["avg_career_war"], alpha=0.75, s=28, color="#1f77b4")
        if len(sub) >= 2:
            x = pd.to_numeric(sub["rarity_score"], errors="coerce")
            y = pd.to_numeric(sub["avg_career_war"], errors="coerce")
            valid = x.notna() & y.notna()
            if valid.sum() >= 2:
                m, b = np.polyfit(x[valid], y[valid], 1)
                x_line = np.linspace(float(x[valid].min()), float(x[valid].max()), 100)
                y_line = m * x_line + b
                ax.plot(x_line, y_line, color="#d62828", linewidth=2, alpha=0.9)
        ax.set_title(f"{submitter} (n={len(sub)})")
        ax.set_xlabel("Immaculate Grid Score (Lower = Rarer)")
        ax.set_ylabel("Avg Career WAR")
        ax.grid(alpha=0.2)

    for extra_ax in axes[len(submitters):]:
        extra_ax.set_visible(False)

    fig.suptitle("Submitter-Grid Points: Rarity Score vs Avg Career WAR", y=1.01)
    fig.tight_layout()
    st.pyplot(fig)

    st.dataframe(
        plot_df.sort_values(["submitter", "grid_number"], ascending=[True, True]),
        use_container_width=True,
        hide_index=True,
    )


def render_player_data_analytics(
    images_df: pd.DataFrame,
    forced_analysis: str | None = None,
) -> None:
    usage_df = _build_usage_df(images_df)
    if usage_df.empty:
        st.info("No parsed player responses available in image metadata.")
        return

    st.caption(
        f"Loaded {len(usage_df):,} player usages across {usage_df['submitter'].nunique()} submitters."
    )

    analysis_options = [
        "Player Search",
        "Tableau Mosaic",
        "Median Year Histogram by User (Usage Weighted)",
        "Career WAR Distribution by User",
        "Avg Career WAR vs Grid Rarity (Scatter)",
    ]
    if forced_analysis and forced_analysis in analysis_options:
        selected_analysis = forced_analysis
    else:
        selected_analysis = st.selectbox(
            "Real Player Data",
            options=analysis_options,
            key="dynamic_analysis_view",
        )

    median_lookup = _load_median_year_lookup()
    war_lookup = _load_career_war_lookup()

    if selected_analysis == "Player Search":
        _render_player_search_chart(usage_df)
    elif selected_analysis == "Tableau Mosaic":
        _render_mosaic_style_plot(usage_df)
    elif selected_analysis == "Median Year Histogram by User (Usage Weighted)":
        _render_median_year_histogram(usage_df, median_lookup)
    elif selected_analysis == "Career WAR Distribution by User":
        _render_war_histogram(usage_df, war_lookup)
    elif selected_analysis == "Avg Career WAR vs Grid Rarity (Scatter)":
        _render_war_vs_rarity_scatter(usage_df, war_lookup)
