from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


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
            player = str(val).strip()
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
    out["player_norm"] = out["player"].map(_normalize_name)
    return out


def _load_player_history(player_history_path: Path) -> pd.DataFrame:
    if not player_history_path.exists():
        return pd.DataFrame()
    try:
        ph = pd.read_csv(player_history_path)
    except Exception:
        return pd.DataFrame()
    if "full_name" not in ph.columns:
        return pd.DataFrame()
    ph["player_norm"] = ph["full_name"].astype(str).map(_normalize_name)
    ph["career_mid_year"] = pd.to_numeric(ph.get("career_mid_year"), errors="coerce")
    ph = ph.drop_duplicates(subset=["player_norm"], keep="first")
    return ph


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


def _render_mid_year_histogram(usage_df: pd.DataFrame, player_hist_df: pd.DataFrame) -> None:
    st.markdown("### Career Mid-Year Histogram by User")
    if player_hist_df.empty:
        st.warning("Player history CSV not found or invalid, so mid-year histogram is unavailable.")
        return

    merged_all = usage_df.merge(
        player_hist_df[["player_norm", "career_mid_year"]],
        on="player_norm",
        how="left",
    )
    matched = merged_all["career_mid_year"].notna().sum()
    st.caption(f"Mapped career mid-year for {matched:,}/{len(merged_all):,} player usages.")

    merged = merged_all.dropna(subset=["career_mid_year"]).copy()
    if merged.empty:
        st.info("No player usages could be mapped to career mid-year.")
        return
    merged["career_mid_year"] = merged["career_mid_year"].astype(int)

    submitters = sorted(merged["submitter"].unique().tolist())
    n = len(submitters)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, max(3.5 * rows, 4)), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    bins = np.arange(1870, 2031, 5)
    for i, submitter in enumerate(submitters):
        ax = axes[i]
        vals = merged.loc[merged["submitter"] == submitter, "career_mid_year"]
        ax.hist(vals, bins=bins, color="#e76f51", alpha=0.85, edgecolor="white")
        ax.set_title(submitter)
        ax.set_xlabel("Career Mid-Year")
        ax.set_ylabel("Count")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Distribution of Players' Career Mid-Years by Submitter", y=1.01)
    fig.tight_layout()
    st.pyplot(fig)


def render_player_data_analytics(images_df: pd.DataFrame, player_history_path: Path) -> None:
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
        "Career Mid-Year Histogram by User",
    ]
    selected_analysis = st.selectbox(
        "Dynamic Analysis View",
        options=analysis_options,
        key="dynamic_analysis_view",
    )

    if selected_analysis == "Player Search":
        _render_player_search_chart(usage_df)
    elif selected_analysis == "Tableau Mosaic":
        _render_mosaic_style_plot(usage_df)
    else:
        player_hist_df = _load_player_history(player_history_path)
        _render_mid_year_histogram(usage_df, player_hist_df)
