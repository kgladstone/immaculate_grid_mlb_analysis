from __future__ import annotations

import streamlit as st


def render_study_guide(
    player_master,
    team_master,
    appearances,
    selectable_team_codes: list[str],
    team_code_to_name: dict[str, str],
    franchid_modern_alignment: dict,
    canonicalize_franchid_fn,
    build_study_guide_df_fn,
) -> None:
    st.markdown("### Study Guide")
    st.caption(
        "Ranks players by total distinct franchIDs. Optionally require a set of franchIDs and show top matches."
    )
    guide_df = build_study_guide_df_fn(player_master, team_master, appearances)
    if guide_df.empty:
        st.info("No study guide data available.")
        return

    ranking_mode = st.selectbox(
        "Ranking mode",
        options=["Overall", "By League", "By Division"],
        index=0,
        key="study_guide_ranking_mode",
    )
    score_col = "distinct_franchids"
    score_label = "distinct_franchids"
    if ranking_mode == "By League":
        leagues = sorted({str(v.get("league")).upper() for v in franchid_modern_alignment.values() if v.get("league")})
        selected_league = st.selectbox("League", options=leagues, index=0, key="study_guide_league")
        score_col = f"teams_in_{selected_league.lower()}"
        score_label = f"teams_in_{selected_league}"
    elif ranking_mode == "By Division":
        divisions = sorted({str(v.get("division")) for v in franchid_modern_alignment.values() if v.get("division")})
        selected_division = st.selectbox("Division", options=divisions, index=0, key="study_guide_division")
        score_col = f"teams_in_{selected_division.lower().replace(' ', '_')}"
        score_label = f"teams_in_{selected_division}"

    selected_franchids = st.multiselect(
        "Filter by required franchID set",
        options=selectable_team_codes,
        default=[],
        format_func=lambda code: f"{team_code_to_name.get(code, code)} ({code})",
        key="study_guide_franchid_filter",
    )
    top_n = st.slider("Top players", min_value=10, max_value=200, value=40, step=10)

    if selected_franchids:
        selected_set = {canonicalize_franchid_fn(fid) for fid in selected_franchids}
        filtered = guide_df[
            guide_df["franchid_list"].apply(
                lambda ids: selected_set.issubset({canonicalize_franchid_fn(fid) for fid in ids})
            )
        ].copy()
        st.write(f"Players who include required set ({', '.join(selected_franchids)}): {len(filtered):,}")
    else:
        filtered = guide_df.copy()
        st.write(f"All ranked players: {len(filtered):,}")

    filtered = filtered.sort_values([score_col, "distinct_franchids", "player_name"], ascending=[False, False, True])
    filtered = filtered.head(top_n).copy()
    if score_col not in filtered.columns:
        filtered[score_col] = 0
    filtered = filtered.rename(columns={score_col: score_label})
    display_cols = ["player_name", "key_bbref", score_label, "distinct_franchids", "franchids"]
    display_cols = list(dict.fromkeys(display_cols))
    st.dataframe(filtered[display_cols], use_container_width=True)

