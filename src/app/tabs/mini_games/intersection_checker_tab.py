from __future__ import annotations

from typing import Callable

import streamlit as st


def render_intersection_checker(
    selectable_team_codes: list[str],
    team_code_to_name: dict[str, str],
    player_master,
    team_master,
    appearances,
    closest_player_matches_fn: Callable,
    get_franchises_for_player_id_fn: Callable,
    canonicalize_franchid_fn: Callable,
) -> None:
    st.markdown("### Pick teams")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        team1 = st.selectbox(
            "Team A",
            options=selectable_team_codes,
            index=selectable_team_codes.index(st.session_state["sim_team1"]),
            key="sim_team1",
            format_func=lambda code: f"{team_code_to_name.get(code, code)} ({code})",
        )
    with col_b:
        team2 = st.selectbox(
            "Team B",
            options=selectable_team_codes,
            index=selectable_team_codes.index(st.session_state["sim_team2"]),
            key="sim_team2",
            format_func=lambda code: f"{team_code_to_name.get(code, code)} ({code})",
        )

    st.markdown("### Enter player")
    col_f, col_l = st.columns(2)
    with col_f:
        first = st.text_input("First name", value="", key="sim_first_name_input").strip()
    with col_l:
        last = st.text_input("Last name", value="", key="sim_last_name_input").strip()

    if not first or not last:
        st.info("Enter first and last name to see closest player matches.")
        return

    matches = closest_player_matches_fn(player_master, first=first, last=last, top_n=5)
    if matches.empty:
        st.warning("No matching players found.")
        return

    st.caption("Select the closest matched player to confirm spelling before checking.")
    match_options = list(matches.index)
    selected_idx = st.selectbox(
        "Closest matched players",
        options=match_options,
        format_func=lambda i: (
            f"{matches.loc[i, 'full_name']} ({matches.loc[i, 'key_bbref']}) "
            f"- {matches.loc[i, 'match_score']:.1%}"
        ),
        key="sim_closest_player_select",
    )

    selected = matches.loc[selected_idx]
    selected_player_id = str(selected["key_bbref"])
    selected_name = str(selected["full_name"])

    if st.button("Confirm player and check", key="sim_confirm_and_check"):
        franchises = get_franchises_for_player_id_fn(team_master, appearances, selected_player_id)
        franchises_set = {canonicalize_franchid_fn(fid) for fid in franchises}
        hits = team1 in franchises_set and team2 in franchises_set
        st.write(f"Confirmed player: **{selected_name}** (`{selected_player_id}`)")
        st.write(f"Teams: {', '.join(sorted(franchises_set)) or 'None'}")
        if hits:
            st.success(f"✅ Yes. {selected_name} played for both {team1} and {team2}.")
        else:
            st.error(f"❌ No. {selected_name} did not play for both {team1} and {team2}.")
    else:
        st.caption("Click **Confirm player and check** to run the Yes/No validation.")

