from __future__ import annotations

import random
import shutil
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
import unidecode
from pathlib import Path


@st.cache_data(show_spinner=True)
def _load_team_master(max_year: int = 2023) -> pd.DataFrame:
    import pybaseball as pb

    frames = []
    for year in range(1876, max_year + 1):
        frames.append(pb.team_ids(year)[["yearID", "teamID", "franchID"]])
    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    return df


@st.cache_data(show_spinner=True)
def _load_player_master() -> pd.DataFrame:
    import pybaseball as pb

    df = pb.chadwick_register()
    df["name_last"] = df["name_last"].apply(lambda x: unidecode.unidecode(str(x)))
    df["name_first"] = df["name_first"].apply(lambda x: unidecode.unidecode(str(x)))
    return df


@st.cache_data(show_spinner=True)
def _load_appearances() -> pd.DataFrame:
    import pybaseball as pb

    return pb.lahman.appearances()


def _get_player_row(player_master: pd.DataFrame, last: str, first: str) -> pd.DataFrame:
    return player_master.loc[
        (player_master["name_last"].str.lower() == last.lower().strip())
        & (player_master["name_first"].str.lower() == first.lower().strip())
    ]


def _get_player_id(player_master: pd.DataFrame, last: str, first: str) -> Optional[str]:
    row = _get_player_row(player_master, last, first)
    if row.empty:
        return None
    return row.key_bbref.iloc[0]


def _get_franchises_for_player(
    player_master: pd.DataFrame, team_master: pd.DataFrame, appearances: pd.DataFrame, last: str, first: str
) -> Tuple[bool, list[str]]:
    player_id = _get_player_id(player_master, last, first)
    if player_id is None:
        return False, []
    rows = appearances.loc[appearances["playerID"] == player_id]
    teams = (
        pd.merge(rows, team_master, on=["teamID", "yearID"], how="left")["franchID"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
        .tolist()
    )
    return True, teams


def _random_team_pair(team_master: pd.DataFrame, max_year: int = 2023) -> Tuple[str, str]:
    current = team_master.loc[team_master["yearID"] == max_year, "franchID"].astype(str).unique().tolist()
    return tuple(random.sample(current, 2))


def render_simulator_tab() -> None:
    st.write("Check if a player fits a team intersection using Lahman/Chadwick data (pybaseball).")

    cache_dir = Path.home() / ".pybaseball"
    cache_dir_alt = Path.home() / ".cache" / "pybaseball"

    def clear_pybaseball_cache():
        for path in [cache_dir, cache_dir_alt]:
            shutil.rmtree(path, ignore_errors=True)
        _load_team_master.clear()
        _load_player_master.clear()
        _load_appearances.clear()
        st.success("Cleared pybaseball cache. Re-run this tab to download fresh data.")

    team_master = player_master = appearances = None
    start_clicked = st.button("Start simulator")

    if start_clicked:
        try:
            team_master = _load_team_master()
            player_master = _load_player_master()
            appearances = _load_appearances()
        except Exception as exc:
            st.error(f"Failed to load pybaseball data: {exc}")
            st.info(f"Try clearing the pybaseball cache and retry. Cache paths: {cache_dir}, {cache_dir_alt}")
            if st.button("Clear pybaseball cache and retry"):
                clear_pybaseball_cache()
            st.stop()

    # If data not loaded, don't render the rest
    if team_master is None or player_master is None or appearances is None:
        return

    max_year = int(team_master["yearID"].max())
    teams_current = sorted(team_master.loc[team_master["yearID"] == max_year, "franchID"].astype(str).unique())

    st.markdown("### Pick teams")
    col_a, col_b, col_btn = st.columns([1, 1, 1])
    with col_a:
        team1 = st.selectbox("Team A", options=teams_current, index=0, key="sim_team1")
    with col_b:
        team2 = st.selectbox("Team B", options=teams_current, index=1, key="sim_team2")
    with col_btn:
        if st.button("Random teams"):
            t1, t2 = _random_team_pair(team_master)
            st.session_state["sim_team1"] = t1
            st.session_state["sim_team2"] = t2
            team1, team2 = t1, t2

    st.markdown("### Enter player")
    col_f, col_l = st.columns(2)
    with col_f:
        first = st.text_input("First name", value="").strip()
    with col_l:
        last = st.text_input("Last name", value="").strip()

    if st.button("Check player"):
        if not first or not last:
            st.warning("Enter both first and last name.")
        else:
            exists, franchises = _get_franchises_for_player(player_master, team_master, appearances, last, first)
            if not exists:
                st.error("Player not found in database (check spelling).")
            else:
                franchises_set = set(franchises)
                hits = team1 in franchises_set and team2 in franchises_set
                st.write(f"Teams for {first} {last}: {', '.join(sorted(franchises_set)) or 'None'}")
                if hits:
                    st.success(f"✅ Yes! {first} {last} played for both {team1} and {team2}.")
                else:
                    st.error(f"❌ No. {first} {last} did not play for both {team1} and {team2}.")

    max_year = int(team_master["yearID"].max())
    teams_current = sorted(team_master.loc[team_master["yearID"] == max_year, "franchID"].astype(str).unique())

    st.markdown("### Pick teams")
    col_a, col_b, col_btn = st.columns([1, 1, 1])
    with col_a:
        team1 = st.selectbox("Team A", options=teams_current, index=0, key="sim_team1")
    with col_b:
        team2 = st.selectbox("Team B", options=teams_current, index=1, key="sim_team2")
    with col_btn:
        if st.button("Random teams"):
            t1, t2 = _random_team_pair(team_master)
            st.session_state["sim_team1"] = t1
            st.session_state["sim_team2"] = t2
            team1, team2 = t1, t2

    st.markdown("### Enter player")
    col_f, col_l = st.columns(2)
    with col_f:
        first = st.text_input("First name", value="").strip()
    with col_l:
        last = st.text_input("Last name", value="").strip()

    if st.button("Check player"):
        if not first or not last:
            st.warning("Enter both first and last name.")
        else:
            exists, franchises = _get_franchises_for_player(player_master, team_master, appearances, last, first)
            if not exists:
                st.error("Player not found in database (check spelling).")
            else:
                franchises_set = set(franchises)
                hits = team1 in franchises_set and team2 in franchises_set
                st.write(f"Teams for {first} {last}: {', '.join(sorted(franchises_set)) or 'None'}")
                if hits:
                    st.success(f"✅ Yes! {first} {last} played for both {team1} and {team2}.")
                else:
                    st.error(f"❌ No. {first} {last} did not play for both {team1} and {team2}.")


__all__ = ["render_simulator_tab"]
