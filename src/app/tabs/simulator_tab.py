from __future__ import annotations

import random
import shutil
import zipfile
from typing import Optional, Tuple, Callable, List

import pandas as pd
import streamlit as st
import unidecode
from pathlib import Path


def clear_pybaseball_cache(paths: Optional[list[Path]] = None):
    paths = paths or [Path.home() / ".pybaseball", Path.home() / ".cache" / "pybaseball"]
    for path in paths:
        shutil.rmtree(path, ignore_errors=True)
    _load_team_master.clear()
    _load_player_master.clear()
    _load_appearances.clear()


def _inspect_cache(paths: List[Path], limit: int = 5) -> List[dict]:
    """
    Inspect cached files to help diagnose corrupt downloads. Returns up to `limit` entries.
    """
    entries = []
    for base in paths:
        if not base.exists():
            continue
        for f in base.rglob("*"):
            if not f.is_file():
                continue
            if len(entries) >= limit:
                return entries
            try:
                size = f.stat().st_size
                header = f.read_bytes()[:120]
                header_preview = header.decode("utf-8", errors="replace")
                is_zip = zipfile.is_zipfile(f)
                entries.append(
                    {
                        "path": str(f),
                        "size_bytes": size,
                        "is_zipfile": is_zip,
                        "header_preview": header_preview,
                    }
                )
            except Exception:
                continue
    return entries


def load_pybaseball_data(
    clear_before: bool = False,
    loaders: Optional[Tuple[Callable, Callable, Callable]] = None,
    clear_fn: Optional[Callable[[list[Path]], None]] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Exception]]:
    team_loader, player_loader, app_loader = loaders or (_load_team_master, _load_player_master, _load_appearances)
    clear_fn = clear_fn or clear_pybaseball_cache
    cache_paths = [Path.home() / ".pybaseball", Path.home() / ".cache" / "pybaseball"]
    if clear_before:
        clear_fn(cache_paths)
    try:
        return team_loader(), player_loader(), app_loader(), None
    except Exception as exc:
        # Retry once after clearing cache (common for corrupted zip)
        clear_fn(cache_paths)
        try:
            return team_loader(), player_loader(), app_loader(), None
        except Exception as exc2:
            return None, None, None, exc2


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

    team_master = player_master = appearances = None
    clear_before = st.checkbox("Clear pybaseball cache before loading", value=False)
    start_clicked = st.button("Start simulator")

    if start_clicked:
        team_master, player_master, appearances, exc = load_pybaseball_data(clear_before=clear_before)
        if exc:
            exc_text = str(exc)
            st.error(f"Failed to load pybaseball data: {exc_text}")
            if "Failed to resolve" in exc_text or "NameResolutionError" in exc_text:
                st.warning(
                    "Network/DNS looks blocked. Pybaseball must download from github.com; "
                    "enable network or manually place a fresh Chadwick/Lahman cache under "
                    f"{cache_dir} (or {cache_dir_alt})."
                )
            st.info(f"Try clearing the pybaseball cache and retry. Cache paths: {cache_dir}, {cache_dir_alt}")
            # Inspect cache to show what kind of file caused trouble
            cache_inspect = _inspect_cache([cache_dir, cache_dir_alt], limit=5)
            if cache_inspect:
                st.warning("Sample of cached files (is_zipfile flag and header preview):")
                for entry in cache_inspect:
                    st.text(f"{entry['path']} | size={entry['size_bytes']} | is_zipfile={entry['is_zipfile']}\n{entry['header_preview']}")
            if st.button("Clear pybaseball cache and retry"):
                clear_pybaseball_cache([cache_dir, cache_dir_alt])
                team_master, player_master, appearances, exc = load_pybaseball_data(clear_before=False)
                if exc:
                    st.error(f"Retry failed: {exc}")
                    st.stop()
            else:
                st.stop()
        else:
            st.success(
                f"Loaded pybaseball data successfully: teams={len(team_master):,}, "
                f"players={len(player_master):,}, appearances={len(appearances):,}"
            )
            with st.expander("Preview loaded data", expanded=False):
                st.write("Teams (head):", team_master.head())
                st.write("Players (head):", player_master.head())
                st.write("Appearances (head):", appearances.head())

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


__all__ = ["render_simulator_tab"]
