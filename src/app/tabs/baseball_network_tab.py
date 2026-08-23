from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.services.baseball_network import BaseballNetwork, TeammateHop
from config.constants import canonicalize_franchid

REPO_ROOT = Path(__file__).resolve().parents[3]
BASEBALL_CACHE_DIR = REPO_ROOT / "bin" / "baseball_cache"


def _cache_file(cache_dir: Path, *names: str) -> Path:
    for name in names:
        path = cache_dir / name
        if path.exists():
            return path
    return cache_dir / names[0]


@st.cache_resource(show_spinner="Building baseball history network...")
def _load_network(cache_dir: str) -> BaseballNetwork:
    root = Path(cache_dir)
    teams = pd.read_csv(_cache_file(root, "teams.csv", "Teams.csv"))
    people = pd.read_csv(_cache_file(root, "People.csv", "people.csv"))
    appearances = pd.read_csv(_cache_file(root, "appearances.csv", "Appearances.csv"))
    return BaseballNetwork(appearances, teams, people, canonicalize_franchid=canonicalize_franchid)


@st.cache_data(show_spinner=False)
def _load_oldest(cache_dir: str) -> pd.DataFrame:
    path = _cache_file(Path(cache_dir), "team_year_oldest_players.csv")
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _path_rows(network: BaseballNetwork, hops: list[TeammateHop]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "from_player": network.name(hop.from_player),
            "team_year": f"{hop.team_name or hop.team_id} ({hop.year})",
            "to_player": network.name(hop.to_player),
            "franchise": hop.franchise_id or "",
        }
        for hop in hops
    ])


def _render_chain(network: BaseballNetwork, hops: list[TeammateHop] | None) -> None:
    if hops is None:
        st.warning("No teammate path found.")
        return
    if not hops:
        st.info("The selected player is already at the destination.")
        return
    st.metric("Teammate hops", len(hops))
    for index, hop in enumerate(hops):
        if index == 0:
            st.markdown(f"**{network.name(hop.from_player)}**")
        st.markdown(f"↓ {hop.team_name or hop.team_id} — **{hop.year}**")
        st.markdown(f"**{network.name(hop.to_player)}**")
    st.dataframe(_path_rows(network, hops), use_container_width=True, hide_index=True)


def render_baseball_network_tab(cache_dir: Path = BASEBALL_CACHE_DIR) -> None:
    st.subheader("⚾ Baseball Network")
    st.caption("Connect MLB history through shared team-season rosters without materializing every teammate pair.")
    required = [_cache_file(cache_dir, "teams.csv", "Teams.csv"), _cache_file(cache_dir, "People.csv", "people.csv"), _cache_file(cache_dir, "appearances.csv", "Appearances.csv")]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        st.warning(f"Baseball cache is missing: {', '.join(missing)}. Build the baseball cache first.")
        return

    network = _load_network(str(cache_dir))
    options = network.player_options()
    player_ids = [player_id for player_id, _ in options]
    labels = {player_id: f"{name} ({player_id})" for player_id, name in options}
    if not player_ids:
        st.warning("No players found in the baseball cache.")
        return

    player_tab, history_tab, oldest_tab, stats_tab = st.tabs(["Player → Player", "Across History", "Oldest Player Walk", "Network Statistics"])

    with player_tab:
        col_a, col_b = st.columns(2)
        start = col_a.selectbox("Player A", player_ids, format_func=lambda player_id: labels[player_id], key="network_player_a")
        target_index = min(1, len(player_ids) - 1)
        target = col_b.selectbox("Player B", player_ids, index=target_index, format_func=lambda player_id: labels[player_id], key="network_player_b")
        if st.button("Find shortest teammate chain", key="network_shortest"):
            _render_chain(network, network.shortest_path(start, target))

    with history_tab:
        start = st.selectbox("Starting player", player_ids, format_func=lambda player_id: labels[player_id], key="network_history_start")
        mode = st.radio("Path mode", ["Fewest teammate hops", "Historical relay"], horizontal=True, key="network_history_mode")
        if st.button("Connect baseball history", key="network_history_go"):
            if mode == "Fewest teammate hops":
                _render_chain(network, network.path_to_earliest_baseball(start))
            else:
                hops = network.historical_relay(start)
                _render_chain(network, hops)
                if hops:
                    st.caption(f"Relay reached a player whose career began in {network.player_years[hops[-1].to_player][0]}.")

    with oldest_tab:
        oldest = _load_oldest(str(cache_dir))
        if oldest.empty:
            st.info("team_year_oldest_players.csv is not available in the baseball cache.")
        else:
            roster_options = sorted({(str(row.teamID), int(row.yearID)) for row in oldest[["teamID", "yearID"]].dropna().itertuples(index=False)}, key=lambda item: (-item[1], item[0]))
            roster = st.selectbox("Starting team-season", roster_options, format_func=lambda item: f"{item[0]} — {item[1]}", key="network_oldest_roster")
            if st.button("Walk backward through oldest players", key="network_oldest_go"):
                walk = network.oldest_player_walk(roster, oldest)
                if walk:
                    st.metric("Transitions", max(0, len(walk) - 1))
                    st.metric("Years traversed", int(walk[0]["yearID"]) - int(walk[-1]["yearID"]))
                    st.dataframe(pd.DataFrame(walk), use_container_width=True, hide_index=True)
                else:
                    st.warning("No oldest-player walk could be constructed from that team-season.")

    with stats_tab:
        stats = network.statistics()
        cols = st.columns(4)
        cols[0].metric("Players", f"{stats['players']:,}")
        cols[1].metric("Team-seasons", f"{stats['team_seasons']:,}")
        cols[2].metric("Memberships", f"{stats['memberships']:,}")
        cols[3].metric("Components", f"{stats['components']:,}")
        cols = st.columns(4)
        cols[0].metric("Largest component", f"{stats['largest_component']:,}")
        cols[1].metric("Players in largest", f"{stats['largest_component_pct']:.2f}%")
        cols[2].metric("Earliest season", stats["earliest_year"])
        cols[3].metric("Latest season", stats["latest_year"])
