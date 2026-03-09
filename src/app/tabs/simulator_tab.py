from __future__ import annotations

import re
import random
import difflib
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
from pathlib import Path
from scripts.build_baseball_cache import build_cache
from utils.constants import FRANCHID_MODERN_ALIGNMENT, TEAM_LIST, canonicalize_franchid


@st.cache_data(show_spinner=True)
def _load_cached_baseball(cache_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    teams = pd.read_csv(cache_dir / "teams.csv")
    players = pd.read_csv(cache_dir / "players.csv")
    appearances = pd.read_csv(cache_dir / "appearances.csv")
    return teams, players, appearances


def load_pybaseball_data(clear_before=False, loaders=None, clear_fn=None):
    """
    Compatibility loader retained for tests/tooling that expect retry-on-bad-zip behavior.
    """
    if loaders is None:
        raise ValueError("loaders must be provided")

    team_loader, player_loader, app_loader = loaders
    clear_fn = clear_fn or (lambda paths: None)
    cache_paths = [
        "~/.pybaseball/cache",
        "~/Library/Caches/pybaseball",
    ]

    if clear_before:
        clear_fn(cache_paths)

    try:
        team_master = team_loader()
        player_master = player_loader()
        appearances = app_loader()
        return team_master, player_master, appearances, None
    except Exception as exc:
        if "zip file" not in str(exc).lower():
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), exc
        clear_fn(cache_paths)
        try:
            team_master = team_loader()
            player_master = player_loader()
            appearances = app_loader()
            return team_master, player_master, appearances, None
        except Exception as retry_exc:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), retry_exc


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
        .apply(canonicalize_franchid)
        .unique()
        .tolist()
    )
    return True, teams


def _get_franchises_for_player_id(
    team_master: pd.DataFrame, appearances: pd.DataFrame, player_id: str
) -> list[str]:
    rows = appearances.loc[appearances["playerID"].astype(str) == str(player_id)]
    teams = (
        pd.merge(rows, team_master, on=["teamID", "yearID"], how="left")["franchID"]
        .dropna()
        .apply(canonicalize_franchid)
        .unique()
        .tolist()
    )
    return teams


def _closest_player_matches(
    player_master: pd.DataFrame,
    first: str,
    last: str,
    top_n: int = 5,
) -> pd.DataFrame:
    target = f"{first} {last}".strip().lower()
    if not target:
        return pd.DataFrame()

    candidates = (
        player_master[["key_bbref", "name_first", "name_last"]]
        .dropna(subset=["key_bbref", "name_first", "name_last"])
        .drop_duplicates(subset=["key_bbref"])
        .copy()
    )
    candidates["full_name"] = (
        candidates["name_first"].astype(str).str.strip() + " " + candidates["name_last"].astype(str).str.strip()
    ).str.strip()

    # Fast vectorized prefilter before fuzzy scoring.
    narrowed = candidates
    first_token = str(first).strip().lower()
    last_token = str(last).strip().lower()
    if last_token:
        narrowed = narrowed[narrowed["name_last"].astype(str).str.lower().str.contains(last_token[:3], na=False)]
    if first_token and not narrowed.empty:
        narrowed = narrowed[narrowed["name_first"].astype(str).str.lower().str.contains(first_token[:2], na=False)]
    if narrowed.empty:
        narrowed = candidates
    narrowed = narrowed.copy()

    narrowed["match_score"] = narrowed["full_name"].str.lower().apply(
        lambda s: difflib.SequenceMatcher(None, target, s).ratio()
    )
    narrowed = narrowed.sort_values(["match_score", "full_name"], ascending=[False, True]).head(top_n).copy()
    return narrowed.reset_index(drop=True)


def _build_study_guide_df(
    player_master: pd.DataFrame, team_master: pd.DataFrame, appearances: pd.DataFrame
) -> pd.DataFrame:
    merged = appearances.merge(
        team_master[["teamID", "yearID", "franchID"]],
        on=["teamID", "yearID"],
        how="left",
    )
    merged["franchID"] = merged["franchID"].apply(canonicalize_franchid)
    merged = merged[merged["franchID"].notna() & (merged["franchID"] != "NAN")]

    grouped = (
        merged.groupby("playerID", dropna=False)["franchID"]
        .agg(lambda s: sorted(set(s)))
        .reset_index()
        .rename(columns={"playerID": "key_bbref", "franchID": "franchid_list"})
    )
    grouped["distinct_franchids"] = grouped["franchid_list"].apply(len)
    grouped["franchids"] = grouped["franchid_list"].apply(lambda ids: ", ".join(ids))

    players = player_master[["key_bbref", "name_first", "name_last"]].drop_duplicates(subset=["key_bbref"])
    out = grouped.merge(players, on="key_bbref", how="left")
    out["player_name"] = (
        out["name_first"].fillna("").astype(str).str.strip() + " " + out["name_last"].fillna("").astype(str).str.strip()
    ).str.strip()

    def _count_in_league(ids: list[str], league: str) -> int:
        return sum(
            1
            for fid in ids
            if str((FRANCHID_MODERN_ALIGNMENT.get(canonicalize_franchid(fid)) or {}).get("league", "")).upper()
            == league.upper()
        )

    def _count_in_division(ids: list[str], division: str) -> int:
        return sum(
            1
            for fid in ids
            if str((FRANCHID_MODERN_ALIGNMENT.get(canonicalize_franchid(fid)) or {}).get("division", "")).lower()
            == division.lower()
        )

    out["teams_in_al"] = out["franchid_list"].apply(lambda ids: _count_in_league(ids, "AL"))
    out["teams_in_nl"] = out["franchid_list"].apply(lambda ids: _count_in_league(ids, "NL"))
    for div in sorted({str(v.get("division")) for v in FRANCHID_MODERN_ALIGNMENT.values() if v.get("division")}):
        safe_col = f"teams_in_{div.lower().replace(' ', '_')}"
        out[safe_col] = out["franchid_list"].apply(lambda ids, d=div: _count_in_division(ids, d))

    out = out.sort_values(["distinct_franchids", "player_name"], ascending=[False, True]).reset_index(drop=True)
    return out


def _build_player_franchise_sets(team_master: pd.DataFrame, appearances: pd.DataFrame) -> list[set[str]]:
    merged = appearances.merge(
        team_master[["teamID", "yearID", "franchID"]],
        on=["teamID", "yearID"],
        how="left",
    )
    merged["franchID"] = merged["franchID"].apply(canonicalize_franchid)
    merged = merged[merged["franchID"].notna() & (merged["franchID"] != "NAN")]
    grouped = merged.groupby("playerID", dropna=False)["franchID"].agg(lambda s: set(map(str, s))).tolist()
    return grouped


def _count_players_with_all(player_sets: list[set[str]], required_franchids: set[str]) -> int:
    return sum(1 for s in player_sets if required_franchids.issubset(s))


def _build_grid_counts(
    row_codes: list[str],
    col_codes: list[str],
    player_sets: list[set[str]],
) -> list[list[int]]:
    return [
        [_count_players_with_all(player_sets, {r, c}) for c in col_codes]
        for r in row_codes
    ]


def _build_cube_counts(
    x_codes: list[str],
    y_codes: list[str],
    z_codes: list[str],
    player_sets: list[set[str]],
) -> list[list[list[int]]]:
    cube = []
    for z in z_codes:
        layer = []
        for x in x_codes:
            row = []
            for y in y_codes:
                row.append(_count_players_with_all(player_sets, {x, y, z}))
            layer.append(row)
        cube.append(layer)
    return cube


def _generate_random_grid_puzzle(
    selectable_codes: list[str],
    player_sets: list[set[str]],
    max_attempts: int = 2000,
) -> tuple[list[str], list[str], list[list[int]]] | None:
    if len(selectable_codes) < 6:
        return None
    for _ in range(max_attempts):
        picked = random.sample(selectable_codes, 6)
        row_codes, col_codes = picked[:3], picked[3:]
        counts = _build_grid_counts(row_codes, col_codes, player_sets)
        if all(v > 0 for row in counts for v in row):
            return row_codes, col_codes, counts
    return None


def _generate_random_cube_puzzle(
    selectable_codes: list[str],
    player_sets: list[set[str]],
    max_attempts: int = 3000,
) -> tuple[list[str], list[str], list[str], list[list[list[int]]]] | None:
    if len(selectable_codes) < 9:
        return None
    for _ in range(max_attempts):
        picked = random.sample(selectable_codes, 9)
        x_codes, y_codes, z_codes = picked[:3], picked[3:6], picked[6:9]
        cube = _build_cube_counts(x_codes, y_codes, z_codes, player_sets)
        if all(v > 0 for layer in cube for row in layer for v in row):
            return x_codes, y_codes, z_codes, cube
    return None


def _build_player_franchise_lookup(
    player_master: pd.DataFrame,
    team_master: pd.DataFrame,
    appearances: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    players = (
        player_master[["key_bbref", "name_first", "name_last"]]
        .dropna(subset=["key_bbref", "name_first", "name_last"])
        .drop_duplicates(subset=["key_bbref"])
        .copy()
    )
    players["full_name"] = (
        players["name_first"].astype(str).str.strip() + " " + players["name_last"].astype(str).str.strip()
    ).str.strip()
    players = players[players["full_name"] != ""].copy()

    merged = appearances.merge(
        team_master[["teamID", "yearID", "franchID"]],
        on=["teamID", "yearID"],
        how="left",
    )
    merged["franchID"] = merged["franchID"].apply(canonicalize_franchid)
    merged = merged[merged["franchID"].notna() & (merged["franchID"] != "NAN")]
    franch_map = (
        merged.groupby("playerID", dropna=False)["franchID"]
        .agg(lambda s: set(map(str, s)))
        .to_dict()
    )
    return players.reset_index(drop=True), franch_map


def _closest_player_matches_by_query(
    players_df: pd.DataFrame,
    query: str,
    top_n: int = 5,
) -> pd.DataFrame:
    target = str(query or "").strip().lower()
    if not target:
        return pd.DataFrame()

    narrowed = players_df
    tokens = [t for t in re.split(r"\s+", target) if t]
    if tokens:
        last_token = tokens[-1][:3]
        if last_token:
            narrowed = narrowed[
                narrowed["name_last"].astype(str).str.lower().str.contains(last_token, na=False)
            ]
    if narrowed.empty:
        narrowed = players_df
    narrowed = narrowed.copy()
    narrowed["match_score"] = narrowed["full_name"].str.lower().apply(
        lambda s: difflib.SequenceMatcher(None, target, s).ratio()
    )
    return narrowed.sort_values(["match_score", "full_name"], ascending=[False, True]).head(top_n).reset_index(drop=True)


def _validate_fuzzy_answer(
    query: str,
    required_franchids: set[str],
    players_df: pd.DataFrame,
    player_franch_map: dict[str, set[str]],
) -> dict[str, object]:
    raw = str(query or "").strip()
    if not raw:
        return {
            "status": "missing",
            "answer": "",
            "matched_player": "",
            "player_id": "",
            "confidence": 0.0,
            "ok": False,
            "missing_franchids": sorted(required_franchids),
        }

    matches = _closest_player_matches_by_query(players_df, raw, top_n=5)
    if matches.empty:
        return {
            "status": "no_match",
            "answer": raw,
            "matched_player": "",
            "player_id": "",
            "confidence": 0.0,
            "ok": False,
            "missing_franchids": sorted(required_franchids),
        }

    best = matches.iloc[0]
    player_id = str(best["key_bbref"])
    player_name = str(best["full_name"])
    confidence = float(best["match_score"])
    player_franchids = {canonicalize_franchid(fid) for fid in player_franch_map.get(player_id, set())}
    missing = sorted(required_franchids.difference(player_franchids))
    ok = len(missing) == 0
    return {
        "status": "matched",
        "answer": raw,
        "matched_player": player_name,
        "player_id": player_id,
        "confidence": confidence,
        "ok": ok,
        "missing_franchids": missing,
    }


def _parse_step_message(message: str) -> tuple[int, int]:
    match = re.match(r"^\[(\d+)/(\d+)\]", str(message).strip())
    if not match:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def _format_dt(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _csv_attrs(path: Path) -> dict[str, str]:
    if not path.exists():
        return {
            "file": path.name,
            "exists": "no",
            "size_kb": "",
            "modified_at": "",
            "rows": "",
            "cols": "",
            "columns_preview": "",
        }

    size_kb = f"{path.stat().st_size / 1024:.1f}"
    modified_at = _format_dt(path.stat().st_mtime)
    try:
        df_head = pd.read_csv(path, nrows=0)
        cols = list(df_head.columns)
        col_count = len(cols)
        col_preview = ", ".join(cols[:8])
        with path.open("rb") as f:
            row_count = max(sum(1 for _ in f) - 1, 0)
    except Exception:
        col_count = 0
        col_preview = ""
        row_count = 0

    return {
        "file": path.name,
        "exists": "yes",
        "size_kb": size_kb,
        "modified_at": modified_at,
        "rows": f"{row_count:,}",
        "cols": str(col_count),
        "columns_preview": col_preview,
    }


def _render_cache_viewer(cache_dir: Path) -> None:
    st.caption(f"Cache directory: `{cache_dir}`")
    meta_path = cache_dir / "cache_meta.csv"
    if meta_path.exists():
        try:
            meta_df = pd.read_csv(meta_path)
            if not meta_df.empty:
                st.write("Last cache metadata")
                st.dataframe(meta_df, use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not read cache_meta.csv: {exc}")
    else:
        st.info("No `cache_meta.csv` found yet. Build the cache first.")

    file_rows = [
        _csv_attrs(cache_dir / "teams.csv"),
        _csv_attrs(cache_dir / "players.csv"),
        _csv_attrs(cache_dir / "appearances.csv"),
    ]
    st.write("Required cache files")
    st.dataframe(pd.DataFrame(file_rows), use_container_width=True)


def _render_data_explorer(cache_dir: Path) -> None:
    st.caption(f"Source directory: `{cache_dir}`")
    people_path = cache_dir / "People.csv"
    players_path = cache_dir / "players.csv"
    appearances_path = cache_dir / "appearances.csv"
    teams_cache_path = cache_dir / "teams.csv"

    df = pd.DataFrame()
    source_name = ""
    if people_path.exists():
        try:
            df = pd.read_csv(people_path)
            source_name = "People.csv"
        except Exception as exc:
            st.warning(f"Could not read People.csv: {exc}")
    elif players_path.exists():
        try:
            df = pd.read_csv(players_path)
            source_name = "players.csv"
        except Exception as exc:
            st.warning(f"Could not read players.csv: {exc}")

    if df.empty:
        st.info("No player dataset found. Build/rebuild cache first.")
        return

    st.write(f"Loaded `{source_name}` with {len(df):,} rows.")
    if {"playerID", "nameFirst", "nameLast"}.issubset(df.columns):
        catalog = df[["playerID", "nameFirst", "nameLast"]].drop_duplicates(subset=["playerID"]).copy()
        catalog = catalog.rename(columns={"playerID": "player_id", "nameFirst": "name_first", "nameLast": "name_last"})
    elif {"key_bbref", "name_first", "name_last"}.issubset(df.columns):
        catalog = df[["key_bbref", "name_first", "name_last"]].drop_duplicates(subset=["key_bbref"]).copy()
        catalog = catalog.rename(columns={"key_bbref": "player_id"})
    else:
        st.warning("No searchable player columns were found in this dataset.")
        return

    catalog["full_name"] = (catalog["name_first"].astype(str).str.strip() + " " + catalog["name_last"].astype(str).str.strip()).str.strip()
    catalog["search_text"] = (catalog["full_name"] + " " + catalog["player_id"].astype(str)).str.lower()

    st.markdown("### Player Search")
    query = st.text_input("Type to filter player dropdown", value="", key="sim_data_explorer_filter").strip().lower()
    if query:
        filtered_catalog = catalog[catalog["search_text"].str.contains(query, na=False)].copy()
        if filtered_catalog.empty:
            # Fuzzy fallback when exact contains finds no rows (e.g., typos like "palmiero" vs "palmeiro").
            fuzzy = catalog.copy()
            fuzzy["match_score"] = fuzzy["full_name"].str.lower().apply(
                lambda s: difflib.SequenceMatcher(None, query, s).ratio()
            )
            filtered_catalog = fuzzy.sort_values(["match_score", "full_name"], ascending=[False, True]).head(30)
        else:
            filtered_catalog["match_score"] = filtered_catalog["full_name"].str.lower().apply(
                lambda s: difflib.SequenceMatcher(None, query, s).ratio()
            )
            filtered_catalog = filtered_catalog.sort_values(["match_score", "full_name"], ascending=[False, True])
    else:
        filtered_catalog = catalog.sort_values("full_name").head(300).copy()

    if filtered_catalog.empty:
        st.info("No players matched your filter.")
        return

    st.caption(f"Matches: {len(filtered_catalog):,}")
    option_rows = filtered_catalog.reset_index(drop=True)
    selected_idx = st.selectbox(
        "Player",
        options=list(option_rows.index),
        format_func=lambda i: f"{option_rows.loc[i, 'full_name']} ({option_rows.loc[i, 'player_id']})",
        key="sim_data_explorer_selected_player",
    )
    selected_player_id = str(option_rows.loc[selected_idx, "player_id"]).strip()
    selected_mask = pd.Series(False, index=df.index)
    if "playerID" in df.columns:
        selected_mask = selected_mask | (df["playerID"].astype(str) == selected_player_id)
    if "key_bbref" in df.columns:
        selected_mask = selected_mask | (df["key_bbref"].astype(str) == selected_player_id)
    selected_row = df[selected_mask].head(1)
    if selected_row.empty:
        selected_row = pd.DataFrame([option_rows.loc[selected_idx, ["player_id", "name_first", "name_last"]].to_dict()])
    else:
        selected_row = selected_row.iloc[0:1]

    st.write("Selected player attributes")
    st.dataframe(selected_row, use_container_width=True)

    if not selected_player_id:
        st.info("Selected row has no player ID to query appearances.")
        return
    if not appearances_path.exists():
        st.info("No `appearances.csv` found yet. Build/rebuild cache first.")
        return

    try:
        app_df = pd.read_csv(appearances_path)
    except Exception as exc:
        st.warning(f"Could not read appearances.csv: {exc}")
        return

    if "playerID" not in app_df.columns:
        st.warning("appearances.csv does not contain `playerID`.")
        return

    player_appearances = app_df[app_df["playerID"].astype(str) == selected_player_id].copy()
    if "yearID" in player_appearances.columns:
        player_appearances["yearID"] = player_appearances["yearID"].astype(str).str.replace(".0", "", regex=False)

    # Enrich with franchID from local teams cache (yearID + teamID -> franchID).
    if teams_cache_path.exists() and {"teamID", "yearID"}.issubset(player_appearances.columns):
        try:
            teams_cache = pd.read_csv(teams_cache_path, usecols=["teamID", "yearID", "franchID"])
            teams_cache["yearID"] = teams_cache["yearID"].astype(str).str.replace(".0", "", regex=False)
            teams_cache["franchID"] = teams_cache["franchID"].apply(canonicalize_franchid)
            player_appearances = player_appearances.merge(
                teams_cache,
                on=["teamID", "yearID"],
                how="left",
            )
        except Exception:
            pass

    st.write(f"Appearances for `{selected_player_id}`: {len(player_appearances):,} rows")
    if player_appearances.empty:
        st.info("No appearances found for this player in cache.")
        return

    summary_cols = [c for c in ["franchID", "teamID", "yearID"] if c in player_appearances.columns]
    if summary_cols:
        code_col = "franchID" if "franchID" in player_appearances.columns else "teamID"
        by_team = (
            player_appearances.groupby([code_col], dropna=False)
            .size()
            .reset_index(name="seasons")
            .sort_values("seasons", ascending=False)
        )

        # Use TEAM_LIST (JSON-backed) as franchID -> display name mapping.
        if code_col in by_team.columns:
            team_code_to_name = {str(code).upper(): str(name) for name, code in TEAM_LIST.items()}
            by_team["team_name"] = (
                by_team[code_col]
                .apply(canonicalize_franchid)
                .map(team_code_to_name)
                .fillna(by_team[code_col].astype(str))
            )
            by_team = by_team[[code_col, "team_name", "seasons"]]

        st.write("Team summary")
        st.dataframe(by_team, use_container_width=True)
    st.write("Appearance records")
    st.dataframe(player_appearances, use_container_width=True)


def render_simulator_tab() -> None:
    st.write("Manage baseball-reference cache data and run team-intersection simulator checks.")

    local_cache_dir = Path("bin/baseball_cache").resolve()
    required_files = ["teams.csv", "players.csv", "appearances.csv"]
    missing_files = [name for name in required_files if not (local_cache_dir / name).exists()]
    data_mgmt_tab, run_simulator_tab, instructions_tab = st.tabs(
        ["Baseball Reference Data Management", "Run Simulator", "Instructions"]
    )

    with data_mgmt_tab:
        builder_tab, metadata_tab, explorer_tab = st.tabs(["Cache Builder", "Metadata", "Data Explorer"])

        with builder_tab:
            st.markdown("### Cache setup")
            st.caption("Build/rebuild cache and monitor each step live in the Streamlit console below.")
            build_clicked = st.button("Build/Rebuild cache", type="primary")

            if "sim_build_logs" not in st.session_state:
                st.session_state["sim_build_logs"] = []

            progress_bar = st.progress(0.0)
            status_placeholder = st.empty()
            build_log_placeholder = st.empty()
            if st.session_state["sim_build_logs"]:
                last = st.session_state["sim_build_logs"][-1]
                status_placeholder.write(f"Latest: `{last}`")
                build_log_placeholder.code("\n".join(st.session_state["sim_build_logs"][-80:]), language="text")
                step, total = _parse_step_message(last)
                if total > 0:
                    progress_bar.progress(min(1.0, step / total))

            if build_clicked:
                try:
                    st.session_state["sim_build_logs"] = []

                    def _ui_progress(message: str) -> None:
                        st.session_state["sim_build_logs"].append(message)
                        status_placeholder.write(f"Current: `{message}`")
                        build_log_placeholder.code("\n".join(st.session_state["sim_build_logs"][-80:]), language="text")
                        step, total = _parse_step_message(message)
                        if total > 0:
                            progress_bar.progress(min(1.0, step / total))

                    with st.spinner("Building baseball cache... this can take a few minutes."):
                        default_max_year = max(datetime.now().year - 1, 1876)
                        build_cache(
                            local_cache_dir,
                            max_year=int(default_max_year),
                            lahman_zip_path=None,
                            lahman_url=None,
                            progress_cb=_ui_progress,
                        )
                    _load_cached_baseball.clear()
                    progress_bar.progress(1.0)
                    status_placeholder.write("Current: `Build complete`")
                    st.success(f"Cache built at `{local_cache_dir}`.")
                    missing_files = [name for name in required_files if not (local_cache_dir / name).exists()]
                    if missing_files:
                        st.warning("Build finished, but some files are still missing: " + ", ".join(missing_files))
                except Exception as exc:
                    status_placeholder.write(f"Current: `Build failed: {exc}`")
                    st.error(f"Cache build failed: {exc}")
                    st.info(
                        "If Lahman download fails, run from terminal with a specific source:\n"
                        "- `python src/scripts/build_baseball_cache.py --lahman-url <working_lahman_zip_url>`\n"
                        "- `python src/scripts/build_baseball_cache.py --lahman-zip /absolute/path/to/lahman.zip`"
                    )

        with metadata_tab:
            _render_cache_viewer(local_cache_dir)

        with explorer_tab:
            _render_data_explorer(local_cache_dir)

    with run_simulator_tab:
        st.write("Check if a player fits a team intersection using locally cached Lahman data.")
        team_master = player_master = appearances = None
        if "simulator_ready" not in st.session_state:
            st.session_state["simulator_ready"] = False
        start_clicked = st.button("Start simulator")
        if start_clicked:
            st.session_state["simulator_ready"] = True
        load_clicked = st.session_state["simulator_ready"]

        if load_clicked:
            missing_files = [name for name in required_files if not (local_cache_dir / name).exists()]
            if missing_files:
                st.error(
                    "Local baseball cache is incomplete. Missing: "
                    + ", ".join(f"`{name}`" for name in missing_files)
                )
                st.info(
                    "Run the cache builder from a terminal that can access pybaseball:\n"
                    "`python src/scripts/build_baseball_cache.py`\n\n"
                    "If `appearances.csv` keeps failing, try one of these:\n"
                    "- `python src/scripts/build_baseball_cache.py --lahman-url <working_lahman_zip_url>`\n"
                    "- `python src/scripts/build_baseball_cache.py --lahman-zip /absolute/path/to/lahman.zip`"
                )
                st.session_state["simulator_ready"] = False
                st.stop()
            try:
                team_master, player_master, appearances = _load_cached_baseball(local_cache_dir)
                st.success(
                    f"Loaded local baseball cache: teams={len(team_master):,}, "
                    f"players={len(player_master):,}, appearances={len(appearances):,}"
                )
                with st.expander("Preview cached data", expanded=False):
                    st.write("Teams (head):", team_master.head())
                    st.write("Players (head):", player_master.head())
                    st.write("Appearances (head):", appearances.head())
            except Exception as exc:
                st.error(f"Failed to load local baseball cache: {exc}")
                st.info("Rebuild the cache with: `python src/scripts/build_baseball_cache.py`")
                st.session_state["simulator_ready"] = False
                st.stop()

        # If data not loaded, don't render the rest
        if team_master is None or player_master is None or appearances is None:
            return

        max_year = int(team_master["yearID"].max())
        teams_current = sorted(team_master.loc[team_master["yearID"] == max_year, "franchID"].apply(canonicalize_franchid).unique())
        team_name_to_code = {str(name): str(code).upper() for name, code in TEAM_LIST.items()}
        team_code_to_name = {code: name for name, code in team_name_to_code.items()}
        teams_from_json = [code for code in team_name_to_code.values() if code in set(teams_current)]
        selectable_team_codes = sorted(teams_from_json or teams_current)
        if teams_from_json:
            st.caption("Team options are sourced from `TEAM_LIST` JSON and filtered to currently available franchises.")
        else:
            st.caption("No overlap with TEAM_LIST JSON found; using available franchises from cache.")
        if len(selectable_team_codes) < 2:
            st.warning("Need at least two teams to run simulator checks.")
            return

        if "sim_team1" not in st.session_state or st.session_state["sim_team1"] not in selectable_team_codes:
            st.session_state["sim_team1"] = selectable_team_codes[0]
        if "sim_team2" not in st.session_state or st.session_state["sim_team2"] not in selectable_team_codes:
            st.session_state["sim_team2"] = selectable_team_codes[1]

        if st.button("Random teams"):
            t1, t2 = tuple(random.sample(selectable_team_codes, 2))
            st.session_state["sim_team1"] = t1
            st.session_state["sim_team2"] = t2

        sim_mode_tab, study_mode_tab, random_grid_tab, random_cube_tab = st.tabs(
            ["Intersection Checker", "Study Guide", "Random Immaculate Grid", "Random Immaculate Cube"]
        )

        with sim_mode_tab:
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
            else:
                matches = _closest_player_matches(player_master, first=first, last=last, top_n=5)
                if matches.empty:
                    st.warning("No matching players found.")
                else:
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
                        franchises = _get_franchises_for_player_id(team_master, appearances, selected_player_id)
                        franchises_set = {canonicalize_franchid(fid) for fid in franchises}
                        hits = team1 in franchises_set and team2 in franchises_set
                        st.write(f"Confirmed player: **{selected_name}** (`{selected_player_id}`)")
                        st.write(f"Teams: {', '.join(sorted(franchises_set)) or 'None'}")
                        if hits:
                            st.success(f"✅ Yes. {selected_name} played for both {team1} and {team2}.")
                        else:
                            st.error(f"❌ No. {selected_name} did not play for both {team1} and {team2}.")
                    else:
                        st.caption("Click **Confirm player and check** to run the Yes/No validation.")

        with study_mode_tab:
            st.markdown("### Study Guide")
            st.caption(
                "Ranks players by total distinct franchIDs. Optionally require a set of franchIDs and show top matches."
            )
            guide_df = _build_study_guide_df(player_master, team_master, appearances)
            if guide_df.empty:
                st.info("No study guide data available.")
            else:
                ranking_mode = st.selectbox(
                    "Ranking mode",
                    options=["Overall", "By League", "By Division"],
                    index=0,
                    key="study_guide_ranking_mode",
                )
                selected_league = None
                selected_division = None
                score_col = "distinct_franchids"
                score_label = "distinct_franchids"
                if ranking_mode == "By League":
                    leagues = sorted({str(v.get("league")).upper() for v in FRANCHID_MODERN_ALIGNMENT.values() if v.get("league")})
                    selected_league = st.selectbox("League", options=leagues, index=0, key="study_guide_league")
                    score_col = f"teams_in_{selected_league.lower()}"
                    score_label = f"teams_in_{selected_league}"
                elif ranking_mode == "By Division":
                    divisions = sorted(
                        {str(v.get("division")) for v in FRANCHID_MODERN_ALIGNMENT.values() if v.get("division")}
                    )
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
                    selected_set = {canonicalize_franchid(fid) for fid in selected_franchids}
                    filtered = guide_df[
                        guide_df["franchid_list"].apply(
                            lambda ids: selected_set.issubset({canonicalize_franchid(fid) for fid in ids})
                        )
                    ].copy()
                    st.write(
                        f"Players who include required set ({', '.join(selected_franchids)}): {len(filtered):,}"
                    )
                else:
                    filtered = guide_df.copy()
                    st.write(f"All ranked players: {len(filtered):,}")

                filtered = filtered.sort_values([score_col, "distinct_franchids", "player_name"], ascending=[False, False, True])
                filtered = filtered.head(top_n).copy()
                if score_col not in filtered.columns:
                    filtered[score_col] = 0
                filtered = filtered.rename(columns={score_col: score_label})
                display_cols = ["player_name", "key_bbref", score_label, "distinct_franchids", "franchids"]
                # Avoid duplicate column names in Overall mode (score_label == distinct_franchids).
                display_cols = list(dict.fromkeys(display_cols))
                st.dataframe(filtered[display_cols], use_container_width=True)

        player_sets = _build_player_franchise_sets(team_master, appearances)
        player_lookup_df, player_franch_map = _build_player_franchise_lookup(player_master, team_master, appearances)

        with random_grid_tab:
            st.markdown("### Random Immaculate Grid (3x3)")
            st.caption("Generates random 3x3 team intersections where every cell has at least 1 valid player.")
            if st.button("Generate random 3x3 puzzle", key="sim_generate_grid"):
                puzzle = _generate_random_grid_puzzle(selectable_team_codes, player_sets)
                st.session_state["sim_random_grid_puzzle"] = puzzle

            puzzle = st.session_state.get("sim_random_grid_puzzle")
            if not puzzle:
                st.info("Click generate to create a guaranteed-solvable 3x3 puzzle.")
            else:
                row_codes, col_codes, counts = puzzle
                df = pd.DataFrame(
                    counts,
                    index=[f"{team_code_to_name.get(c, c)} ({c})" for c in row_codes],
                    columns=[f"{team_code_to_name.get(c, c)} ({c})" for c in col_codes],
                )
                st.write("Cell solution counts")
                st.dataframe(df, use_container_width=True)
                st.success("All 9 cells have at least one valid solution.")

                st.markdown("#### Fill out puzzle")
                st.caption("Type any player spelling; checker uses fuzzy matching to resolve to the closest player.")
                for r_idx, r_code in enumerate(row_codes):
                    cols = st.columns(3)
                    for c_idx, c_code in enumerate(col_codes):
                        cell_key = f"sim_grid_answer_{r_idx}_{c_idx}"
                        cell_label = (
                            f"R{r_idx + 1}C{c_idx + 1}: "
                            f"{team_code_to_name.get(r_code, r_code)} + {team_code_to_name.get(c_code, c_code)}"
                        )
                        cols[c_idx].text_input(cell_label, key=cell_key)

                if st.button("Check puzzle (3x3)", key="sim_check_random_grid"):
                    result_rows = []
                    solved_count = 0
                    for r_idx, r_code in enumerate(row_codes):
                        for c_idx, c_code in enumerate(col_codes):
                            required = {canonicalize_franchid(r_code), canonicalize_franchid(c_code)}
                            answer = st.session_state.get(f"sim_grid_answer_{r_idx}_{c_idx}", "")
                            check = _validate_fuzzy_answer(answer, required, player_lookup_df, player_franch_map)
                            solved_count += int(check["ok"])
                            result_rows.append(
                                {
                                    "cell": f"R{r_idx + 1}C{c_idx + 1}",
                                    "required": " + ".join(sorted(required)),
                                    "answer": check["answer"],
                                    "matched_player": check["matched_player"],
                                    "player_id": check["player_id"],
                                    "confidence": f"{check['confidence']:.1%}",
                                    "ok": "yes" if check["ok"] else "no",
                                    "missing_franchids": ", ".join(check["missing_franchids"]),
                                }
                            )
                    st.write(f"Score: **{solved_count}/9**")
                    if solved_count == 9:
                        st.success("Puzzle solved.")
                    else:
                        st.warning("Puzzle not fully solved yet.")
                    st.dataframe(pd.DataFrame(result_rows), use_container_width=True)

        with random_cube_tab:
            st.markdown("### Random Immaculate Cube (3x3x3)")
            st.caption("Generates a random 3D puzzle where each cell (X,Y,Z) has at least 1 valid player.")
            if st.button("Generate random 3x3x3 puzzle", key="sim_generate_cube"):
                puzzle = _generate_random_cube_puzzle(selectable_team_codes, player_sets)
                st.session_state["sim_random_cube_puzzle"] = puzzle

            puzzle = st.session_state.get("sim_random_cube_puzzle")
            if not puzzle:
                st.info("Click generate to create a guaranteed-solvable 3x3x3 puzzle.")
            else:
                x_codes, y_codes, z_codes, cube = puzzle
                st.write("Axes")
                st.write("X-axis:", ", ".join(f"{team_code_to_name.get(c, c)} ({c})" for c in x_codes))
                st.write("Y-axis:", ", ".join(f"{team_code_to_name.get(c, c)} ({c})" for c in y_codes))
                st.write("Z-axis:", ", ".join(f"{team_code_to_name.get(c, c)} ({c})" for c in z_codes))

                for z_idx, z_code in enumerate(z_codes):
                    st.markdown(f"**Layer Z = {team_code_to_name.get(z_code, z_code)} ({z_code})**")
                    layer_df = pd.DataFrame(
                        cube[z_idx],
                        index=[f"{team_code_to_name.get(c, c)} ({c})" for c in x_codes],
                        columns=[f"{team_code_to_name.get(c, c)} ({c})" for c in y_codes],
                    )
                    st.dataframe(layer_df, use_container_width=True)
                st.success("All 27 cube cells have at least one valid solution.")

                st.markdown("#### Fill out cube")
                st.caption("Each cell requires a player who played for all 3 franchises at that coordinate.")
                for z_idx, z_code in enumerate(z_codes):
                    st.markdown(f"**Answer layer: Z = {team_code_to_name.get(z_code, z_code)} ({z_code})**")
                    for x_idx, x_code in enumerate(x_codes):
                        cols = st.columns(3)
                        for y_idx, y_code in enumerate(y_codes):
                            cell_key = f"sim_cube_answer_{z_idx}_{x_idx}_{y_idx}"
                            cell_label = (
                                f"Z{z_idx + 1}R{x_idx + 1}C{y_idx + 1}: "
                                f"{team_code_to_name.get(x_code, x_code)} + "
                                f"{team_code_to_name.get(y_code, y_code)} + "
                                f"{team_code_to_name.get(z_code, z_code)}"
                            )
                            cols[y_idx].text_input(cell_label, key=cell_key)

                if st.button("Check puzzle (3x3x3)", key="sim_check_random_cube"):
                    result_rows = []
                    solved_count = 0
                    for z_idx, z_code in enumerate(z_codes):
                        for x_idx, x_code in enumerate(x_codes):
                            for y_idx, y_code in enumerate(y_codes):
                                required = {
                                    canonicalize_franchid(x_code),
                                    canonicalize_franchid(y_code),
                                    canonicalize_franchid(z_code),
                                }
                                answer = st.session_state.get(f"sim_cube_answer_{z_idx}_{x_idx}_{y_idx}", "")
                                check = _validate_fuzzy_answer(answer, required, player_lookup_df, player_franch_map)
                                solved_count += int(check["ok"])
                                result_rows.append(
                                    {
                                        "cell": f"Z{z_idx + 1}R{x_idx + 1}C{y_idx + 1}",
                                        "required": " + ".join(sorted(required)),
                                        "answer": check["answer"],
                                        "matched_player": check["matched_player"],
                                        "player_id": check["player_id"],
                                        "confidence": f"{check['confidence']:.1%}",
                                        "ok": "yes" if check["ok"] else "no",
                                        "missing_franchids": ", ".join(check["missing_franchids"]),
                                    }
                                )
                    st.write(f"Score: **{solved_count}/27**")
                    if solved_count == 27:
                        st.success("Cube solved.")
                    else:
                        st.warning("Cube not fully solved yet.")
                    st.dataframe(pd.DataFrame(result_rows), use_container_width=True)

    with instructions_tab:
        st.markdown(
            """
            ### What this does
            The Simulator lets you test whether a player has appeared for both selected MLB franchises.

            ### Required local data
            This tab reads three local cache files from `bin/baseball_cache/`:
            - `teams.csv`
            - `players.csv`
            - `appearances.csv`

            If these files are missing, build them with:
            `python src/scripts/build_baseball_cache.py`

            ### How to use
            1. Open `Baseball Reference Data Management` and click **Build/Rebuild cache**.
            2. Open `Run Simulator` and click **Start simulator**.
            3. Pick **Team A** and **Team B** (or use **Random teams**).
            4. Enter player first and last name.
            5. Click **Check player** to verify if the player matches the intersection.

            ### Notes
            - Name matching depends on the cached player records (spelling matters).
            - Team checks are franchise-based (`franchID`) from Lahman data.
            """
        )


__all__ = ["render_simulator_tab", "load_pybaseball_data"]
