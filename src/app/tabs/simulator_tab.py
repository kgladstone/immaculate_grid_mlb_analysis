from __future__ import annotations

import re
import math
import random
import difflib
import itertools
from collections import defaultdict
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
from pathlib import Path
from app.services.data_loaders import load_image_metadata_df
from scripts.build_baseball_cache import build_cache
from scripts.build_career_war_cache import build_career_war_cache
from config.constants import FRANCHID_MODERN_ALIGNMENT, GRID_PLAYERS, TEAM_LIST, canonicalize_franchid


@st.cache_data(show_spinner=True)
def _load_cached_baseball(cache_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    teams = pd.read_csv(cache_dir / "teams.csv")
    players = pd.read_csv(cache_dir / "People.csv")
    players = players[
        ["playerID", "nameFirst", "nameLast", "birthYear", "birthMonth", "birthDay"]
    ].rename(
        columns={
            "playerID": "key_bbref",
            "nameFirst": "name_first",
            "nameLast": "name_last",
            "birthYear": "birth_year",
            "birthMonth": "birth_month",
            "birthDay": "birth_day",
        }
    )
    appearances = pd.read_csv(cache_dir / "appearances.csv")
    return teams, players, appearances


@st.cache_data(show_spinner=True)
def _load_oldest_player_cache(cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    first_team_path = cache_dir / "player_first_team.csv"
    oldest_path = cache_dir / "team_year_oldest_players.csv"
    first_team = pd.read_csv(first_team_path) if first_team_path.exists() else pd.DataFrame()
    oldest = pd.read_csv(oldest_path) if oldest_path.exists() else pd.DataFrame()
    return first_team, oldest


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


def _sample_reverse_grid_puzzle(
    selectable_codes: list[str],
    player_franch_map: dict[str, set[str]],
    player_lookup_df: pd.DataFrame,
    allowed_player_ids: set[str] | None = None,
    max_attempts: int = 300,
) -> dict[str, object] | None:
    """
    Generate a reverse immaculate grid puzzle:
    - pick a solvable 3x3 team grid
    - pick one valid player for each cell
    - return 9 players (shuffled) plus the hidden solution mapping
    """
    if len(selectable_codes) < 6:
        return None

    id_to_name = (
        player_lookup_df[["key_bbref", "full_name"]]
        .dropna(subset=["key_bbref", "full_name"])
        .drop_duplicates(subset=["key_bbref"])
        .set_index("key_bbref")["full_name"]
        .to_dict()
    )

    allowed_ids = {str(pid) for pid in allowed_player_ids} if allowed_player_ids else set()
    all_player_ids = [
        str(pid)
        for pid, teams in player_franch_map.items()
        if teams and (not allowed_ids or str(pid) in allowed_ids)
    ]
    if len(all_player_ids) < 9:
        return None

    def _candidate_ids(required: set[str]) -> list[str]:
        return [
            str(pid)
            for pid, teams in player_franch_map.items()
            if required.issubset(teams)
            and str(pid) in id_to_name
            and (not allowed_ids or str(pid) in allowed_ids)
        ]

    def _pick_unique(cells: list[tuple[str, str, list[str]]]) -> dict[tuple[str, str], str] | None:
        # Backtracking over smallest candidate sets first.
        ordered = sorted(cells, key=lambda c: len(c[2]))
        used: set[str] = set()
        out: dict[tuple[str, str], str] = {}

        def rec(i: int) -> bool:
            if i >= len(ordered):
                return True
            r, c, candidates = ordered[i]
            pool = candidates[:]
            random.shuffle(pool)
            for pid in pool:
                if pid in used:
                    continue
                used.add(pid)
                out[(r, c)] = pid
                if rec(i + 1):
                    return True
                used.remove(pid)
                out.pop((r, c), None)
            return False

        return out if rec(0) else None

    for _ in range(max_attempts):
        picked = random.sample(selectable_codes, 6)
        row_codes, col_codes = picked[:3], picked[3:]
        cells = []
        valid = True
        for r in row_codes:
            for c in col_codes:
                req = {canonicalize_franchid(r), canonicalize_franchid(c)}
                candidates = _candidate_ids(req)
                if not candidates:
                    valid = False
                    break
                cells.append((r, c, candidates))
            if not valid:
                break
        if not valid:
            continue

        assignment = _pick_unique(cells)
        if not assignment:
            continue

        solution_grid_ids = [[assignment[(r, c)] for c in col_codes] for r in row_codes]
        solution_grid_names = [[id_to_name.get(pid, pid) for pid in row] for row in solution_grid_ids]
        players = [pid for row in solution_grid_ids for pid in row]
        player_cards = [{"player_id": pid, "player_name": id_to_name.get(pid, pid)} for pid in players]
        random.shuffle(player_cards)

        counts = _build_grid_counts(row_codes, col_codes, list(player_franch_map.values()))
        return {
            "row_codes": row_codes,
            "col_codes": col_codes,
            "counts": counts,
            "player_cards": player_cards,
            "solution_grid_ids": solution_grid_ids,
            "solution_grid_names": solution_grid_names,
        }

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


def _build_used_player_id_set(
    images_df: pd.DataFrame,
    players_df: pd.DataFrame,
    submitters: list[str] | None = None,
) -> set[str]:
    if images_df is None or images_df.empty or players_df is None or players_df.empty:
        return set()
    if submitters is not None:
        allowed = {str(s).strip() for s in submitters}
        images_df = images_df[images_df["submitter"].astype(str).isin(allowed)]
        if images_df.empty:
            return set()

    players = players_df.copy()
    players["full_name_norm"] = players["full_name"].astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    players = players[players["full_name_norm"] != ""]
    name_to_id = dict(zip(players["full_name_norm"], players["key_bbref"].astype(str)))

    used_names: set[str] = set()
    for _, row in images_df.iterrows():
        responses = row.get("responses")
        if not isinstance(responses, dict):
            continue
        for val in responses.values():
            name = str(val or "").strip().lower()
            name = re.sub(r"\s+", " ", name)
            if name:
                used_names.add(name)

    used_ids: set[str] = set()
    for nm in used_names:
        if nm in name_to_id:
            used_ids.add(name_to_id[nm])
    return used_ids


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


def _style_yes_no_columns(df: pd.DataFrame, columns: list[str]):
    def _cell_style(value: object) -> str:
        v = str(value).strip().lower()
        if v == "yes":
            return "background-color: #d1fae5; color: #065f46; font-weight: 700;"
        if v == "no":
            return "background-color: #fee2e2; color: #991b1b; font-weight: 700;"
        return ""

    if df is None or df.empty:
        return df
    valid_cols = [c for c in columns if c in df.columns]
    if not valid_cols:
        return df
    return df.style.map(_cell_style, subset=valid_cols)


def _safe_int(value: object) -> Optional[int]:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def _format_birth_text(birth_year: object, birth_month: object, birth_day: object) -> str:
    by = _safe_int(birth_year)
    bm = _safe_int(birth_month)
    bd = _safe_int(birth_day)
    if by is None:
        return "unknown"
    if bm is None or bd is None:
        return str(by)
    return f"{by:04d}-{bm:02d}-{bd:02d}"


def _player_label(row: pd.Series) -> str:
    first = str(row.get("nameFirst", row.get("name_first", "")) or "").strip()
    last = str(row.get("nameLast", row.get("name_last", "")) or "").strip()
    pid = str(row.get("playerID", row.get("key_bbref", "")) or "").strip()
    name = f"{first} {last}".strip() or pid or "Unknown"
    birth_text = _format_birth_text(
        row.get("birthYear", row.get("birth_year")),
        row.get("birthMonth", row.get("birth_month")),
        row.get("birthDay", row.get("birth_day")),
    )
    label = f"{name} (`{pid}`)" if pid else name
    return f"{label}, born {birth_text}"


def _oldest_player_for_team_year(oldest_df: pd.DataFrame, franchid: str, year: int) -> Optional[pd.Series]:
    team_code = canonicalize_franchid(franchid)
    if not team_code:
        return None
    subset = oldest_df[
        (oldest_df["franchID"].astype(str).str.upper() == str(team_code).upper())
        & (pd.to_numeric(oldest_df["yearID"], errors="coerce").astype("Int64") == int(year))
    ].copy()
    if subset.empty:
        return None
    ordered = subset.sort_values(
        ["age_on_july_1", "birthYear", "birthMonth", "birthDay", "playerID"],
        ascending=[False, True, True, True, True],
    )
    return ordered.head(1).iloc[0]


def _league_for_franchid(franchid: str) -> str:
    return str((FRANCHID_MODERN_ALIGNMENT.get(canonicalize_franchid(franchid)) or {}).get("league", "")).upper()


def _build_oldest_chain(
    oldest_df: pd.DataFrame,
    start_franchid: str,
    start_year: int,
    max_hops: int = 12,
    allowed_leagues: Optional[set[str]] = None,
) -> tuple[pd.DataFrame, str]:
    records: list[dict[str, object]] = []
    visited_team_year: set[tuple[str, int]] = set()
    current_team = canonicalize_franchid(start_franchid)
    current_year = int(start_year)
    stop_reason = f"Stopped after reaching max hops ({max_hops})."

    for hop in range(1, max_hops + 1):
        if allowed_leagues and _league_for_franchid(current_team) not in allowed_leagues:
            stop_reason = f"Stopped: {current_team} is outside selected leagues ({', '.join(sorted(allowed_leagues))})."
            break
        key = (str(current_team), int(current_year))
        if key in visited_team_year:
            stop_reason = f"Stopped on loop: {current_team} {current_year} already visited."
            break
        visited_team_year.add(key)

        oldest_row = _oldest_player_for_team_year(oldest_df, current_team, current_year)
        if oldest_row is None:
            stop_reason = f"Stopped: no oldest-player row for {current_team} {current_year}."
            break

        first_team = canonicalize_franchid(oldest_row.get("first_franchID", ""))
        first_year = _safe_int(oldest_row.get("first_yearID"))
        if allowed_leagues and (not first_team or _league_for_franchid(first_team) not in allowed_leagues):
            records.append(
                {
                    "hop": hop,
                    "oldest_team": current_team,
                    "oldest_year": current_year,
                    "oldest_player": _player_label(oldest_row),
                    "first_team": "",
                    "first_year": "",
                    "next_oldest_player": "",
                }
            )
            stop_reason = (
                "Stopped: next first team is outside selected leagues "
                f"({', '.join(sorted(allowed_leagues))})."
            )
            break
        next_allowed = not allowed_leagues or (first_team and _league_for_franchid(first_team) in allowed_leagues)
        next_oldest = (
            _oldest_player_for_team_year(oldest_df, first_team or "", first_year)
            if first_team and first_year and next_allowed
            else None
        )

        records.append(
            {
                "hop": hop,
                "oldest_team": current_team,
                "oldest_year": current_year,
                "oldest_player": _player_label(oldest_row),
                "first_team": first_team or "",
                "first_year": first_year if first_year is not None else "",
                "next_oldest_player": _player_label(next_oldest) if next_oldest is not None else "",
            }
        )

        if not first_team or first_year is None:
            stop_reason = f"Stopped: first-team metadata missing for {_player_label(oldest_row)}."
            break
        if (first_team, int(first_year)) in visited_team_year:
            stop_reason = f"Stopped on loop: next node {first_team} {first_year} already visited."
            break
        if next_oldest is None:
            stop_reason = f"Stopped: no oldest-player row for next node {first_team} {first_year}."
            break

        current_team, current_year = first_team, int(first_year)

    return pd.DataFrame(records), stop_reason


def _build_oldest_teammate_chain(
    start_player_id: str,
    appearances: pd.DataFrame,
    team_master: pd.DataFrame,
    player_master: pd.DataFrame,
    max_hops: int = 12,
    allowed_leagues: Optional[set[str]] = None,
    start_team_years: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, str]:
    apps = appearances[["playerID", "teamID", "yearID"]].copy()
    apps["playerID"] = apps["playerID"].astype(str)
    apps = apps.merge(team_master[["teamID", "yearID", "franchID"]], on=["teamID", "yearID"], how="left")
    apps["franchID"] = apps["franchID"].apply(canonicalize_franchid)
    if allowed_leagues:
        apps = apps[
            apps["franchID"].apply(lambda code: _league_for_franchid(code) in allowed_leagues)
        ].copy()

    players = player_master.copy()
    players["key_bbref"] = players["key_bbref"].astype(str)
    players["birth_year_sort"] = pd.to_numeric(players["birth_year"], errors="coerce").fillna(9999).astype(int)
    players["birth_month_sort"] = pd.to_numeric(players["birth_month"], errors="coerce").fillna(13).astype(int)
    players["birth_day_sort"] = pd.to_numeric(players["birth_day"], errors="coerce").fillna(32).astype(int)
    player_index = players.set_index("key_bbref", drop=False)

    def _birth_sort_tuple(row: pd.Series) -> tuple[int, int, int]:
        return (
            int(row.get("birth_year_sort", 9999)),
            int(row.get("birth_month_sort", 13)),
            int(row.get("birth_day_sort", 32)),
        )

    def _player_series(pid: str) -> pd.Series:
        if pid in player_index.index:
            return player_index.loc[pid]
        return pd.Series({"key_bbref": pid, "name_first": "", "name_last": ""})

    records: list[dict[str, object]] = []
    visited_players: set[str] = set()
    current_player = str(start_player_id)
    stop_reason = f"Stopped after reaching max hops ({max_hops})."

    for hop in range(1, max_hops + 1):
        if current_player in visited_players:
            stop_reason = f"Stopped on loop: {current_player} already visited."
            break
        visited_players.add(current_player)

        current_row = _player_series(current_player)
        current_birth = _birth_sort_tuple(current_row)
        current_apps = apps[apps["playerID"] == current_player][["teamID", "yearID"]].drop_duplicates()
        if hop == 1 and start_team_years is not None and not start_team_years.empty:
            current_apps = start_team_years[["teamID", "yearID"]].drop_duplicates().copy()
        if current_apps.empty:
            stop_reason = f"Stopped: no roster records for {current_player} with selected filters."
            break

        overlap = apps.merge(current_apps, on=["teamID", "yearID"], how="inner")
        overlap = overlap[overlap["playerID"] != current_player].copy()
        if overlap.empty:
            stop_reason = f"Stopped: no teammates found for {current_player}."
            break

        overlap = overlap.merge(
            players[
                [
                    "key_bbref",
                    "name_first",
                    "name_last",
                    "birth_year",
                    "birth_month",
                    "birth_day",
                    "birth_year_sort",
                    "birth_month_sort",
                    "birth_day_sort",
                ]
            ],
            left_on="playerID",
            right_on="key_bbref",
            how="left",
        )
        overlap = overlap.sort_values(
            ["birth_year_sort", "birth_month_sort", "birth_day_sort", "playerID", "yearID", "teamID"],
            ascending=[True, True, True, True, True, True],
        )
        oldest_teammate = overlap.head(1).iloc[0]
        next_player_id = str(oldest_teammate["playerID"])

        # Terminate if current player is already as old or older than any teammate.
        if _birth_sort_tuple(current_row) <= _birth_sort_tuple(oldest_teammate):
            stop_reason = f"Stopped: {current_player} is already the oldest rostered player in this chain step."
            break

        next_row = _player_series(next_player_id)
        shared_code = canonicalize_franchid(oldest_teammate.get("franchID", ""))
        shared_year = _safe_int(oldest_teammate.get("yearID"))

        records.append(
            {
                "hop": hop,
                "player": _player_label(current_row),
                "shared_team": shared_code,
                "shared_year": shared_year if shared_year is not None else "",
                "oldest_teammate": _player_label(next_row),
                "next_player_id": next_player_id,
            }
        )

        if next_player_id in visited_players:
            stop_reason = f"Stopped on loop: next player {next_player_id} already visited."
            break
        current_player = next_player_id

    return pd.DataFrame(records), stop_reason


def _extract_player_id_from_label(label: object) -> str:
    text = str(label or "")
    match = re.search(r"\(`([^`]+)`\)", text)
    return str(match.group(1)).strip() if match else ""


def _player_bbr_url(player_id: str) -> str:
    pid = str(player_id or "").strip()
    if not pid:
        return ""
    return f"https://www.baseball-reference.com/players/{pid[0].lower()}/{pid}.shtml"


def _player_link_markdown_from_label(label: object) -> str:
    text = str(label or "")
    pid = _extract_player_id_from_label(text)
    if not pid:
        return text
    name_part = text.split(" (`", 1)[0].strip()
    born_part = ""
    if ", born " in text:
        born_part = text.split(", born ", 1)[1].strip()
    linked_id = f"[{pid}]({_player_bbr_url(pid)})"
    out = f"**{name_part}** ({linked_id})"
    return f"{out}, born {born_part}" if born_part else out


def _bbr_id_from_url(url: object) -> str:
    m = re.search(r"/([a-z0-9]+)\.shtml$", str(url or ""))
    return str(m.group(1)) if m else ""


def _render_html_summary_table(df: pd.DataFrame, columns: list[str], link_columns: list[str]) -> None:
    show_df = df[columns].copy()
    for col in link_columns:
        if col not in show_df.columns:
            continue
        show_df[col] = show_df[col].apply(
            lambda u: (
                f'<a href="{u}" target="_blank" rel="noopener noreferrer">{_bbr_id_from_url(u)}</a>'
                if str(u).strip()
                else ""
            )
        )
    st.markdown(show_df.to_html(index=False, escape=False), unsafe_allow_html=True)


def _build_chain_player_ids(chain_df: pd.DataFrame, chain_mode: str, source_player_id: str = "") -> list[str]:
    if chain_df is None or chain_df.empty:
        return []

    ids: list[str] = []
    if chain_mode == "First-year oldest on first team":
        first_id = _extract_player_id_from_label(chain_df.iloc[0].get("oldest_player", ""))
        if first_id:
            ids.append(first_id)
        for _, row in chain_df.iterrows():
            next_id = _extract_player_id_from_label(row.get("next_oldest_player", ""))
            if next_id:
                ids.append(next_id)
    else:
        first_id = _extract_player_id_from_label(chain_df.iloc[0].get("player", ""))
        if first_id:
            ids.append(first_id)
        for _, row in chain_df.iterrows():
            next_id = _extract_player_id_from_label(row.get("oldest_teammate", ""))
            if next_id:
                ids.append(next_id)

    if str(source_player_id).strip():
        ids = [str(source_player_id).strip()] + ids

    # Keep first occurrence order while removing duplicates from accidental repeats.
    seen: set[str] = set()
    out: list[str] = []
    for pid in ids:
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
    return out


def _render_interactive_chain_graph(
    chain_df: pd.DataFrame,
    chain_mode: str,
    player_master: pd.DataFrame,
    player_first_team_df: pd.DataFrame,
    team_code_to_name: dict[str, str],
    source_player_id: str = "",
) -> None:
    if chain_df is None or chain_df.empty:
        return
    try:
        import plotly.graph_objects as go
    except Exception:
        st.info("Install plotly to enable interactive chain graph (`pip install plotly`).")
        return

    player_ids = _build_chain_player_ids(chain_df, chain_mode, source_player_id=source_player_id)
    if not player_ids:
        return

    pm = player_master[["key_bbref", "name_first", "name_last", "birth_year"]].copy()
    pm["key_bbref"] = pm["key_bbref"].astype(str)
    pft = player_first_team_df[["playerID", "first_franchID", "first_yearID"]].copy()
    pft["playerID"] = pft["playerID"].astype(str)
    pft["first_franchID"] = pft["first_franchID"].apply(canonicalize_franchid)
    pft["first_yearID"] = pd.to_numeric(pft["first_yearID"], errors="coerce").astype("Int64")

    points = []
    for idx, pid in enumerate(player_ids):
        prow = pm.loc[pm["key_bbref"] == str(pid)].head(1)
        name = pid
        born_year = "unknown"
        if not prow.empty:
            pr = prow.iloc[0]
            full_name = f"{str(pr.get('name_first', '')).strip()} {str(pr.get('name_last', '')).strip()}".strip()
            name = full_name or pid
            by = _safe_int(pr.get("birth_year"))
            born_year = str(by) if by is not None else "unknown"

        frow = pft.loc[pft["playerID"] == str(pid)].sort_values(["first_yearID", "first_franchID"], ascending=[True, True]).head(1)
        if frow.empty:
            first_team_text = "unknown"
        else:
            fr = frow.iloc[0]
            first_code = str(fr.get("first_franchID", "") or "")
            first_year = _safe_int(fr.get("first_yearID"))
            team_label = f"{team_code_to_name.get(first_code, first_code)} ({first_code})" if first_code else "unknown"
            first_team_text = f"{team_label} in {first_year}" if first_year is not None else team_label

        points.append(
            {
                "x": idx,
                "y": 0,
                "player_name": name,
                "player_id": pid,
                "born_year": born_year,
                "first_team": first_team_text,
            }
        )

    x = [p["x"] for p in points]
    y = [p["y"] for p in points]
    hover = [
        f"Player: {p['player_name']} ({p['player_id']})<br>Born: {p['born_year']}<br>First MLB team: {p['first_team']}"
        for p in points
    ]
    text = [p["player_name"] for p in points]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line={"width": 2, "color": "#64748b"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers+text",
            marker={"size": 14, "color": "#0f766e"},
            text=text,
            textposition="top center",
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover,
            showlegend=False,
        )
    )
    fig.update_layout(
        height=360,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis={"showgrid": False, "zeroline": False, "showticklabels": False, "title": ""},
        yaxis={"showgrid": False, "zeroline": False, "showticklabels": False, "title": ""},
        dragmode="pan",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})


def _render_full_web_graph(
    team_master: pd.DataFrame,
    appearances: pd.DataFrame,
    player_master: pd.DataFrame,
    player_first_team_df: pd.DataFrame,
    team_code_to_name: dict[str, str],
    available_years: list[int],
    selected_year_override: Optional[int] = None,
    allowed_leagues_override: Optional[set[str]] = None,
    source_team_override: str = "",
    source_player_override: str = "",
) -> None:
    if not available_years:
        st.info("No years available for full web.")
        return
    try:
        import plotly.graph_objects as go
    except Exception:
        st.info("Install plotly to enable Full Web (`pip install plotly`).")
        return

    st.caption(
        "Focused full web: select a source player, then show all inlinks to that player, "
        "then inlinks to the next outlink target, and so on."
    )
    if selected_year_override is not None:
        selected_year = int(selected_year_override)
        limit_to_major = allowed_leagues_override is not None and {"AL", "NL"}.issubset(set(allowed_leagues_override))
    else:
        selected_year = st.selectbox(
            "Season year (Full Web)",
            options=available_years,
            index=0,
            key="sim_fullweb_year",
        )
        limit_to_major = st.toggle(
            "Limit to NL/AL (Full Web)",
            value=True,
            key="sim_fullweb_limit_nl_al",
        )
    max_steps = st.slider("Cascade depth", min_value=1, max_value=12, value=6, step=1, key="sim_fullweb_depth")
    direction_mode = st.radio(
        "Direction",
        options=["Forward", "Backward", "Both"],
        index=0,
        key="sim_fullweb_direction",
        help="Forward follows outlinks, Backward follows reverse outlinks, Both expands in both directions.",
    )
    show_labels = st.toggle("Show player labels", value=False, key="sim_fullweb_show_labels")

    tm = team_master.copy()
    tm["franchID"] = tm["franchID"].apply(canonicalize_franchid)
    tm = tm[pd.to_numeric(tm["yearID"], errors="coerce").astype("Int64") == int(selected_year)].copy()
    if limit_to_major:
        tm = tm[tm["franchID"].apply(lambda code: _league_for_franchid(code) in {"AL", "NL"})].copy()
    if tm.empty:
        st.info("No teams found for the selected year/filter.")
        return

    team_options = sorted(
        tm["franchID"].dropna().astype(str).unique().tolist(),
        key=lambda code: str(team_code_to_name.get(code, code)).lower(),
    )
    if not team_options:
        st.info("No teams available for selected filters.")
        return
    if source_team_override and source_team_override in team_options:
        selected_team = str(source_team_override)
    else:
        selected_team = st.selectbox(
            "Team (Full Web source)",
            options=team_options,
            index=0,
            key="sim_fullweb_team",
            format_func=lambda code: f"{team_code_to_name.get(code, code)} ({code})",
        )
    year_team_ids = tm.loc[tm["franchID"] == selected_team, "teamID"].dropna().astype(str).unique().tolist()
    if not year_team_ids:
        st.info("No team IDs found for selected team/year.")
        return

    # Source picker context: selected season/team roster.
    apps_year = appearances.copy()
    apps_year = apps_year[pd.to_numeric(apps_year["yearID"], errors="coerce").astype("Int64") == int(selected_year)].copy()
    apps_year = apps_year[apps_year["teamID"].astype(str).isin(set(tm["teamID"].dropna().astype(str).unique().tolist()))].copy()
    apps_year["playerID"] = apps_year["playerID"].astype(str)
    if apps_year.empty:
        st.info("No appearances found for selected year/filter.")
        return

    source_roster_apps = apps_year[apps_year["teamID"].astype(str).isin(set(year_team_ids))].copy()
    source_player_ids = source_roster_apps["playerID"].dropna().astype(str).drop_duplicates().tolist()
    if not source_player_ids:
        st.info("No rostered players for selected source team/year.")
        return

    pm = player_master[["key_bbref", "name_first", "name_last", "birth_year", "birth_month", "birth_day"]].copy()
    pm["key_bbref"] = pm["key_bbref"].astype(str)
    pm = pm.drop_duplicates(subset=["key_bbref"])
    pm["display_name"] = (pm["name_first"].astype(str).str.strip() + " " + pm["name_last"].astype(str).str.strip()).str.strip()
    pm["birth_year_sort"] = pd.to_numeric(pm["birth_year"], errors="coerce").fillna(9999).astype(int)
    pm["birth_month_sort"] = pd.to_numeric(pm["birth_month"], errors="coerce").fillna(13).astype(int)
    pm["birth_day_sort"] = pd.to_numeric(pm["birth_day"], errors="coerce").fillna(32).astype(int)

    source_catalog = pm[pm["key_bbref"].isin(set(source_player_ids))].copy()
    source_catalog = source_catalog.sort_values(["name_last", "name_first", "key_bbref"], ascending=[True, True, True])
    option_ids = source_catalog["key_bbref"].astype(str).tolist()
    if not option_ids:
        st.info("No source players available after filters.")
        return
    if source_player_override and source_player_override in option_ids:
        source_player_id = str(source_player_override)
    else:
        source_player_id = st.selectbox(
            "Player (Full Web source)",
            options=option_ids,
            index=0,
            key="sim_fullweb_player",
            format_func=lambda pid: f"{source_catalog.loc[source_catalog['key_bbref'] == pid, 'display_name'].iloc[0]} ({pid})",
        )

    # Graph/outlink context: all-years teammate data (same basis as Specific Search).
    apps_all = appearances.copy()
    apps_all = apps_all.merge(team_master[["teamID", "yearID", "franchID"]], on=["teamID", "yearID"], how="left")
    apps_all["franchID"] = apps_all["franchID"].apply(canonicalize_franchid)
    if limit_to_major:
        apps_all = apps_all[
            apps_all["franchID"].apply(lambda code: _league_for_franchid(code) in {"AL", "NL"})
        ].copy()
    apps_all["playerID"] = apps_all["playerID"].astype(str)
    if apps_all.empty:
        st.info("No all-years appearances found for selected filters.")
        return

    roster_by_team: dict[tuple[str, int], list[str]] = {}
    for (tid, yid), grp in apps_all.groupby(
        [apps_all["teamID"].astype(str), pd.to_numeric(apps_all["yearID"], errors="coerce").astype("Int64")]
    ):
        if pd.isna(yid):
            continue
        pids = grp["playerID"].dropna().astype(str).drop_duplicates().tolist()
        if pids:
            roster_by_team[(str(tid), int(yid))] = pids
    if not roster_by_team:
        st.info("No team rosters found.")
        return

    teammates: dict[str, set[str]] = defaultdict(set)
    for pids in roster_by_team.values():
        unique_p = list(dict.fromkeys(pids))
        for p in unique_p:
            mates = [m for m in unique_p if m != p]
            if mates:
                teammates[p].update(mates)

    birth_sort_map = {
        str(r["key_bbref"]): (
            int(r["birth_year_sort"]),
            int(r["birth_month_sort"]),
            int(r["birth_day_sort"]),
        )
        for _, r in pm.iterrows()
    }

    outlink: dict[str, str] = {}
    for pid, mates in teammates.items():
        mate_list = sorted(set(mates), key=lambda m: (birth_sort_map.get(m, (9999, 13, 32)), m))
        if mate_list:
            outlink[pid] = mate_list[0]

    reverse_links: dict[str, set[str]] = defaultdict(set)
    for src, dst in outlink.items():
        reverse_links[dst].add(src)

    def _next_forward(pid: str) -> str:
        return str(outlink.get(pid, "") or "")

    def _next_backward(pid: str) -> str:
        parents = sorted(
            reverse_links.get(pid, set()),
            key=lambda m: (birth_sort_map.get(m, (9999, 13, 32)), m),
        )
        return str(parents[0]) if parents else ""

    source_pid = str(source_player_id)
    forward_chain: list[str] = [source_pid]
    seen_f = {source_pid}
    for _ in range(max_steps):
        nxt = _next_forward(forward_chain[-1])
        if not nxt or nxt in seen_f:
            break
        forward_chain.append(nxt)
        seen_f.add(nxt)

    backward_chain: list[str] = [source_pid]
    seen_b = {source_pid}
    for _ in range(max_steps):
        nxt = _next_backward(backward_chain[-1])
        if not nxt or nxt in seen_b:
            break
        backward_chain.append(nxt)
        seen_b.add(nxt)

    if direction_mode == "Forward":
        target_chain = forward_chain
    elif direction_mode == "Backward":
        target_chain = backward_chain
    else:
        # left side (oldest reverse path) -> source -> right side (forward path)
        left = list(reversed(backward_chain[1:]))
        right = forward_chain
        target_chain = left + right

    edge_set: set[tuple[str, str]] = set()
    node_layers: dict[str, set[int]] = defaultdict(set)
    for i, target in enumerate(target_chain):
        center_layer = i * 2
        node_layers[target].add(center_layer)
        parents = sorted(reverse_links.get(target, set()))
        for parent in parents:
            edge_set.add((parent, target))
            node_layers[parent].add(center_layer - 1)
        if i < len(target_chain) - 1:
            edge_set.add((target, target_chain[i + 1]))
            node_layers[target_chain[i + 1]].add((i + 1) * 2)

    nodes = sorted(node_layers.keys())
    if not nodes:
        st.info("No nodes found for selected player in this season.")
        return

    # Layered layout: targets on even columns, inlink feeders on odd columns.
    x_map: dict[str, float] = {}
    y_map: dict[str, float] = {}
    layers_to_nodes: dict[int, list[str]] = defaultdict(list)
    for n in nodes:
        layer = min(node_layers[n])
        layers_to_nodes[layer].append(n)
    for layer, layer_nodes in layers_to_nodes.items():
        ordered = sorted(
            layer_nodes,
            key=lambda pid: (
                pm.loc[pm["key_bbref"] == pid, "birth_year_sort"].head(1).iloc[0]
                if not pm.loc[pm["key_bbref"] == pid].empty
                else 9999,
                pid,
            ),
        )
        count = len(ordered)
        for j, pid in enumerate(ordered):
            x_map[pid] = float(layer)
            y_map[pid] = float(j - (count - 1) / 2.0)

    pft = player_first_team_df[["playerID", "first_franchID", "first_yearID"]].copy()
    pft["playerID"] = pft["playerID"].astype(str)
    pft["first_franchID"] = pft["first_franchID"].apply(canonicalize_franchid)
    pft["first_yearID"] = pd.to_numeric(pft["first_yearID"], errors="coerce").astype("Int64")

    node_x = [x_map[n] for n in nodes]
    node_y = [y_map[n] for n in nodes]
    labels = []
    hover = []
    for pid in nodes:
        prow = pm.loc[pm["key_bbref"] == pid].head(1)
        if prow.empty:
            name = pid
            born_year = "unknown"
        else:
            pr = prow.iloc[0]
            nm = str(pr.get("display_name", "")).strip()
            name = nm or pid
            born_year = str(_safe_int(pr.get("birth_year")) or "unknown")
        frow = pft.loc[pft["playerID"] == pid].sort_values(["first_yearID", "first_franchID"]).head(1)
        if frow.empty:
            first_text = "unknown"
        else:
            fr = frow.iloc[0]
            code = str(fr.get("first_franchID", "") or "")
            yr = _safe_int(fr.get("first_yearID"))
            team_label = f"{team_code_to_name.get(code, code)} ({code})" if code else "unknown"
            first_text = f"{team_label} in {yr}" if yr is not None else team_label
        labels.append(name if show_labels else "")
        hover.append(
            f"Player: {name} ({pid})<br>Born: {born_year}<br>First MLB team: {first_text}<br>"
            f"Outlink: {outlink.get(pid, 'None')}"
        )

    edge_x = []
    edge_y = []
    for a, b in edge_set:
        if a not in x_map or b not in x_map:
            continue
        edge_x.extend([x_map[a], x_map[b], None])
        edge_y.extend([y_map[a], y_map[b], None])

    st.write(
        f"Source season {selected_year} (graph basis: all-years): source={source_player_id}, direction={direction_mode}, "
        f"targets={len(target_chain):,}, nodes={len(nodes):,}, edges={len(edge_set):,}"
    )
    source_out = outlink.get(str(source_player_id), "")
    if source_out:
        st.write(f"Source outlink: `{source_player_id}` -> `{source_out}`")
    else:
        st.write(f"Source outlink: `{source_player_id}` -> `None`")
    if len(target_chain) > 1:
        st.write(f"First cascade hop: `{target_chain[0]}` -> `{target_chain[1]}`")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"width": 1.1, "color": "rgba(100,116,139,0.45)"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text" if show_labels else "markers",
            marker={
                "size": [12 if n in set(target_chain) else 8 for n in nodes],
                "color": ["#0f766e" if n in set(target_chain) else "#334155" for n in nodes],
                "line": {"width": 1, "color": "#ffffff"},
            },
            text=labels if show_labels else None,
            textposition="top center",
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        height=700,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis={
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "range": [
                x_map.get(str(source_player_id), 0.0) - 1.5,
                x_map.get(str(source_player_id), 0.0) + 3.5,
            ],
        },
        yaxis={
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "range": [
                y_map.get(str(source_player_id), 0.0) - 6.0,
                y_map.get(str(source_player_id), 0.0) + 6.0,
            ],
        },
        dragmode="pan",
        template="plotly_white",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})


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
        _csv_attrs(cache_dir / "People.csv"),
        _csv_attrs(cache_dir / "appearances.csv"),
        _csv_attrs(cache_dir / "player_first_team.csv"),
        _csv_attrs(cache_dir / "team_year_oldest_players.csv"),
    ]
    st.write("Required cache files")
    st.dataframe(pd.DataFrame(file_rows), use_container_width=True)


def _render_data_explorer(cache_dir: Path) -> None:
    st.caption(f"Source directory: `{cache_dir}`")
    people_path = cache_dir / "People.csv"
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

    if df.empty:
        st.info("No player dataset found. Build/rebuild cache first.")
        return

    st.write(f"Loaded `{source_name}` with {len(df):,} rows.")
    if {"playerID", "nameFirst", "nameLast"}.issubset(df.columns):
        catalog = df[["playerID", "nameFirst", "nameLast"]].drop_duplicates(subset=["playerID"]).copy()
        catalog = catalog.rename(columns={"playerID": "player_id", "nameFirst": "name_first", "nameLast": "name_last"})
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
    required_files = ["teams.csv", "People.csv", "appearances.csv"]
    missing_files = [name for name in required_files if not (local_cache_dir / name).exists()]
    run_simulator_tab, cache_tools_tab, instructions_tab = st.tabs(
        ["🎮 Play", "🧾 Cache Tools", "📘 Instructions"]
    )

    with cache_tools_tab:
        metadata_tab, explorer_tab = st.tabs(["🧾 Metadata", "🔎 Data Explorer"])
        with metadata_tab:
            _render_cache_viewer(local_cache_dir)
        with explorer_tab:
            _render_data_explorer(local_cache_dir)

    with run_simulator_tab:
        st.write("Check if a player fits a team intersection using locally cached Lahman data.")
        team_master = player_master = appearances = None
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
            st.stop()

        # If data not loaded, don't render the rest
        if team_master is None or player_master is None or appearances is None:
            return
        player_first_team_df, oldest_by_team_year_df = _load_oldest_player_cache(local_cache_dir)

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

        sim_mode_tab, study_mode_tab, oldest_mode_tab, random_grid_tab, random_cube_tab, reverse_grid_tab = st.tabs(
            [
                "✅ Intersection Checker",
                "📚 Study Guide",
                "🌿 Oldest Player on Roster",
                "🎲 Random Immaculate Grid",
                "🧊 Random Immaculate Cube",
                "🔁 Reverse Immaculate Grid",
            ]
        )

        with sim_mode_tab:
            st.markdown("### Pick teams")
            col_a, col_b = st.columns([1, 1])
            with col_a:
                team1 = st.selectbox(
                    "Team A",
                    options=selectable_team_codes,
                    key="sim_team1",
                    format_func=lambda code: f"{team_code_to_name.get(code, code)} ({code})",
                )
            with col_b:
                team2 = st.selectbox(
                    "Team B",
                    options=selectable_team_codes,
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

        with oldest_mode_tab:
            st.markdown("### Oldest Player on Roster")
            st.caption(
                "For a selected team-year, show the oldest rostered player and draw arrows "
                "from their first team to every franchise they played on."
            )
            st.info(
                "How to use this section:\n"
                "1. Use **Specific Search** to run chain analysis from a team-year or selected player.\n"
                "2. Use **Full Web** to explore a season-wide MLB teammate network.\n"
                "3. Use **Refresh cache** if any derived files are missing or stale."
            )
            if st.button("Refresh cache", type="primary", key="sim_refresh_oldest_cache"):
                try:
                    with st.spinner("Refreshing baseball cache... this can take a few minutes."):
                        default_max_year = max(datetime.now().year - 1, 1876)
                        build_cache(
                            local_cache_dir,
                            max_year=int(default_max_year),
                            lahman_zip_path=None,
                            lahman_url=None,
                            progress_cb=None,
                        )
                    st.cache_data.clear()
                    st.success("Cache refreshed. Reloading oldest-player view.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Cache refresh failed: {exc}")
                    st.info(
                        "If Lahman download fails, run from terminal with a specific source:\n"
                        "- `python src/scripts/build_baseball_cache.py --lahman-url <working_lahman_zip_url>`\n"
                        "- `python src/scripts/build_baseball_cache.py --lahman-zip /absolute/path/to/lahman.zip`"
                    )
            st.caption(
                "Specific Search steps: choose year, optional NL/AL limit, pick start mode "
                "(team-year oldest player or selected player), then choose chain mode."
            )
            if player_first_team_df.empty or oldest_by_team_year_df.empty:
                st.error(
                    "Derived cache files are missing: `player_first_team.csv` and/or `team_year_oldest_players.csv`."
                )
                st.info("Rebuild cache: `python src/scripts/build_baseball_cache.py`")
            else:
                oldest = oldest_by_team_year_df.copy()
                oldest["yearID"] = pd.to_numeric(oldest["yearID"], errors="coerce").astype("Int64")
                oldest["franchID"] = oldest["franchID"].apply(canonicalize_franchid)
                oldest = oldest[oldest["yearID"].notna() & oldest["franchID"].notna()].copy()
                oldest["yearID"] = oldest["yearID"].astype(int)

                available_years = sorted(oldest["yearID"].unique().tolist(), reverse=True)
                if not available_years:
                    st.info("No oldest-player records found in cache.")
                else:
                    selected_year = st.selectbox(
                        "Season year",
                        options=available_years,
                        index=0,
                        key="sim_oldest_year",
                    )
                    limit_to_major = st.toggle(
                        "Limit to NL/AL",
                        value=True,
                        key="sim_oldest_limit_nl_al",
                        help="When enabled, only American League and National League franchises are included.",
                    )
                    allowed_leagues = {"AL", "NL"} if limit_to_major else set()
                    start_mode = st.radio(
                        "Start from",
                        options=["Team-year oldest player", "Selected player"],
                        index=0,
                        key="sim_oldest_start_mode",
                    )

                    oldest_year = oldest[oldest["yearID"] == int(selected_year)].copy()
                    if allowed_leagues:
                        oldest_year = oldest_year[
                            oldest_year["franchID"].apply(
                                lambda code: str(
                                    (FRANCHID_MODERN_ALIGNMENT.get(canonicalize_franchid(code)) or {}).get("league", "")
                                ).upper()
                                in allowed_leagues
                            )
                        ].copy()
                    team_options = sorted(
                        oldest_year["franchID"].dropna().astype(str).unique().tolist(),
                        key=lambda code: str(team_code_to_name.get(code, code)).lower(),
                    )
                    start_player_id = ""
                    first_chain_start_team = ""
                    first_chain_start_year: Optional[int] = None
                    prefix_line = ""
                    source_player_label = ""
                    source_team_for_fullweb = ""

                    if start_mode == "Team-year oldest player":
                        if not team_options:
                            st.info("No teams available for this season with the selected league filter.")
                        else:
                            selected_team = st.selectbox(
                                "Team",
                                options=team_options,
                                index=team_options.index(st.session_state.get("sim_team1", team_options[0]))
                                if st.session_state.get("sim_team1", team_options[0]) in team_options
                                else 0,
                                key="sim_oldest_team",
                                format_func=lambda code: f"{team_code_to_name.get(code, code)} ({code})",
                            )
                            source_team_for_fullweb = str(selected_team)
                            candidates = oldest_year[oldest_year["franchID"] == selected_team].copy()
                            if candidates.empty:
                                st.info("No oldest-player record found for this team-year.")
                            else:
                                candidate = candidates.sort_values(
                                    ["age_on_july_1", "birthYear", "birthMonth", "birthDay"],
                                    ascending=[False, True, True, True],
                                ).head(1).iloc[0]
                                start_player_id = str(candidate["playerID"])
                                source_player_label = _player_label(candidate)
                                first_chain_start_team = str(selected_team)
                                first_chain_start_year = int(selected_year)
                                player_name = f"{str(candidate.get('nameFirst', '')).strip()} {str(candidate.get('nameLast', '')).strip()}".strip()
                                st.write(
                                    f"**{team_code_to_name.get(selected_team, selected_team)} ({selected_team}) in {selected_year}:** "
                                    f"{player_name} (`{start_player_id}`)"
                                )
                                birth_text = _format_birth_text(
                                    candidate.get("birthYear"),
                                    candidate.get("birthMonth"),
                                    candidate.get("birthDay"),
                                )
                                st.caption(
                                    f"Born: {birth_text} | Age on July 1: {candidate.get('age_on_july_1', '')} | "
                                    f"First team: {candidate.get('first_franchID', '')} ({candidate.get('first_yearID', '')})"
                                )
                    else:
                        selected_player_id = ""
                        if not team_options:
                            st.info("No teams available for this season with the selected league filter.")
                        else:
                            selected_team_for_player = st.selectbox(
                                "Team",
                                options=team_options,
                                index=team_options.index(st.session_state.get("sim_team1", team_options[0]))
                                if st.session_state.get("sim_team1", team_options[0]) in team_options
                                else 0,
                                key="sim_oldest_player_team",
                                format_func=lambda code: f"{team_code_to_name.get(code, code)} ({code})",
                            )
                            source_team_for_fullweb = str(selected_team_for_player)
                            team_master_year = team_master.copy()
                            team_master_year["franchID"] = team_master_year["franchID"].apply(canonicalize_franchid)
                            year_team_ids = (
                                team_master_year[
                                    (pd.to_numeric(team_master_year["yearID"], errors="coerce").astype("Int64") == int(selected_year))
                                    & (team_master_year["franchID"] == selected_team_for_player)
                                ]["teamID"]
                                .dropna()
                                .astype(str)
                                .unique()
                                .tolist()
                            )
                            if not year_team_ids:
                                st.info("No roster team IDs found for this team-year.")
                                option_ids = []
                                roster_catalog = pd.DataFrame()
                            else:
                                roster_apps = appearances[
                                    (pd.to_numeric(appearances["yearID"], errors="coerce").astype("Int64") == int(selected_year))
                                    & (appearances["teamID"].astype(str).isin(set(year_team_ids)))
                                ].copy()
                                roster_player_ids = (
                                    roster_apps["playerID"].dropna().astype(str).drop_duplicates().tolist()
                                )
                                roster_catalog = player_master[
                                    player_master["key_bbref"].astype(str).isin(set(roster_player_ids))
                                ][["key_bbref", "name_first", "name_last", "birth_year", "birth_month", "birth_day"]].copy()
                                roster_catalog = roster_catalog.drop_duplicates(subset=["key_bbref"])
                                roster_catalog["display_name"] = (
                                    roster_catalog["name_first"].astype(str).str.strip()
                                    + " "
                                    + roster_catalog["name_last"].astype(str).str.strip()
                                ).str.strip()
                                roster_catalog = roster_catalog.sort_values(
                                    ["name_last", "name_first", "key_bbref"],
                                    ascending=[True, True, True],
                                )
                                option_ids = roster_catalog["key_bbref"].astype(str).tolist()

                            if not option_ids:
                                st.info("No rostered players found for this team-year.")
                                selected_player_id = ""
                            else:
                                selected_player_id = st.selectbox(
                                    "Player (from selected team-year roster)",
                                    options=option_ids,
                                    index=0,
                                    key="sim_oldest_selected_player",
                                    format_func=lambda pid: (
                                        f"{roster_catalog.loc[roster_catalog['key_bbref'] == pid, 'display_name'].iloc[0]} ({pid})"
                                    ),
                                )

                        if selected_player_id:
                            selected_player_row = player_master[
                                player_master["key_bbref"].astype(str) == str(selected_player_id)
                            ].head(1)
                            if selected_player_row.empty:
                                selected_player_row = pd.DataFrame([{"key_bbref": selected_player_id}])
                            selected_player_row = selected_player_row.iloc[0]
                            start_player_id = str(selected_player_id)
                            source_player_label = _player_label(selected_player_row)
                            st.write(f"**Selected player:** {source_player_label}")

                            pft_rows = player_first_team_df[
                                player_first_team_df["playerID"].astype(str) == str(start_player_id)
                            ].copy()
                            if not pft_rows.empty:
                                pft_rows["first_yearID"] = pd.to_numeric(pft_rows["first_yearID"], errors="coerce").astype("Int64")
                                pft_rows = pft_rows[pft_rows["first_yearID"].notna()].copy()
                                if not pft_rows.empty:
                                    pft_rows["first_yearID"] = pft_rows["first_yearID"].astype(int)
                                    pft_rows["first_franchID"] = pft_rows["first_franchID"].apply(canonicalize_franchid)
                                    pft_row = pft_rows.sort_values(["first_yearID", "first_franchID"]).head(1).iloc[0]
                                    first_chain_start_team = str(pft_row.get("first_franchID", "") or "")
                                    first_chain_start_year = _safe_int(pft_row.get("first_yearID"))
                                    if first_chain_start_team and first_chain_start_year:
                                        team_label = f"{team_code_to_name.get(first_chain_start_team, first_chain_start_team)} ({first_chain_start_team})"
                                        prefix_line = (
                                            f"{_player_label(selected_player_row)} -> {team_label} in {first_chain_start_year} (first year)"
                                        )

                    if start_player_id:
                        specific_search_subtab, full_web_subtab = st.tabs(["Specific Search", "Full Web"])
                        with specific_search_subtab:
                                chain_mode = st.radio(
                                    "Chain mode",
                                    options=[
                                        "First-year oldest on first team",
                                        "Oldest teammate ever (any roster year)",
                                    ],
                                    index=0,
                                    key="sim_oldest_chain_mode",
                                    horizontal=False,
                                )
                                st.markdown("#### Chain View")
                                if chain_mode == "First-year oldest on first team":
                                    if not first_chain_start_team or first_chain_start_year is None:
                                        st.info("No first-team record found for this starting player.")
                                    elif allowed_leagues and _league_for_franchid(first_chain_start_team) not in allowed_leagues:
                                        st.info("Starting player's first team is outside the NL/AL filter.")
                                    else:
                                        chain_df, stop_reason = _build_oldest_chain(
                                            oldest,
                                            start_franchid=first_chain_start_team,
                                            start_year=int(first_chain_start_year),
                                            max_hops=12,
                                            allowed_leagues=allowed_leagues,
                                        )
                                        if chain_df.empty:
                                            st.info("No chain data available for this start.")
                                        else:
                                            if prefix_line:
                                                first_oldest = str(chain_df.iloc[0].get("oldest_player", ""))
                                                st.markdown(
                                                    f"{_player_link_markdown_from_label(prefix_line.split(' -> ', 1)[0])} -> "
                                                    f"{prefix_line.split(' -> ', 1)[1]} -> "
                                                    f"{_player_link_markdown_from_label(first_oldest)}"
                                                )
                                            for row in chain_df.itertuples(index=False):
                                                team_label = f"{team_code_to_name.get(row.first_team, row.first_team)} ({row.first_team})"
                                                next_player = row.next_oldest_player or "No oldest-player row found"
                                                st.markdown(
                                                    f"{_player_link_markdown_from_label(row.oldest_player)} -> "
                                                    f"{team_label} in {row.first_year} (first year) -> "
                                                    f"{_player_link_markdown_from_label(next_player)}"
                                                )
                                            st.caption(stop_reason)
                                            st.markdown("#### Interactive Chain Graph")
                                            _render_interactive_chain_graph(
                                                chain_df=chain_df,
                                                chain_mode=chain_mode,
                                                player_master=player_master,
                                                player_first_team_df=player_first_team_df,
                                                team_code_to_name=team_code_to_name,
                                                source_player_id=start_player_id if start_mode == "Selected player" else "",
                                            )
                                            chain_display = chain_df.copy()
                                            chain_display["oldest_team"] = chain_display["oldest_team"].apply(
                                                lambda code: f"{team_code_to_name.get(code, code)} ({code})"
                                            )
                                            chain_display["first_team"] = chain_display["first_team"].apply(
                                                lambda code: f"{team_code_to_name.get(code, code)} ({code})" if str(code).strip() else ""
                                            )
                                            chain_display["oldest_year"] = chain_display["oldest_year"].astype(str)
                                            chain_display["first_year"] = chain_display["first_year"].astype(str)
                                            if start_mode == "Selected player" and source_player_label:
                                                source_row = pd.DataFrame(
                                                    [
                                                        {
                                                            "hop": "0",
                                                            "oldest_player": source_player_label,
                                                            "oldest_team": "",
                                                            "oldest_year": "",
                                                            "first_team": "",
                                                            "first_year": "",
                                                            "next_oldest_player": "",
                                                        }
                                                    ]
                                                )
                                                chain_display = pd.concat([source_row, chain_display], ignore_index=True)
                                            chain_display["oldest_player_bbr"] = chain_display["oldest_player"].apply(
                                                lambda v: _player_bbr_url(_extract_player_id_from_label(v))
                                            )
                                            chain_display["next_oldest_player_bbr"] = chain_display["next_oldest_player"].apply(
                                                lambda v: _player_bbr_url(_extract_player_id_from_label(v))
                                            )
                                        _render_html_summary_table(
                                            chain_display,
                                            columns=[
                                                "hop",
                                                "oldest_player",
                                                "oldest_player_bbr",
                                                "oldest_team",
                                                "oldest_year",
                                                "first_team",
                                                "first_year",
                                                "next_oldest_player",
                                                "next_oldest_player_bbr",
                                            ],
                                            link_columns=["oldest_player_bbr", "next_oldest_player_bbr"],
                                        )
                                else:
                                    chain_df, stop_reason = _build_oldest_teammate_chain(
                                        start_player_id=start_player_id,
                                        appearances=appearances,
                                        team_master=team_master,
                                        player_master=player_master,
                                        max_hops=12,
                                        allowed_leagues=allowed_leagues,
                                    )
                                    if chain_df.empty:
                                        st.info(stop_reason or "No teammate chain data available for this player.")
                                    else:
                                        for row in chain_df.itertuples(index=False):
                                            team_label = f"{team_code_to_name.get(row.shared_team, row.shared_team)} ({row.shared_team})"
                                            st.markdown(
                                                f"{_player_link_markdown_from_label(row.player)} -> "
                                                f"{team_label} in {row.shared_year} (shared roster) -> "
                                                f"{_player_link_markdown_from_label(row.oldest_teammate)}"
                                            )
                                        st.caption(stop_reason)
                                        st.markdown("#### Interactive Chain Graph")
                                        _render_interactive_chain_graph(
                                            chain_df=chain_df,
                                            chain_mode=chain_mode,
                                            player_master=player_master,
                                            player_first_team_df=player_first_team_df,
                                            team_code_to_name=team_code_to_name,
                                            source_player_id=start_player_id if start_mode == "Selected player" else "",
                                        )
                                        chain_display = chain_df.copy()
                                        chain_display["shared_team"] = chain_display["shared_team"].apply(
                                            lambda code: f"{team_code_to_name.get(code, code)} ({code})"
                                        )
                                        chain_display["shared_year"] = chain_display["shared_year"].astype(str)
                                        if start_mode == "Selected player" and source_player_label:
                                            source_row = pd.DataFrame(
                                                [
                                                    {
                                                        "hop": "0",
                                                        "player": source_player_label,
                                                        "shared_team": "",
                                                        "shared_year": "",
                                                        "oldest_teammate": "",
                                                    }
                                                ]
                                            )
                                            chain_display = pd.concat([source_row, chain_display], ignore_index=True)
                                        chain_display["player_bbr"] = chain_display["player"].apply(
                                            lambda v: _player_bbr_url(_extract_player_id_from_label(v))
                                        )
                                        chain_display["oldest_teammate_bbr"] = chain_display["oldest_teammate"].apply(
                                            lambda v: _player_bbr_url(_extract_player_id_from_label(v))
                                        )
                                    _render_html_summary_table(
                                        chain_display,
                                        columns=[
                                            "hop",
                                            "player",
                                            "player_bbr",
                                            "shared_team",
                                            "shared_year",
                                            "oldest_teammate",
                                            "oldest_teammate_bbr",
                                        ],
                                        link_columns=["player_bbr", "oldest_teammate_bbr"],
                                    )

                        with full_web_subtab:
                            _render_full_web_graph(
                                team_master=team_master,
                                appearances=appearances,
                                player_master=player_master,
                                player_first_team_df=player_first_team_df,
                                team_code_to_name=team_code_to_name,
                                available_years=available_years,
                                selected_year_override=int(selected_year),
                                allowed_leagues_override=allowed_leagues,
                                source_team_override=source_team_for_fullweb,
                                source_player_override=str(start_player_id),
                            )

        player_sets = _build_player_franchise_sets(team_master, appearances)
        player_lookup_df, player_franch_map = _build_player_franchise_lookup(player_master, team_master, appearances)
        images_df = load_image_metadata_df()

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
                    result_df = pd.DataFrame(result_rows)
                    st.dataframe(_style_yes_no_columns(result_df, ["ok"]), use_container_width=True)

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
                    result_df = pd.DataFrame(result_rows)
                    st.dataframe(_style_yes_no_columns(result_df, ["ok"]), use_container_width=True)

        with reverse_grid_tab:
            st.markdown("### Reverse Immaculate Grid (Guess Axes)")
            st.caption(
                "A 3x3 player grid is pre-filled. Your job is to guess the hidden row and column teams "
                "that make every cell valid."
            )
            all_submitters = sorted({str(s) for s in images_df.get("submitter", pd.Series(dtype=str)).dropna().unique().tolist()})
            default_submitters = sorted(
                [
                    name
                    for name, details in GRID_PLAYERS.items()
                    if str(details.get("restricted", "")).lower() != "true" and name in all_submitters
                ]
            )
            selected_submitters = st.multiselect(
                "Include submissions from",
                options=all_submitters,
                default=default_submitters,
                key="sim_reverse_submitter_filter",
            )
            used_player_ids = _build_used_player_id_set(images_df, player_lookup_df, selected_submitters)
            st.caption(f"Eligible players from selected submitters: {len(used_player_ids):,}")
            if st.button("Generate reverse 3x3 puzzle", key="sim_generate_reverse_grid"):
                reverse_puzzle = _sample_reverse_grid_puzzle(
                    selectable_team_codes,
                    player_franch_map,
                    player_lookup_df,
                    allowed_player_ids=used_player_ids,
                )
                st.session_state["sim_reverse_grid_puzzle"] = reverse_puzzle

            reverse_puzzle = st.session_state.get("sim_reverse_grid_puzzle")
            if not selected_submitters:
                st.info("Select at least one submitter to generate a reverse puzzle.")
            elif not reverse_puzzle:
                st.info("Click generate to create a reverse puzzle with guaranteed solutions.")
            else:
                row_codes = reverse_puzzle["row_codes"]
                col_codes = reverse_puzzle["col_codes"]
                player_grid_df = pd.DataFrame(
                    reverse_puzzle["solution_grid_names"],
                    index=["Row 1", "Row 2", "Row 3"],
                    columns=["Col 1", "Col 2", "Col 3"],
                )
                st.write("Puzzle grid (players)")
                st.dataframe(player_grid_df, use_container_width=True)

                st.markdown("#### Guess Row Axes")
                guessed_rows = []
                row_cols = st.columns(3)
                for i in range(3):
                    with row_cols[i]:
                        guessed_rows.append(
                            st.selectbox(
                                f"Row {i+1} team",
                                options=selectable_team_codes,
                                key=f"sim_reverse_row_guess_{i}",
                                format_func=lambda code: f"{team_code_to_name.get(code, code)} ({code})",
                            )
                        )

                st.markdown("#### Guess Column Axes")
                guessed_cols = []
                col_cols = st.columns(3)
                for i in range(3):
                    with col_cols[i]:
                        guessed_cols.append(
                            st.selectbox(
                                f"Col {i+1} team",
                                options=selectable_team_codes,
                                key=f"sim_reverse_col_guess_{i}",
                                format_func=lambda code: f"{team_code_to_name.get(code, code)} ({code})",
                            )
                        )

                if st.button("Check axes", key="sim_check_reverse_axes"):
                    row_hits = sum(1 for i in range(3) if guessed_rows[i] == row_codes[i])
                    col_hits = sum(1 for i in range(3) if guessed_cols[i] == col_codes[i])
                    total_hits = row_hits + col_hits
                    st.write(f"Axis score: **{total_hits}/6** (rows: {row_hits}/3, cols: {col_hits}/3)")

                    axes_result = pd.DataFrame(
                        {
                            "axis": [f"Row {i+1}" for i in range(3)] + [f"Col {i+1}" for i in range(3)],
                            "guess": guessed_rows + guessed_cols,
                            "answer": row_codes + col_codes,
                            "correct": [
                                "yes" if guessed_rows[i] == row_codes[i] else "no" for i in range(3)
                            ] + [
                                "yes" if guessed_cols[i] == col_codes[i] else "no" for i in range(3)
                            ],
                        }
                    )
                    st.dataframe(_style_yes_no_columns(axes_result, ["correct"]), use_container_width=True)

                if st.button("Reveal answer axes", key="sim_reveal_reverse_grid_axes"):
                    solved_df = pd.DataFrame(
                        reverse_puzzle["solution_grid_names"],
                        index=[f"{team_code_to_name.get(code, code)} ({code})" for code in row_codes],
                        columns=[f"{team_code_to_name.get(code, code)} ({code})" for code in col_codes],
                    )
                    st.write("Solved grid (answer axes + player cells)")
                    st.dataframe(solved_df, use_container_width=True)

    with instructions_tab:
        st.markdown(
            """
            ### What this does
            Mini Games let you test intersections, explore study guides, play random/reverse immaculate puzzle modes,
            and view oldest-roster-player career arrows.

            ### Required local data
            This tab reads local cache files from `bin/baseball_cache/`:
            - `teams.csv`
            - `People.csv`
            - `appearances.csv`
            - `player_first_team.csv`
            - `team_year_oldest_players.csv`

            If these files are missing, build them with:
            `python src/scripts/build_baseball_cache.py`

            ### How to use
            1. Open **Add / Update Data**, select **MLB Player Cache**, and click **Build/Rebuild cache**.
            2. Open `Play`.
            3. Pick **Team A** and **Team B** (or use **Random teams**).
            4. Enter player first and last name.
            5. Click **Confirm player and check** to verify if the player matches the intersection.

            ### Notes
            - Name matching depends on the cached player records (spelling matters).
            - Team checks are franchise-based (`franchID`) from Lahman data.
            """
        )


__all__ = ["render_simulator_tab", "load_pybaseball_data"]
