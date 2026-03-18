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

from data.io.mlb_reference import MANUAL_ALIASES, clean_name, load_mlb_player_names
from config.constants import MESSAGES_CSV_PATH, PROMPTS_CSV_PATH


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


def _prompt_cell_parts(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (tuple, list)):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (tuple, list)):
                return [str(v) for v in parsed if str(v).strip()]
        except Exception:
            pass
        return [raw]
    return [str(value)]


def _prompt_text_to_position_code(prompt_text: str) -> str | None:
    text = re.sub(r"\s+", " ", str(prompt_text).lower()).strip()
    if not text:
        return None

    if "played center field" in text or text in {"cf", "center field"}:
        return "CF"
    if "played left field" in text or text in {"lf", "left field"}:
        return "LF"
    if "played right field" in text or text in {"rf", "right field"}:
        return "RF"
    if "played outfield" in text or text in {"of", "outfield"}:
        return "OF"
    if "played shortstop" in text or text in {"ss", "shortstop"}:
        return "SS"
    if "played third base" in text or text in {"3b", "third base"}:
        return "3B"
    if "played second base" in text or text in {"2b", "second base"}:
        return "2B"
    if "played first base" in text or text in {"1b", "first base"}:
        return "1B"
    if "played catcher" in text or text == "catcher":
        return "C"
    if "pitched" in text or " pitching" in text:
        return "P"
    return None


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

    def _last_token(s: str) -> str:
        parts = [p for p in str(s).strip().split() if p]
        return parts[-1].lower() if parts else ""

    if name in MANUAL_ALIASES:
        return MANUAL_ALIASES[name]

    known = _known_player_names()
    known_set = set(known)
    if name in known_set:
        return name

    # For non-exact matches, single-token inputs are too ambiguous for reliable fuzzy mapping
    # (e.g., "Rodriguez", "Blue"), so treat them as unresolved.
    if len(name.split()) < 2:
        return ""

    # Common OCR artifact: leading stray character token (e.g., "Z Ian Happ" -> "Ian Happ")
    parts = name.split()
    if len(parts) >= 3 and len(parts[0]) == 1:
        trimmed = " ".join(parts[1:])
        if trimmed in known_set:
            return trimmed
        name = trimmed

    best = process.extractOne(name, known, score_cutoff=88)
    canonical = str(best[0]) if best else name
    if best:
        score = float(best[1]) if len(best) > 1 else 0.0
        if _last_token(name) and _last_token(canonical) and _last_token(name) != _last_token(canonical):
            # Block aggressive fuzzy jumps across different surnames.
            if score < 97:
                return name
    return canonical


def _build_usage_df(images_df: pd.DataFrame) -> pd.DataFrame:
    if images_df.empty or "responses" not in images_df.columns or "submitter" not in images_df.columns:
        return pd.DataFrame(columns=["submitter", "player", "submission_date", "grid_number", "position_key"])

    rows = []
    for _, row in images_df.iterrows():
        submitter = row.get("submitter")
        responses = _parse_responses(row.get("responses"))
        raw_date = row.get("date") if pd.notna(row.get("date")) else row.get("image_date")
        submission_date = pd.to_datetime(raw_date, errors="coerce")
        grid_number = row.get("grid_number")
        if not submitter or not responses:
            continue
        for position_key, val in responses.items():
            player = _canonicalize_player_name_for_ui(str(val).strip())
            if player:
                rows.append(
                    {
                        "submitter": str(submitter),
                        "player": player,
                        "submission_date": submission_date,
                        "grid_number": grid_number,
                        "position_key": str(position_key),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["submitter", "player", "submission_date", "grid_number", "position_key"])
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


@lru_cache(maxsize=1)
def _load_prompt_position_lookup() -> dict[tuple[str, str], str]:
    path = Path(PROMPTS_CSV_PATH)
    if not path.exists():
        return {}
    try:
        prompts = pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return {}
    if "grid_id" not in prompts.columns:
        return {}

    pos_cols = [
        "top_left",
        "top_center",
        "top_right",
        "middle_left",
        "middle_center",
        "middle_right",
        "bottom_left",
        "bottom_center",
        "bottom_right",
    ]
    existing_cols = [c for c in pos_cols if c in prompts.columns]
    if not existing_cols:
        return {}

    lookup: dict[tuple[str, str], str] = {}
    for _, row in prompts.iterrows():
        grid_number = _normalize_grid_number(row.get("grid_id"))
        if not grid_number:
            continue
        for pos in existing_cols:
            parts = _prompt_cell_parts(row.get(pos))
            mapped = None
            for part in parts:
                mapped = _prompt_text_to_position_code(part)
                if mapped:
                    break
            if mapped:
                lookup[(grid_number, pos)] = mapped
    return lookup


@lru_cache(maxsize=1)
def _load_career_position_fraction_table(cache_dir: str = "bin/baseball_cache") -> pd.DataFrame:
    cache_path = Path(cache_dir)
    appearances_candidates = [cache_path / "appearances.csv", cache_path / "Appearances.csv"]
    people_candidates = [cache_path / "People.csv", cache_path / "people.csv"]
    appearances_path = next((p for p in appearances_candidates if p.exists()), None)
    people_path = next((p for p in people_candidates if p.exists()), None)
    if appearances_path is None or people_path is None:
        return pd.DataFrame(columns=["player_norm", "position", "fractional_appearance", "total_games"])

    try:
        appearances = pd.read_csv(appearances_path, dtype=str)
    except Exception:
        return pd.DataFrame(columns=["player_norm", "position", "fractional_appearance", "total_games"])
    if appearances.empty or "playerID" not in appearances.columns:
        return pd.DataFrame(columns=["player_norm", "position", "fractional_appearance", "total_games"])

    position_col_to_code = {
        "G_p": "P",
        "G_c": "C",
        "G_1b": "1B",
        "G_2b": "2B",
        "G_3b": "3B",
        "G_ss": "SS",
        "G_lf": "LF",
        "G_cf": "CF",
        "G_rf": "RF",
    }
    available_position_cols = [c for c in position_col_to_code if c in appearances.columns]
    if not available_position_cols:
        return pd.DataFrame(columns=["player_norm", "position", "fractional_appearance", "total_games"])

    num_cols = ["G_all"] + available_position_cols
    for col in num_cols:
        if col not in appearances.columns:
            appearances[col] = 0
        appearances[col] = pd.to_numeric(appearances[col], errors="coerce").fillna(0.0)

    per_player = appearances.groupby("playerID", as_index=False)[num_cols].sum()
    if per_player.empty:
        return pd.DataFrame(columns=["player_norm", "position", "fractional_appearance", "total_games"])

    per_player["total_games"] = pd.to_numeric(per_player["G_all"], errors="coerce").fillna(0.0)
    fallback_games = per_player[available_position_cols].sum(axis=1)
    per_player.loc[per_player["total_games"] <= 0, "total_games"] = fallback_games
    per_player = per_player[per_player["total_games"] > 0].copy()
    if per_player.empty:
        return pd.DataFrame(columns=["player_norm", "position", "fractional_appearance", "total_games"])

    long_frames = []
    for col in available_position_cols:
        code = position_col_to_code[col]
        chunk = per_player[["playerID", "total_games", col]].copy()
        chunk = chunk.rename(columns={col: "position_games"})
        chunk["position"] = code
        chunk["fractional_appearance"] = (
            pd.to_numeric(chunk["position_games"], errors="coerce").fillna(0.0)
            / pd.to_numeric(chunk["total_games"], errors="coerce").replace(0, np.nan)
        )
        long_frames.append(chunk[["playerID", "position", "fractional_appearance", "total_games"]])

    # For OF prompts, use LF+CF+RF appearances explicitly (can exceed 1.0 by design).
    outfield_cols = [c for c in ("G_lf", "G_cf", "G_rf") if c in per_player.columns]
    if outfield_cols:
        of_chunk = per_player[["playerID", "total_games"] + outfield_cols].copy()
        of_chunk["position_games"] = of_chunk[outfield_cols].sum(axis=1)
        of_chunk["position"] = "OF"
        of_chunk["fractional_appearance"] = (
            pd.to_numeric(of_chunk["position_games"], errors="coerce").fillna(0.0)
            / pd.to_numeric(of_chunk["total_games"], errors="coerce").replace(0, np.nan)
        )
        long_frames.append(of_chunk[["playerID", "position", "fractional_appearance", "total_games"]])
    frac_df = pd.concat(long_frames, ignore_index=True)

    try:
        people = pd.read_csv(people_path, dtype=str)
    except Exception:
        return pd.DataFrame(columns=["player_norm", "position", "fractional_appearance", "total_games"])
    if people.empty:
        return pd.DataFrame(columns=["player_norm", "position", "fractional_appearance", "total_games"])

    id_col = next((c for c in people.columns if c.lower() in {"playerid", "key_bbref"}), None)
    first_col = next((c for c in people.columns if c.lower() in {"namefirst", "name_first"}), None)
    last_col = next((c for c in people.columns if c.lower() in {"namelast", "name_last"}), None)
    if not id_col or not first_col or not last_col:
        return pd.DataFrame(columns=["player_norm", "position", "fractional_appearance", "total_games"])

    people = people[[id_col, first_col, last_col]].copy()
    people.columns = ["playerID", "name_first", "name_last"]
    people["playerID"] = people["playerID"].astype(str).str.strip()
    people["name_first"] = people["name_first"].fillna("").astype(str).str.strip()
    people["name_last"] = people["name_last"].fillna("").astype(str).str.strip()
    people["player_name"] = (people["name_first"] + " " + people["name_last"]).str.strip()
    people["player_norm"] = people["player_name"].map(clean_name).map(_normalize_name)
    people = people[people["player_norm"] != ""]

    merged = frac_df.merge(people[["playerID", "player_norm"]], on="playerID", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["player_norm", "position", "fractional_appearance", "total_games"])

    merged["fractional_appearance"] = pd.to_numeric(merged["fractional_appearance"], errors="coerce")
    merged["total_games"] = pd.to_numeric(merged["total_games"], errors="coerce")
    merged = merged.dropna(subset=["fractional_appearance", "total_games"])
    if merged.empty:
        return pd.DataFrame(columns=["player_norm", "position", "fractional_appearance", "total_games"])

    # Be generous with duplicate names: if multiple Lahman playerIDs map to one normalized
    # name, keep the highest observed career fraction for each position.
    merged = (
        merged.groupby(["player_norm", "position"], as_index=False)
        .agg(
            fractional_appearance=("fractional_appearance", "max"),
            total_games=("total_games", "max"),
        )
    )
    return merged[["player_norm", "position", "fractional_appearance", "total_games"]]


def _render_fudged_position_usage(usage_df: pd.DataFrame) -> None:
    st.markdown("### Fudged Position Usage")
    st.caption(
        "How this is calculated: "
        "`fractional_appearance = career_position_games / career_total_games` "
        "(for `OF`, position games are `LF + CF + RF`), "
        "`log_fractional_appearance = ln(fractional_appearance)`, "
        "`fudge_score = grid_uses / fractional_appearance` (higher = more fudged), "
        "and submitter rollup uses "
        "`weighted_mean_log_fractional_appearance = weighted_mean(log_fractional_appearance, weights=grid_uses)` "
        "with `submitter_fudge_index = -weighted_mean_log_fractional_appearance`."
    )
    if usage_df.empty:
        st.info("No parsed player responses available.")
        return

    if "grid_number" not in usage_df.columns or "position_key" not in usage_df.columns:
        st.info("Usage data does not include grid/position metadata yet.")
        return

    prompt_lookup = _load_prompt_position_lookup()
    if not prompt_lookup:
        st.warning("No prompt position mapping available from `bin/csv/prompts.csv`.")
        return

    pos_frac = _load_career_position_fraction_table()
    if pos_frac.empty:
        st.warning("No career position fractions available. Rebuild `bin/baseball_cache/appearances.csv` and `People.csv`.")
        return

    work = usage_df.copy()
    work["grid_number_norm"] = work["grid_number"].map(_normalize_grid_number)
    work["position_key"] = work["position_key"].astype(str).str.strip()
    work["prompt_position"] = work.apply(
        lambda r: prompt_lookup.get((r.get("grid_number_norm", ""), r.get("position_key", ""))),
        axis=1,
    )
    work = work.dropna(subset=["prompt_position", "player_norm", "submitter", "player"]).copy()
    if work.empty:
        st.info("No usages intersected with position-specific prompts.")
        return

    grouped = (
        work.groupby(["submitter", "player", "player_norm", "prompt_position"], as_index=False)
        .agg(
            grid_uses=("grid_number_norm", "size"),
            grid_ids=("grid_number_norm", lambda s: ", ".join(sorted({str(v) for v in s if str(v).strip()}, key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x))))),
        )
        .rename(columns={"prompt_position": "position", "player": "mlb_player"})
    )

    merged = grouped.merge(
        pos_frac[["player_norm", "position", "fractional_appearance"]],
        on=["player_norm", "position"],
        how="left",
    )
    mapped = int(merged["fractional_appearance"].notna().sum())
    total = int(len(merged))
    st.caption(f"Mapped career position fractions for {mapped:,}/{total:,} submitter-player-position rows.")

    merged = merged.dropna(subset=["fractional_appearance"]).copy()
    if merged.empty:
        st.info("No submitter-player-position rows could be mapped to career position fractions.")
        return

    merged["fractional_appearance"] = pd.to_numeric(merged["fractional_appearance"], errors="coerce")
    merged = merged.dropna(subset=["fractional_appearance"]).copy()
    zero_frac = merged[merged["fractional_appearance"] <= 0].copy()
    merged = merged[merged["fractional_appearance"] > 0].copy()
    if not zero_frac.empty:
        st.caption(
            f"Excluded {len(zero_frac):,} row(s) with zero career fractional appearance from ranking (likely mismatches or edge cases)."
        )
    if merged.empty:
        st.info("No rows remain after excluding zero-fraction appearances.")
        return
    merged["fudge_score"] = np.where(
        merged["fractional_appearance"] > 0,
        merged["grid_uses"] / merged["fractional_appearance"],
        np.inf,
    )
    merged["log_fractional_appearance"] = np.log(merged["fractional_appearance"])
    merged["fractional_appearance"] = merged["fractional_appearance"].round(4)
    merged["log_fractional_appearance"] = pd.to_numeric(
        merged["log_fractional_appearance"], errors="coerce"
    ).round(4)
    merged = merged.sort_values(
        ["fudge_score", "grid_uses", "submitter", "mlb_player", "position"],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)
    merged["rank"] = np.arange(1, len(merged) + 1)

    submitter_agg = (
        merged.groupby("submitter", as_index=False)
        .agg(
            grid_uses_total=("grid_uses", "sum"),
            unique_player_positions=("position", "size"),
            weighted_mean_log_fractional_appearance=(
                "log_fractional_appearance",
                lambda s: np.average(s, weights=merged.loc[s.index, "grid_uses"]) if len(s) else np.nan,
            ),
            mean_fractional_appearance=("fractional_appearance", "mean"),
        )
    )
    # Higher "fudge index" means more fudgy usage behavior.
    submitter_agg["submitter_fudge_index"] = -submitter_agg["weighted_mean_log_fractional_appearance"]
    submitter_agg["weighted_mean_log_fractional_appearance"] = pd.to_numeric(
        submitter_agg["weighted_mean_log_fractional_appearance"], errors="coerce"
    ).round(4)
    submitter_agg["mean_fractional_appearance"] = pd.to_numeric(
        submitter_agg["mean_fractional_appearance"], errors="coerce"
    ).round(4)
    submitter_agg["submitter_fudge_index"] = pd.to_numeric(
        submitter_agg["submitter_fudge_index"], errors="coerce"
    ).round(4)
    submitter_agg = submitter_agg.sort_values(
        ["submitter_fudge_index", "grid_uses_total", "submitter"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    submitter_agg["cheat_rank"] = np.arange(1, len(submitter_agg) + 1)
    submitter_agg["saint_rank"] = (
        submitter_agg["submitter_fudge_index"].rank(method="min", ascending=True).astype(int)
    )

    st.markdown("#### Submitter Rollup")
    st.caption(
        "Aggregate is usage-weighted: weighted_mean(log(fractional_appearance), weights=grid_uses). "
        "`submitter_fudge_index = -weighted_mean_log_fractional_appearance`."
    )
    st.dataframe(
        submitter_agg[
            [
                "cheat_rank",
                "saint_rank",
                "submitter",
                "submitter_fudge_index",
                "weighted_mean_log_fractional_appearance",
                "mean_fractional_appearance",
                "grid_uses_total",
                "unique_player_positions",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Submitter | Player | Position Detail")
    st.dataframe(
        merged[
            [
                "rank",
                "submitter",
                "mlb_player",
                "position",
                "fractional_appearance",
                "log_fractional_appearance",
                "grid_uses",
                "grid_ids",
                "fudge_score",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


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
        "Fudged Position Usage",
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
    elif selected_analysis == "Fudged Position Usage":
        _render_fudged_position_usage(usage_df)
