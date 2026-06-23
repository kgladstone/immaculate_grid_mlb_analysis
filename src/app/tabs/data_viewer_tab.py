from __future__ import annotations

import ast
import json
import re
from difflib import SequenceMatcher
from typing import List

import pandas as pd
from pathlib import Path
import streamlit as st
from PIL import Image
from PIL import UnidentifiedImageError

from config.constants import (
    GRID_PLAYERS,
    GRID_PLAYERS_RESTRICTED,
    IMAGES_PATH,
    IMAGES_METADATA_CSV_PATH,
    MESSAGES_CSV_PATH,
    MY_NAME,
    PROMPTS_CSV_PATH,
    RULE5_FULL_BANS_CSV_PATH,
)
from utils.grid_utils import ImmaculateGridUtils
from app.services.data_loaders import resolve_path
from app.services.player_links import player_link_html, player_link_html_table

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    _HEIF_ENABLED = True
except Exception:
    _HEIF_ENABLED = False


def _norm_grid_id(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _grid_sort_key(value: str):
    text = _norm_grid_id(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def _norm_match_text(value) -> str:
    text = str(value or "").lower().replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\b(jr|sr)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_prompt_pair(value) -> tuple[str, str]:
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (tuple, list)) and len(parsed) == 2:
            return _clean_prompt_label(parsed[0]), _clean_prompt_label(parsed[1])
    except Exception:
        pass
    text = str(value)
    if " + " in text:
        left, right = text.split(" + ", 1)
        return _clean_prompt_label(left), _clean_prompt_label(right)
    return _clean_prompt_label(text), ""


def _clean_prompt_label(value) -> str:
    text = re.sub(r"\s+", " ", str(value).strip())
    return text.replace("≤", "<=")


def _format_prompt_pair(pair: tuple[str, str]) -> str:
    left, right = pair
    return f"{left} / {right}" if right else left


def _parse_responses_value(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            try:
                parsed = ast.literal_eval(value)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
    return {}


def _best_note_prompt_position(note: str, prompt_pairs: dict[str, tuple[str, str]]) -> str | None:
    aliases = {
        "ari": "Arizona Diamondbacks",
        "atl": "Atlanta Braves",
        "bal": "Baltimore Orioles",
        "bos": "Boston Red Sox",
        "chc": "Chicago Cubs",
        "chw": "Chicago White Sox",
        "cin": "Cincinnati Reds",
        "cle": "Cleveland Guardians",
        "col": "Colorado Rockies",
        "det": "Detroit Tigers",
        "dr": "Dominican Republic",
        "hou": "Houston Astros",
        "kc": "Kansas City Royals",
        "kcr": "Kansas City Royals",
        "laa": "Los Angeles Angels",
        "lad": "Los Angeles Dodgers",
        "mil": "Milwaukee Brewers",
        "nyy": "New York Yankees",
        "phi": "Philadelphia Phillies",
        "pit": "Pittsburgh Pirates",
        "sd": "San Diego Padres",
        "sea": "Seattle Mariners",
        "sf": "San Francisco Giants",
        "sfg": "San Francisco Giants",
        "stl": "St. Louis Cardinals",
        "tex": "Texas Rangers",
        "tor": "Toronto Blue Jays",
        "wsh": "Washington Nationals",
    }
    note_norm = _norm_match_text(note)
    alias_by_label = {_norm_match_text(full): code for code, full in aliases.items()}

    def score_pair(pair: tuple[str, str]) -> float:
        score = 0.0
        for label in pair:
            label_norm = _norm_match_text(label)
            if label_norm and label_norm in note_norm:
                score += 10 + len(label_norm) / 100
            alias = alias_by_label.get(label_norm)
            if alias and re.search(rf"\b{re.escape(alias)}\b", note_norm):
                score += 10 + len(alias) / 100
        for token in re.findall(r"[a-z0-9]+", note_norm):
            for label in pair:
                label_norm = _norm_match_text(label)
                if token and token in label_norm:
                    score += 1
        return score

    scored = sorted(
        ((score_pair(pair), position) for position, pair in prompt_pairs.items()),
        reverse=True,
    )
    if not scored or scored[0][0] <= 0:
        return None
    return scored[0][1]


def standardize_rule5_response(
    grid_number: int,
    player: str,
    note: str,
    prompts_df: pd.DataFrame | None,
    images_df: pd.DataFrame | None,
) -> str:
    if prompts_df is None or prompts_df.empty or "grid_id" not in prompts_df.columns:
        return str(note).strip()

    grid_id = _norm_grid_id(grid_number)
    prompt_rows = prompts_df[prompts_df["grid_id"].map(_norm_grid_id) == grid_id]
    if prompt_rows.empty:
        return str(note).strip()

    positions = [
        "top_left", "top_center", "top_right",
        "middle_left", "middle_center", "middle_right",
        "bottom_left", "bottom_center", "bottom_right",
    ]
    prompt_row = prompt_rows.iloc[0]
    prompt_pairs = {
        position: _parse_prompt_pair(prompt_row[position])
        for position in positions
        if position in prompt_row
    }

    chosen_position = None
    if images_df is not None and not images_df.empty and {"grid_number", "submitter", "responses"}.issubset(images_df.columns):
        image_rows = images_df[
            (images_df["grid_number"].map(_norm_grid_id) == grid_id)
            & (images_df["submitter"].astype(str) == MY_NAME)
        ]
        if not image_rows.empty:
            responses = _parse_responses_value(image_rows.iloc[0].get("responses"))
            player_norm = _norm_match_text(player)
            best_match: tuple[float, str] | None = None
            for position, response_player in responses.items():
                response_norm = _norm_match_text(response_player)
                if not response_norm:
                    continue
                ratio = SequenceMatcher(None, player_norm, response_norm).ratio()
                if player_norm == response_norm or player_norm in response_norm or response_norm in player_norm or ratio > 0.86:
                    if best_match is None or ratio > best_match[0]:
                        best_match = (ratio, position)
            if best_match is not None and best_match[1] in prompt_pairs:
                chosen_position = best_match[1]

    if chosen_position is None:
        chosen_position = _best_note_prompt_position(note, prompt_pairs)

    if chosen_position in prompt_pairs:
        return _format_prompt_pair(prompt_pairs[chosen_position])
    return str(note).strip()


def _upsert_rule5_full_ban(path: Path, grid_number: int, player: str, response: str) -> tuple[str, pd.DataFrame]:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["player", "grid_number", "response"]
    if path.exists():
        df = pd.read_csv(path, dtype=str)
    else:
        df = pd.DataFrame(columns=columns)

    for col in columns:
        if col not in df.columns:
            df[col] = ""
    df = df[columns].copy()
    df["player"] = df["player"].astype(str).str.strip()
    df["grid_number"] = df["grid_number"].astype(str).str.strip().str.removesuffix(".0")
    df["response"] = df["response"].astype(str).str.strip()

    grid_id = str(int(grid_number))
    player_clean = str(player).strip()
    response_clean = str(response).strip()
    mask = (
        (df["grid_number"] == grid_id)
        & (df["player"].str.casefold() == player_clean.casefold())
    )

    if mask.any():
        df.loc[mask, "player"] = player_clean
        df.loc[mask, "response"] = response_clean
        action = "updated"
    else:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [{"player": player_clean, "grid_number": grid_id, "response": response_clean}]
                ),
            ],
            ignore_index=True,
        )
        action = "added"

    df["grid_sort"] = pd.to_numeric(df["grid_number"], errors="coerce").fillna(-1)
    df = df.sort_values(["grid_sort", "player"], ascending=[True, True]).drop(columns=["grid_sort"])
    df.to_csv(path, index=False)
    return action, df


def format_prompt_cell(value: str) -> str:
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (tuple, list)) and len(parsed) == 2:
            return f"{parsed[0]}\n{parsed[1]}"
    except Exception:
        pass
    return str(value).replace(" + ", "\n")


def prompts_row_to_grid(row: pd.Series) -> List[List[str]]:
    return [
        [
            format_prompt_cell(row["top_left"]),
            format_prompt_cell(row["top_center"]),
            format_prompt_cell(row["top_right"]),
        ],
        [
            format_prompt_cell(row["middle_left"]),
            format_prompt_cell(row["middle_center"]),
            format_prompt_cell(row["middle_right"]),
        ],
        [
            format_prompt_cell(row["bottom_left"]),
            format_prompt_cell(row["bottom_center"]),
            format_prompt_cell(row["bottom_right"]),
        ],
    ]


def render_prompts_grid(grid: List[List[str]]) -> None:
    html_parts = [
        "<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:6px;max-width:680px;'>"
    ]
    for row in grid:
        for cell in row:
            safe_cell = str(cell).replace("\n", "<br>")
            html_parts.append(
                "<div style='border:1px solid #ccc;border-radius:6px;padding:10px;"
                "background:#f4f6fa;color:#111;min-height:70px;display:flex;align-items:center;justify-content:center;"
                "text-align:center;font-weight:600;'>"
                f"{safe_cell}"
                "</div>"
            )
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def parse_matrix(matrix_raw) -> List[List[bool]] | None:
    if isinstance(matrix_raw, list):
        matrix = matrix_raw
    else:
        try:
            matrix = json.loads(matrix_raw)
        except Exception:
            try:
                normalized = str(matrix_raw).replace("true", "True").replace("false", "False")
                matrix = ast.literal_eval(normalized)
            except Exception:
                return None

    if not isinstance(matrix, list) or any(not isinstance(row, list) for row in matrix):
        return None
    return matrix


def render_matrix(matrix_raw: str) -> None:
    matrix = parse_matrix(matrix_raw)
    if matrix is None:
        st.warning("Could not parse matrix.")
        return

    html_parts = ["<div style='display: grid; grid-template-columns: repeat(3, 42px); gap: 4px;'>"]
    for row in matrix:
        for cell in row:
            is_true = bool(cell)
            color = "#2ecc71" if is_true else "#ffffff"
            border = "1px solid #ccc"
            html_parts.append(
                f"<div style='width: 42px; height: 42px; background: {color}; border: {border};'></div>"
            )
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def responses_to_grid_cells(responses) -> List[str]:
    positions = extract_player_positions(responses)
    label_lookup = {label: value for label, value in positions}
    labels = [
        "Top Left",
        "Top Center",
        "Top Right",
        "Middle Left",
        "Middle Center",
        "Middle Right",
        "Bottom Left",
        "Bottom Center",
        "Bottom Right",
    ]
    ordered = []
    for label in labels:
        ordered.append((label, label_lookup.get(label, "<Empty>")))
    return ordered


def render_responses_grid(responses) -> None:
    cells = responses_to_grid_cells(responses)
    html_parts = [
        "<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:6px;'>"
    ]
    for label, value in cells:
        value_str = str(value).strip()
        if value_str in {"", "<Empty>", "nan", "NaN", "None"}:
            safe_value = "<span style='color:#c0392b;font-weight:700;font-size:18px;'>{}</span>"
        else:
            safe_value = player_link_html(value_str).replace("\n", "<br>")
        html_parts.append(
            "<div style='border:1px solid #ddd;border-radius:6px;padding:8px;"
            "background:#fff;min-height:70px;display:flex;flex-direction:column;justify-content:center;'>"
            f"<div style='font-size:11px;color:#555;'>{label}</div>"
            f"<div style='font-weight:600;color:#111;'>{safe_value}</div>"
            "</div>"
        )
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def extract_player_positions(responses) -> List[tuple]:
    parsed = responses
    if isinstance(responses, str):
        try:
            parsed = json.loads(responses)
        except Exception:
            try:
                parsed = ast.literal_eval(responses)
            except Exception:
                parsed = None

    label_order = [
        ("top_left", "Top Left"),
        ("top_center", "Top Center"),
        ("top_right", "Top Right"),
        ("middle_left", "Middle Left"),
        ("middle_center", "Middle Center"),
        ("middle_right", "Middle Right"),
        ("bottom_left", "Bottom Left"),
        ("bottom_center", "Bottom Center"),
        ("bottom_right", "Bottom Right"),
    ]

    positions: List[tuple] = []
    if isinstance(parsed, dict):
        for key, label in label_order:
            val = parsed.get(key)
            positions.append((label, _clean_display_player_name(val)))
    elif isinstance(parsed, list):
        for idx, label in enumerate(label_order):
            if idx < len(parsed):
                val = parsed[idx]
                positions.append((label[1], _clean_display_player_name(val)))
            else:
                positions.append((label[1], "<Empty>"))
    return positions


def _clean_display_player_name(value) -> str:
    """Normalize OCR/parser noise for UI display only; do not persist changes."""
    if value is None or pd.isna(value):
        return "<Empty>"
    text = str(value).strip()
    if not text:
        return "<Empty>"

    # Trim simple punctuation artifacts around names.
    text = re.sub(r"^[\s\-\.,;:!?]+", "", text)
    text = re.sub(r"[\s\-\.,;:!?]+$", "", text)

    # Drop single-letter OCR artifacts like "Z Justin Turner" or "Justin Turner Z".
    text = re.sub(r"^[A-Za-z]\s+(?=[A-Za-z])", "", text)
    text = re.sub(r"(?<=[A-Za-z])\s+[A-Za-z]$", "", text)

    return text or "<Empty>"


def load_image_for_display(img_path: Path):
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            if height > 2 * width:
                crop_size = width
                top = max((height - crop_size) // 2, 0)
                bottom = top + crop_size
                img = img.crop((0, top, crop_size, bottom))
            return img.copy()
    except UnidentifiedImageError:
        # Some uploads are HEIC files mislabeled as .jpg.
        try:
            with img_path.open("rb") as f:
                header = f.read(16)
            if b"ftypheic" in header or b"ftypheix" in header:
                if not _HEIF_ENABLED:
                    return None
                with Image.open(img_path) as img:
                    return img.copy()
        except Exception:
            return None
        return None
    except Exception:
        return None


def render_prompts_and_texts(prompts_df: pd.DataFrame, texts_df: pd.DataFrame, selected_grid=None, on_select=None, show_selector=True) -> None:
    st.caption(
        f"Prompt source: `{resolve_path(PROMPTS_CSV_PATH)}` | Texts source: `{resolve_path(MESSAGES_CSV_PATH)}`"
    )

    grids_prompts = set(prompts_df["grid_id"].map(_norm_grid_id).unique()) if not prompts_df.empty else set()
    grids_texts = set(texts_df["grid_number"].map(_norm_grid_id).unique()) if not texts_df.empty else set()
    grid_ids = sorted(grids_prompts | grids_texts, key=_grid_sort_key, reverse=True)

    if not grid_ids:
        st.info("No prompts or texts available.")
        return

    if show_selector:
        default_index = 0
        if selected_grid in grid_ids:
            default_index = grid_ids.index(selected_grid)
        def _format_grid_label(grid_id):
            try:
                grid_date = ImmaculateGridUtils._fixed_date_from_grid_number(int(grid_id))
                return f"{grid_id} ({grid_date})"
            except Exception:
                return str(grid_id)

        selected_grid = st.selectbox(
            "Grid ID",
            grid_ids,
            index=default_index,
            key="masked_grid_select",
            format_func=_format_grid_label,
        )
        if on_select:
            on_select(selected_grid)
    elif selected_grid is None:
        selected_grid = grid_ids[0]

    col_prompt, col_texts = st.columns([1, 2])

    with col_prompt:
        st.subheader("Prompt")
        if not prompts_df.empty and selected_grid in grids_prompts:
            row = prompts_df[prompts_df["grid_id"].map(_norm_grid_id) == _norm_grid_id(selected_grid)].iloc[0]
            grid = prompts_row_to_grid(row)
            render_prompts_grid(grid)
        else:
            st.info("Prompt not available for this grid.")

    with col_texts:
        st.subheader("Player Results")
        entries = (
            texts_df[texts_df["grid_number"].map(_norm_grid_id) == _norm_grid_id(selected_grid)]
            if not texts_df.empty
            else pd.DataFrame()
        )
        if entries.empty:
            st.info("No results for this grid.")
            return

        restricted_players = {
            name for name, details in GRID_PLAYERS.items()
            if str(details.get("restricted", "")).lower() == "true"
        }

        entries = entries.assign(
            restricted_flag=entries["name"].apply(lambda x: 1 if x in restricted_players else 0)
        ).sort_values(["restricted_flag", "name"])

        cards_per_row = 4
        rows = [
            entries.iloc[i: i + cards_per_row] for i in range(0, len(entries), cards_per_row)
        ]

        for row_df in rows:
            cols = st.columns(cards_per_row)
            row_entries = [row_df.iloc[i] for i in range(len(row_df))]
            for idx, col in enumerate(cols):
                if idx >= len(row_entries):
                    col.markdown("&nbsp;")
                    continue
                entry = row_entries[idx]
                with col:
                    st.markdown(
                        f"**{entry['name']}**  \n"
                        f"Correct: {entry['correct']} | Score: {entry['score']}  \n"
                        f"Date: {entry['date']}"
                    )
                    render_matrix(entry["matrix"])


def render_image_metadata(
    df: pd.DataFrame,
    source_path: Path,
    texts_df: pd.DataFrame | None = None,
    rule5_df: pd.DataFrame | None = None,
    selected_grid=None,
    on_select=None,
    show_selector=True,
) -> None:
    st.caption(f"Source: `{source_path}`")
    if df.empty:
        st.info("No image metadata found at this path.")
        return
    if "grid_number" not in df.columns:
        st.warning("grid_number column missing; cannot filter.")
        return

    if texts_df is None:
        texts_df = pd.DataFrame()

    images_dir = resolve_path(IMAGES_PATH)
    derived_responses_df = pd.DataFrame()
    derived_path = resolve_path(IMAGES_METADATA_CSV_PATH)
    if derived_path.exists():
        try:
            derived_responses_df = pd.read_csv(
                derived_path,
                dtype={"grid_number": str, "submitter": str, "position": str, "response": str},
            )
            if {"grid_number", "submitter", "position", "response"}.issubset(derived_responses_df.columns):
                derived_responses_df["grid_number"] = derived_responses_df["grid_number"].map(_norm_grid_id)
                derived_responses_df["submitter"] = derived_responses_df["submitter"].astype(str)
            else:
                derived_responses_df = pd.DataFrame()
        except Exception:
            derived_responses_df = pd.DataFrame()

    def _display_responses(entry: dict):
        if derived_responses_df.empty:
            return entry.get("responses")
        grid_id = _norm_grid_id(entry.get("grid_number"))
        submitter = str(entry.get("submitter", ""))
        subset = derived_responses_df[
            (derived_responses_df["grid_number"] == grid_id)
            & (derived_responses_df["submitter"] == submitter)
        ]
        if subset.empty:
            return entry.get("responses")
        return {
            str(row["position"]): "" if pd.isna(row.get("response")) else str(row.get("response", ""))
            for _, row in subset.iterrows()
            if str(row.get("position", "")).strip()
        }

    grid_ids = sorted(df["grid_number"].dropna().map(_norm_grid_id).unique(), key=_grid_sort_key, reverse=True)
    if show_selector:
        default_index = 0
        if selected_grid in grid_ids:
            default_index = grid_ids.index(selected_grid)
        def _format_grid_label(grid_id):
            try:
                grid_date = ImmaculateGridUtils._fixed_date_from_grid_number(int(grid_id))
                return f"{grid_id} ({grid_date})"
            except Exception:
                return str(grid_id)

        selected_grid = st.selectbox(
            "Grid ID",
            grid_ids,
            index=default_index,
            key="full_grid_select",
            format_func=_format_grid_label,
        )
        if on_select:
            on_select(selected_grid)
    elif selected_grid is None:
        selected_grid = grid_ids[0]

    rule5_options = []
    rule5_lookup: dict[str, pd.Series] = {}
    if rule5_df is not None and not rule5_df.empty and {"player", "grid_number", "response"}.issubset(rule5_df.columns):
        rule5_display_df = rule5_df.copy()
        rule5_display_df["grid_id"] = rule5_display_df["grid_number"].map(_norm_grid_id)
        rule5_display_df["grid_sort"] = rule5_display_df["grid_id"].map(
            lambda value: int(value) if str(value).isdigit() else -1
        )
        rule5_display_df = rule5_display_df.sort_values(
            by=["grid_sort", "player"],
            ascending=[False, True],
        )
        rule5_options = ["__current__", "__all_rule5__"]
        for idx, row in rule5_display_df.reset_index(drop=True).iterrows():
            option_key = f"rule5::{idx}"
            rule5_options.append(option_key)
            rule5_lookup[option_key] = row

    rule5_choice = "__current__"
    if rule5_options:
        def _format_rule5_option(option: str) -> str:
            if option == "__current__":
                return "Current grid selection"
            if option == "__all_rule5__":
                return "All Rule 5 ban dates"
            row = rule5_lookup[option]
            grid_id = row["grid_id"]
            try:
                grid_date = ImmaculateGridUtils._fixed_date_from_grid_number(int(grid_id))
                grid_label = f"Grid {grid_id} ({grid_date})"
            except Exception:
                grid_label = f"Grid {grid_id}"
            return f"{grid_label} - {row['player']}: {row['response']}"

        rule5_choice = st.selectbox(
            "Rule 5 ban grid",
            rule5_options,
            format_func=_format_rule5_option,
            key="full_rule5_grid_select",
        )

    if rule5_choice == "__all_rule5__":
        rule5_grid_ids = set(rule5_df["grid_number"].map(_norm_grid_id))
        filtered = df[df["grid_number"].map(_norm_grid_id).isin(rule5_grid_ids)]
    elif rule5_choice.startswith("rule5::"):
        selected_grid = rule5_lookup[rule5_choice]["grid_id"]
        if on_select:
            on_select(selected_grid)
        filtered = df[df["grid_number"].map(_norm_grid_id) == _norm_grid_id(selected_grid)]
    else:
        filtered = df[df["grid_number"].map(_norm_grid_id) == _norm_grid_id(selected_grid)]

    if filtered.empty:
        available = ", ".join(grid_ids[:10])
        st.caption(f"Available image grids (latest): {available}")
        st.info("No images for this selection.")
        return

    submitter_series_all = df.get("submitter", pd.Series(dtype=str)).astype(str).str.strip().str.lower()
    undefined_total = int((submitter_series_all == "undefined").sum())
    submitter_series_selected = filtered.get("submitter", pd.Series(dtype=str)).astype(str).str.strip().str.lower()
    undefined_selected = int((submitter_series_selected == "undefined").sum())
    if undefined_total > 0:
        st.warning(
            f"Undefined uploads: {undefined_selected} in this selection, {undefined_total} total. "
            "Use Add / Update Data -> Assign Users to resolve them."
        )
        show_only_undefined = st.checkbox(
            "Show only Undefined uploads for this grid",
            value=False,
            key=f"full_show_only_undefined_{selected_grid}",
        )
        if show_only_undefined:
            filtered = filtered[filtered["submitter"].astype(str).str.strip().str.lower() == "undefined"]
            if filtered.empty:
                st.info("No Undefined uploads for this selection.")
                return

    all_texts = texts_df if not texts_df.empty and "grid_number" in texts_df.columns else pd.DataFrame()

    def _lookup_stats(submitter, grid_number):
        if all_texts.empty or "name" not in all_texts.columns:
            return None, None
        subset = all_texts[
            (all_texts["grid_number"].map(_norm_grid_id) == _norm_grid_id(grid_number))
            & (all_texts["name"] == submitter)
        ]
        if subset.empty:
            return None, None
        subset = subset.sort_values(["correct", "score"], ascending=[False, True])
        best = subset.iloc[0]
        return best.get("correct"), best.get("score")

    filtered = filtered.copy()
    # Normalize date column; prefer image_date if present, else date, else derive from grid_number
    def _normalize_date(row):
        if pd.notna(row.get("image_date")) and str(row.get("image_date")).strip().lower() != "nan":
            return str(row.get("image_date"))
        if pd.notna(row.get("date")) and str(row.get("date")).strip().lower() != "nan":
            return str(row.get("date"))
        try:
            return ImmaculateGridUtils._fixed_date_from_grid_number(int(row.get("grid_number"))).strftime("%Y-%m-%d")
        except Exception:
            return ""

    filtered["date_display"] = filtered.apply(_normalize_date, axis=1)
    filtered[["correct", "score"]] = filtered.apply(
        lambda row: pd.Series(_lookup_stats(row.get("submitter"), row.get("grid_number"))), axis=1
    )

    restricted_players = {
        name for name, details in GRID_PLAYERS.items()
        if str(details.get("restricted", "")).lower() == "true"
    }
    filtered = filtered.assign(
        restricted_flag=filtered["submitter"].apply(lambda x: 1 if x in restricted_players else 0)
    ).sort_values(["grid_number", "restricted_flag", "submitter", "date"], ascending=[False, True, True, True])

    cards_per_row = 4
    rows = [
        filtered.iloc[i: i + cards_per_row] for i in range(0, len(filtered), cards_per_row)
    ]

    for row_df in rows:
        cols = st.columns(cards_per_row)
        entries = [row_df.iloc[i].to_dict() for i in range(len(row_df))]
        for idx, col in enumerate(cols):
            if idx >= len(entries):
                col.markdown("&nbsp;")
                continue
            entry = entries[idx]
            with col:
                st.markdown(
                    f"**{entry.get('submitter', 'Unknown')}**  \n"
                    f"Grid: {entry.get('grid_number', '')} · Date: {entry.get('date_display', entry.get('date', ''))}  \n"
                    f"Correct: {entry.get('correct', '')} · Score: {entry.get('score', '')}"
                )
                filename = entry.get("image_filename")
                # Fall back to basename from stored path if filename missing
                if not filename and entry.get("path"):
                    filename = Path(entry["path"]).name
                if filename:
                    img_path = images_dir / filename
                else:
                    img_path = None

                if img_path and img_path.exists():
                    display_img = load_image_for_display(img_path)
                    if display_img is not None:
                        st.image(display_img, caption=filename, use_container_width=True)
                    else:
                        st.warning(f"Could not render image file: {filename}")
                else:
                    st.warning(f"Image not found for entry (filename={filename}).")

                st.caption("Parsed players (fuzzy-normalized for display):")
                render_responses_grid(_display_responses(entry))


def render_rule5_bans(rule5_df: pd.DataFrame, prompts_df: pd.DataFrame, images_df: pd.DataFrame) -> None:
    st.write("Add or update a Rule 5 full ban in the local master list.")
    rule5_path = resolve_path(RULE5_FULL_BANS_CSV_PATH)
    st.caption(f"Master list: `{rule5_path}`")

    existing_rule5_df = rule5_df.copy() if rule5_df is not None else pd.DataFrame()
    default_grid = ImmaculateGridUtils.get_today_grid_id()
    if not existing_rule5_df.empty and "grid_number" in existing_rule5_df.columns:
        latest_grid = pd.to_numeric(existing_rule5_df["grid_number"], errors="coerce").max()
        if pd.notna(latest_grid):
            default_grid = int(latest_grid)

    with st.form("rule5_full_ban_form", clear_on_submit=False):
        col_grid, col_player = st.columns([1, 2])
        with col_grid:
            ban_grid = st.number_input(
                "Grid number",
                min_value=0,
                step=1,
                value=int(default_grid),
                key="rule5_ban_grid_number",
            )
        with col_player:
            ban_player = st.text_input(
                "Player",
                placeholder="Jose Ramirez",
                key="rule5_ban_player",
            )
        ban_response = st.text_input(
            "Response / reason",
            placeholder="CLE 100 RBI",
            key="rule5_ban_response",
        )
        submitted = st.form_submit_button("Save Rule 5 ban", type="primary")

    if submitted:
        if not str(ban_player).strip():
            st.error("Player is required.")
        elif not str(ban_response).strip():
            st.error("Response / reason is required.")
        else:
            try:
                standardized_response = standardize_rule5_response(
                    int(ban_grid),
                    str(ban_player),
                    str(ban_response),
                    prompts_df,
                    images_df,
                )
                action, saved_df = _upsert_rule5_full_ban(
                    rule5_path,
                    int(ban_grid),
                    str(ban_player),
                    standardized_response,
                )
                st.cache_data.clear()
                st.success(
                    f"Rule 5 ban {action}: Grid {int(ban_grid)} - "
                    f"{str(ban_player).strip()} ({standardized_response})."
                )
                existing_rule5_df = saved_df
            except Exception as exc:
                st.error(f"Failed to save Rule 5 ban: {exc}")

    if existing_rule5_df.empty:
        st.info("No Rule 5 bans found.")
        return

    display_df = existing_rule5_df.copy()
    display_df["grid_number"] = display_df["grid_number"].map(_norm_grid_id)
    display_df["date"] = display_df["grid_number"].map(
        lambda gid: ImmaculateGridUtils._fixed_date_from_grid_number(int(gid))
        if str(gid).isdigit()
        else ""
    )
    display_df = display_df[["grid_number", "date", "player", "response"]]
    display_df = display_df.sort_values(
        by=["grid_number", "player"],
        key=lambda col: pd.to_numeric(col, errors="coerce") if col.name == "grid_number" else col,
        ascending=[False, True],
    )
    st.markdown(player_link_html_table(display_df, max_height_px=420, dark=True), unsafe_allow_html=True)

    st.markdown("### Ban Screenshots")
    st.caption("Shows screenshots for the selected ban grid from the ban-review user set only.")
    if images_df is None or images_df.empty or "grid_number" not in images_df.columns:
        st.info("No image metadata available.")
        return

    review_submitters = sorted(GRID_PLAYERS_RESTRICTED.keys())
    review_options_df = display_df.copy()
    review_options_df["label"] = review_options_df.apply(
        lambda row: f"Grid {row['grid_number']} ({row['date']}) - {row['player']}: {row['response']}",
        axis=1,
    )
    selected_label = st.selectbox(
        "Show all images for ban grid",
        review_options_df["label"].tolist(),
        key="rule5_ban_image_grid_select",
    )
    selected_row = review_options_df[review_options_df["label"] == selected_label].iloc[0]
    selected_grid = _norm_grid_id(selected_row["grid_number"])

    screenshot_df = images_df[
        (images_df["grid_number"].map(_norm_grid_id) == selected_grid)
        & (images_df["submitter"].astype(str).isin(review_submitters))
    ].copy()

    if screenshot_df.empty:
        st.info("No screenshots found for the ban-review users on this grid.")
        return

    images_dir = resolve_path(IMAGES_PATH)
    screenshot_df["submitter_order"] = screenshot_df["submitter"].map(
        {name: idx for idx, name in enumerate(review_submitters)}
    )
    screenshot_df = screenshot_df.sort_values(["submitter_order", "submitter"])
    st.caption(f"Users shown: {', '.join(review_submitters)}")

    cards_per_row = 4
    for start in range(0, len(screenshot_df), cards_per_row):
        cols = st.columns(cards_per_row)
        entries = [screenshot_df.iloc[i].to_dict() for i in range(start, min(start + cards_per_row, len(screenshot_df)))]
        for idx, entry in enumerate(entries):
            with cols[idx]:
                st.markdown(f"**{entry.get('submitter', 'Unknown')}**")
                filename = entry.get("image_filename")
                if not filename and entry.get("path"):
                    filename = Path(str(entry["path"])).name
                img_path = images_dir / filename if filename else None
                if img_path and img_path.exists():
                    display_img = load_image_for_display(img_path)
                    if display_img is not None:
                        st.image(display_img, caption=filename, use_container_width=True)
                    else:
                        st.warning(f"Could not render image file: {filename}")
                else:
                    st.warning(f"Image not found for entry (filename={filename}).")
                render_responses_grid(entry.get("responses"))


def render_data_availability(prompts_df: pd.DataFrame, texts_df: pd.DataFrame, images_df: pd.DataFrame) -> None:
    st.caption(
        f"Prompts: `{resolve_path(PROMPTS_CSV_PATH)}` | "
        f"Texts: `{resolve_path(MESSAGES_CSV_PATH)}` | "
        f"Images: `{resolve_path(IMAGES_PATH)}`"
    )

    grid_ids = set()
    if not prompts_df.empty and "grid_id" in prompts_df.columns:
        grid_ids |= set(prompts_df["grid_id"].unique())
    if not texts_df.empty and "grid_number" in texts_df.columns:
        grid_ids |= set(texts_df["grid_number"].unique())
    if not images_df.empty and "grid_number" in images_df.columns:
        grid_ids |= set(images_df["grid_number"].unique())

    if not grid_ids:
        st.info("No data available to summarize.")
        return

    grid_ids = sorted(grid_ids, key=_grid_sort_key, reverse=True)

    players = list(GRID_PLAYERS.keys())

    def status_color(grid_id, player):
        has_text = False
        has_image = False
        if not texts_df.empty:
            has_text = not texts_df[
                (texts_df.get("grid_number") == grid_id) & (texts_df.get("name") == player)
            ].empty
        if not images_df.empty:
            has_image = not images_df[
                (images_df.get("grid_number") == grid_id) & (images_df.get("submitter") == player)
            ].empty
        if has_text and has_image:
            return "#2ecc71"
        if has_image:
            return "#f39c12"
        if has_text:
            return "#f1c40f"
        return "#ffffff"

    table_rows = []
    for gid in grid_ids:
        grid_date = ImmaculateGridUtils._fixed_date_from_grid_number(gid)
        cells = "".join(
            f"<td style='padding:6px;text-align:center;'><div title='Grid {gid} · {player}' "
            f"style='width:18px;height:18px;margin:auto;border:1px solid #ccc;background:{status_color(gid, player)};'></div></td>"
            for player in players
        )
        table_rows.append(
            f"<tr>"
            f"<td style='padding:6px;font-weight:600;'>{gid}</td>"
            f"<td style='padding:6px;color:#555;'>{grid_date}</td>"
            f"{cells}"
            f"</tr>"
        )

    legend_html = (
        "<div style='margin-bottom:8px; font-size:13px;'>"
        "<span style='display:inline-block;width:14px;height:14px;border:1px solid #ccc;background:#2ecc71;margin-right:6px;vertical-align:middle;'></span>Text + Image"
        "<span style='display:inline-block;width:14px;height:14px;border:1px solid #ccc;background:#f39c12;margin-left:12px;margin-right:6px;vertical-align:middle;'></span>Image only"
        "<span style='display:inline-block;width:14px;height:14px;border:1px solid #ccc;background:#f1c40f;margin-left:12px;margin-right:6px;vertical-align:middle;'></span>Text only"
        "<span style='display:inline-block;width:14px;height:14px;border:1px solid #ccc;background:#ffffff;margin-left:12px;margin-right:6px;vertical-align:middle;'></span>None"
        "</div>"
    )

    header_cells = "".join(
        f"<th style='padding:6px;background:#f4f6fa;color:#111;position:sticky;top:0;z-index:2;'>{p}</th>"
        for p in players
    )
    table_html = legend_html + (
        "<div style='max-height:600px;overflow-y:auto;border:1px solid #ddd;border-radius:6px;'>"
        "<table style='border-collapse:collapse;width:100%;'>"
        f"<thead><tr>"
        f"<th style='padding:6px;background:#f4f6fa;color:#111;position:sticky;top:0;z-index:3;'>Grid</th>"
        f"<th style='padding:6px;background:#f4f6fa;color:#111;position:sticky;top:0;z-index:3;'>Date</th>"
        f"{header_cells}</tr></thead>"
        "<tbody>"
        + "".join(table_rows)
        + "</tbody></table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def render_scores_matrix(texts_df: pd.DataFrame) -> None:
    st.caption("Scores by player, grid, and date (blank means missing).")
    if texts_df.empty:
        st.info("No text results available.")
        return
    if not {"grid_number", "name", "score"}.issubset(texts_df.columns):
        st.warning("Missing required columns in text results.")
        return

    matrix_df = (
        texts_df.pivot_table(index="grid_number", columns="name", values="score", aggfunc="first")
        .sort_index(ascending=False)
    )
    player_columns = sorted(GRID_PLAYERS.keys())
    matrix_df = matrix_df.reindex(columns=player_columns)

    def _date_from_grid(grid_number) -> str:
        try:
            return str(ImmaculateGridUtils._fixed_date_from_grid_number(int(grid_number)))
        except Exception:
            return ""

    matrix_df.insert(0, "date", [_date_from_grid(grid_number) for grid_number in matrix_df.index])
    styler = matrix_df.style.background_gradient(cmap="RdYlGn_r", axis=None, subset=player_columns)
    styler = styler.format(
        {column: (lambda val: "" if pd.isna(val) else f"{int(val)}") for column in player_columns}
        | {"date": lambda val: val}
    )
    st.dataframe(styler, height=min(600, 30 * len(matrix_df) + 40), use_container_width=True)
