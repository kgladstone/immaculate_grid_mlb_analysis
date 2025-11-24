from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st
from PIL import Image

from utils.constants import GRID_PLAYERS, IMAGES_PATH, MESSAGES_CSV_PATH, PROMPTS_CSV_PATH
from .data_loaders import resolve_path


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
        safe_value = str(value).replace("\n", "<br>")
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
            positions.append((label, str(val) if val else "<Empty>"))
    elif isinstance(parsed, list):
        for idx, label in enumerate(label_order):
            if idx < len(parsed):
                val = parsed[idx]
                positions.append((label[1], str(val) if val else "<Empty>"))
            else:
                positions.append((label[1], "<Empty>"))
    return positions


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
    except Exception:
        return None


def render_prompts_and_texts(prompts_df: pd.DataFrame, texts_df: pd.DataFrame, selected_grid=None, on_select=None, show_selector=True) -> None:
    st.caption(
        f"Prompt source: `{resolve_path(PROMPTS_CSV_PATH)}` | Texts source: `{resolve_path(MESSAGES_CSV_PATH)}`"
    )

    grids_prompts = set(prompts_df["grid_id"].unique()) if not prompts_df.empty else set()
    grids_texts = set(texts_df["grid_number"].unique()) if not texts_df.empty else set()
    grid_ids = sorted(grids_prompts | grids_texts, reverse=True)

    if not grid_ids:
        st.info("No prompts or texts available.")
        return

    if show_selector:
        default_index = 0
        if selected_grid in grid_ids:
            default_index = grid_ids.index(selected_grid)
        selected_grid = st.selectbox("Grid ID", grid_ids, index=default_index, key="masked_grid_select")
        if on_select:
            on_select(selected_grid)
    elif selected_grid is None:
        selected_grid = grid_ids[0]

    col_prompt, col_texts = st.columns([1, 2])

    with col_prompt:
        st.subheader("Prompt")
        if not prompts_df.empty and selected_grid in grids_prompts:
            row = prompts_df[prompts_df["grid_id"] == selected_grid].iloc[0]
            grid = prompts_row_to_grid(row)
            render_prompts_grid(grid)
        else:
            st.info("Prompt not available for this grid.")

    with col_texts:
        st.subheader("Player Results")
        entries = texts_df[texts_df["grid_number"] == selected_grid] if not texts_df.empty else pd.DataFrame()
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


def render_image_metadata(df: pd.DataFrame, source_path: Path, texts_df: pd.DataFrame | None = None, selected_grid=None, on_select=None, show_selector=True) -> None:
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
    grid_ids = sorted(df["grid_number"].dropna().unique(), reverse=True)
    if show_selector:
        default_index = 0
        if selected_grid in grid_ids:
            default_index = grid_ids.index(selected_grid)
        selected_grid = st.selectbox("Grid ID", grid_ids, index=default_index, key="full_grid_select")
        if on_select:
            on_select(selected_grid)
    elif selected_grid is None:
        selected_grid = grid_ids[0]
    filtered = df[df["grid_number"] == selected_grid]

    if filtered.empty:
        st.info("No images for this grid.")
        return

    grid_texts = (
        texts_df[texts_df["grid_number"] == selected_grid]
        if not texts_df.empty and "grid_number" in texts_df.columns
        else pd.DataFrame()
    )

    def _lookup_stats(submitter):
        if grid_texts.empty or "name" not in grid_texts.columns:
            return None, None
        subset = grid_texts[grid_texts["name"] == submitter]
        if subset.empty:
            return None, None
        subset = subset.sort_values(["correct", "score"], ascending=[False, True])
        best = subset.iloc[0]
        return best.get("correct"), best.get("score")

    filtered = filtered.copy()
    filtered[["correct", "score"]] = filtered.apply(
        lambda row: pd.Series(_lookup_stats(row.get("submitter"))), axis=1
    )

    restricted_players = {
        name for name, details in GRID_PLAYERS.items()
        if str(details.get("restricted", "")).lower() == "true"
    }
    filtered = filtered.assign(
        restricted_flag=filtered["submitter"].apply(lambda x: 1 if x in restricted_players else 0)
    ).sort_values(["restricted_flag", "submitter", "date"])

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
                    f"Grid: {entry.get('grid_number', '')} · Date: {entry.get('date', '')}  \n"
                    f"Correct: {entry.get('correct', '')} · Score: {entry.get('score', '')}"
                )
                img_path = images_dir / entry.get("image_filename", "")
                if img_path.exists():
                    display_img = load_image_for_display(img_path)
                    if display_img is not None:
                        st.image(display_img, caption=entry.get("image_filename", ""), use_container_width=True)
                    else:
                        st.image(str(img_path), caption=entry.get("image_filename", ""), use_container_width=True)
                else:
                    st.warning(f"Image not found: {entry.get('image_filename', '')}")

                st.caption("Parsed players (by position):")
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

    grid_ids = sorted(grid_ids, reverse=True)

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
        if has_text:
            return "#f1c40f"
        return "#ffffff"

    table_rows = []
    for gid in grid_ids:
        cells = "".join(
            f"<td style='padding:6px;text-align:center;'><div title='Grid {gid} · {player}' "
            f"style='width:18px;height:18px;margin:auto;border:1px solid #ccc;background:{status_color(gid, player)};'></div></td>"
            for player in players
        )
        table_rows.append(f"<tr><td style='padding:6px;font-weight:600;'>{gid}</td>{cells}</tr>")

    legend_html = (
        "<div style='margin-bottom:8px; font-size:13px;'>"
        "<span style='display:inline-block;width:14px;height:14px;border:1px solid #ccc;background:#2ecc71;margin-right:6px;vertical-align:middle;'></span>Text + Image"
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
        f"<thead><tr><th style='padding:6px;background:#f4f6fa;color:#111;position:sticky;top:0;z-index:3;'>Grid</th>{header_cells}</tr></thead>"
        "<tbody>"
        + "".join(table_rows)
        + "</tbody></table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)
