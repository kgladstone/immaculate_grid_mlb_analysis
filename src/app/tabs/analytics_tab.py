from __future__ import annotations

import contextlib
import io
from numbers import Number
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import textwrap
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from PIL import UnidentifiedImageError

import tempfile

from config.constants import GRID_PLAYERS, GRID_PLAYERS_RESTRICTED, IMAGES_METADATA_FUZZY_LOG_PATH
from config.constants import IMAGES_METADATA_PATH, IMAGES_PATH, MESSAGES_CSV_PATH, PROMPTS_CSV_PATH, RULE5_FULL_BANS_CSV_PATH
from scripts.build_career_war_cache import build_career_war_cache
from app.services.player_links import player_link_columns, player_link_html, player_link_html_table
from app.services.report_bank import load_report_bank, run_report
from data.transforms.data_prep import preprocess_data_into_texts_structure, make_color_map, build_category_structure
from data.io.mlb_reference import correct_typos_with_fuzzy_matching
from app.tabs.player_data_tab import (
    render_player_data_analytics,
    _build_usage_df,
    _load_career_position_fraction_table,
    _load_career_war_lookup,
    _load_grid_rarity_scores,
    _load_median_year_lookup,
    _load_prompt_position_lookup,
    _normalize_grid_number,
)

ANALYTICS_CACHE_DIR = Path(".cache")
PDF_PERSON_REPORT_SUBMITTERS = sorted(GRID_PLAYERS_RESTRICTED.keys())
EXCEL_REPORT_TITLES = {
    "Full Week Usage",
    "Low Bit High Reward",
    "Raw Prompts",
    "Raw Results",
    "Raw Images Metadata",
}
DYNAMIC_REPORT_FUNCS = {
    "dynamic_player_search_view": "Player Search",
    "dynamic_tableau_mosaic_view": "Tableau Mosaic",
    "dynamic_median_year_hist_view": "Median Year Histogram by User (Usage Weighted)",
    "dynamic_war_hist_view": "Career WAR Distribution by User",
    "dynamic_war_rarity_scatter_view": "Avg Career WAR vs Grid Rarity (Scatter)",
    "dynamic_fudged_position_usage_view": "Fudged Position Usage",
    "dynamic_median_year_hist_unique_view": "Median Year Histogram by Submitter (Unique Players)",
}
PDF_DYNAMIC_REPORT_FUNCS = set(DYNAMIC_REPORT_FUNCS) - {
    "dynamic_player_search_view",
}
RULE5_PDF_TITLE = "Rule 5 Bans + Screenshots"
PDF_ONLY_SECTION_TITLES = {RULE5_PDF_TITLE}

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    _HEIF_ENABLED = True
except Exception:
    _HEIF_ENABLED = False


def _safe_open_image(img_path: Path):
    try:
        with Image.open(img_path) as img:
            return img.copy(), None
    except UnidentifiedImageError as exc:
        try:
            with img_path.open("rb") as f:
                header = f.read(16)
            if b"ftypheic" in header or b"ftypheix" in header:
                if _HEIF_ENABLED:
                    with Image.open(img_path) as img:
                        return img.copy(), None
                return None, (
                    "HEIC image detected but HEIC decoder is not installed. "
                    "Install `pillow-heif` to render this file."
                )
        except Exception:
            pass
        return None, str(exc)
    except Exception as exc:
        return None, str(exc)


def _render_responses_preview(responses: object) -> None:
    ordered = [
        "top_left", "top_center", "top_right",
        "middle_left", "middle_center", "middle_right",
        "bottom_left", "bottom_center", "bottom_right",
    ]
    if not isinstance(responses, dict):
        st.caption("No parsed responses.")
        return
    html_parts = ["<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:4px;'>"]
    for row in (ordered[0:3], ordered[3:6], ordered[6:9]):
        for pos in row:
            value = str(responses.get(pos, "")).strip()
            linked_value = player_link_html(value) if value else "{}"
            html_parts.append(
                "<div style='border:1px solid #ddd;border-radius:4px;padding:6px;"
                "background:#fff;min-height:38px;font-size:12px;font-weight:600;'>"
                f"{linked_value}"
                "</div>"
            )
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def _render_dataframe_with_player_links(df: pd.DataFrame, **kwargs) -> None:
    if player_link_columns(df):
        height = kwargs.pop("height", 520)
        st.markdown(player_link_html_table(df, max_height_px=height), unsafe_allow_html=True)
    else:
        st.dataframe(df, **kwargs)


def _render_overlap_drilldown(ctx: dict, grid_number: int, submitter_1: str, submitter_2: str) -> None:
    images_df = ctx.get("images_raw")
    if images_df is None or images_df.empty:
        st.warning("No image metadata available for drill-down.")
        return

    st.markdown(f"### Drill Down: Grid {grid_number} · {submitter_1} vs {submitter_2}")
    pair = [submitter_1, submitter_2]
    cols = st.columns(2)
    for idx, submitter in enumerate(pair):
        with cols[idx]:
            st.markdown(f"**{submitter}**")
            subset = images_df[
                (pd.to_numeric(images_df.get("grid_number"), errors="coerce") == int(grid_number))
                & (images_df.get("submitter").astype(str) == str(submitter))
            ]
            subset = subset[subset.get("responses").apply(lambda x: isinstance(x, dict))] if not subset.empty else subset
            if subset.empty:
                st.warning("No entry found for this submitter/grid.")
                continue
            row = subset.iloc[-1]

            img_path = None
            if pd.notna(row.get("path")) and str(row.get("path")).strip():
                img_path = Path(str(row.get("path")))
            elif pd.notna(row.get("image_filename")) and str(row.get("image_filename")).strip():
                img_path = Path("images") / str(row.get("image_filename"))

            if img_path and img_path.exists():
                image_obj, err = _safe_open_image(img_path)
                if image_obj is not None:
                    st.image(image_obj, caption=img_path.name, use_container_width=True)
                else:
                    st.warning(f"Could not render image `{img_path.name}`: {err}")
            else:
                st.warning("Image file path is missing or not found.")
            st.caption(f"Date: {row.get('date', '')}")
            st.caption("Parsed responses")
            _render_responses_preview(row.get("responses"))


def _build_analytics_context(
    prompts_df: pd.DataFrame,
    texts_df: pd.DataFrame,
    images_df: pd.DataFrame,
    progress_cb=None,
    phase_cb=None,
):
    total_steps = 5
    typo_log = pd.DataFrame()

    if phase_cb:
        phase_cb(1, total_steps, "Loading submitter color map")
    color_map = make_color_map(GRID_PLAYERS)

    if phase_cb:
        phase_cb(2, total_steps, "Preparing text results structure")
    texts_struct = preprocess_data_into_texts_structure(texts_df)

    if phase_cb:
        phase_cb(3, total_steps, "Building category structure")
    categories = build_category_structure(texts_struct, prompts_df)

    if phase_cb:
        phase_cb(4, total_steps, "Normalizing image response names (always) for analytics")
    with contextlib.redirect_stdout(io.StringIO()):
        image_metadata, typo_log = correct_typos_with_fuzzy_matching(
            images_df,
            "responses",
            progress_callback=progress_cb,
            verbose=False,
        )

    if phase_cb:
        phase_cb(total_steps, total_steps, "Finalizing analytics context")

    return {
        "color_map": color_map,
        "texts": texts_struct,
        "texts_raw": texts_df,
        "categories": categories,
        "prompts": prompts_df,
        "images": image_metadata,
        "images_raw": images_df,
        "typo_log": typo_log,
    }


def _write_fuzzy_log_csv(ctx: dict) -> None:
    typo_log = ctx.get("typo_log")
    if not isinstance(typo_log, pd.DataFrame):
        return
    IMAGES_METADATA_FUZZY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    typo_log.to_csv(IMAGES_METADATA_FUZZY_LOG_PATH, index=False)


def _analytics_cache_path() -> Path:
    return ANALYTICS_CACHE_DIR / "analytics_ctx.pkl"


def _path_mtime_ns(path_like) -> int:
    path = Path(path_like).expanduser()
    return path.stat().st_mtime_ns if path.exists() else -1


def _analytics_input_fingerprint() -> dict:
    return {
        "prompts_mtime_ns": _path_mtime_ns(PROMPTS_CSV_PATH),
        "texts_mtime_ns": _path_mtime_ns(MESSAGES_CSV_PATH),
        "images_mtime_ns": _path_mtime_ns(IMAGES_METADATA_PATH),
    }


def _load_analytics_cache(path: Path):
    if not path.exists():
        return None
    try:
        raw = pd.read_pickle(path)
        if isinstance(raw, dict) and "__ctx__" in raw:
            return raw
        # Backward compatibility with legacy cache format (ctx-only payload).
        return {"__ctx__": raw, "__fingerprint__": None}
    except Exception:
        return None


def _cache_matches_inputs(cache_payload: dict | None, fingerprint: dict) -> bool:
    if not isinstance(cache_payload, dict):
        return False
    cached_fp = cache_payload.get("__fingerprint__")
    return isinstance(cached_fp, dict) and cached_fp == fingerprint


def _rehydrate_analytics_ctx(ctx: dict):
    """
    Ensure cached ctx has runtime-only objects that may not be pickle-safe.
    """
    if not isinstance(ctx, dict):
        return ctx
    if "texts" not in ctx or ctx.get("texts") is None:
        texts_raw = ctx.get("texts_raw", pd.DataFrame())
        prompts = ctx.get("prompts", pd.DataFrame())
        texts_struct = preprocess_data_into_texts_structure(texts_raw)
        ctx["texts"] = texts_struct
        if "categories" not in ctx or ctx.get("categories") is None:
            ctx["categories"] = build_category_structure(texts_struct, prompts)
    return ctx


def _save_analytics_cache(ctx: dict, path: Path, fingerprint: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pd.to_pickle({"__ctx__": ctx, "__fingerprint__": fingerprint}, path)
    except Exception:
        # Some runtime objects (e.g., custom grid result classes) can fail to pickle.
        # Persist a pickle-safe subset and rehydrate on load.
        safe_ctx = dict(ctx)
        safe_ctx["texts"] = None
        pd.to_pickle({"__ctx__": safe_ctx, "__fingerprint__": fingerprint}, path)

def _write_excel_reports(reports, ctx, excel_path: Path, status_text=None, progress_bar=None) -> None:
    if status_text:
        status_text.write("Building Excel workbook...")
    with pd.ExcelWriter(excel_path) as writer:
        exportable_reports = [r for r in reports if r.get("func") not in DYNAMIC_REPORT_FUNCS]
        total = len(exportable_reports)
        for idx, report in enumerate(exportable_reports, start=1):
            if status_text:
                status_text.write(f"Building Excel {idx}/{total}: {report['title']}")
            if progress_bar:
                progress_bar.progress(min(1.0, idx / max(total, 1)))
            sheet_name = report["title"][:31]
            result = run_report(report, ctx, None)
            if isinstance(result, dict) and "__error__" in result:
                df = pd.DataFrame([{"error": result["__error__"]}])
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                continue
            if result is None:
                df = pd.DataFrame([{"info": "No data."}])
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                continue
            if isinstance(result, pd.DataFrame):
                result.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                df = pd.DataFrame([{"value": str(result)}])
                df.to_excel(writer, sheet_name=sheet_name, index=False)


def _format_pdf_cell_value(value: object) -> str:
    if isinstance(value, Number) and not isinstance(value, bool):
        if float(value).is_integer():
            return str(int(value))
        rounded = round(float(value), 3)
        return f"{rounded:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _wrap_pdf_cell(value: object, wrap_width: int = 28) -> str:
    if pd.isna(value):
        return ""
    text = _format_pdf_cell_value(value)
    wrapped_lines = []
    for line in text.splitlines() or [""]:
        if line == "":
            wrapped_lines.append("")
        else:
            wrapped_lines.extend(textwrap.wrap(line, width=wrap_width))
    return "\n".join(wrapped_lines)


def _pdf_wrap_width_for_column(column: object, wide: bool) -> int:
    name = str(column).lower()
    if name == "rank":
        return 6
    if name in {"repeated_players", "players_used", "grid_ids", "submitter_positions", "position_prompts"}:
        return 22 if wide else 26
    if "players" in name or "positions" in name or "prompts" in name:
        return 24 if wide else 28
    return 18 if wide else 24


def _wrap_pdf_frame(frame: pd.DataFrame, wide: bool) -> pd.DataFrame:
    wrapped = pd.DataFrame(index=frame.index)
    for col in frame.columns:
        wrap_width = _pdf_wrap_width_for_column(col, wide)
        wrapped[col] = frame[col].map(lambda value: _wrap_pdf_cell(value, wrap_width=wrap_width))
    return wrapped[frame.columns]


def _pdf_line_count(value: object) -> int:
    return len(str(value).splitlines() or [""])


def _pdf_max_line_len(value: object) -> int:
    return max((len(line) for line in str(value).splitlines()), default=0)


def _pdf_column_widths(frame: pd.DataFrame, max_total_width: float = 0.92) -> list[float]:
    widths = []
    for col in frame.columns:
        name = str(col).lower()
        if name == "rank":
            widths.append(0.055)
            continue
        max_cell = max((_pdf_max_line_len(v) for v in frame[col].astype(str)), default=0)
        target = 0.018 + min(max(max(len(str(col)), max_cell), 6), 28) * 0.006
        widths.append(min(max(target, 0.075), 0.18))

    total = sum(widths)
    if total > max_total_width:
        scale = max_total_width / total
        widths = [w * scale for w in widths]
    return widths


def _pdf_row_heights(frame: pd.DataFrame, max_total_height: float = 0.9) -> list[float]:
    header_lines = max((_pdf_line_count(c) for c in frame.columns), default=1)
    row_weights = [header_lines]
    for _, row in frame.iterrows():
        row_weights.append(max((_pdf_line_count(value) for value in row.astype(str)), default=1))
    total = sum(row_weights) or 1
    base = max_total_height / total
    return [base * w for w in row_weights]


def _paginate_pdf_table(frame: pd.DataFrame, max_lines_per_page: int) -> list[pd.DataFrame]:
    header_lines = max((_pdf_line_count(c) for c in frame.columns), default=1)
    pages: list[pd.DataFrame] = []
    start_idx = 0
    current_lines = header_lines
    for i, (_, row) in enumerate(frame.iterrows()):
        row_lines = max((_pdf_line_count(value) for value in row.astype(str)), default=1)
        if current_lines + row_lines > max_lines_per_page and i > start_idx:
            pages.append(frame.iloc[start_idx:i])
            start_idx = i
            current_lines = header_lines + row_lines
        else:
            current_lines += row_lines
    if start_idx < len(frame):
        pages.append(frame.iloc[start_idx:])
    return pages


def _save_pdf_table(pdf: PdfPages, title: str, df: pd.DataFrame, max_pages: int | None = None) -> int:
    base_df = df.reset_index(drop=True)
    wide = base_df.shape[1] > 8
    df_wrapped = _wrap_pdf_frame(base_df, wide=wide)
    max_lines_per_page = 24 if wide else 30
    page_slices = _paginate_pdf_table(df_wrapped, max_lines_per_page)
    if max_pages is not None:
        page_slices = page_slices[:max_pages]
    total_pages = len(page_slices)
    for page, slice_df in enumerate(page_slices, start=1):
        fig_size = (11, 8.5) if wide else (8.5, 11)
        fig, ax = plt.subplots(figsize=fig_size)
        ax.axis("off")
        suffix = f" (Page {page}/{total_pages})" if total_pages > 1 else ""
        ax.text(0.5, 0.97, f"{title}{suffix}", ha="center", va="top", fontsize=12, fontweight="bold")
        table = ax.table(
            cellText=slice_df.values,
            colLabels=slice_df.columns,
            cellLoc="left",
            colWidths=_pdf_column_widths(slice_df),
            bbox=[0, 0, 1, 0.9],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6.2 if wide else 7.2)
        numeric_cols = {idx for idx, col in enumerate(df.columns) if pd.api.types.is_numeric_dtype(df[col])}
        row_heights = _pdf_row_heights(slice_df)
        for (r, c), cell in table.get_celld().items():
            cell.get_text().set_fontfamily("DejaVu Sans Mono")
            if r == 0:
                cell.set_facecolor("#f0f0f0")
                cell.set_fontsize(8 if wide else 9)
                cell.get_text().set_ha("center")
            else:
                cell.get_text().set_ha("right" if c in numeric_cols else "left")
            if r < len(row_heights):
                cell.set_height(row_heights[r])
        pdf.savefig(fig)
        plt.close(fig)
    return total_pages


def _load_rule5_bans_for_pdf() -> pd.DataFrame:
    if not RULE5_FULL_BANS_CSV_PATH.exists():
        return pd.DataFrame(columns=["player", "grid_number", "response"])
    df = pd.read_csv(RULE5_FULL_BANS_CSV_PATH, dtype=str)
    for col in ["player", "grid_number", "response"]:
        if col not in df.columns:
            df[col] = ""
    df = df[["player", "grid_number", "response"]].copy()
    df["grid_number"] = df["grid_number"].astype(str).str.strip().str.removesuffix(".0")
    df["grid_sort"] = pd.to_numeric(df["grid_number"], errors="coerce")
    return df.sort_values(["grid_sort", "player"], ascending=[False, True]).drop(columns=["grid_sort"])


def _image_path_from_row(row: pd.Series) -> Path | None:
    path_val = str(row.get("path") or "").strip()
    if path_val and path_val.lower() != "nan":
        path = Path(path_val)
        if path.exists():
            return path
    filename = str(row.get("image_filename") or "").strip()
    if filename and filename.lower() != "nan":
        path = Path(IMAGES_PATH) / filename
        if path.exists():
            return path
    return None


def _write_basic_rule5_pages(pdf: PdfPages, images_df: pd.DataFrame) -> int:
    rule5_df = _load_rule5_bans_for_pdf()
    if rule5_df.empty:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(0.5, 0.5, "No Rule 5 bans found.", ha="center", va="center")
        pdf.savefig(fig)
        plt.close(fig)
        return 1

    pages = _save_pdf_table(pdf, "Rule 5 Bans", rule5_df.rename(columns={"response": "intersection"}))
    if images_df is None or images_df.empty or "grid_number" not in images_df.columns:
        return pages

    review_submitters = sorted(GRID_PLAYERS_RESTRICTED.keys())
    images = images_df.copy()
    images["grid_norm"] = images["grid_number"].astype(str).str.strip().str.removesuffix(".0")
    images["submitter"] = images["submitter"].astype(str)

    for grid_id, grid_bans in rule5_df.groupby("grid_number", sort=False):
        grid_id = str(grid_id)
        subset = images[
            (images["grid_norm"] == grid_id)
            & (images["submitter"].isin(review_submitters))
        ].copy()
        if subset.empty:
            continue
        subset["submitter_order"] = subset["submitter"].map({name: idx for idx, name in enumerate(review_submitters)})
        subset = subset.sort_values(["submitter_order", "submitter"]).head(4)

        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        axes_flat = axes.flatten()
        ban_lines = [f"{row.player}: {row.response}" for row in grid_bans.itertuples(index=False)]
        title = f"Grid {grid_id}: " + " | ".join(ban_lines)
        fig.suptitle("\n".join(textwrap.wrap(title, width=110)), fontsize=11, fontweight="bold")
        for ax in axes_flat:
            ax.axis("off")
        for ax, (_, row) in zip(axes_flat, subset.iterrows()):
            img_path = _image_path_from_row(row)
            ax.set_title(str(row.get("submitter", "")), fontsize=10)
            if img_path is None:
                ax.text(0.5, 0.5, "Image not found", ha="center", va="center")
                continue
            img, err = _safe_open_image(img_path)
            if img is None:
                ax.text(0.5, 0.5, f"Could not render image:\n{err}", ha="center", va="center", wrap=True)
                continue
            img.thumbnail((900, 900))
            ax.imshow(img)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)
        pages += 1
    return pages


def _write_pdf_message(pdf: PdfPages, title: str, message: str) -> int:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(0.5, 0.48, message, ha="center", va="center", wrap=True)
    pdf.savefig(fig)
    plt.close(fig)
    return 1


def _write_pdf_table_of_contents(pdf: PdfPages, title: str, entries: list[str]) -> int:
    if not entries:
        return 0
    pages = 0
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.text(0.5, 0.95, title, ha="center", va="top", fontsize=18, fontweight="bold")
    y = 0.9
    line_height = 0.028
    for i, entry in enumerate(entries, start=1):
        lines = textwrap.wrap(f"{i}. {entry}", width=92) or [f"{i}. {entry}"]
        if y - (len(lines) * line_height) < 0.05:
            pdf.savefig(fig)
            plt.close(fig)
            pages += 1
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")
            ax.text(0.5, 0.95, f"{title} (continued)", ha="center", va="top", fontsize=18, fontweight="bold")
            y = 0.9
        for line in lines:
            ax.text(0.1, y, line, ha="left", va="top", fontsize=10)
            y -= line_height
        y -= 0.006
    pdf.savefig(fig)
    plt.close(fig)
    return pages + 1


def _write_pdf_text_table(pdf: PdfPages, title: str, df: pd.DataFrame, rows_per_page: int = 32, max_pages: int | None = None) -> int:
    if df.empty:
        return _write_pdf_message(pdf, title, "No data.")
    pages = [
        df.iloc[start : start + rows_per_page].copy()
        for start in range(0, len(df), rows_per_page)
    ]
    if max_pages is not None:
        pages = pages[:max_pages]
    total_pages = len(pages)
    for page_num, page_df in enumerate(pages, start=1):
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        suffix = f" (Page {page_num}/{total_pages})" if total_pages > 1 else ""
        ax.text(0.5, 0.96, f"{title}{suffix}", ha="center", va="top", fontsize=13, fontweight="bold")
        text = page_df.to_string(index=False, max_colwidth=28)
        text = text.encode("ascii", errors="replace").decode("ascii")
        ax.text(
            0.02,
            0.90,
            text,
            ha="left",
            va="top",
            fontsize=7.5,
            linespacing=1.25,
        )
        pdf.savefig(fig)
        plt.close(fig)
    return total_pages


def _write_pdf_text_pages(pdf: PdfPages, title: str, text: str, max_pages: int | None = None) -> int:
    lines: list[str] = []
    for raw_line in str(text).splitlines() or [""]:
        wrapped = textwrap.wrap(raw_line, width=105, replace_whitespace=False) if raw_line else [""]
        lines.extend(wrapped)
    lines_per_page = 44
    pages = [
        lines[start : start + lines_per_page]
        for start in range(0, len(lines), lines_per_page)
    ] or [[""]]
    if max_pages is not None:
        pages = pages[:max_pages]
    total_pages = len(pages)
    for page_num, page_lines in enumerate(pages, start=1):
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        suffix = f" (Page {page_num}/{total_pages})" if total_pages > 1 else ""
        ax.text(0.5, 0.97, f"{title}{suffix}", ha="center", va="top", fontsize=12, fontweight="bold")
        y = 0.92
        for line in page_lines:
            safe_line = line.encode("ascii", errors="replace").decode("ascii")
            ax.text(0.06, y, safe_line, ha="left", va="top", fontsize=8.5)
            y -= 0.019
        pdf.savefig(fig)
        plt.close(fig)
    return total_pages


def _expanded_pdf_reports(reports: list[dict]) -> list[dict]:
    expanded: list[dict] = []
    for report in reports:
        if report.get("needs_person"):
            for person in PDF_PERSON_REPORT_SUBMITTERS:
                report_for_person = dict(report)
                report_for_person["person"] = person
                report_for_person["title"] = f"{report['title']} - {person}"
                expanded.append(report_for_person)
        else:
            expanded.append(report)
    return expanded


def _usage_for_pdf(ctx: dict) -> pd.DataFrame:
    return _build_usage_df(ctx.get("images", pd.DataFrame()))


def _write_pdf_tableau_mosaic(pdf: PdfPages, usage_df: pd.DataFrame) -> int:
    submitters = sorted(usage_df["submitter"].dropna().astype(str).unique().tolist())
    if not submitters:
        return _write_pdf_message(pdf, "Tableau Mosaic", "No submitters available.")
    cols = 2
    rows = int(np.ceil(len(submitters) / cols)) or 1
    fig, axes = plt.subplots(rows, cols, figsize=(11, max(4.0 * rows, 4.5)))
    axes = np.array(axes).reshape(-1)
    for idx, submitter in enumerate(submitters):
        ax = axes[idx]
        counts = usage_df.loc[usage_df["submitter"] == submitter, "player"].value_counts().head(12)
        if counts.empty:
            ax.set_visible(False)
            continue
        plot_counts = counts.sort_values(ascending=True)
        labels = [textwrap.shorten(str(label), width=24, placeholder="...") for label in plot_counts.index]
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, plot_counts.values, color="#4c78a8", alpha=0.9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Uses")
        ax.grid(axis="x", alpha=0.2)
        ax.set_title(f"{submitter}: Top Player Usage", fontsize=10, fontweight="bold")
    for ax in axes[len(submitters):]:
        ax.set_visible(False)
    fig.suptitle("Tableau Mosaic: Top 12 Players by Submitter", y=0.995, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig)
    plt.close(fig)
    return 1


def _write_pdf_median_year_histogram(pdf: PdfPages, usage_df: pd.DataFrame, unique_players: bool = False) -> int:
    median_lookup = _load_median_year_lookup()
    title = (
        "Median Year Histogram by Submitter (Unique Players)"
        if unique_players
        else "Median Year Histogram by User (Usage Weighted)"
    )
    if not median_lookup:
        return _write_pdf_message(pdf, title, "Median-year cache is missing. Run `python src/scripts/build_baseball_cache.py`.")
    merged = usage_df.assign(median_year=usage_df["player_norm"].map(median_lookup))
    if unique_players:
        merged = merged.drop_duplicates(subset=["submitter", "player_norm"]).copy()
    merged = merged.dropna(subset=["median_year"]).copy()
    if merged.empty:
        return _write_pdf_message(pdf, title, "No player usages could be mapped to median year.")
    merged["median_year"] = merged["median_year"].astype(int)
    submitters = sorted(merged["submitter"].unique().tolist())
    cols = 2
    rows = int(np.ceil(len(submitters) / cols)) or 1
    fig, axes = plt.subplots(rows, cols, figsize=(11, max(3.8 * rows, 4.5)))
    axes = np.array(axes).reshape(-1)
    bins = np.arange(1870, 2031, 5)
    color = "#2a9d8f" if unique_players else "#e76f51"
    for idx, submitter in enumerate(submitters):
        ax = axes[idx]
        vals = merged.loc[merged["submitter"] == submitter, "median_year"]
        weights = np.ones(len(vals)) * (100.0 / max(len(vals), 1))
        ax.hist(vals, bins=bins, weights=weights, color=color, alpha=0.85, edgecolor="white")
        ax.set_title(submitter)
        ax.set_xlabel("Median Year")
        ax.set_ylabel("Freq %")
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xticks(np.arange(1870, 2031, 20))
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    for ax in axes[len(submitters):]:
        ax.set_visible(False)
    fig.suptitle(title, y=0.995, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig)
    plt.close(fig)
    return 1


def _write_pdf_war_histogram(pdf: PdfPages, usage_df: pd.DataFrame) -> int:
    title = "Career WAR Distribution by User"
    war_lookup = _load_career_war_lookup()
    if not war_lookup:
        return _write_pdf_message(pdf, title, "Career WAR cache is missing. Run `python src/scripts/build_baseball_cache.py`.")
    merged = usage_df.assign(career_war=usage_df["player_norm"].map(war_lookup)).dropna(subset=["career_war"]).copy()
    if merged.empty:
        return _write_pdf_message(pdf, title, "No player usages could be mapped to career WAR.")
    submitters = sorted(merged["submitter"].unique().tolist())
    cols = 2
    rows = int(np.ceil(len(submitters) / cols)) or 1
    fig, axes = plt.subplots(rows, cols, figsize=(11, max(3.8 * rows, 4.5)))
    axes = np.array(axes).reshape(-1)
    bins = np.arange(-10, 106, 5)
    for idx, submitter in enumerate(submitters):
        ax = axes[idx]
        vals = merged.loc[merged["submitter"] == submitter, "career_war"]
        weights = np.ones(len(vals)) * (100.0 / max(len(vals), 1))
        ax.hist(vals, bins=bins, weights=weights, color="#264653", alpha=0.85, edgecolor="white")
        ax.set_title(submitter)
        ax.set_xlabel("Career WAR")
        ax.set_ylabel("Freq %")
    for ax in axes[len(submitters):]:
        ax.set_visible(False)
    fig.suptitle("Career WAR Distribution by Submitter (Usage Weighted)", y=0.995, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig)
    plt.close(fig)
    return 1


def _write_pdf_war_rarity_scatter(pdf: PdfPages, usage_df: pd.DataFrame) -> int:
    title = "Avg Career WAR vs Grid Rarity (Scatter)"
    war_lookup = _load_career_war_lookup()
    rarity_df = _load_grid_rarity_scores()
    if not war_lookup:
        return _write_pdf_message(pdf, title, "Career WAR cache is missing. Run `python src/scripts/build_baseball_cache.py`.")
    if rarity_df.empty:
        return _write_pdf_message(pdf, title, "Rarity score data is missing. Refresh Texts in Add/Update Data.")
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
        .dropna(subset=["avg_career_war"])
    )
    merged = by_submitter_grid.merge(rarity_df, on=["submitter", "grid_number"], how="inner", validate="1:1")
    if merged.empty:
        return _write_pdf_message(pdf, title, "No submitter-grid rows had both rarity score and mapped career WAR.")
    max_mapped = int(merged["mapped_players"].max()) if not merged.empty else 0
    min_mapped = min(6, max(1, max_mapped))
    plot_df = merged[merged["mapped_players"] >= min_mapped].copy()
    if plot_df.empty:
        return _write_pdf_message(pdf, title, f"No rows remain after requiring at least {min_mapped} mapped players.")
    submitters = sorted(plot_df["submitter"].astype(str).unique().tolist())
    cols = 2
    rows = int(np.ceil(len(submitters) / cols)) or 1
    fig, axes = plt.subplots(rows, cols, figsize=(11, max(4.0 * rows, 4.5)), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    for idx, submitter in enumerate(submitters):
        ax = axes[idx]
        sub = plot_df[plot_df["submitter"] == submitter].copy()
        ax.scatter(sub["rarity_score"], sub["avg_career_war"], alpha=0.75, s=28, color="#1f77b4")
        x = pd.to_numeric(sub["rarity_score"], errors="coerce")
        y = pd.to_numeric(sub["avg_career_war"], errors="coerce")
        valid = x.notna() & y.notna()
        if valid.sum() >= 2:
            slope, intercept = np.polyfit(x[valid], y[valid], 1)
            x_line = np.linspace(float(x[valid].min()), float(x[valid].max()), 100)
            ax.plot(x_line, slope * x_line + intercept, color="#d62828", linewidth=2, alpha=0.9)
        ax.set_title(f"{submitter} (n={len(sub)})")
        ax.set_xlabel("Grid Score (Lower = Rarer)")
        ax.set_ylabel("Avg Career WAR")
        ax.grid(alpha=0.2)
    for ax in axes[len(submitters):]:
        ax.set_visible(False)
    fig.suptitle(f"{title} | min mapped players: {min_mapped}", y=0.995, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig)
    plt.close(fig)
    return 1


def _build_fudged_position_tables(usage_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    if usage_df.empty or "grid_number" not in usage_df.columns or "position_key" not in usage_df.columns:
        return pd.DataFrame(), pd.DataFrame(), "No parsed player responses with grid/position metadata are available."
    prompt_lookup = _load_prompt_position_lookup()
    if not prompt_lookup:
        return pd.DataFrame(), pd.DataFrame(), "No prompt position mapping is available from `bin/csv/prompts.csv`."
    pos_frac = _load_career_position_fraction_table()
    if pos_frac.empty:
        return pd.DataFrame(), pd.DataFrame(), "No career position fractions are available in `bin/baseball_cache`."
    work = usage_df.copy()
    work["grid_number_norm"] = work["grid_number"].map(_normalize_grid_number)
    work["position_key"] = work["position_key"].astype(str).str.strip()
    work["prompt_position"] = work.apply(
        lambda r: prompt_lookup.get((r.get("grid_number_norm", ""), r.get("position_key", ""))),
        axis=1,
    )
    work = work.dropna(subset=["prompt_position", "player_norm", "submitter", "player"]).copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame(), "No usages intersected with position-specific prompts."
    grouped = (
        work.groupby(["submitter", "player", "player_norm", "prompt_position"], as_index=False)
        .agg(
            grid_uses=("grid_number_norm", "size"),
            grid_ids=("grid_number_norm", lambda s: ", ".join(sorted({str(v) for v in s if str(v).strip()}, key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x))))),
        )
        .rename(columns={"prompt_position": "position", "player": "mlb_player"})
    )
    merged = grouped.merge(pos_frac[["player_norm", "position", "fractional_appearance"]], on=["player_norm", "position"], how="left")
    merged = merged.dropna(subset=["fractional_appearance"]).copy()
    merged["fractional_appearance"] = pd.to_numeric(merged["fractional_appearance"], errors="coerce")
    merged = merged[merged["fractional_appearance"] > 0].copy()
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame(), "No rows remain after mapping to positive career position fractions."
    merged["fudge_score"] = merged["grid_uses"] / merged["fractional_appearance"]
    merged["log_fractional_appearance"] = np.log(merged["fractional_appearance"])
    merged["fractional_appearance"] = merged["fractional_appearance"].round(4)
    merged["log_fractional_appearance"] = merged["log_fractional_appearance"].round(4)
    merged["fudge_score"] = merged["fudge_score"].round(2)
    merged = merged.sort_values(["fudge_score", "grid_uses", "submitter", "mlb_player", "position"], ascending=[False, False, True, True, True]).reset_index(drop=True)
    merged["rank"] = np.arange(1, len(merged) + 1)
    submitter_agg = (
        merged.groupby("submitter", as_index=False)
        .agg(
            grid_uses_total=("grid_uses", "sum"),
            unique_player_positions=("position", "size"),
            weighted_mean_log_fractional_appearance=("log_fractional_appearance", lambda s: np.average(s, weights=merged.loc[s.index, "grid_uses"]) if len(s) else np.nan),
            mean_fractional_appearance=("fractional_appearance", "mean"),
        )
    )
    submitter_agg["submitter_fudge_index"] = -submitter_agg["weighted_mean_log_fractional_appearance"]
    for col in ["weighted_mean_log_fractional_appearance", "mean_fractional_appearance", "submitter_fudge_index"]:
        submitter_agg[col] = pd.to_numeric(submitter_agg[col], errors="coerce").round(4)
    submitter_agg = submitter_agg.sort_values(["submitter_fudge_index", "grid_uses_total", "submitter"], ascending=[False, False, True]).reset_index(drop=True)
    submitter_agg["cheat_rank"] = np.arange(1, len(submitter_agg) + 1)
    submitter_agg["saint_rank"] = submitter_agg["submitter_fudge_index"].rank(method="min", ascending=True).astype(int)
    rollup = submitter_agg[
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
    ]
    detail = merged[
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
    ].head(120)
    return rollup, detail, None


def _write_pdf_fudged_position_usage(pdf: PdfPages, usage_df: pd.DataFrame) -> int:
    rollup, detail, message = _build_fudged_position_tables(usage_df)
    if message:
        return _write_pdf_message(pdf, "Fudged Position Usage", message)
    pages = _write_pdf_text_table(pdf, "Fudged Position Usage: Submitter Rollup", rollup, rows_per_page=18)
    pages += _write_pdf_text_table(
        pdf,
        "Fudged Position Usage: Top 120 Player-Position Rows",
        detail,
        rows_per_page=28,
        max_pages=4,
    )
    return pages


def _write_dynamic_pdf_report(pdf: PdfPages, report: dict, ctx: dict) -> int:
    func = report.get("func")
    if func not in PDF_DYNAMIC_REPORT_FUNCS:
        return _write_pdf_message(pdf, report.get("title", "Report"), "This interactive report is available in Streamlit preview only.")
    usage_df = _usage_for_pdf(ctx)
    if usage_df.empty:
        return _write_pdf_message(pdf, report.get("title", "Report"), "No parsed player responses available in image metadata.")
    if func == "dynamic_tableau_mosaic_view":
        return _write_pdf_tableau_mosaic(pdf, usage_df)
    if func == "dynamic_median_year_hist_view":
        return _write_pdf_median_year_histogram(pdf, usage_df, unique_players=False)
    if func == "dynamic_median_year_hist_unique_view":
        return _write_pdf_median_year_histogram(pdf, usage_df, unique_players=True)
    if func == "dynamic_war_hist_view":
        return _write_pdf_war_histogram(pdf, usage_df)
    if func == "dynamic_war_rarity_scatter_view":
        return _write_pdf_war_rarity_scatter(pdf, usage_df)
    if func == "dynamic_fudged_position_usage_view":
        return _write_pdf_fudged_position_usage(pdf, usage_df)
    return _write_pdf_message(pdf, report.get("title", "Report"), "No PDF renderer is available for this report.")


def _write_pdf_report_result(pdf: PdfPages, report: dict, result, max_pages: int | None = None) -> int:
    if isinstance(result, dict) and "__error__" in result:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(0.5, 0.5, f"{report['title']} failed:\n{result['__error__']}", ha="center", va="center", wrap=True)
        pdf.savefig(fig)
        plt.close(fig)
        return 1
    if report["type"] == "chart":
        fig = result if result is not None else plt.gcf()
        pdf.savefig(fig)
        plt.close(fig)
        return 1
    if result is None:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(0.5, 0.5, f"{report['title']}: No data.", ha="center", va="center")
        pdf.savefig(fig)
        plt.close(fig)
        return 1
    if isinstance(result, pd.DataFrame):
        return _save_pdf_table(pdf, report["title"], result.reset_index(drop=True), max_pages=max_pages)
    return _write_pdf_text_pages(pdf, report["title"], str(result), max_pages=max_pages)


def render_analytics(prompts_df: pd.DataFrame, texts_df: pd.DataFrame, images_df: pd.DataFrame) -> None:
    st.write("Dynamic preview of all assets that would go into the PDF report.")

    if "analytics_enabled" not in st.session_state:
        st.session_state["analytics_enabled"] = False
    if not st.session_state["analytics_enabled"]:
        if st.button("Load analytics workspace"):
            st.session_state["analytics_enabled"] = True
            st.rerun()
        st.info("Analytics is not loaded by default. Click 'Load analytics workspace' to initialize it.")
        return

    if prompts_df.empty or texts_df.empty:
        st.info("Prompts or texts data is missing; analytics cannot be generated.")
        return
    st.markdown(
        "Prepare analytics data (image metadata is normalized via fuzzy correction; cached locally). "
        "Rebuilding the cache now also downloads the latest Baseball-Reference WAR ZIP (first link alphabetically, Z→A) to refresh `bin/baseball_cache/war.csv`."
    )
    cache_path = _analytics_cache_path()
    input_fingerprint = _analytics_input_fingerprint()
    cache_payload = _load_analytics_cache(cache_path)
    cached_ctx = cache_payload.get("__ctx__") if isinstance(cache_payload, dict) else None
    cache_ready = cached_ctx is not None and _cache_matches_inputs(cache_payload, input_fingerprint)
    status_emoji = "🟢" if cache_ready else "🔴"
    st.write(f"Cache status: {status_emoji} {'ready' if cache_ready else 'not ready'}")
    if cached_ctx is not None and not cache_ready:
        st.info("Cached analytics context is stale relative to current data files. Rebuild to refresh reports.")

    build_clicked = st.button("Build/refresh analytics cache")
    if build_clicked:
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        total_steps = 5
        step4_label = "Normalizing image response names (always) for analytics"

        def _phase_cb(step, total, label, done=None, done_total=None):
            suffix = ""
            if done is not None and done_total:
                suffix = f" ({done}/{done_total} assets built)"
            status_text.write(
                f"({step} of {total}) : Building analytics cache... Step {step}/{total}: {label}{suffix}"
            )

            # Smooth progress across phases; if in-phase counts exist, use them.
            phase_start = (step - 1) / max(total, 1)
            phase_width = 1.0 / max(total, 1)
            if done is not None and done_total:
                in_phase = min(1.0, max(0.0, done / done_total))
            else:
                in_phase = 1.0 if step == total else 0.0
            progress_bar.progress(min(1.0, phase_start + in_phase * phase_width))

        def _step4_progress(done, done_total):
            _phase_cb(4, total_steps, step4_label, done=done, done_total=done_total)

        with st.spinner("Building analytics cache..."):
            ctx = _build_analytics_context(
                prompts_df,
                texts_df,
                images_df,
                progress_cb=_step4_progress,
                phase_cb=_phase_cb,
            )
            _save_analytics_cache(ctx, cache_path, _analytics_input_fingerprint())
            _write_fuzzy_log_csv(ctx)
        _phase_cb(total_steps, total_steps, "Complete")
        st.success(
            f"Cache built. Fuzzy log updated at `{IMAGES_METADATA_FUZZY_LOG_PATH}` "
            f"({len(ctx.get('typo_log', pd.DataFrame()))} changes)."
        )
        try:
            war_status = st.empty()

            def _war_progress(message: str) -> None:
                war_status.write(f"WAR cache: `{message}`")

            build_career_war_cache(
                None,
                None,
                Path("bin/baseball_cache"),
                auto_download=True,
                progress_cb=_war_progress,
            )
            st.success("Career WAR cache saved to `bin/baseball_cache/war.csv`.")
        except Exception as exc:
            st.warning(f"Could not refresh career WAR cache: {exc}")
        cache_ready = True
        cached_ctx = ctx
    if not cache_ready:
        st.info("Build the analytics cache first.")
        return

    ctx = cached_ctx if cached_ctx is not None else _build_analytics_context(prompts_df, texts_df, images_df)
    ctx = _rehydrate_analytics_ctx(ctx)

    all_reports = load_report_bank()
    report_tab, export_tab = st.tabs(
        ["📄 Report Preview", "📤 Report Export Selection"]
    )

    with report_tab:
        categories = sorted({r.get("category", "Misc") for r in all_reports})
        if "analytics_category" not in st.session_state:
            st.session_state["analytics_category"] = categories[0] if categories else ""
        elif categories and st.session_state["analytics_category"] not in categories:
            # Handle renamed/removed categories in persisted session state.
            st.session_state["analytics_category"] = categories[0]
        selected_category = st.selectbox(
            "Report category",
            options=categories,
            index=categories.index(st.session_state["analytics_category"]) if categories else 0,
        )
        if selected_category != st.session_state.get("analytics_category"):
            st.session_state["analytics_category"] = selected_category
            st.session_state["analytics_report_idx"] = 0

        reports = [r for r in all_reports if r.get("category", "Misc") == st.session_state["analytics_category"]]
        report_titles = [r["title"] for r in reports]
        report_titles_sorted = sorted(report_titles, key=lambda t: str(t).lower())
        if "analytics_report_idx" not in st.session_state:
            st.session_state["analytics_report_idx"] = 0
        if st.session_state["analytics_report_idx"] >= len(report_titles):
            st.session_state["analytics_report_idx"] = 0

        current_title = report_titles[st.session_state["analytics_report_idx"]] if report_titles else None
        go_to_index = report_titles_sorted.index(current_title) if current_title in report_titles_sorted else 0
        go_to_title = st.selectbox(
            "Go to report",
            options=report_titles_sorted,
            index=go_to_index if report_titles_sorted else 0,
            key="analytics_go_to",
        ) if report_titles_sorted else None
        if go_to_title:
            st.session_state["analytics_report_idx"] = report_titles.index(go_to_title)

        col_prev, col_curr, col_next = st.columns([1, 3, 1])
        with col_prev:
            if st.button("Previous") and report_titles:
                st.session_state["analytics_report_idx"] = (st.session_state["analytics_report_idx"] - 1) % len(reports)
        with col_curr:
            if report_titles:
                st.markdown(f"**Current report:** {report_titles[st.session_state['analytics_report_idx']]}")
            else:
                st.markdown("**Current report:** None")
        with col_next:
            if st.button("Next") and report_titles:
                st.session_state["analytics_report_idx"] = (st.session_state["analytics_report_idx"] + 1) % len(reports)

        if not reports:
            st.warning("No reports available for this category.")
        else:
            selected_report = reports[st.session_state["analytics_report_idx"]]

            selected_person = None
            if selected_report["needs_person"]:
                selected_person = st.selectbox("Select submitter", options=sorted(GRID_PLAYERS.keys()))

            dynamic_view = DYNAMIC_REPORT_FUNCS.get(selected_report.get("func"))
            if dynamic_view is not None:
                st.caption(
                    "Median-year histograms link the refreshed responses in `bin/csv/images_metadata.csv` to Lahman appearances cached in `bin/baseball_cache` (keep it up-to-date with `python src/scripts/build_baseball_cache.py`)."
                )
                render_player_data_analytics(
                    ctx["images"],
                    forced_analysis=dynamic_view,
                )
            else:
                # Auto-generate current report
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    output = run_report(selected_report, ctx, selected_person)
                if isinstance(output, dict) and "__error__" in output:
                    st.error(f"Failed to render report: {output['__error__']}")
                elif selected_report["type"] == "chart":
                    fig = output
                    if fig is None:
                        fig = plt.gcf()
                    st.pyplot(fig)
                else:
                    if output is None:
                        st.info("No data.")
                    elif isinstance(output, pd.DataFrame):
                        df_out = output.reset_index(drop=True)
                        if selected_report.get("func") == "analyze_grid_overlap_submitters":
                            df_show = df_out.copy()
                            if {"grid_number", "submitter_1", "submitter_2"}.issubset(df_show.columns):
                                _render_dataframe_with_player_links(df_show, use_container_width=True)
                                st.markdown("#### Drill Down")
                                drill_idx = st.selectbox(
                                    "Select a row",
                                    options=list(df_show.index),
                                    format_func=lambda i: (
                                        f"Grid {int(df_show.loc[i, 'grid_number'])} | "
                                        f"{df_show.loc[i, 'submitter_1']} vs {df_show.loc[i, 'submitter_2']} | "
                                        f"{df_show.loc[i, 'overlap_cells']}"
                                    ),
                                    key="overlap_drill_row_idx",
                                )
                                if st.button("Show Drill Down", key="overlap_show_drilldown"):
                                    st.session_state["overlap_drilldown_selection"] = int(drill_idx)
                                selected_idx = st.session_state.get("overlap_drilldown_selection")
                                if selected_idx is not None and selected_idx in df_show.index:
                                    sel = df_show.loc[selected_idx]
                                    _render_overlap_drilldown(
                                        ctx,
                                        int(sel["grid_number"]),
                                        str(sel["submitter_1"]),
                                        str(sel["submitter_2"]),
                                    )
                            else:
                                _render_dataframe_with_player_links(df_show, use_container_width=True)
                        else:
                            _render_dataframe_with_player_links(df_out)
                    else:
                        st.text(output)
                log_text = buf.getvalue()
                if log_text:
                    with st.expander("Logs", expanded=False):
                        st.text(log_text)

    with export_tab:
        st.markdown("### Report export selection")
        st.caption("Choose which reports go into the PDF and which go into the Excel workbook.")
        pdf_exportable_reports = [
            r
            for r in all_reports
            if r.get("func") not in DYNAMIC_REPORT_FUNCS or r.get("func") in PDF_DYNAMIC_REPORT_FUNCS
        ]
        excel_exportable_reports = [r for r in all_reports if r.get("func") not in DYNAMIC_REPORT_FUNCS]
        pdf_disabled_titles = {
            r["title"]
            for r in all_reports
            if r.get("func") in DYNAMIC_REPORT_FUNCS and r.get("func") not in PDF_DYNAMIC_REPORT_FUNCS
        }
        excel_disabled_titles = {r["title"] for r in all_reports if r.get("func") in DYNAMIC_REPORT_FUNCS}
        default_pdf = {r["title"] for r in pdf_exportable_reports if r.get("title") not in EXCEL_REPORT_TITLES}
        default_pdf.update(PDF_ONLY_SECTION_TITLES)
        default_excel = {r["title"] for r in excel_exportable_reports if r.get("title") in EXCEL_REPORT_TITLES}
        if "export_pdf_titles" not in st.session_state:
            st.session_state["export_pdf_titles"] = set(default_pdf)
        else:
            st.session_state["export_pdf_titles"] = set(st.session_state["export_pdf_titles"]) - pdf_disabled_titles
        if "export_excel_titles" not in st.session_state:
            st.session_state["export_excel_titles"] = set(default_excel)
        else:
            st.session_state["export_excel_titles"] = set(st.session_state["export_excel_titles"]) - excel_disabled_titles

        col_pdf, col_excel = st.columns(2)
        with col_pdf:
            st.markdown("**Include in PDF**")
            if st.button("PDF: Select all"):
                st.session_state["export_pdf_titles"] = {r["title"] for r in all_reports}
                st.session_state["export_pdf_titles"].update(PDF_ONLY_SECTION_TITLES)
            if st.button("PDF: Clear all"):
                st.session_state["export_pdf_titles"] = set()
        with col_excel:
            st.markdown("**Include in Excel**")
            if st.button("Excel: Select defaults"):
                st.session_state["export_excel_titles"] = set(default_excel)
            if st.button("Excel: Clear all"):
                st.session_state["export_excel_titles"] = set()

        header = st.columns([2, 3, 1, 1])
        header[0].markdown("**Category**")
        header[1].markdown("**Report**")
        header[2].markdown("**PDF**")
        header[3].markdown("**Excel**")

        sorted_reports = sorted(
            all_reports,
            key=lambda r: (r.get("category", "Misc"), r.get("title", "")),
        )
        pdf_only_rows = [
            {
                "title": RULE5_PDF_TITLE,
                "category": "Controls & Restrictions",
                "func": "__rule5_pdf_section__",
            }
        ]
        for report in pdf_only_rows:
            title = report["title"]
            row = st.columns([2, 3, 1, 1])
            row[0].write(report.get("category", "PDF Only"))
            row[1].write(title)
            pdf_checked = title in st.session_state["export_pdf_titles"]
            if row[2].checkbox(
                "Include in PDF",
                value=pdf_checked,
                key=f"pdf_{title}",
                label_visibility="collapsed",
            ):
                st.session_state["export_pdf_titles"].add(title)
            else:
                st.session_state["export_pdf_titles"].discard(title)
            row[3].checkbox(
                "Include in Excel",
                value=False,
                key=f"excel_{title}",
                label_visibility="collapsed",
                disabled=True,
            )
        for report in sorted_reports:
            title = report["title"]
            category = report.get("category", "Misc")
            is_dynamic_report = report.get("func") in DYNAMIC_REPORT_FUNCS
            is_pdf_only_dynamic = report.get("func") in PDF_DYNAMIC_REPORT_FUNCS
            pdf_disabled = is_dynamic_report and report.get("func") not in PDF_DYNAMIC_REPORT_FUNCS
            excel_disabled = is_dynamic_report
            row = st.columns([2, 3, 1, 1])
            row[0].write(category)
            if is_pdf_only_dynamic:
                title_label = f"{title} (PDF visual)"
            elif is_dynamic_report:
                title_label = f"{title} (preview only)"
            else:
                title_label = title
            row[1].write(title_label)
            pdf_checked = title in st.session_state["export_pdf_titles"]
            excel_checked = title in st.session_state["export_excel_titles"]
            if row[2].checkbox(
                "Include in PDF",
                value=pdf_checked and not pdf_disabled,
                key=f"pdf_{title}",
                label_visibility="collapsed",
                disabled=pdf_disabled,
            ):
                st.session_state["export_pdf_titles"].add(title)
            else:
                st.session_state["export_pdf_titles"].discard(title)
            if row[3].checkbox(
                "Include in Excel",
                value=excel_checked and not excel_disabled,
                key=f"excel_{title}",
                label_visibility="collapsed",
                disabled=excel_disabled,
            ):
                st.session_state["export_excel_titles"].add(title)
            else:
                st.session_state["export_excel_titles"].discard(title)

        st.markdown("### Presets")
        st.caption(
            "Basic Export Mode includes the Rule 5 bans table, relevant ban screenshots, "
            "and all non-Excel analyses capped at 5 pages each."
        )
        generate_basic_pdf = st.button("Generate Basic Export Mode PDF")

        export_col_pdf, export_col_excel = st.columns(2)
        with export_col_pdf:
            generate_pdf = st.button("Generate PDF 📄")
        with export_col_excel:
            generate_excel = st.button("Generate Excel 📊")

    if generate_basic_pdf:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf_path = Path(tmp.name)
        tmp.close()
        include_rule5_pdf = RULE5_PDF_TITLE in st.session_state["export_pdf_titles"]
        basic_reports = [
            r
            for r in all_reports
            if r.get("title") not in EXCEL_REPORT_TITLES
            and (
                r.get("func") not in DYNAMIC_REPORT_FUNCS
                or r.get("func") in PDF_DYNAMIC_REPORT_FUNCS
            )
        ]
        basic_reports = _expanded_pdf_reports(basic_reports)
        total_steps = len(basic_reports) + 1
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        with st.spinner("Generating Basic Export Mode PDF..."):
            with PdfPages(pdf_path) as pdf:
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis("off")
                ax.text(0.5, 0.7, "Immaculate Grid Basic Export", ha="center", va="center", fontsize=24, fontweight="bold")
                ax.text(0.5, 0.62, "Rule 5 bans, relevant screenshots, and capped analytics", ha="center", va="center", fontsize=12)
                ax.text(0.5, 0.56, pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), ha="center", va="center", fontsize=10)
                pdf.savefig(fig)
                plt.close(fig)

                toc_entries = [report["title"] for report in basic_reports]
                if include_rule5_pdf:
                    toc_entries = [RULE5_PDF_TITLE] + toc_entries
                _write_pdf_table_of_contents(
                    pdf,
                    "Table of Contents",
                    toc_entries,
                )

                if include_rule5_pdf:
                    status_text.write("Generating Basic Export: Rule 5 bans and screenshots")
                    _write_basic_rule5_pages(pdf, images_df)
                progress_bar.progress(1 / max(total_steps, 1))

                for idx, report in enumerate(basic_reports, start=1):
                    status_text.write(f"Generating Basic Export {idx}/{len(basic_reports)}: {report['title']}")
                    if report.get("func") in PDF_DYNAMIC_REPORT_FUNCS:
                        _write_dynamic_pdf_report(pdf, report, ctx)
                    else:
                        result = run_report(report, ctx, report.get("person"))
                        _write_pdf_report_result(pdf, report, result, max_pages=5)
                    progress_bar.progress(min(1.0, (idx + 1) / max(total_steps, 1)))
        status_text.write("Completed Basic Export Mode PDF.")
        with open(pdf_path, "rb") as f:
            st.download_button(
                "Download Basic Export Mode PDF",
                data=f,
                file_name="basic_export_mode.pdf",
                mime="application/pdf",
            )

    if generate_pdf:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf_path = Path(tmp.name)
        tmp.close()
        include_rule5_pdf = RULE5_PDF_TITLE in st.session_state["export_pdf_titles"]
        pdf_reports = [
            r
            for r in all_reports
            if r.get("title") in st.session_state["export_pdf_titles"]
            and (
                r.get("func") not in DYNAMIC_REPORT_FUNCS
                or r.get("func") in PDF_DYNAMIC_REPORT_FUNCS
            )
        ]
        pdf_reports = _expanded_pdf_reports(pdf_reports)
        total_pdf = len(pdf_reports) + (1 if include_rule5_pdf else 0)
        progress_bar = st.progress(0.0) if total_pdf > 0 else None
        status_text = st.empty()
        if not pdf_reports and not include_rule5_pdf:
            st.warning("No reports selected for PDF.")
            return
        with st.spinner("Generating exports..."):
            if pdf_reports:
                with PdfPages(pdf_path) as pdf:
                    # Cover page
                    fig, ax = plt.subplots(figsize=(8.5, 11))
                    ax.axis("off")
                    ax.text(0.5, 0.7, "Immaculate Grid Analytics", ha="center", va="center", fontsize=24, fontweight="bold")
                    ax.text(0.5, 0.6, pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), ha="center", va="center", fontsize=12)
                    pdf.savefig(fig)
                    plt.close(fig)

                    _write_pdf_table_of_contents(
                        pdf,
                        "Table of Contents",
                        ([RULE5_PDF_TITLE] if include_rule5_pdf else []) + [report["title"] for report in pdf_reports],
                    )

                    start_idx = 1
                    if include_rule5_pdf:
                        status_text.write("Generating PDF: Rule 5 bans and screenshots")
                        _write_basic_rule5_pages(pdf, images_df)
                        if progress_bar:
                            progress_bar.progress(min(1.0, start_idx / max(total_pdf, 1)))
                        start_idx += 1

                    for idx, report in enumerate(pdf_reports, start=start_idx):
                        status_text.write(f"Generating PDF {idx}/{total_pdf}: {report['title']}")
                        if progress_bar:
                            progress_bar.progress(min(1.0, idx / max(total_pdf, 1)))
                        if report.get("func") in PDF_DYNAMIC_REPORT_FUNCS:
                            _write_dynamic_pdf_report(pdf, report, ctx)
                            continue
                        result = run_report(report, ctx, report.get("person"))
                        if isinstance(result, dict) and "__error__" in result:
                            fig, ax = plt.subplots(figsize=(8.5, 11))
                            ax.axis("off")
                            ax.text(0.5, 0.5, f"{report['title']} failed:\n{result['__error__']}", ha="center", va="center", wrap=True)
                            pdf.savefig(fig)
                            plt.close(fig)
                            continue

                        if report["type"] == "chart":
                            fig = result
                            if fig is None:
                                fig = plt.gcf()
                            pdf.savefig(fig)
                            plt.close(fig)
                        else:
                            if result is None:
                                fig, ax = plt.subplots(figsize=(8.5, 11))
                                ax.axis("off")
                                ax.text(0.5, 0.5, f"{report['title']}: No data.", ha="center", va="center")
                                pdf.savefig(fig)
                                plt.close(fig)
                                continue
                            if isinstance(result, pd.DataFrame):
                                df = result.reset_index(drop=True)
                                # Wrap long cell contents to keep tables readable in the PDF
                                wrap_width = 28
                                def _format_cell_value(value: object) -> str:
                                    if isinstance(value, Number) and not isinstance(value, bool):
                                        if float(value).is_integer():
                                            return str(int(value))
                                        rounded = round(float(value), 3)
                                        text = f"{rounded:.3f}".rstrip("0").rstrip(".")
                                        return text
                                    return str(value)

                                def _wrap_cell(value: object) -> str:
                                    if pd.isna(value):
                                        return ""
                                    text = _format_cell_value(value)
                                    lines = text.splitlines()
                                    if not lines:
                                        return ""
                                    wrapped_lines = []
                                    for line in lines:
                                        if line == "":
                                            wrapped_lines.append("")
                                        else:
                                            wrapped_lines.extend(textwrap.wrap(line, width=wrap_width))
                                    return "\n".join(wrapped_lines)

                                df_wrapped = df.applymap(_wrap_cell)

                                def _max_line_len(value: str) -> int:
                                    lines = str(value).splitlines() or [""]
                                    return max(len(line) for line in lines)

                                def _line_count(value: str) -> int:
                                    return len(str(value).splitlines() or [""])

                                def _column_widths(frame: pd.DataFrame, max_total_width: float = 0.98) -> list[float]:
                                    col_lens = []
                                    for col in frame.columns:
                                        max_cell = max((_max_line_len(v) for v in frame[col].astype(str)), default=0)
                                        col_lens.append(max(len(str(col)), max_cell, 1))
                                    total = sum(col_lens) or 1
                                    widths = [max_total_width * (length / total) for length in col_lens]
                                    min_col = 0.06
                                    max_col = 0.24
                                    widths = [min(max(w, min_col), max_col) for w in widths]
                                    scale = (max_total_width / sum(widths)) if sum(widths) > 0 else 1.0
                                    return [w * scale for w in widths]

                                def _row_heights(frame: pd.DataFrame, max_total_height: float = 0.9) -> list[float]:
                                    header_lines = max((_line_count(c) for c in frame.columns), default=1)
                                    row_weights = [header_lines]
                                    for _, row in frame.iterrows():
                                        max_lines = 1
                                        for value in row.astype(str):
                                            max_lines = max(max_lines, _line_count(value))
                                        row_weights.append(max_lines)
                                    total = sum(row_weights) or 1
                                    base = max_total_height / total
                                    return [base * w for w in row_weights]

                                def _paginate_by_lines(frame: pd.DataFrame, max_lines_per_page: int) -> list[pd.DataFrame]:
                                    header_lines = max((_line_count(c) for c in frame.columns), default=1)
                                    pages: list[pd.DataFrame] = []
                                    start_idx = 0
                                    current_lines = header_lines
                                    for i, (_, row) in enumerate(frame.iterrows()):
                                        row_lines = 1
                                        for value in row.astype(str):
                                            row_lines = max(row_lines, _line_count(value))
                                        if current_lines + row_lines > max_lines_per_page and i > start_idx:
                                            pages.append(frame.iloc[start_idx:i])
                                            start_idx = i
                                            current_lines = header_lines + row_lines
                                        else:
                                            current_lines += row_lines
                                    if start_idx < len(frame):
                                        pages.append(frame.iloc[start_idx:])
                                    return pages

                                # Adjust chunk and page orientation based on width
                                cols = df_wrapped.shape[1]
                                wide = cols > 8
                                max_lines = 28 if wide else 36
                                page_slices = _paginate_by_lines(df_wrapped, max_lines)
                                pages = len(page_slices)
                                for page, slice_df in enumerate(page_slices, start=1):
                                    fig_size = (11, 8.5) if wide else (8.5, 11)
                                    fig, ax = plt.subplots(figsize=fig_size)
                                    ax.axis("off")
                                    ax.text(0.5, 0.97, f"{report['title']} (Page {page}/{pages})", ha="center", va="top", fontsize=12, fontweight="bold")
                                    col_widths = _column_widths(slice_df)
                                    table = ax.table(
                                        cellText=slice_df.values,
                                        colLabels=slice_df.columns,
                                        cellLoc="left",
                                        colWidths=col_widths,
                                        bbox=[0, 0, 1, 0.9],
                                    )
                                    table.auto_set_font_size(False)
                                    table.set_fontsize(7 if wide else 8)
                                    numeric_cols = {idx for idx, col in enumerate(slice_df.columns) if pd.api.types.is_numeric_dtype(df[col])}
                                    row_heights = _row_heights(slice_df)
                                    for (r, c), cell in table.get_celld().items():
                                        cell.get_text().set_fontfamily("DejaVu Sans Mono")
                                        if r == 0:
                                            cell.set_facecolor("#f0f0f0")
                                            cell.set_fontsize(8 if wide else 9)
                                            cell.get_text().set_ha("center")
                                        else:
                                            cell.get_text().set_ha("right" if c in numeric_cols else "left")
                                        if r < len(row_heights):
                                            cell.set_height(row_heights[r])
                                    pdf.savefig(fig)
                                    plt.close(fig)
                            else:
                                fig, ax = plt.subplots(figsize=(8.5, 11))
                                ax.axis("off")
                                ax.text(0.5, 0.5, str(result), ha="center", va="center", wrap=True)
                                pdf.savefig(fig)
                                plt.close(fig)
        if total_pdf:
            status_text.write(f"Completed PDF reports: {total_pdf}/{total_pdf}.")
        if total_pdf:
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF", data=f, file_name="analytics_report.pdf", mime="application/pdf")
    if generate_excel:
        excel_reports = [
            r
            for r in all_reports
            if r.get("title") in st.session_state["export_excel_titles"]
            and r.get("func") not in DYNAMIC_REPORT_FUNCS
        ]
        if not excel_reports:
            st.warning("No reports selected for Excel.")
            return
        excel_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
        excel_path = Path(excel_tmp.name)
        excel_tmp.close()
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        with st.spinner("Generating Excel..."):
            _write_excel_reports(excel_reports, ctx, excel_path, status_text=status_text, progress_bar=progress_bar)
        with open(excel_path, "rb") as f:
            st.download_button(
                "Download Excel",
                data=f,
                file_name="analytics_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
