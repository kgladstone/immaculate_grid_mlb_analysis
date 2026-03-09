from __future__ import annotations

import contextlib
import io
from numbers import Number
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import textwrap
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from PIL import UnidentifiedImageError

import tempfile

from utils.constants import GRID_PLAYERS
from app.operations.report_bank import load_report_bank, run_report
from data.data_prep import preprocess_data_into_texts_structure, make_color_map, build_category_structure
from data.mlb_reference import correct_typos_with_fuzzy_matching
from app.tabs.player_data_tab import render_player_data_analytics
from utils.constants import PLAYER_HISTORY_CSV_PATH

ANALYTICS_CACHE_DIR = Path(".cache")
EXCEL_REPORT_TITLES = {
    "Full Week Usage",
    "Low Bit High Reward",
    "Raw Prompts",
    "Raw Results",
    "Raw Images Metadata",
}

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
    lines = []
    for row in (ordered[0:3], ordered[3:6], ordered[6:9]):
        vals = [str(responses.get(pos, "")).strip() for pos in row]
        lines.append(" | ".join(vals))
    st.code("\n".join(lines), language="text")


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
    fix_typos: bool,
    progress_cb=None,
    phase_cb=None,
):
    total_steps = 5 if fix_typos else 4

    if phase_cb:
        phase_cb(1, total_steps, "Loading submitter color map")
    color_map = make_color_map(GRID_PLAYERS)

    if phase_cb:
        phase_cb(2, total_steps, "Preparing text results structure")
    texts_struct = preprocess_data_into_texts_structure(texts_df)

    if phase_cb:
        phase_cb(3, total_steps, "Building category structure")
    categories = build_category_structure(texts_struct, prompts_df)

    if fix_typos:
        if phase_cb:
            phase_cb(4, total_steps, "Correcting image response names")
        with contextlib.redirect_stdout(io.StringIO()):
            image_metadata, _ = correct_typos_with_fuzzy_matching(
                images_df,
                "responses",
                progress_callback=progress_cb,
                verbose=False,
            )
    else:
        image_metadata = images_df

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
    }


def _analytics_cache_path(fix_typos: bool) -> Path:
    label = "typos" if fix_typos else "raw"
    return ANALYTICS_CACHE_DIR / f"analytics_ctx_{label}.pkl"


def _load_analytics_cache(path: Path):
    if not path.exists():
        return None
    try:
        return pd.read_pickle(path)
    except Exception:
        return None


def _save_analytics_cache(ctx: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(ctx, path)

def _write_excel_reports(reports, ctx, excel_path: Path, status_text=None, progress_bar=None) -> None:
    if status_text:
        status_text.write("Building Excel workbook...")
    with pd.ExcelWriter(excel_path) as writer:
        total = len(reports)
        for idx, report in enumerate(reports, start=1):
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
    st.markdown("Prepare analytics data (typo correction is expensive; cached locally).")
    cache_path = None
    fix_typos = st.checkbox("Apply typo correction to image responses (slow)", value=False)
    if fix_typos:
        st.caption("Typo correction can increase overlap metrics by normalizing OCR misspellings.")
    cache_path = _analytics_cache_path(fix_typos)
    cached_ctx = _load_analytics_cache(cache_path)
    cache_ready = cached_ctx is not None
    status_emoji = "🟢" if cache_ready else "🔴"
    st.write(f"Cache status: {status_emoji} {'ready' if cache_ready else 'not ready'}")

    build_clicked = st.button("Build/refresh analytics cache")
    if build_clicked:
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        total_steps = 5 if fix_typos else 4

        def _phase_cb(step, total, label, done=None, done_total=None):
            suffix = ""
            if done is not None and done_total:
                suffix = f" ({done}/{done_total} assets built)"
            status_text.write(f"Building analytics cache... Step {step}/{total}: {label}{suffix}")

            # Smooth progress across phases; if in-phase counts exist, use them.
            phase_start = (step - 1) / max(total, 1)
            phase_width = 1.0 / max(total, 1)
            if done is not None and done_total:
                in_phase = min(1.0, max(0.0, done / done_total))
            else:
                in_phase = 1.0 if step == total else 0.0
            progress_bar.progress(min(1.0, phase_start + in_phase * phase_width))

        def _cb(done, total):
            if total > 0:
                _phase_cb(4, total_steps, "Correcting image response names", done, total)

        with st.spinner("Building analytics cache..."):
            ctx = _build_analytics_context(
                prompts_df,
                texts_df,
                images_df,
                fix_typos,
                progress_cb=_cb,
                phase_cb=_phase_cb,
            )
            _save_analytics_cache(ctx, cache_path)
        _phase_cb(total_steps, total_steps, "Complete")
        st.success("Cache built.")
        cache_ready = True
        cached_ctx = ctx
    if not cache_ready:
        st.info("Build the analytics cache first.")
        return

    ctx = cached_ctx if cached_ctx is not None else _build_analytics_context(prompts_df, texts_df, images_df, fix_typos)

    all_reports = load_report_bank()
    report_tab, dynamic_analysis_tab, export_tab = st.tabs(
        ["Report Preview", "Dynamic Analysis", "Report Export Selection"]
    )

    with report_tab:
        categories = sorted({r.get("category", "Misc") for r in all_reports})
        if "analytics_category" not in st.session_state:
            st.session_state["analytics_category"] = categories[0] if categories else ""
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
        if "analytics_report_idx" not in st.session_state:
            st.session_state["analytics_report_idx"] = 0
        if st.session_state["analytics_report_idx"] >= len(report_titles):
            st.session_state["analytics_report_idx"] = 0

        go_to_title = st.selectbox(
            "Go to report",
            options=report_titles,
            index=st.session_state["analytics_report_idx"] if report_titles else 0,
            key="analytics_go_to",
        ) if report_titles else None
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
                            st.dataframe(df_show, use_container_width=True)
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
                            st.dataframe(df_show, use_container_width=True)
                    else:
                        st.dataframe(df_out)
                else:
                    st.text(output)
            log_text = buf.getvalue()
            if log_text:
                with st.expander("Logs", expanded=False):
                    st.text(log_text)

    with dynamic_analysis_tab:
        st.markdown("### Dynamic Analysis")
        st.caption(
            "For historical player fields (teams/years/positions/awards), build the CSV with: "
            "`python src/scripts/build_player_history_database.py`"
        )
        render_player_data_analytics(ctx["images"], Path(PLAYER_HISTORY_CSV_PATH))

    with export_tab:
        st.markdown("### Report export selection")
        st.caption("Choose which reports go into the PDF and which go into the Excel workbook.")
        default_pdf = {r["title"] for r in all_reports if r.get("title") not in EXCEL_REPORT_TITLES}
        default_excel = {r["title"] for r in all_reports if r.get("title") in EXCEL_REPORT_TITLES}
        if "export_pdf_titles" not in st.session_state:
            st.session_state["export_pdf_titles"] = set(default_pdf)
        if "export_excel_titles" not in st.session_state:
            st.session_state["export_excel_titles"] = set(default_excel)

        col_pdf, col_excel = st.columns(2)
        with col_pdf:
            st.markdown("**Include in PDF**")
            if st.button("PDF: Select all"):
                st.session_state["export_pdf_titles"] = {r["title"] for r in all_reports}
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
        for report in sorted_reports:
            title = report["title"]
            category = report.get("category", "Misc")
            row = st.columns([2, 3, 1, 1])
            row[0].write(category)
            row[1].write(title)
            pdf_checked = title in st.session_state["export_pdf_titles"]
            excel_checked = title in st.session_state["export_excel_titles"]
            if row[2].checkbox("Include in PDF", value=pdf_checked, key=f"pdf_{title}", label_visibility="collapsed"):
                st.session_state["export_pdf_titles"].add(title)
            else:
                st.session_state["export_pdf_titles"].discard(title)
            if row[3].checkbox(
                "Include in Excel",
                value=excel_checked,
                key=f"excel_{title}",
                label_visibility="collapsed",
            ):
                st.session_state["export_excel_titles"].add(title)
            else:
                st.session_state["export_excel_titles"].discard(title)

        export_col_pdf, export_col_excel = st.columns(2)
        with export_col_pdf:
            generate_pdf = st.button("Generate PDF 📄")
        with export_col_excel:
            generate_excel = st.button("Generate Excel 📊")

    if generate_pdf:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf_path = Path(tmp.name)
        tmp.close()
        pdf_reports = [r for r in all_reports if r.get("title") in st.session_state["export_pdf_titles"]]
        total_pdf = len(pdf_reports)
        progress_bar = st.progress(0.0) if total_pdf > 0 else None
        status_text = st.empty()
        if not pdf_reports:
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

                    # Table of contents
                    fig, ax = plt.subplots(figsize=(8.5, 11))
                    ax.axis("off")
                    ax.text(0.5, 0.95, "Table of Contents", ha="center", va="top", fontsize=18, fontweight="bold")
                    y = 0.9
                    for i, rep in enumerate(pdf_reports, start=1):
                        ax.text(0.1, y, f"{i}. {rep['title']}", ha="left", va="top", fontsize=10)
                        y -= 0.03
                        if y < 0.05:
                            pdf.savefig(fig)
                            plt.close(fig)
                            fig, ax = plt.subplots(figsize=(8.5, 11))
                            ax.axis("off")
                            y = 0.95
                    pdf.savefig(fig)
                    plt.close(fig)

                    for idx, report in enumerate(pdf_reports, start=1):
                        status_text.write(f"Generating PDF {idx}/{total_pdf}: {report['title']}")
                        if progress_bar:
                            progress_bar.progress(min(1.0, idx / max(total_pdf, 1)))
                        result = run_report(report, ctx, None)
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
        excel_reports = [r for r in all_reports if r.get("title") in st.session_state["export_excel_titles"]]
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
