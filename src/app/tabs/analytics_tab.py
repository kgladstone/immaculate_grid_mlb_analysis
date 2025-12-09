from __future__ import annotations

import contextlib
import io
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import textwrap
from matplotlib.backends.backend_pdf import PdfPages

import tempfile

from utils.constants import GRID_PLAYERS
from app.operations.report_bank import load_report_bank, run_report
from data.data_prep import preprocess_data_into_texts_structure, make_color_map, build_category_structure
from data.mlb_reference import correct_typos_with_fuzzy_matching

ANALYTICS_CACHE_DIR = Path(".cache")


def _build_analytics_context(
    prompts_df: pd.DataFrame,
    texts_df: pd.DataFrame,
    images_df: pd.DataFrame,
    fix_typos: bool,
    progress_cb=None,
):
    color_map = make_color_map(GRID_PLAYERS)
    texts_struct = preprocess_data_into_texts_structure(texts_df)
    categories = build_category_structure(texts_struct, prompts_df)
    if fix_typos:
        with contextlib.redirect_stdout(io.StringIO()):
            image_metadata, _ = correct_typos_with_fuzzy_matching(
                images_df,
                "responses",
                progress_callback=progress_cb,
                verbose=False,
            )
    else:
        image_metadata = images_df
    return {
        "color_map": color_map,
        "texts": texts_struct,
        "texts_raw": texts_df,
        "categories": categories,
        "prompts": prompts_df,
        "images": image_metadata,
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


def render_analytics(prompts_df: pd.DataFrame, texts_df: pd.DataFrame, images_df: pd.DataFrame) -> None:
    st.write("Dynamic preview of all assets that would go into the PDF report.")

    if prompts_df.empty or texts_df.empty:
        st.info("Prompts or texts data is missing; analytics cannot be generated.")
        return
    st.markdown("Prepare analytics data (typo correction is expensive; cached locally).")
    cache_path = None
    fix_typos = st.checkbox("Apply typo correction to image responses (slow)", value=True)
    cache_path = _analytics_cache_path(fix_typos)
    cached_ctx = _load_analytics_cache(cache_path)
    cache_ready = cached_ctx is not None
    status_emoji = "🟢" if cache_ready else "🔴"
    st.write(f"Cache status: {status_emoji} {'ready' if cache_ready else 'not ready'}")

    build_clicked = st.button("Build/refresh analytics cache")
    if build_clicked:
        progress_bar = st.progress(0.0) if fix_typos else None

        def _cb(done, total):
            if progress_bar and total > 0:
                progress_bar.progress(min(1.0, done / total))

        with st.spinner("Building analytics cache..."):
            ctx = _build_analytics_context(prompts_df, texts_df, images_df, fix_typos, progress_cb=_cb)
            _save_analytics_cache(ctx, cache_path)
        st.success("Cache built.")
        cache_ready = True
        cached_ctx = ctx
    if not cache_ready:
        st.info("Build the analytics cache first.")
        return

    ctx = cached_ctx if cached_ctx is not None else _build_analytics_context(prompts_df, texts_df, images_df, fix_typos)

    all_reports = load_report_bank()
    categories = sorted({r.get("category", "Misc") for r in all_reports})
    if "analytics_category" not in st.session_state:
        st.session_state["analytics_category"] = categories[0] if categories else ""
    selected_category = st.selectbox("Report category", options=categories, index=categories.index(st.session_state["analytics_category"]))
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
        return
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
            st.dataframe(output.reset_index(drop=True))
        else:
            st.text(output)
    log_text = buf.getvalue()
    if log_text:
        with st.expander("Logs", expanded=False):
            st.text(log_text)

    if st.button("Generate PDF 📄"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf_path = Path(tmp.name)
        tmp.close()
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        with st.spinner("Generating PDF..."):
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
                for i, rep in enumerate(all_reports, start=1):
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

                total = len(all_reports)
                for idx, report in enumerate(all_reports, start=1):
                    status_text.write(f"Generating {idx}/{total}: {report['title']}")
                    progress_bar.progress(min(1.0, idx / total))
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
                            df_wrapped = df.applymap(lambda v: "\n".join(textwrap.wrap(str(v), width=wrap_width)) if pd.notna(v) else "")
                            # Adjust chunk and page orientation based on width
                            cols = df_wrapped.shape[1]
                            chunk = 30 if cols > 8 else 40
                            pages = max(1, (len(df) + chunk - 1) // chunk)
                            for page in range(pages):
                                slice_df = df_wrapped.iloc[page * chunk : (page + 1) * chunk]
                                wide = cols > 8
                                fig_size = (11, 8.5) if wide else (8.5, 11)
                                fig, ax = plt.subplots(figsize=fig_size)
                                ax.axis("off")
                                ax.text(0.5, 0.97, f"{report['title']} (Page {page+1}/{pages})", ha="center", va="top", fontsize=12, fontweight="bold")
                                col_width = min(0.95, max(0.08, 0.9 / max(cols, 1)))
                                table = ax.table(
                                    cellText=slice_df.values,
                                    colLabels=slice_df.columns,
                                    cellLoc="center",
                                    colWidths=[col_width] * cols,
                                    bbox=[0, 0, 1, 0.9],
                                )
                                table.auto_set_font_size(False)
                                table.set_fontsize(7 if wide else 8)
                                # Header styling and minimum width enforcement
                                min_width = 0.12 if cols > 6 else 0.1
                                for (r, c), cell in table.get_celld().items():
                                    if r == 0:
                                        cell.set_facecolor("#f0f0f0")
                                        cell.set_fontsize(8 if wide else 9)
                                    cell.set_width(max(cell.get_width(), min_width))
                                pdf.savefig(fig)
                                plt.close(fig)
                        else:
                            fig, ax = plt.subplots(figsize=(8.5, 11))
                            ax.axis("off")
                            ax.text(0.5, 0.5, str(result), ha="center", va="center", wrap=True)
                            pdf.savefig(fig)
                            plt.close(fig)
        status_text.write(f"Completed {total}/{total} reports.")
        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF", data=f, file_name="analytics_report.pdf", mime="application/pdf")
