from __future__ import annotations

import contextlib
import io
import os
import shutil
import traceback
from pathlib import Path
from typing import Dict, Iterable, Tuple
import pandas as pd

import streamlit as st

from data.data_prep import create_disaggregated_results_df
from data.image_processor import ImageProcessor
from data.messages_loader import MessagesLoader
from data.mlb_reference import correct_typos_with_fuzzy_matching
from data.prompts_loader import PromptsLoader
from utils.constants import (
    APPLE_TEXTS_DB_PATH,
    IMAGES_METADATA_CSV_PATH,
    IMAGES_METADATA_FUZZY_LOG_PATH,
    IMAGES_METADATA_PATH,
    IMAGES_PATH,
    MESSAGES_CSV_PATH,
    PROMPTS_CSV_PATH,
)
from app.operations.data_loaders import resolve_path


def _default_messages_db_path() -> Path:
    """
    Prefer a workspace copy of chat.db (or backup) to avoid macOS Full Disk Access issues.
    Falls back to the live Messages DB path if no copy is present.
    """
    cwd = Path.cwd()
    for candidate in [
        cwd / "chat_snapshot" / "chat_backup.db",
        cwd / "chat_snapshot" / "chat.db",
        cwd / "chat.db",
        resolve_path(APPLE_TEXTS_DB_PATH),
    ]:
        if candidate.exists():
            return candidate
    return resolve_path(APPLE_TEXTS_DB_PATH)


def _resolve_messages_db_path(texts_db_path: Path | str | None = None) -> Path:
    """Expand a user-supplied path or pick the best available default."""
    if texts_db_path:
        return Path(texts_db_path).expanduser()
    return _default_messages_db_path()


def copy_messages_db(src: Path, dest: Path) -> Path:
    if not src.exists():
        raise FileNotFoundError(f"Messages DB not found at {src}")
    if not os.access(src, os.R_OK):
        raise PermissionError(
            f"Messages DB not readable at {src}. "
            "Grant Full Disk Access to the terminal/IDE running Streamlit or copy from an FDA-enabled shell."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest.resolve()


def refresh_selected_data(
    selected: Iterable[str],
    texts_db_path: Path | str | None = None,
    image_date_range=None,
    preview_only: bool = False,
    progress_cb=None,
) -> Tuple[Dict[str, int], Iterable[str], Dict[str, dict]]:
    results: Dict[str, int] = {}
    errors = []
    diagnostics: Dict[str, dict] = {}
    prompts_df = None
    db_path = None
    db_exists = None
    db_readable = None

    if any(choice in selected for choice in ("Texts", "Images")):
        db_path = _resolve_messages_db_path(texts_db_path)
        db_exists = db_path.exists()
        db_readable = os.access(db_path, os.R_OK)

    if "Prompts" in selected:
        try:
            prompts_loader = PromptsLoader(str(PROMPTS_CSV_PATH))
            prompts_loader.load().validate()
            prompts_df = prompts_loader.get_data()
            results["Prompts"] = len(prompts_df)
        except Exception as exc:  # pragma: no cover - surfacing to UI
            errors.append(f"Prompts refresh failed: {exc}\n{traceback.format_exc()}")

    if "Texts" in selected:
        try:
            if not db_exists:
                raise FileNotFoundError(f"Messages DB not found at {db_path}")
            if not db_readable:
                raise PermissionError(
                    f"Messages DB not readable at {db_path}. "
                    "Grant Full Disk Access to the terminal/IDE running Streamlit or point to a readable copy (e.g., workspace chat.db)."
                )
            messages_loader = MessagesLoader(str(db_path), str(MESSAGES_CSV_PATH))
            messages_loader.load().validate()
            results["Texts"] = len(messages_loader.get_data())
            diagnostics["Texts"] = messages_loader.diagnostics
        except Exception as exc:  # pragma: no cover - surfacing to UI
            diag = f"Path: {db_path} | exists={db_exists} | readable={db_readable}"
            errors.append(f"Texts refresh failed: {exc}\n{traceback.format_exc()}\n{diag}")

    if "Images" in selected:
        try:
            if not db_exists:
                raise FileNotFoundError(f"Messages DB not found at {db_path}")
            if not db_readable:
                raise PermissionError(
                    f"Messages DB not readable at {db_path}. "
                    "Grant Full Disk Access to the terminal/IDE running Streamlit or point to a readable copy (e.g., workspace chat.db)."
                )
            image_processor = ImageProcessor(str(db_path), str(IMAGES_METADATA_PATH), str(IMAGES_PATH))

            # Preview counts by date before processing
            preview_df = image_processor._fetch_images()
            preview_df["image_date"] = pd.to_datetime(preview_df["image_date"]).dt.date
            if image_date_range:
                start_date, end_date = image_date_range
                if start_date:
                    start_date = pd.to_datetime(start_date).date()
                    preview_df = preview_df[preview_df["image_date"] >= start_date]
                if end_date:
                    end_date = pd.to_datetime(end_date).date()
                    preview_df = preview_df[preview_df["image_date"] <= end_date]
            # Crosstab counts by day and sender
            preview_df["image_date"] = preview_df["image_date"].astype(str)
            pivot = (
                preview_df.pivot_table(
                    index="image_date", columns="submitter", values="path", aggfunc="count", fill_value=0
                )
                .reset_index()
                .sort_values(by="image_date", ascending=False)
            )
            diagnostics["Images_counts_by_day"] = pivot.to_dict(orient="records")
            diagnostics["Images_total_in_range"] = len(preview_df)

            if preview_only:
                results["Images (preview only)"] = len(preview_df)
            else:
                image_processor.process_images(date_range=image_date_range, progress_callback=progress_cb)
                image_metadata_df = image_processor.load_image_metadata()
                parser_data_df = image_processor.load_parser_metadata()

                if prompts_df is None:
                    prompts_df = None
                    try:
                        prompts_loader = PromptsLoader(str(PROMPTS_CSV_PATH))
                        prompts_loader.load().validate()
                        prompts_df = prompts_loader.get_data()
                    except Exception:
                        prompts_df = None

                if prompts_df is not None and not image_metadata_df.empty:
                    disagg_df = create_disaggregated_results_df(image_metadata_df, prompts_df)
                    disagg_df, typo_log = correct_typos_with_fuzzy_matching(disagg_df, "response")
                    disagg_df.to_csv(resolve_path(IMAGES_METADATA_CSV_PATH), index=False)
                    typo_log.to_csv(resolve_path(IMAGES_METADATA_FUZZY_LOG_PATH), index=False)

                results["Images metadata entries"] = len(image_metadata_df)
                results["Parser log entries"] = len(parser_data_df)
        except Exception as exc:  # pragma: no cover - surfacing to UI
            diag = f"Path: {db_path} | exists={db_exists} | readable={db_readable}"
            errors.append(f"Images refresh failed: {exc}\n{traceback.format_exc()}\n{diag}")

    return results, errors, diagnostics


def render_refresh_tab() -> None:
    st.subheader("Refresh Data")
    st.write(
        "Select a dataset to refresh. Images will also rebuild the derived CSV and fuzzy matching log when metadata is available."
    )
    choices = ["Prompts", "Texts", "Images"]
    selected = st.radio("Dataset to refresh", options=choices, index=0)
    texts_db_input = None
    image_start_date = None
    image_end_date = None
    if selected in ("Texts", "Images"):
        default_path = str(_default_messages_db_path())
        texts_db_input = st.text_input(
            "Messages DB path",
            value=default_path,
            help="Absolute path to chat.db; must be readable (Full Disk Access on macOS).",
        )
    if selected == "Images":
        image_start_date = st.date_input("Image start date (optional)", value=None)
        image_end_date = st.date_input("Image end date (optional)", value=None)
        st.markdown("**Images workflow:** Preview counts, then process.")

    buf = io.StringIO()
    results = {}
    errors = []
    diags = {}

    if selected == "Images":
        col1, col2 = st.columns(2)
        with col1:
            preview_clicked = st.button("Show image preview", key="preview_images")
        with col2:
            process_clicked = st.button("Process images", key="process_images")

        if preview_clicked:
            with st.spinner("Computing image preview..."):
                try:
                    with contextlib.redirect_stdout(buf):
                        results, errors, diags = refresh_selected_data(
                            [selected],
                            texts_db_path=texts_db_input,
                            image_date_range=(image_start_date, image_end_date),
                            preview_only=True,
                            progress_cb=None,
                        )
                    st.session_state["images_preview"] = {
                        "counts": diags.get("Images_counts_by_day", []),
                        "total": diags.get("Images_total_in_range", 0),
                        "range": (str(image_start_date), str(image_end_date)),
                    }
                except Exception as exc:
                    errors.append(f"Preview failed: {exc}\n{traceback.format_exc()}")
                finally:
                    st.cache_data.clear()

        if process_clicked:
            preview = st.session_state.get("images_preview")
            expected_range = (str(image_start_date), str(image_end_date))
            if not preview or preview.get("range") != expected_range:
                errors.append("Please run a preview for the current date range before processing.")
            else:
                progress_text = st.empty()
                progress_bar = st.progress(0.0)

                def _progress_cb(done, total, current_date=None, current_submitter=None):
                    if total <= 0:
                        return
                    pct = done / total
                    progress_bar.progress(min(1.0, pct))
                    extra = ""
                    if current_date or current_submitter:
                        extra = f" | {current_date or '?'} / {current_submitter or '?'}"
                    progress_text.write(f"Processing images {done}/{total} ({pct*100:.1f}%){extra}")

                with st.spinner("Processing images..."):
                    try:
                        with contextlib.redirect_stdout(buf):
                            results, errors, diags = refresh_selected_data(
                                [selected],
                                texts_db_path=texts_db_input,
                                image_date_range=(image_start_date, image_end_date),
                                preview_only=False,
                                progress_cb=_progress_cb,
                            )
                        st.session_state["images_preview"] = {
                            "counts": diags.get("Images_counts_by_day", []),
                            "total": diags.get("Images_total_in_range", 0),
                            "range": (str(image_start_date), str(image_end_date)),
                        }
                    except Exception as exc:
                        errors.append(f"Processing failed: {exc}\n{traceback.format_exc()}")
                    finally:
                        st.cache_data.clear()
    else:
        if st.button("Refresh selected datasets"):
            with st.spinner("Refreshing..."):
                try:
                    with contextlib.redirect_stdout(buf):
                        results, errors, diags = refresh_selected_data(
                            [selected],
                            texts_db_path=texts_db_input,
                            image_date_range=(image_start_date, image_end_date),
                            preview_only=False,
                            progress_cb=None,
                        )
                except Exception as exc:
                    errors.append(f"Refresh failed: {exc}\n{traceback.format_exc()}")
                finally:
                    st.cache_data.clear()

    if results:
        for label, count in results.items():
            st.success(f"{label} refreshed ({count} rows).")
    if errors:
        for message in errors:
            st.error(message)
    # Offer a quick rerun so Data Viewer picks up new metadata immediately
    if selected == "Images" and results:
        st.info("Reload the app to see updated images in the Data Viewer tab.")
        st.button("Reload app", on_click=lambda: st.session_state.update({"_reload_flag": True}))
        if st.session_state.get("_reload_flag"):
            st.session_state["_reload_flag"] = False
            st.rerun()

    # Single preview table for Images (current run or cached)
    if selected == "Images":
        preview_counts = None
        preview_total = None
        preview_range = None
        if diags.get("Images_counts_by_day") is not None:
            preview_counts = diags.get("Images_counts_by_day")
            preview_total = diags.get("Images_total_in_range", 0)
            preview_range = (str(image_start_date), str(image_end_date))
        elif st.session_state.get("images_preview"):
            cached = st.session_state["images_preview"]
            preview_counts = cached.get("counts")
            preview_total = cached.get("total")
            preview_range = cached.get("range")

        if preview_counts is not None:
            st.subheader("Images preview (counts by day and sender)")
            counts_df = pd.DataFrame(preview_counts)
            if not counts_df.empty:
                if "image_date" in counts_df.columns:
                    counts_df = counts_df.rename(columns={"image_date": "date"})
                if "date" in counts_df.columns:
                    counts_df = counts_df.sort_values(by="date", ascending=False)
            st.dataframe(counts_df, height=min(400, 30 * len(counts_df) + 40))
            if preview_total is not None:
                st.info(f"Images in selected date range: {preview_total}")
            if preview_range:
                st.caption(f"Preview range: {preview_range}")

    if diags.get("Texts"):
        diag = diags["Texts"]
        st.subheader("Texts diagnostics (last 90 days)")
        st.write(f"Cutoff: {diag.get('cutoff')}")
        if diag.get("counts_recent"):
            st.markdown("**Counts by date and sender**")
            counts_df = pd.DataFrame(diag["counts_recent"])
            if not counts_df.empty:
                pivot = counts_df.pivot_table(index="date_str", columns="name", values="count", fill_value=0)
                pivot = pivot.sort_index(ascending=False)
                def emoji_square(val):
                    if val >= 2:
                        return "🟦"
                    if val >= 1:
                        return "🟩"
                    return "⬜"
                emoji_df = pivot.applymap(emoji_square)
                st.dataframe(emoji_df, height=min(600, 30 * len(pivot) + 40))
    if buf.getvalue():
        with st.expander("Console output", expanded=False):
            st.text_area("stdout", value=buf.getvalue(), height=260)
