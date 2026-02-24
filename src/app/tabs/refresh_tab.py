from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
import streamlit as st

from data.data_prep import create_disaggregated_results_df
from data.image_processor import ImageProcessor
from data.messages_loader import MessagesLoader
from data.mlb_reference import correct_typos_with_fuzzy_matching
from data.prompts_loader import PromptsLoader
from utils.grid_utils import ImmaculateGridUtils
from utils.constants import (
    APPLE_TEXTS_DB_PATH,
    IMAGES_METADATA_CSV_PATH,
    IMAGES_METADATA_FUZZY_LOG_PATH,
    IMAGES_METADATA_PATH,
    IMAGES_PATH,
    MESSAGES_CSV_PATH,
    PROMPTS_CSV_PATH,
    GRID_PLAYERS,
)
from app.operations.data_loaders import resolve_path

UPLOAD_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".heic"}


def _norm_grid_id(value) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _is_hidden_zip_member(member_name: str) -> bool:
    path = Path(member_name)
    parts = [p for p in path.parts if p not in ("", ".")]
    if any(part == "__MACOSX" for part in parts):
        return True
    if any(part.startswith(".") for part in parts):
        return True
    if path.name.startswith("._"):
        return True
    if path.name in {".DS_Store", "Thumbs.db"}:
        return True
    return False


def _safe_token(text: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", str(text)).strip("_")
    return token[:60] if token else "upload"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_image_path_from_row(row: pd.Series | dict) -> Path | None:
    path_val = str((row.get("path") if isinstance(row, dict) else row.get("path")) or "").strip()
    if path_val:
        p = Path(path_val)
        if p.exists():
            return p
    fn_val = str((row.get("image_filename") if isinstance(row, dict) else row.get("image_filename")) or "").strip()
    if fn_val:
        p2 = Path(IMAGES_PATH) / fn_val
        if p2.exists():
            return p2
    return None


def _backfill_source_hashes_in_metadata(path: Path) -> tuple[int, int]:
    """
    Ensure each metadata row has source_hash when the image exists.
    Returns (updated_rows, missing_image_rows).
    """
    if not path.exists():
        return 0, 0
    try:
        with path.open() as f:
            raw = json.load(f)
    except Exception:
        return 0, 0

    rows = raw if isinstance(raw, list) else ([raw] if isinstance(raw, dict) else [])
    updated = 0
    missing = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        img_path = _resolve_image_path_from_row(row)
        if img_path is None:
            missing += 1
            continue
        try:
            digest = _file_sha256(img_path)
        except Exception:
            missing += 1
            continue
        if str(row.get("source_hash") or "") != digest:
            row["source_hash"] = digest
            updated += 1

    if updated > 0:
        with path.open("w") as f:
            json.dump(rows, f, indent=4)
    return updated, missing


def _rebuild_image_metadata_derivatives() -> tuple[bool, str]:
    """
    Rebuild images_metadata.csv and fuzzy log from current images_metadata.json + prompts.
    """
    try:
        prompts_loader = PromptsLoader(str(PROMPTS_CSV_PATH))
        prompts_loader.load().validate()
        prompts_df = prompts_loader.get_data()

        ip = ImageProcessor(str(_default_messages_db_path()), str(IMAGES_METADATA_PATH), str(IMAGES_PATH))
        image_metadata_df = ip.load_image_metadata()
        if image_metadata_df.empty:
            pd.DataFrame().to_csv(resolve_path(IMAGES_METADATA_CSV_PATH), index=False)
            pd.DataFrame().to_csv(resolve_path(IMAGES_METADATA_FUZZY_LOG_PATH), index=False)
            return True, "Rebuilt derivatives (empty metadata)."

        disagg_df = create_disaggregated_results_df(image_metadata_df, prompts_df)
        disagg_df, typo_log = correct_typos_with_fuzzy_matching(disagg_df, "response")
        disagg_df.to_csv(resolve_path(IMAGES_METADATA_CSV_PATH), index=False)
        typo_log.to_csv(resolve_path(IMAGES_METADATA_FUZZY_LOG_PATH), index=False)
        return True, f"Rebuilt derivatives: rows={len(disagg_df)}, fuzzy_changes={len(typo_log)}."
    except Exception as exc:
        return False, str(exc)


def _resolve_entry_image_path(entry: pd.Series | dict) -> Path | None:
    path_val = str((entry.get("path") if isinstance(entry, dict) else entry.get("path")) or "").strip()
    if path_val:
        p = Path(path_val)
        if p.exists():
            return p
    fn_val = str((entry.get("image_filename") if isinstance(entry, dict) else entry.get("image_filename")) or "").strip()
    if fn_val:
        p2 = Path(IMAGES_PATH) / fn_val
        if p2.exists():
            return p2
    # Fallback for legacy rows where metadata path/filename drifted from what ImageProcessor wrote.
    submitter = str((entry.get("submitter") if isinstance(entry, dict) else entry.get("submitter")) or "").strip()
    grid_val = _norm_grid_id(entry.get("grid_number") if isinstance(entry, dict) else entry.get("grid_number"))
    date_val = str(
        (entry.get("date") if isinstance(entry, dict) else entry.get("date"))
        or (entry.get("image_date") if isinstance(entry, dict) else entry.get("image_date"))
        or ""
    ).strip()
    if submitter and grid_val:
        fallback_name = f"{submitter}_{date_val}_grid_{grid_val}.jpg" if date_val else f"{submitter}_grid_{grid_val}.jpg"
        p3 = Path(IMAGES_PATH) / fallback_name
        if p3.exists():
            return p3
    return None


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


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
            if "image_date" not in preview_df.columns:
                preview_df = pd.DataFrame(columns=["path", "submitter", "image_date"])
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
                updated_hashes, missing_hash_images = _backfill_source_hashes_in_metadata(resolve_path(IMAGES_METADATA_PATH))
                diagnostics["Images_source_hash_backfill"] = {
                    "updated_rows": int(updated_hashes),
                    "missing_image_rows": int(missing_hash_images),
                }
                # Reload after hash backfill so downstream artifacts see current hashes.
                image_metadata_df = image_processor.load_image_metadata()

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
    st.subheader("Add / Update Data")
    main_tab, upload_tab, assign_tab, manual_tab = st.tabs(
        ["Refresh Data", "Upload & Register Image", "Assign Users", "Manual Data Entry"]
    )

    # --- Main refresh tab ---
    with main_tab:
        st.write(
            "Select a dataset to refresh. Images will also rebuild the derived CSV and fuzzy matching log when metadata is available."
        )
        buf = io.StringIO()
        results: Dict[str, int] = {}
        errors: list[str] = []
        diags: Dict[str, dict] = {}

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
                    checklist_box = st.empty()
                    prev_parse_box = st.empty()
                    checklist_state = {
                        "image_key": None,
                        "hash_checked": False,
                        "grid_identified": False,
                        "grid_number": None,
                        "parsing_started": False,
                    }
                    prev_success = {
                        "responses": None,
                        "meta": "",
                    }

                    def _responses_to_text_grid(responses_obj) -> str:
                        if not isinstance(responses_obj, dict):
                            return "No parsed responses."
                        order = [
                            "top_left", "top_center", "top_right",
                            "middle_left", "middle_center", "middle_right",
                            "bottom_left", "bottom_center", "bottom_right",
                        ]
                        vals = []
                        for key in order:
                            v = str(responses_obj.get(key, "")).strip()
                            vals.append(v if v else "X")
                        rows = [vals[0:3], vals[3:6], vals[6:9]]
                        # simple fixed-width text table for easy visual scanning
                        width = 18
                        lines = []
                        for row in rows:
                            lines.append(" | ".join(s[:width].ljust(width) for s in row))
                        return "\n".join(lines)

                    def _render_prev_success():
                        if isinstance(prev_success["responses"], dict):
                            grid_txt = _responses_to_text_grid(prev_success["responses"])
                            prev_parse_box.markdown(
                                "**Previous successful parse (3x3)**  \n"
                                f"{prev_success['meta']}\n"
                                f"```text\n{grid_txt}\n```"
                            )

                    def _render_checklist(current_date, current_submitter):
                        grid_label = checklist_state["grid_number"] if checklist_state["grid_number"] is not None else "?"
                        lines = [
                            f"**Current image:** `{current_date or '?'} / {current_submitter or '?'}`",
                            f"- [{'x' if checklist_state['hash_checked'] else ' '}] Check hash",
                            f"- [{'x' if checklist_state['grid_identified'] else ' '}] Identify grid ID",
                            f"- [{'x' if checklist_state['parsing_started'] else ' '}] Parse grid (grid: `{grid_label}`)",
                        ]
                        checklist_box.markdown("\n".join(lines))

                    def _progress_cb(
                        done,
                        total,
                        current_date=None,
                        current_submitter=None,
                        stage=None,
                        grid_number=None,
                        image_path=None,
                        image_done=False,
                        result_message=None,
                        parsed_responses=None,
                    ):
                        image_key = f"{current_date}|{current_submitter}|{image_path}"
                        if checklist_state["image_key"] != image_key and not image_done:
                            checklist_state["image_key"] = image_key
                            checklist_state["hash_checked"] = False
                            checklist_state["grid_identified"] = False
                            checklist_state["grid_number"] = None
                            checklist_state["parsing_started"] = False

                        if stage in {"hash_checked"}:
                            checklist_state["hash_checked"] = True
                        if stage in {"grid_identified"}:
                            checklist_state["grid_identified"] = True
                            checklist_state["grid_number"] = grid_number
                        if stage in {"parsing_start", "parsing_success", "parsing_finished"}:
                            checklist_state["parsing_started"] = True
                            if grid_number is not None:
                                checklist_state["grid_number"] = grid_number

                        if total <= 0:
                            return
                        pct = done / total
                        progress_bar.progress(min(1.0, pct))
                        extra = ""
                        if current_date or current_submitter:
                            extra = f" | {current_date or '?'} / {current_submitter or '?'}"
                        stage_msg = f" | {stage}" if stage else ""
                        progress_text.write(f"Processing images {done}/{total} ({pct*100:.1f}%){extra}{stage_msg}")

                        if not image_done:
                            _render_checklist(current_date, current_submitter)
                            _render_prev_success()
                        else:
                            checklist_box.empty()
                            if isinstance(parsed_responses, dict) and len(parsed_responses) > 0:
                                prev_success["responses"] = parsed_responses
                                prev_success["meta"] = (
                                    f"`{current_date or '?'} / {current_submitter or '?'} / "
                                    f"grid {grid_number if grid_number is not None else '?'} / Success`"
                                )
                            _render_prev_success()

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
                        except Exception as exc:
                            errors.append(f"Processing failed: {exc}\n{traceback.format_exc()}")
                        finally:
                            st.cache_data.clear()

        else:
            if st.button("Run refresh"):
                with st.spinner(f"Refreshing {selected}..."):
                    try:
                        with contextlib.redirect_stdout(buf):
                            results, errors, diags = refresh_selected_data([selected], texts_db_path=texts_db_input)
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
        if diags.get("Images_source_hash_backfill"):
            hdiag = diags["Images_source_hash_backfill"]
            st.caption(
                "Image hash backfill: "
                f"updated {hdiag.get('updated_rows', 0)} row(s), "
                f"missing image files for {hdiag.get('missing_image_rows', 0)} row(s)."
            )
        if buf.getvalue():
            with st.expander("Console output", expanded=False):
                st.text_area("stdout", value=buf.getvalue(), height=260)

    # --- Upload/registry tab ---
    with upload_tab:
        st.write("Upload a grid screenshot, assign a submitter, and auto-parse/store it in the image registry.")
        st.caption(
            f"Registry target: `{resolve_path(IMAGES_METADATA_PATH)}` "
            f"(this does not directly write to `{resolve_path(MESSAGES_CSV_PATH)}`)."
        )

        def _process_and_register_image(
            ip: ImageProcessor,
            raw_image_path: Path,
            submitter_name: str,
            skip_if_grid_exists: bool = False,
        ) -> dict:
            source_hash = _file_sha256(raw_image_path)
            ocr_text = ip.extract_text_from_image(str(raw_image_path))
            grid_number = ip.grid_number_from_image_text(ocr_text)
            if grid_number is None:
                return {
                    "file": raw_image_path.name,
                    "submitter": submitter_name,
                    "status": "failed",
                    "reason": "Could not detect grid number",
                }

            image_date = ImmaculateGridUtils._fixed_date_from_grid_number(grid_number)
            matrix_all_true = "[[true,true,true],[true,true,true],[true,true,true]]"
            if str(submitter_name).strip().lower() == "undefined":
                source_token = source_hash[:12]
                canonical_name = f"Undefined_{image_date}_grid_{grid_number}_{source_token}.jpg"
            else:
                canonical_name = f"{submitter_name}_{image_date}_grid_{grid_number}.jpg"
            canonical_path = Path(IMAGES_PATH) / canonical_name

            meta_df = ip.load_image_metadata()
            # Always dedupe exact duplicate image content regardless of filename.
            if not meta_df.empty:
                if "source_hash" in meta_df.columns:
                    existing = meta_df["source_hash"].astype(str) == source_hash
                else:
                    # Backward-compatible fallback for older metadata rows.
                    existing = meta_df.get("path", pd.Series(dtype=str)).astype(str) == str(canonical_path)
                if existing.any():
                    return {
                        "file": raw_image_path.name,
                        "submitter": submitter_name,
                        "status": "skipped",
                        "grid_number": grid_number,
                        "reason": "Exact image already exists (content hash match)",
                    }

            parser_msg = ip.process_image_with_dynamic_grid(
                str(raw_image_path),
                submitter_name,
                str(image_date),
                grid_number,
                matrix_all_true,
                skip_validation=True,
            )
            # ImageProcessor writes to {submitter}_{date}_grid_{id}.jpg. If we use a richer canonical
            # name (e.g., Undefined + hash token), rename the written file so metadata path matches disk.
            default_written_path = Path(IMAGES_PATH) / f"{submitter_name}_{image_date}_grid_{grid_number}.jpg"
            if default_written_path != canonical_path and default_written_path.exists():
                final_path = _unique_path(canonical_path)
                try:
                    default_written_path.rename(final_path)
                    canonical_path = final_path
                    canonical_name = final_path.name
                except Exception:
                    # Keep metadata consistent with actual file if rename fails.
                    canonical_path = default_written_path
                    canonical_name = default_written_path.name

            parser_entry = {
                "path": str(canonical_path),
                "submitter": submitter_name,
                "grid_number": grid_number,
                "image_date": str(image_date),
                "parser_message": parser_msg,
                "source_hash": source_hash,
            }
            ip.save_parser_metadata(parser_entry)

            responses_from_cells = {}
            try:
                logo_pos = ip.find_logo_position(str(canonical_path))
                if logo_pos:
                    grid_cells = ip.divide_image_into_grid(str(canonical_path), logo_pos[0], logo_pos[1], 1, logo_pos[2])
                    cell_texts = ip.extract_text_from_cells(str(canonical_path), grid_cells)
                    position_mapping = {
                        (0, 0): "top_left",
                        (0, 1): "top_center",
                        (0, 2): "top_right",
                        (1, 0): "middle_left",
                        (1, 1): "middle_center",
                        (1, 2): "middle_right",
                        (2, 0): "bottom_left",
                        (2, 1): "bottom_center",
                        (2, 2): "bottom_right",
                    }
                    responses_from_cells = {
                        position_mapping.get((r, c)): val
                        for (r, c), val in cell_texts.items()
                        if position_mapping.get((r, c))
                    }
            except Exception:
                responses_from_cells = {}

            # For Undefined placeholders we keep one row per uploaded image (no grid-level overwrite).
            if str(submitter_name).strip().lower() == "undefined":
                mask = meta_df.get("path", pd.Series(dtype=str)).astype(str) == str(canonical_path)
            else:
                mask = (meta_df["grid_number"] == grid_number) & (meta_df["submitter"] == submitter_name)
            if mask.any():
                meta_df.loc[mask, "responses"] = [responses_from_cells or {}]
                meta_df.loc[mask, "date"] = str(image_date)
                meta_df.loc[mask, "image_date"] = str(image_date)
                meta_df.loc[mask, "image_filename"] = meta_df.loc[mask, "image_filename"].fillna(canonical_name)
                meta_df.loc[mask, "path"] = str(canonical_path)
                entry_out = meta_df.loc[mask].iloc[0].to_dict()
            else:
                entry_out = {
                    "path": str(canonical_path),
                    "submitter": submitter_name,
                    "grid_number": grid_number,
                    "image_date": str(image_date),
                    "parser_message": parser_msg,
                    "image_filename": canonical_name,
                    "date": str(image_date),
                    "responses": responses_from_cells or {},
                    "source_hash": source_hash,
                }
                meta_df = pd.concat([meta_df, pd.DataFrame([entry_out])], ignore_index=True)

            if "source_hash" not in meta_df.columns:
                meta_df["source_hash"] = ""
            if mask.any():
                meta_df.loc[mask, "source_hash"] = source_hash

            meta_df.to_json(IMAGES_METADATA_PATH, orient="records", indent=4)
            return {
                "file": raw_image_path.name,
                "submitter": submitter_name,
                "status": "success",
                "grid_number": grid_number,
                "date": str(image_date),
                "parser_message": parser_msg,
                "registered_entry": entry_out,
            }

        uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "heic"])
        submitter = st.selectbox("Submitter", options=sorted(GRID_PLAYERS.keys()))
        process_upload = st.button("Process uploaded image")
        upload_status = st.empty()
        if process_upload:
            if not uploaded:
                st.error("Please upload an image first.")
            else:
                temp_dir = Path(IMAGES_PATH) / "uploads"
                temp_dir.mkdir(parents=True, exist_ok=True)
                dest_path = temp_dir / uploaded.name
                dest_path.write_bytes(uploaded.getbuffer())

                ip = ImageProcessor(str(_default_messages_db_path()), str(IMAGES_METADATA_PATH), str(IMAGES_PATH))
                try:
                    result = _process_and_register_image(ip, dest_path, submitter)
                    if result["status"] == "success":
                        upload_status.success(
                            f"Parsed grid #{result['grid_number']} for {submitter} (date {result['date']}). "
                            f"Message: {result.get('parser_message', '')}"
                        )
                        st.write("Registered metadata entry:")
                        st.json(result.get("registered_entry", {}))
                        st.cache_data.clear()
                        ok, msg = _rebuild_image_metadata_derivatives()
                        if ok:
                            st.caption(msg)
                        else:
                            st.warning(f"Registered metadata, but failed to rebuild images_metadata.csv: {msg}")
                        st.info("Upload saved to image registry. Use Reload app to refresh cached views.")
                        if st.button("Reload app", key="reload_after_single_upload"):
                            st.rerun()
                    else:
                        upload_status.error(f"{result.get('reason', 'Failed to process image')}")
                except Exception as exc:
                    st.error(f"Could not process upload: {exc}")
                finally:
                    dest_path.unlink(missing_ok=True)

        st.markdown("---")
        st.write("Bulk upload: process a `.zip` archive of images. These are tagged as `Undefined` for later assignment.")
        uploaded_zip = st.file_uploader("Upload zip archive", type=["zip"], key="bulk_zip_uploader")
        process_zip = st.button("Process zip archive")
        if process_zip:
            if not uploaded_zip:
                st.error("Please upload a zip archive first.")
            else:
                temp_root = Path(IMAGES_PATH) / "uploads" / f"bulk_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                temp_root.mkdir(parents=True, exist_ok=True)
                zip_path = temp_root / uploaded_zip.name
                zip_path.write_bytes(uploaded_zip.getbuffer())

                extracted_files = []
                skipped_unsupported = 0
                skipped_hidden = 0
                total_members = 0
                try:
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        for idx, member in enumerate(zf.infolist(), start=1):
                            total_members += 1
                            member_path = Path(member.filename)
                            if member.is_dir():
                                continue
                            if _is_hidden_zip_member(member.filename):
                                skipped_hidden += 1
                                continue
                            if member_path.suffix.lower() not in UPLOAD_IMAGE_EXTENSIONS:
                                skipped_unsupported += 1
                                continue
                            # Preserve every file from the archive, even if basenames repeat.
                            safe_name = f"{idx:05d}_{member_path.name}"
                            out_path = temp_root / safe_name
                            with zf.open(member, "r") as src_f, out_path.open("wb") as dst_f:
                                shutil.copyfileobj(src_f, dst_f)
                            extracted_files.append(out_path)
                except Exception as exc:
                    st.error(f"Could not read zip archive: {exc}")
                    shutil.rmtree(temp_root, ignore_errors=True)
                    extracted_files = []

                if not extracted_files:
                    st.warning("No supported image files found in the zip.")
                    shutil.rmtree(temp_root, ignore_errors=True)
                else:
                    st.caption(
                        f"Zip scan: {total_members} entries, {len(extracted_files)} supported images, "
                        f"{skipped_unsupported} unsupported skipped, {skipped_hidden} hidden skipped."
                    )
                    ip = ImageProcessor(str(_default_messages_db_path()), str(IMAGES_METADATA_PATH), str(IMAGES_PATH))
                    progress_bar = st.progress(0.0)
                    progress_text = st.empty()
                    results_rows = []
                    total = len(extracted_files)
                    for i, img_path in enumerate(extracted_files, start=1):
                        progress_text.write(f"Processing {i}/{total}: {img_path.name}")
                        try:
                            result = _process_and_register_image(
                                ip,
                                img_path,
                                "Undefined",
                                skip_if_grid_exists=False,
                            )
                        except Exception as exc:
                            result = {
                                "file": img_path.name,
                                "submitter": "Undefined",
                                "status": "failed",
                                "reason": str(exc),
                            }
                        results_rows.append(result)
                        progress_bar.progress(i / total)

                    st.cache_data.clear()
                    summary_df = pd.DataFrame(results_rows)
                    st.write("Bulk upload results:")
                    st.dataframe(summary_df, use_container_width=True)
                    successes = int((summary_df["status"] == "success").sum()) if "status" in summary_df.columns else 0
                    skipped = int((summary_df["status"] == "skipped").sum()) if "status" in summary_df.columns else 0
                    failed = int((summary_df["status"] == "failed").sum()) if "status" in summary_df.columns else 0
                    st.success(
                        f"Processed {len(summary_df)} files. "
                        f"Successful registrations: {successes}, skipped: {skipped}, failed: {failed}."
                    )
                    if successes > 0:
                        ok, msg = _rebuild_image_metadata_derivatives()
                        if ok:
                            st.caption(msg)
                        else:
                            st.warning(f"Bulk metadata saved, but failed to rebuild images_metadata.csv: {msg}")
                    if successes > 0:
                        st.info("Successful bulk uploads were tagged as `Undefined`. Use the Assign Users tab to map them.")
                    else:
                        st.info(
                            "No new Undefined entries were created. "
                            "Review failed rows in the bulk results table for details."
                        )
                    if st.button("Reload app", key="reload_after_bulk_upload"):
                        st.rerun()
                    shutil.rmtree(temp_root, ignore_errors=True)

    # --- Assign users tab ---
    with assign_tab:
        st.write("Review `Undefined` uploads and assign each image to a submitter.")
        ip = ImageProcessor(str(_default_messages_db_path()), str(IMAGES_METADATA_PATH), str(IMAGES_PATH))
        try:
            meta_df = ip.load_image_metadata()
        except Exception as exc:
            st.error(f"Could not load image metadata: {exc}")
            meta_df = pd.DataFrame()

        if meta_df.empty:
            st.info("No image metadata entries found.")
        else:
            unresolved = meta_df[meta_df.get("submitter", "").astype(str).str.lower() == "undefined"].copy()
            unresolved = unresolved.reset_index(drop=True)
            if unresolved.empty:
                st.success("No Undefined entries. Everything is assigned.")
            else:
                unresolved["display_label"] = unresolved.apply(
                    lambda r: (
                        f"Grid {r.get('grid_number', '?')} | "
                        f"{r.get('date', r.get('image_date', ''))} | "
                        f"{r.get('image_filename', Path(str(r.get('path', ''))).name)}"
                    ),
                    axis=1,
                )
                st.caption(f"Undefined entries: {len(unresolved)}")
                if "assign_users_idx" in st.session_state:
                    if st.session_state["assign_users_idx"] >= len(unresolved):
                        st.session_state["assign_users_idx"] = max(0, len(unresolved) - 1)
                selected_idx = st.selectbox(
                    "Select image to assign",
                    options=list(range(len(unresolved))),
                    format_func=lambda i: unresolved.iloc[i]["display_label"],
                    key="assign_users_idx",
                )
                selected = unresolved.iloc[int(selected_idx)]

                img_path = _resolve_entry_image_path(selected)

                if img_path and img_path.exists():
                    st.image(str(img_path), caption=img_path.name, width=260)
                else:
                    st.warning("Image file not found on disk for this entry.")

                st.write("Current metadata:")
                st.json(selected.to_dict())

                assignee = st.selectbox(
                    "Assign to submitter",
                    options=sorted(GRID_PLAYERS.keys()),
                    key="assign_users_submitter",
                )
                col_assign, col_delete = st.columns(2)
                if col_assign.button("Save assignment", key="assign_users_save"):
                    # Match by path if available, else by grid/date/image_filename triple.
                    if selected.get("path"):
                        mask = meta_df["path"].astype(str) == str(selected.get("path"))
                    else:
                        mask = (
                            (meta_df.get("grid_number").astype(str) == str(selected.get("grid_number")))
                            & (meta_df.get("image_filename").astype(str) == str(selected.get("image_filename")))
                            & (meta_df.get("date").astype(str) == str(selected.get("date")))
                        )
                    if mask.any():
                        # Always rename underlying image file from Undefined_* to assigned submitter naming.
                        target_row = meta_df.loc[mask].iloc[0]
                        src_img_path = _resolve_entry_image_path(target_row)
                        if src_img_path is not None:
                            grid_val = str(target_row.get("grid_number", "")).strip()
                            grid_norm = _norm_grid_id(grid_val)
                            date_val = str(target_row.get("date") or target_row.get("image_date") or "").strip()
                            if not date_val and grid_norm.isdigit():
                                date_val = str(ImmaculateGridUtils._fixed_date_from_grid_number(int(grid_norm)))
                            base_name = (
                                f"{assignee}_{date_val}_grid_{grid_norm}"
                                if date_val
                                else f"{assignee}_grid_{grid_norm}"
                            )
                            dest_img_path = _unique_path((Path(IMAGES_PATH) / base_name).with_suffix(".jpg"))
                            try:
                                src_img_path.rename(dest_img_path)
                                meta_df.loc[mask, "path"] = str(dest_img_path)
                                meta_df.loc[mask, "image_filename"] = dest_img_path.name
                            except Exception as exc:
                                st.warning(f"Submitter updated but file rename failed: {exc}")
                        meta_df.loc[mask, "submitter"] = assignee
                        meta_df.to_json(IMAGES_METADATA_PATH, orient="records", indent=4)
                        st.cache_data.clear()
                        ok, msg = _rebuild_image_metadata_derivatives()
                        if ok:
                            st.caption(msg)
                        else:
                            st.warning(f"Assigned metadata, but failed to rebuild images_metadata.csv: {msg}")
                        st.success(f"Assigned entry to {assignee}.")
                        st.rerun()
                    else:
                        st.error("Could not find matching metadata row to update.")
                if col_delete.button("Delete selected entry", key="assign_users_delete"):
                    if selected.get("path"):
                        mask = meta_df["path"].astype(str) == str(selected.get("path"))
                    else:
                        mask = (
                            (meta_df.get("grid_number").astype(str) == str(selected.get("grid_number")))
                            & (meta_df.get("image_filename").astype(str) == str(selected.get("image_filename")))
                            & (meta_df.get("date").astype(str) == str(selected.get("date")))
                        )
                    if mask.any():
                        rows_to_delete = meta_df.loc[mask].copy()
                        meta_df = meta_df.loc[~mask].copy()
                        meta_df.to_json(IMAGES_METADATA_PATH, orient="records", indent=4)
                        # Always delete underlying image file(s) when deleting metadata entries.
                        for _, row in rows_to_delete.iterrows():
                            img_path = _resolve_entry_image_path(row)
                            if img_path is not None:
                                img_path.unlink(missing_ok=True)
                        st.cache_data.clear()
                        ok, msg = _rebuild_image_metadata_derivatives()
                        if ok:
                            st.caption(msg)
                        else:
                            st.warning(f"Deleted metadata, but failed to rebuild images_metadata.csv: {msg}")
                        st.success("Deleted selected metadata entry.")
                        st.rerun()
                    else:
                        st.error("Could not find matching metadata row to delete.")

    # --- Manual entry tab ---
    with manual_tab:
        st.write("Add a manual result entry directly to results.csv.")
        results_path = resolve_path(MESSAGES_CSV_PATH)
        results_path.parent.mkdir(parents=True, exist_ok=True)

        if "manual_matrix" not in st.session_state:
            st.session_state["manual_matrix"] = [[False, False, False] for _ in range(3)]
        if "manual_grid_number" not in st.session_state:
            st.session_state["manual_grid_number"] = ImmaculateGridUtils.get_today_grid_id()
        if "manual_date" not in st.session_state:
            auto_date = ImmaculateGridUtils._fixed_date_from_grid_number(st.session_state["manual_grid_number"])
            st.session_state["manual_date"] = datetime.strptime(auto_date, "%Y-%m-%d").date()
        if "manual_date_auto" not in st.session_state:
            st.session_state["manual_date_auto"] = True
        with st.form("manual_entry_form", clear_on_submit=False):
            st.selectbox("Player (name)", options=sorted(GRID_PLAYERS.keys()), key="manual_player")
            st.number_input("Grid number", min_value=0, step=1, key="manual_grid_number")
            st.checkbox("Auto-update date from grid number", key="manual_date_auto")
            st.date_input("Date", key="manual_date", format="MM/DD/YYYY")

            st.markdown("**Matrix**")
            matrix_df = pd.DataFrame(
                st.session_state["manual_matrix"],
                columns=["Left", "Center", "Right"],
                index=["Top", "Middle", "Bottom"],
            )
            edited_df = st.data_editor(
                matrix_df,
                num_rows="fixed",
                use_container_width=False,
                key="manual_matrix_editor",
            )
            st.number_input("Score", min_value=0, step=1, key="manual_score")
            col_submit, col_reset = st.columns(2)
            add_clicked = col_submit.form_submit_button("Add entry to results.csv")
            reset_clicked = col_reset.form_submit_button("Reset matrix")

        if reset_clicked:
            st.session_state["manual_matrix"] = [[False, False, False] for _ in range(3)]
            st.rerun()

        matrix = (
            edited_df.astype(bool).values.tolist()
            if isinstance(edited_df, pd.DataFrame)
            else st.session_state["manual_matrix"]
        )
        st.session_state["manual_matrix"] = matrix
        correct = sum(sum(1 for cell in row if cell) for row in matrix)
        st.write(f"Correct cells: {correct}")

        def _existing_entry_mask(results_df: pd.DataFrame, name: str, grid_number: int):
            if results_df.empty or not {"name", "grid_number"}.issubset(results_df.columns):
                return None
            return (results_df["name"] == name) & (results_df["grid_number"] == grid_number)

        def _write_manual_entry(entry: dict, overwrite: bool) -> None:
            if results_path.exists():
                results_df = pd.read_csv(results_path)
            else:
                results_df = pd.DataFrame()
            if overwrite:
                mask = _existing_entry_mask(results_df, entry["name"], entry["grid_number"])
                if mask is not None and mask.any():
                    results_df = results_df[~mask]
            entry_df = pd.DataFrame([entry])
            results_df = pd.concat([results_df, entry_df], ignore_index=True)
            results_df.to_csv(results_path, index=False)
            st.success(f"Added manual entry for {entry['name']} (grid {entry['grid_number']}).")
            st.cache_data.clear()
            st.rerun()

        pending_entry = st.session_state.get("manual_pending_entry")
        if st.session_state.get("manual_overwrite_pending") and pending_entry:
            st.warning("An entry for this player and grid already exists. Confirm to overwrite it.")
            col_confirm, col_cancel = st.columns(2)
            with col_confirm:
                if st.button("Confirm overwrite", key="manual_confirm_overwrite"):
                    _write_manual_entry(pending_entry, overwrite=True)
                    st.session_state.pop("manual_overwrite_pending", None)
                    st.session_state.pop("manual_pending_entry", None)
            with col_cancel:
                if st.button("Cancel", key="manual_cancel_overwrite"):
                    st.session_state.pop("manual_overwrite_pending", None)
                    st.session_state.pop("manual_pending_entry", None)

        if add_clicked:
            name = st.session_state.get("manual_player")
            grid_number = int(st.session_state.get("manual_grid_number", 0))
            score = int(st.session_state.get("manual_score", 0))
            if st.session_state.get("manual_date_auto"):
                date_str = ImmaculateGridUtils._fixed_date_from_grid_number(grid_number)
            else:
                date_val = st.session_state.get("manual_date")
                date_str = date_val.strftime("%Y-%m-%d") if date_val else ""

            entry = {
                "grid_number": grid_number,
                "correct": correct,
                "score": score,
                "date": date_str,
                "matrix": json.dumps(matrix),
                "name": name,
            }
            if results_path.exists():
                results_df = pd.read_csv(results_path)
            else:
                results_df = pd.DataFrame()
            mask = _existing_entry_mask(results_df, name, grid_number)
            if mask is not None and mask.any():
                st.session_state["manual_pending_entry"] = entry
                st.session_state["manual_overwrite_pending"] = True
                st.rerun()
            else:
                _write_manual_entry(entry, overwrite=False)
