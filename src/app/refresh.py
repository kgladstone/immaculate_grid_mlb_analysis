from __future__ import annotations

import contextlib
import io
import os
import shutil
import traceback
from pathlib import Path
from typing import Dict, Iterable, Tuple

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
from .data_loaders import resolve_path


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


def refresh_selected_data(selected: Iterable[str], texts_db_path: Path | str | None = None) -> Tuple[Dict[str, int], Iterable[str], Dict[str, dict]]:
    results: Dict[str, int] = {}
    errors = []
    diagnostics: Dict[str, dict] = {}
    prompts_df = None

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
            db_path = Path(texts_db_path).expanduser() if texts_db_path else resolve_path(APPLE_TEXTS_DB_PATH)
            exists = db_path.exists()
            readable = os.access(db_path, os.R_OK)
            if not exists:
                raise FileNotFoundError(f"Messages DB not found at {db_path}")
            if not readable:
                raise PermissionError(
                    f"Messages DB not readable at {db_path}. "
                    "Grant Full Disk Access to the terminal/IDE running Streamlit or point to a readable copy (e.g., workspace chat.db)."
                )
            messages_loader = MessagesLoader(str(db_path), str(MESSAGES_CSV_PATH))
            messages_loader.load().validate()
            results["Texts"] = len(messages_loader.get_data())
            diagnostics["Texts"] = messages_loader.diagnostics
        except Exception as exc:  # pragma: no cover - surfacing to UI
            diag = f"Path: {db_path} | exists={exists} | readable={readable}"
            errors.append(f"Texts refresh failed: {exc}\n{traceback.format_exc()}\n{diag}")

    if "Images" in selected:
        try:
            image_processor = ImageProcessor(APPLE_TEXTS_DB_PATH, str(IMAGES_METADATA_PATH), str(IMAGES_PATH))
            image_processor.process_images()
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
            errors.append(f"Images refresh failed: {exc}\n{traceback.format_exc()}")

    return results, errors, diagnostics


def render_refresh_tab() -> None:
    st.subheader("Refresh Data")
    st.write(
        "Select a dataset to refresh. Images will also rebuild the derived CSV and fuzzy matching log when metadata is available."
    )
    choices = ["Prompts", "Texts", "Images"]
    selected = st.radio("Dataset to refresh", options=choices, index=0)
    texts_db_input = None
    if selected == "Texts":
        snapshot_db = Path.cwd() / "chat_snapshot" / "chat.db"
        workspace_db = Path.cwd() / "chat.db"
        if snapshot_db.exists():
            default_path = str(snapshot_db)
        elif workspace_db.exists():
            default_path = str(workspace_db)
        else:
            default_path = str(resolve_path(APPLE_TEXTS_DB_PATH))
        texts_db_input = st.text_input(
            "Messages DB path",
            value=default_path,
            help="Absolute path to chat.db; must be readable (Full Disk Access on macOS).",
        )

    if st.button("Refresh selected datasets"):
        buf = io.StringIO()
        results = {}
        errors = []
        diags = {}

        with st.spinner("Refreshing..."):
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
        if diags.get("Texts"):
            diag = diags["Texts"]
            st.subheader("Texts diagnostics (last 90 days)")
            st.write(f"Cutoff: {diag.get('cutoff')}")
            if diag.get("counts_recent"):
                st.markdown("**Counts by date and sender**")
                import pandas as pd
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
        with st.expander("Console output", expanded=False):
            st.text_area("stdout", value=buf.getvalue(), height=260)
