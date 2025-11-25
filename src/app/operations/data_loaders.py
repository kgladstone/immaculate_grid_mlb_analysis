from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.constants import (
    IMAGES_METADATA_PATH,
    MESSAGES_CSV_PATH,
    PROMPTS_CSV_PATH,
)


def resolve_path(path_like) -> Path:
    return Path(path_like).expanduser()


@st.cache_data(show_spinner=False)
def load_prompts_df() -> pd.DataFrame:
    path = resolve_path(PROMPTS_CSV_PATH)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_texts_df() -> pd.DataFrame:
    path = resolve_path(MESSAGES_CSV_PATH)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_image_metadata_df() -> pd.DataFrame:
    path = resolve_path(IMAGES_METADATA_PATH)
    if not path.exists():
        return pd.DataFrame()
    try:
        with path.open() as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return pd.DataFrame()

    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        return pd.DataFrame([data])
    return pd.DataFrame()
