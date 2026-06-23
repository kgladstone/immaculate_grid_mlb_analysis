from __future__ import annotations

from functools import lru_cache
from html import escape
from pathlib import Path
import re

import pandas as pd

from data.io.mlb_reference import MANUAL_ALIASES, clean_name

MANUAL_BBREF_IDS = {
    "A J Pierzynski": "pierza.01",
    "A.J. Pierzynski": "pierza.01",
    "AJ Pierzynski": "pierza.01",
    "Bobby Witt Jr.": "wittbo02",
    "Cal Ripken Jr.": "ripkeca01",
    "Fernando Tatis": "tatisfe01",
    "Fernando Tatis Jr.": "tatisfe02",
    "Ken Griffey Jr.": "griffke02",
    "Ronald Acuna Jr.": "acunaro01",
    "Ronald Acuña Jr.": "acunaro01",
    "Vlad Guerrero": "guerrvl01",
    "Vlad Guerrero Jr.": "guerrvl02",
    "Vladimir Guerrero": "guerrvl01",
    "Vladimir Guerrero Jr.": "guerrvl02",
}

PLAYER_LINK_STYLE_LIGHT_BG = "color:#0b5cad;font-weight:700;text-decoration:underline;"
PLAYER_LINK_STYLE_DARK_BG = "color:#ffd166;font-weight:700;text-decoration:underline;"


def _normalize_name(value: object) -> str:
    text = clean_name(str(value or "")).casefold().replace(".", "")
    return re.sub(r"\s+", " ", text).strip()


def baseball_reference_url_from_id(player_id: object) -> str:
    pid = str(player_id or "").strip()
    if not pid:
        return ""
    return f"https://www.baseball-reference.com/players/{pid[0].lower()}/{pid}.shtml"


@lru_cache(maxsize=1)
def player_url_lookup(cache_dir: str = "bin/baseball_cache") -> dict[str, str]:
    people_path = Path(cache_dir) / "People.csv"
    if not people_path.exists():
        return {}
    try:
        people = pd.read_csv(people_path, dtype=str)
    except Exception:
        return {}

    id_col = next((c for c in people.columns if c.lower() in {"bbrefid", "playerid", "key_bbref"}), None)
    first_col = next((c for c in people.columns if c.lower() in {"namefirst", "name_first"}), None)
    last_col = next((c for c in people.columns if c.lower() in {"namelast", "name_last"}), None)
    if not id_col or not first_col or not last_col:
        return {}

    lookup: dict[str, str] = {}
    slim = people[[id_col, first_col, last_col]].dropna(subset=[id_col, first_col, last_col]).copy()
    for _, row in slim.iterrows():
        player_id = str(row.get(id_col, "")).strip()
        first = str(row.get(first_col, "")).strip()
        last = str(row.get(last_col, "")).strip()
        full_name = clean_name(f"{first} {last}")
        url = baseball_reference_url_from_id(player_id)
        if not full_name or not url:
            continue
        lookup.setdefault(_normalize_name(full_name), url)

    for alias, canonical in MANUAL_ALIASES.items():
        url = lookup.get(_normalize_name(canonical))
        if url:
            lookup.setdefault(_normalize_name(alias), url)
    for alias, player_id in MANUAL_BBREF_IDS.items():
        url = baseball_reference_url_from_id(player_id)
        if url:
            lookup[_normalize_name(alias)] = url
    return lookup


def player_url(name: object) -> str:
    return player_url_lookup().get(_normalize_name(name), "")


def player_link_markdown(name: object) -> str:
    label = str(name or "").strip()
    url = player_url(label)
    return f"[{label}]({url})" if label and url else label


def player_link_html(name: object, *, on_dark: bool = False) -> str:
    label = str(name or "").strip()
    url = player_url(label)
    safe_label = escape(label)
    if not safe_label or not url:
        return safe_label
    safe_url = escape(url, quote=True)
    style = PLAYER_LINK_STYLE_DARK_BG if on_dark else PLAYER_LINK_STYLE_LIGHT_BG
    return (
        f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer" '
        f'style="{style}">{safe_label}</a>'
    )


def player_link_columns(df: pd.DataFrame) -> list:
    return [
        col
        for col in df.columns
        if str(col).strip().casefold() in {"player", "mlb_player", "player_name"}
    ]


def player_link_html_table(df: pd.DataFrame, max_height_px: int = 420, *, dark: bool = False) -> str:
    columns = list(df.columns)
    link_columns = set(player_link_columns(df))
    header_bg = "#1f2933" if dark else "#f4f6fa"
    header_color = "#f8fafc" if dark else "#111"
    body_bg = "#0e1117" if dark else "#fff"
    body_color = "#f8fafc" if dark else "#111"
    border_color = "#334155" if dark else "#ddd"
    row_border_color = "#334155" if dark else "#eee"
    header = "".join(
        f"<th style='position:sticky;top:0;background:{header_bg};color:{header_color};"
        f"padding:6px;border-bottom:1px solid {border_color};text-align:left;z-index:1;'>"
        f"{escape(str(col))}</th>"
        for col in columns
    )

    rows = []
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            value = row.get(col, "")
            if pd.isna(value):
                value = ""
            if col in link_columns:
                rendered = player_link_html(value, on_dark=dark)
            else:
                rendered = escape(str(value))
            cells.append(
                f"<td style='padding:6px;border-bottom:1px solid {row_border_color};"
                f"vertical-align:top;color:{body_color};background:{body_bg};'>"
                f"{rendered}</td>"
            )
        rows.append(f"<tr>{''.join(cells)}</tr>")

    return (
        f"<div style='max-height:{int(max_height_px)}px;overflow:auto;"
        f"border:1px solid {border_color};border-radius:6px;'>"
        "<table style='border-collapse:collapse;width:100%;font-size:13px;'>"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></div>"
    )
