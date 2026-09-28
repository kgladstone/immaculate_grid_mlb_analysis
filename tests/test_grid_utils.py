import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.grid_utils import ImmaculateGridUtils  # noqa: E402


def test_grid_number_from_text_valid():
    text = "Immaculate Grid 123 8/9\nRarity: 145"
    assert ImmaculateGridUtils._grid_number_from_text(text) == 123


def test_grid_number_from_text_invalid_returns_none():
    assert ImmaculateGridUtils._grid_number_from_text("not a grid message") is None


def test_matrix_from_text_ignores_truncated_emoji_fragments():
    text = (
        "⚾️ Immaculate Grid 123 8/9:\n"
        "Rarity: 145\n"
        "🟩🟩🟩\n"
        "⬜️🟩🟩\n"
        "🟩⬜️🟩\n"
        '🟩…”'
    )

    assert ImmaculateGridUtils._matrix_from_text(text) == (
        "[[true, true, true], [false, true, true], [true, false, true]]"
    )


def test_matrix_from_text_rejects_incomplete_grid():
    text = (
        "⚾️ Immaculate Grid 123 8/9:\n"
        "Rarity: 145\n"
        "🟩🟩🟩\n"
        "⬜️🟩🟩\n"
        '🟩…”'
    )

    with pytest.raises(ValueError, match="found 2"):
        ImmaculateGridUtils._matrix_from_text(text)


def test_is_valid_message_rejects_incomplete_grid():
    text = (
        "⚾️ Immaculate Grid 123 8/9:\n"
        "Rarity: 145\n"
        "🟩🟩🟩\n"
        "⬜️🟩🟩\n"
        '🟩…”'
    )

    assert ImmaculateGridUtils._is_valid_message("Known Player", text) is False
