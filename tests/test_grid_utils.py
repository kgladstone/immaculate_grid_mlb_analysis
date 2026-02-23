import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.grid_utils import ImmaculateGridUtils  # noqa: E402


def test_grid_number_from_text_valid():
    text = "Immaculate Grid 123 8/9\nRarity: 145"
    assert ImmaculateGridUtils._grid_number_from_text(text) == 123


def test_grid_number_from_text_invalid_returns_none():
    assert ImmaculateGridUtils._grid_number_from_text("not a grid message") is None
