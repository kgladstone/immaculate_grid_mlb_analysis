import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from app.tabs.player_data_tab import _build_usage_df, _parse_responses  # noqa: E402


def test_parse_responses_handles_json_and_literal_dict():
    json_val = '{"top_left":"Ken Griffey Jr.","top_center":""}'
    lit_val = "{'top_left': 'Barry Bonds', 'top_center': ''}"
    assert _parse_responses(json_val)["top_left"] == "Ken Griffey Jr."
    assert _parse_responses(lit_val)["top_left"] == "Barry Bonds"


def test_build_usage_df_extracts_nonempty_player_usages():
    images_df = pd.DataFrame(
        [
            {
                "submitter": "Alice",
                "responses": {"top_left": "Ken Griffey Jr.", "top_center": ""},
            },
            {
                "submitter": "Bob",
                "responses": '{"top_left":"Barry Bonds","top_center":"Willie Mays"}',
            },
        ]
    )
    out = _build_usage_df(images_df)
    assert len(out) == 3
    assert sorted(out["submitter"].unique().tolist()) == ["Alice", "Bob"]
