import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.io.messages_loader import MessagesLoader


def test_will_correction_replaces_cached_submission_and_survives_refresh(tmp_path):
    original = dict(name="Will", grid_number=1274, correct=9, score=42,
                    date="2026-09-27", matrix="[[true,true,true],[true,true,true],[true,true,true]]")
    corrected = dict(original, correct=8, score=140, matrix="[[true,false,true],[true,true,true],[true,true,true]]")
    other = dict(original, name="Sam")
    cache = tmp_path / "messages.csv"
    pd.DataFrame([original, other]).to_csv(cache, index=False)
    loader = MessagesLoader("unused", cache)
    loader.fetch_function = lambda _: pd.DataFrame([original, other, dict(corrected, correct=9)])
    for _ in range(2):
        result = loader.load().get_data()
        assert result[result.name.eq("Will")].to_dict("records") == [corrected]
        assert result[result.name.eq("Sam")].to_dict("records") == [other]
        assert len(pd.read_csv(cache)) == 2

    # A partial snapshot must not undo an already accepted correction.
    loader.fetch_function = lambda _: pd.DataFrame([original])
    result = loader.load().get_data()
    assert result[result.name.eq("Will")].to_dict("records") == [corrected]


def test_missing_correction_does_not_invent_score_or_matrix(tmp_path):
    original = dict(name="Will", grid_number=1274, correct=9, score=42,
                    date="2026-09-27", matrix="[[true,true,true],[true,true,true],[true,true,true]]")
    loader = MessagesLoader("unused", tmp_path / "messages.csv")
    loader.fetch_function = lambda _: pd.DataFrame([original])
    assert loader.load().get_data().to_dict("records") == [original]
