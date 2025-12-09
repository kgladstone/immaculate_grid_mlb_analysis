import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from app.tabs import simulator_tab  # noqa: E402


def test_load_pybaseball_data_recovers_from_bad_zip(tmp_path):
    calls = []

    def team_loader():
        if not calls:
            calls.append("fail")
            raise ValueError("File is not a zip file")
        calls.append("ok")
        return pd.DataFrame({"yearID": [2023], "teamID": ["AAA"], "franchID": ["AAA"]})

    def player_loader():
        return pd.DataFrame({"name_last": ["Doe"], "name_first": ["John"], "key_bbref": ["doejo01"]})

    def app_loader():
        return pd.DataFrame({"playerID": ["doejo01"], "yearID": [2023], "teamID": ["AAA"]})

    cleared = []

    def clear_fn(paths):
        cleared.append(tuple(paths))

    team_master, player_master, appearances, exc = simulator_tab.load_pybaseball_data(
        clear_before=False,
        loaders=(team_loader, player_loader, app_loader),
        clear_fn=clear_fn,
    )

    assert exc is None
    assert calls == ["fail", "ok"], "Should retry after clearing corrupted cache"
    assert cleared, "Cache clear should be invoked on failure"
    assert not team_master.empty
    assert not player_master.empty
    assert not appearances.empty
