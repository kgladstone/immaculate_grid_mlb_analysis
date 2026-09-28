import sys
import os

os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from analytics import analysis


@pytest.mark.parametrize('uses,expected', [([2], 0), ([1, 1], 1), ([2, 1], 1), ([2, 2], 2), ([3], 1)])
def test_frank_thomas_requires_same_grid_evidence(monkeypatch, tmp_path, uses, expected):
    monkeypatch.setattr(analysis, 'RULE5_FULL_BANS_CSV_PATH', tmp_path / 'no_bans.csv')
    # 1240 is Monday; all grids stay in the same reporting week.
    rows = [{'submitter': 'Keith', 'grid_number': 1240 + offset,
             'responses': {str(i): 'Frank Thomas' for i in range(count)}}
            for offset, count in enumerate(uses)]
    result = analysis.analyze_shame_index(pd.DataFrame(rows))
    if expected == 0:
        assert result.empty
    else:
        assert result.shame_index.sum() == expected
        assert result.total_uses.sum() == sum(uses)


def test_exception_does_not_hide_other_repeats_or_bans(monkeypatch, tmp_path):
    bans = tmp_path / 'bans.csv'
    pd.DataFrame([{'player': 'Frank Thomas', 'grid_number': 1230}]).to_csv(bans, index=False)
    monkeypatch.setattr(analysis, 'RULE5_FULL_BANS_CSV_PATH', bans)
    data = pd.DataFrame([{'submitter': 'Keith', 'grid_number': 1245,
        'responses': {'a': 'Frank Thomas', 'b': 'Frank Thomas', 'c': 'Tom Seaver', 'd': 'Tom Seaver'}}])
    row = analysis.analyze_shame_index(data).iloc[0]
    assert row.shame_index == 2
    assert row.rule5_banned_count == 1
    assert 'Frank Thomas' not in row.repeated_players
    assert 'Tom Seaver' in row.repeated_players
    assert row.total_uses == 4
