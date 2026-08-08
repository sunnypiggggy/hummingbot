from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from backtest_xgboost_long_risk_gate_180d import (  # noqa: E402
    Candidate,
    DAY,
    END_TS,
    START_TS,
    _future_min_return,
    diagnostic_windows,
    frozen_grid_sequence,
)


def test_diagnostic_windows_cover_exactly_180_days_without_gaps():
    windows = diagnostic_windows()
    assert len(windows) == 26
    assert int(windows.test_start.iloc[0]) == START_TS
    assert int(windows.test_end.iloc[-1]) == END_TS
    assert int((windows.test_end - windows.test_start).sum()) == 180 * DAY
    assert (windows.test_start.iloc[1:].to_numpy() == windows.test_end.iloc[:-1].to_numpy()).all()
    assert (windows.train_end == windows.test_start).all()
    assert (windows.train_start == windows.train_end - 14 * DAY).all()


def test_future_min_return_excludes_current_bar_and_requires_full_horizon():
    frame = pd.DataFrame({
        "close": [100.0, 99.0, 98.0, 97.0],
        "low": [1.0, 95.0, 90.0, 96.0],
    })
    result = _future_min_return(frame, 2)
    assert result.iloc[0] == pytest.approx(-0.10)  # current low=1 is excluded
    assert result.iloc[1] == pytest.approx(90.0 / 99.0 - 1.0)
    assert pd.isna(result.iloc[2])
    assert pd.isna(result.iloc[3])


def test_frozen_grid_sequence_never_uses_a_future_approval(tmp_path):
    source = tmp_path / "weekly.csv"
    pd.DataFrame([
        {"scenario": "new", "test_start": START_TS + 7 * DAY,
         "half_range": 0.04, "min_spread": 0.008, "take_profit": 0.008,
         "move_threshold": 0.02, "move_cooldown_seconds": 1800},
    ]).to_csv(source, index=False)
    windows = diagnostic_windows().head(3)
    selected, audit = frozen_grid_sequence(windows, source)
    assert selected.iloc[0].half_range == Candidate(0.03, 0.006, 0.006, 0.015, 1800).half_range
    assert selected.iloc[1].half_range == 0.04
    assert audit.approval_not_after_fold_start.all()
