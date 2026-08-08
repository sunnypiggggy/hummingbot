from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import revalidate_eth_xgboost_long_risk_gate_v15_250d as study


def test_period_is_exactly_250_days() -> None:
    assert study.END_TS - study.START_TS == 250 * 86400
    folds = study.windows()
    assert len(folds) == 36
    assert int((folds.test_end - folds.test_start).sum()) == 250 * 86400


def test_locked_model_jobs_do_not_search_250d() -> None:
    class Args:
        locked_v15 = study.LOCKED_V15
    jobs = study.model_jobs(Args())
    assert len(jobs) == 4
    eth_long = [job for job in jobs if job["pair"] == "ETH-FDUSD" and job["target"].startswith("long")]
    assert len(eth_long) == 1
    assert eth_long[0]["config_id"] == "xgb_35"
