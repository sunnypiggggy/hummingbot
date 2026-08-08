from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import optimize_eth_xgboost_long_risk_gate_v15 as v15


def test_search_contract_is_eth_long_only_and_deterministic() -> None:
    assert v15.PAIR == "ETH-FDUSD"
    assert v15.TARGETS == ("long_72h", "long_120h")
    assert len(v15.tune.xgb_configurations()) == 40
    gates = v15.gate_candidates()
    assert len(gates) == 128
    assert len({v15.engine.gate_id("long", gate) for gate in gates}) == 128


def test_lock_prediction_hash_is_enforced(tmp_path: Path) -> None:
    target = tmp_path / "prediction.csv.gz"
    pd.DataFrame({"probability": [0.1]}).to_csv(target, index=False, compression="gzip")
    first = v15.sha256_file(target)
    pd.DataFrame({"probability": [0.2]}).to_csv(target, index=False, compression="gzip")
    assert v15.sha256_file(target) != first


def test_lock_schema_disallows_deployment() -> None:
    assert "lock-v1" in v15.LOCK_SCHEMA
    assert v15.MODEL_VERSION.endswith("v15")
