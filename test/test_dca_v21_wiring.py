import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "live_guard"))

from dca_live_guard import Guard  # noqa: E402
from backtest_dca_momentum_guard import gate_for_frame  # noqa: E402


def test_dca_v22_reuses_grid_guard_without_a_new_container() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    dockerfile = (ROOT / "Dockerfile.dca-live-guard").read_text(encoding="utf-8")
    macro_compose = (ROOT / "ops/dca-macro/docker-compose.yml").read_text(encoding="utf-8")
    assert "  grid-xgboost-v21-shadow:" not in compose
    assert "  grid-xgboost-v22:" not in compose
    assert "./grid-live-fdusd-data:/workspace/technical:ro" in compose
    assert "DCA_V22_GATE_PATH: /workspace/technical/xgboost_risk_gate.json" in compose
    assert "./dca-macro-data:/workspace/macro:ro" in compose
    assert "COPY scripts/ethbtc_forced_exit_contract.py" in dockerfile
    assert 'DCA_MACRO_EXECUTION_ENABLED: "false"' in macro_compose


def test_roc_sqz_runtime_configuration_is_removed() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    source = (ROOT / "live_guard/dca_live_guard.py").read_text(encoding="utf-8")
    for legacy in (
        "DCA_ROC_BUY_GUARD_ENABLED",
        "DCA_ROC_BUY_GUARD_REFRESH_SECONDS",
        "DCA_ROC_BUY_GUARD_TRIGGER_PCT",
        "DCA_SQZ_BUY_GUARD_TRIGGER_PCT",
        "_roc_sqz_signal_from_klines",
        "_apply_roc_buy_guard",
    ):
        assert legacy not in compose
        assert legacy not in source


def test_legacy_roc_state_is_retired_without_becoming_v22_state(tmp_path) -> None:
    state_path = tmp_path / "guard_state.json"
    state_path.write_text(json.dumps({
        "version": 1, "bots": {},
        "roc_buy_guard": {"active": True, "controlled_bots": ["btc"]},
    }), encoding="utf-8")
    guard = Guard.__new__(Guard)
    guard.state_path = state_path
    guard.mechanisms = {"v22_weekly_buy_gate": True}
    state = guard._load_state()
    assert state["version"] == 2
    assert state["roc_buy_guard"] == {
        "retired": True,
        "retired_reason": "replaced_by_ethbtc_forced_exit_v22",
        "previous_active": True,
    }
    assert "active" not in state["roc_buy_guard"]


def test_dca_backtest_maps_fdusd_v21_state_by_base_asset() -> None:
    frame = pd.DataFrame({"timestamp": [1500, 2500], "close": [100, 101]})
    states = pd.DataFrame({
        "pair": ["BTC-FDUSD", "BTC-FDUSD"],
        "signal_ts": [1000, 2000],
        "recommended_buy_enabled": [False, True],
    })
    gate = gate_for_frame(
        frame, pd.DataFrame(), "v21", pair="BTC-USDT", v21_states=states,
    )
    assert gate.tolist() == [False, True]
