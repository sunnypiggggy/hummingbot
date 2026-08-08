import json
from pathlib import Path

import joblib
import pandas as pd


OUT = Path("results/backtests/lightgbm_regime_spike_pair_risk_gate_v10")


def test_lightgbm_v10_search_and_feature_contract():
    schema = pd.read_csv(OUT / "feature_schema.csv")
    assert set(schema[schema.target.str.startswith("long_")].feature) == {
        "adx_14", "di_spread", "atr_pct", "btc_volatility_20",
    }
    assert set(schema[schema.target.eq("short_1h_6h")].feature) == {
        "price_to_ema20_atr", "volume_zscore", "di_spread",
    }
    assert len(pd.read_csv(OUT / "lightgbm_40_parameters.csv")) == 40
    assert len(pd.read_csv(OUT / "model_screen_40x2pairsx3targetsx8.csv")) == 1920
    assert len(pd.read_csv(OUT / "single_pair_channel_refined_search.csv")) == 1920
    assert len(pd.read_csv(OUT / "pair_independent_long_short_search.csv")) == 200
    assert len(pd.read_csv(OUT / "btc_eth_independent_portfolio_search.csv")) == 100


def test_lightgbm_v10_models_are_pair_independent_and_serialized():
    lock = json.loads((OUT / "locked_configuration.json").read_text(encoding="utf-8"))
    bundle = joblib.load(lock["model_path"])
    assert bundle["model_version"] == "lightgbm-regime-spike-pair-risk-gate-v10"
    assert set(bundle["pairs"]) == {"BTC-FDUSD", "ETH-FDUSD"}
    assert bundle["pairs"]["BTC-FDUSD"]["channels"]["long"]["model"] is not bundle["pairs"]["ETH-FDUSD"]["channels"]["long"]["model"]
    check = json.loads((OUT / "model_serialization_check.json").read_text(encoding="utf-8"))
    assert check["passed"] is True


def test_lightgbm_v10_rejected_contract_is_buy_only_fail_closed():
    signal = json.loads((OUT / "grid_xgboost_risk_gate_v3_sample.json").read_text(encoding="utf-8"))
    assert signal["deployment_allowed"] is False
    assert signal["market_sell_action"] is False
    assert signal["mechanism1_runtime_fallback"] is False
    assert all(not row["buy_enabled"] for row in signal["pairs"].values())
