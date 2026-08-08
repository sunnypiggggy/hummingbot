import json
from pathlib import Path

import joblib
import pandas as pd


OUT = Path("results/backtests/xgboost_regime_spike_pair_risk_gate_v9")


def test_v9_feature_contract_is_channel_specific():
    schema = pd.read_csv(OUT / "feature_schema.csv")
    long_features = set(schema[schema.target.str.startswith("long_")].feature)
    short_features = set(schema[schema.target.eq("short_1h_6h")].feature)
    assert long_features == {"adx_14", "di_spread", "atr_pct", "btc_volatility_20"}
    assert short_features == {"price_to_ema20_atr", "volume_zscore", "di_spread"}


def test_v9_search_counts_and_independent_pair_models():
    assert len(pd.read_csv(OUT / "model_screen_40x2pairsx3targetsx8.csv")) == 1920
    assert len(pd.read_csv(OUT / "single_pair_channel_refined_search.csv")) == 1920
    assert len(pd.read_csv(OUT / "pair_independent_long_short_search.csv")) == 200
    assert len(pd.read_csv(OUT / "btc_eth_independent_portfolio_search.csv")) == 100
    lock = json.loads((OUT / "locked_configuration.json").read_text(encoding="utf-8"))
    bundle = joblib.load(lock["model_path"])
    assert set(bundle["pairs"]) == {"BTC-FDUSD", "ETH-FDUSD"}
    assert bundle["pairs"]["BTC-FDUSD"]["channels"]["long"]["model"] is not bundle["pairs"]["ETH-FDUSD"]["channels"]["long"]["model"]


def test_v9_rejected_signal_fails_closed_without_sell_action():
    payload = json.loads((OUT / "grid_xgboost_risk_gate_v3_sample.json").read_text(encoding="utf-8"))
    assert payload["deployment_allowed"] is False
    assert payload["mechanism1_runtime_fallback"] is False
    assert payload["market_sell_action"] is False
    assert all(not value["buy_enabled"] for value in payload["pairs"].values())
