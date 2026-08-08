#!/usr/bin/env python3
"""Search independent long-regime and short-spike XGBoost Grid BUY gates."""

from pathlib import Path

import optimize_xgboost_roc_sqz_pair_risk_gate_v8 as engine


LONG_FEATURES = (
    "adx_14",
    "di_spread",
    "atr_pct",
    "btc_volatility_20",
)
SHORT_FEATURES = (
    "price_to_ema20_atr",
    "volume_zscore",
    "di_spread",
)

engine.MODEL_VERSION = "xgboost-regime-spike-pair-risk-gate-v9"
engine.OUTPUT_DIR = Path("results/backtests/xgboost_regime_spike_pair_risk_gate_v9")
engine.ROC_SQZ_FEATURES = tuple(dict.fromkeys((*LONG_FEATURES, *SHORT_FEATURES)))
engine.FEATURES_BY_TARGET = {
    "long_72h": LONG_FEATURES,
    "long_120h": LONG_FEATURES,
    "short_1h_6h": SHORT_FEATURES,
}
engine.MODEL_ARTIFACT_FILENAME = "xgboost_regime_spike_pair_risk_gate_v9.joblib"
engine.MODEL_SCHEMA = "xgboost-regime-spike-pair-risk-gate-v9-model-v1"
engine.LOCK_SCHEMA = "xgboost-regime-spike-pair-risk-gate-v9-lock-v1"
engine.SUMMARY_SCHEMA = "xgboost-regime-spike-pair-risk-gate-v9-summary-v1"
engine.PREDICTION_CACHE_SCHEMA = "xgboost-regime-spike-pair-v9-prediction-cache-v1"
engine.STRATEGY_LABEL = "XGBoost v9 independent regime/spike BUY gate"
engine.PLOT_FILENAME = "xgboost_v9_regime_spike_pair_riskoff_plotly.html"
engine.PLOT_TITLE = "XGBoost v9：BTC/ETH独立长期趋势与1h插针Risk-off驱动Grid"
engine.FEATURE_NOTE = (
    "长期特征=ADX/DI spread/ATR%/BTC volatility；"
    "短期特征=EMA距离/成交量Z-score/DI spread"
)
engine.LONG_CHANNEL_LABEL = "长期趋势风险"
engine.SHORT_CHANNEL_LABEL = "1h快速下跌"


if __name__ == "__main__":
    raise SystemExit(engine.main())
