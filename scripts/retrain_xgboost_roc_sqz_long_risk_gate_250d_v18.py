#!/usr/bin/env python3
"""ROC/SQZMOM-only 250-day XGBoost long-risk retraining adapter."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import retrain_xgboost_long_risk_gate_250d_v17 as engine


engine.MODEL_VERSION = "xgboost-roc-sqz-independent-long-risk-gate-v18-250d"
engine.OUTPUT_DIR = Path("results/backtests/xgboost_roc_sqz_long_risk_gate_v18_250d")
engine.LOCK_SCHEMA = "xgboost-roc-sqz-independent-long-risk-gate-v18-lock-v1"
engine.FEATURE_SETS = {
    "roc_sqz_core": (
        "roc_5", "roc_20", "roc_48h_4h", "sqzmom_pct_4h",
        "sqzmom_slope_4h", "sqzmom_improving_4h",
    ),
    "roc_sqz_threshold_structure": (
        "roc_20", "roc_48h_4h", "sqzmom_value", "sqzmom_slope",
        "sqzmom_pct_4h", "sqzmom_slope_4h", "roc_to_entry_4h",
        "sqz_to_entry_4h", "roc_to_recovery_4h", "sqz_to_recovery_4h",
    ),
}
engine.WEIGHT_PROFILES = ("balanced", "persistent_severity")

_base_plot = engine.plot


def build_v18_plot(args):
    source = _base_plot(args)
    target = args.output_dir / "xgboost_v18_roc_sqz_250d_riskoff_plotly.html"
    page = source.read_text(encoding="utf-8").replace(
        "XGBoost v17：BTC/ETH独立长期Risk-off（250天）",
        "XGBoost v18：ROC/SQZ长期Risk-off（250天）",
    )
    target.write_text(page, encoding="utf-8")
    return target


engine.plot = build_v18_plot


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(engine.main())
