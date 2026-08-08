#!/usr/bin/env python3
"""Build LightGBM v10 research manifest, notebook, and disabled Grid signal."""

from pathlib import Path

import build_xgboost_regime_spike_pair_risk_gate_v9_artifacts as builder


builder.OUT = Path("results/backtests/lightgbm_regime_spike_pair_risk_gate_v10")
builder.ARTIFACT_TITLE = "LightGBM v10 独立长期趋势/短期插针 Risk-off BUY门"
builder.PLOT_FILENAME = "lightgbm_v10_regime_spike_pair_riskoff_plotly.html"
builder.NOTEBOOK_FILENAME = "lightgbm_regime_spike_pair_risk_gate_v10_executed.ipynb"
builder.NOTEBOOK_TITLE = "LightGBM v10：长期趋势与1h插针Risk-off"
builder.MODEL_LABEL = "LightGBM v10"


if __name__ == "__main__":
    builder.main()
