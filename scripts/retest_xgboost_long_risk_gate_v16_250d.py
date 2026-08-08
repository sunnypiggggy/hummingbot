#!/usr/bin/env python3
"""Run the corrected v16 BTC/ETH long-only search over exactly 250 days."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import pandas as pd

import prepare_xgboost_long_risk_gate_v16 as engine


START_TS = int(pd.Timestamp("2025-11-23T15:00:00Z").timestamp())
END_TS = int(pd.Timestamp("2026-07-31T15:00:00Z").timestamp())

engine.MODEL_VERSION = "xgboost-grid-long-risk-gate-v16-250d-retest"
engine.LOCK_SCHEMA = "xgboost-grid-long-risk-gate-v16-250d-lock-v1"
engine.PREDICTION_SCHEMA = "xgboost-grid-long-risk-gate-v16-250d-prediction-v1"
engine.OUTPUT_DIR = Path("results/backtests/xgboost_grid_long_risk_gate_v16_250d")
engine.SOURCE_DIR = Path("results/backtests/eth_xgboost_long_risk_gate_v15_250d")
engine.V15_DIR = engine.SOURCE_DIR


def configure_period() -> None:
    if END_TS - START_TS != 250 * 86400:
        raise AssertionError("250-day interval is not exact")
    engine.engine.START_TS = START_TS
    engine.engine.END_TS = END_TS
    engine.engine.v7.START_TS = START_TS
    engine.engine.v7.END_TS = END_TS


if __name__ == "__main__":
    mp.freeze_support()
    configure_period()
    raise SystemExit(engine.main())
