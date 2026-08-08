#!/usr/bin/env python3
"""Focused v21 refinement: high-quantile arming plus adaptive strong recovery."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import refine_xgboost_long_risk_gate_v20 as engine


def focused_candidates() -> list[engine.Gate]:
    result = [engine.Gate(q, entry, 48, 48, cooldown, 3, confirmation, recovery)
              for q in (.98, .985, .99)
              for entry in (1, 2)
              for cooldown in (24, 48)
              for confirmation in ("directional_relaxed", "persistent_bearish")
              for recovery in ("structural_relief", "adaptive_relief")]
    if len(result) != 48 or len({tuple(asdict(item).values()) for item in result}) != 48:
        raise AssertionError("v21 must contain 48 deterministic focused gates")
    return result


engine.MODEL_VERSION = "xgboost-grid-long-risk-gate-v21-250d"
engine.OUTPUT_DIR = Path("results/backtests/xgboost_grid_long_risk_gate_v21_250d")
engine.candidates = focused_candidates


if __name__ == "__main__":
    raise SystemExit(engine.main())
