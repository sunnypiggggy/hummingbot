#!/usr/bin/env python3
"""Compatibility entrypoint for the v12 deep long-risk study.

The shared candle loader returns ``(candles, quality)`` while the v12 research
main consumes only the candle mapping.  Keep this adaptation outside the
hashed trainer source so completed walk-forward predictions remain valid.
"""

from __future__ import annotations

import optimize_deep_learning_long_risk_gate_v12 as study


_load_candles = study.load_candles


def _candles_only(cache_dir):
    candles, _quality = _load_candles(cache_dir)
    return candles


study.load_candles = _candles_only


if __name__ == "__main__":
    raise SystemExit(study.main())
