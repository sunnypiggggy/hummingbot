#!/usr/bin/env python3
"""Freeze the v21 BTC/ETH long-risk models into a non-authorizing shadow bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import retrain_xgboost_long_risk_gate_250d_v19 as research
import tune_xgboost_momentum_stop_v2 as tune
from xgboost_long_risk_gate_v21 import (
    FEATURES, GATES, MODEL_BUNDLE_SCHEMA, MODEL_VERSION, PAIRS,
    feature_schema_sha256, strategy_schema_sha256, strategy_spec,
)


ROOT = Path("results/backtests/xgboost_grid_long_risk_gate_v21_250d")
PANEL = Path("results/backtests/xgboost_grid_long_risk_gate_v19_250d/feature_panel.csv.gz")
TRAINING_CUTOFF = research.END_TS
SELECTIONS = {
    "BTC-FDUSD": {"target": "long_event_72h", "config_id": "xgb_34"},
    "ETH-FDUSD": {"target": "long_event_72h", "config_id": "xgb_16"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "shadow_package")
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--candidate-lock", type=Path, default=ROOT / "locked_configuration.json")
    return parser.parse_args()


def build_pair(panel: pd.DataFrame, pair: str) -> tuple[Any, dict[str, Any], float, pd.DataFrame]:
    selection = SELECTIONS[pair]
    config = next(item for item in tune.xgb_configurations() if item["config_id"] == selection["config_id"])
    frame = research.target_frame(panel, selection["target"], pair)
    model, calibration, audit = research.fit_leakage_safe(frame, TRAINING_CUTOFF, config, FEATURES[pair])
    threshold = float(calibration.probability.quantile(GATES[pair].entry_quantile))
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise RuntimeError(f"invalid production threshold for {pair}")
    audit.update({
        "pair": pair, "target": selection["target"], "config_id": selection["config_id"],
        "entry_quantile": GATES[pair].entry_quantile, "entry_threshold": threshold,
        "features": list(FEATURES[pair]), "training_cutoff_ts": TRAINING_CUTOFF,
        "label_maturity_hours": 96, "calibration_excluded_from_final_fit": True,
    })
    return model, config, threshold, pd.DataFrame([audit])


def main() -> int:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(args.panel)
    candidate = json.loads(args.candidate_lock.read_text(encoding="utf-8"))
    if candidate.get("model_version") != MODEL_VERSION or candidate.get("verdict") != "DIAGNOSTIC_ONLY":
        raise RuntimeError("v21 candidate lock is not the frozen diagnostic selection")
    models, configs, thresholds, tree_counts, audits = {}, {}, {}, {}, []
    for pair in PAIRS:
        model, config, threshold, audit = build_pair(panel, pair)
        models[pair] = model; configs[pair] = config; thresholds[pair] = threshold
        tree_counts[pair] = int(audit.iloc[0]["best_tree_count"]); audits.append(audit)
    bundle = {
        "schema": MODEL_BUNDLE_SCHEMA, "model_version": MODEL_VERSION,
        "strategy_spec": strategy_spec(),
        "strategy_schema_sha256": strategy_schema_sha256(),
        "pairs": {
            pair: {"model": models[pair], "features": list(FEATURES[pair]),
                   "config": configs[pair], "entry_threshold": thresholds[pair],
                   "best_tree_count": tree_counts[pair], "gate": GATES[pair].__dict__,
                   "target": SELECTIONS[pair]["target"]}
            for pair in PAIRS
        },
        "training_cutoff_ts": TRAINING_CUTOFF,
        "feature_schema_sha256": feature_schema_sha256(),
    }
    model_path = args.output_dir / "models" / "xgboost_long_risk_gate_v21.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(bundle, model_path)
    loaded = joblib.load(model_path); maximum_error = 0.0
    sample = panel[panel.signal_ts < TRAINING_CUTOFF].groupby("pair", group_keys=False).tail(64)
    for pair in PAIRS:
        rows = sample[sample.pair.eq(pair)]
        before = models[pair].predict_proba(rows[list(FEATURES[pair])])[:, 1]
        after = loaded["pairs"][pair]["model"].predict_proba(rows[list(FEATURES[pair])])[:, 1]
        maximum_error = max(maximum_error, float(np.max(np.abs(before - after))))
    if maximum_error > 1e-12:
        raise RuntimeError(f"model serialization drift: {maximum_error}")
    candidate_hash = research.sha256_file(args.candidate_lock)
    panel_hash = research.sha256_file(args.panel)
    lock = {
        "schema": "xgboost-grid-long-risk-gate-v21-shadow-lock-v1",
        "model_version": MODEL_VERSION, "package_status": "BUILT_UNVERIFIED",
        "historical_verdict": "NO-GO", "shadow_mode": True,
        "deployment_allowed": False, "promotion_authorized": False,
        "short_spike_enabled": False, "market_sell_action": False,
        "mechanism1_fallback_allowed": False, "forward_shadow_weeks_required": 8,
        "candidate_lock_path": args.candidate_lock.as_posix(), "candidate_lock_sha256": candidate_hash,
        "model_path": model_path.as_posix(), "model_sha256": research.sha256_file(model_path),
        "feature_schema_sha256": feature_schema_sha256(),
        "strategy_schema_sha256": strategy_schema_sha256(),
        "training_panel_sha256": panel_hash, "training_cutoff_ts": TRAINING_CUTOFF,
        "pairs": {
            pair: {"target": SELECTIONS[pair]["target"], "config_id": SELECTIONS[pair]["config_id"],
                   "features": list(FEATURES[pair]), "entry_threshold": thresholds[pair],
                   "best_tree_count": tree_counts[pair],
                   "gate": GATES[pair].__dict__}
            for pair in PAIRS
        },
        "serialization_check": {"maximum_probability_absolute_error": maximum_error, "passed": True},
    }
    research.atomic_json(args.output_dir / "shadow_lock.json", lock)
    pd.concat(audits, ignore_index=True).to_csv(args.output_dir / "final_training_audit.csv", index=False)
    research.atomic_json(args.output_dir / "model_inventory.json", {
        "model_version": MODEL_VERSION, "python": "3.12", "xgboost": "3.3.0", "cpu_only": True,
        "bundle_schema": MODEL_BUNDLE_SCHEMA,
        "strategy_schema_sha256": strategy_schema_sha256(),
        "models": {pair: {"config_id": SELECTIONS[pair]["config_id"],
                           "feature_count": len(FEATURES[pair]),
                           "best_tree_count": tree_counts[pair]}
                   for pair in PAIRS},
    })
    (args.output_dir / "SHADOW_RUNBOOK.md").write_text(
        "# v21 long-only Risk-off shadow runbook\n\n"
        "This package is advisory only and must never be copied to a Hummingbot instance or the v16 filename. "
        "Run `python scripts/validate_xgboost_v21_shadow_package.py`, then after Docker is available run "
        "`powershell -File scripts/run_xgboost_v21_shadow_container_validation.ps1`. Start only with "
        "`docker compose --profile risk-shadow-v21 up -d grid-xgboost-v21-shadow`. The eight-week clock starts "
        "at the first complete Monday 00:00 UTC after explicit launch. SHADOW_READY is not deployment authorization.\n",
        encoding="utf-8")
    (args.output_dir / "STOP_AND_CLEANUP.md").write_text(
        "# Stop and cleanup\n\nStop only `grid-xgboost-v21-shadow` and preserve its isolated state, heartbeat "
        "and daily metrics. Never alter `grid-live-fdusd-data`.\n",
        encoding="utf-8")
    (args.output_dir / "ROLLBACK.md").write_text(
        "# Rollback\n\nThere is no trading-path rollback because v21 is not connected. Stop the isolated "
        "producer, preserve evidence, never copy its file to the v16 filename, never use Mechanism 1 fallback, "
        "and never emit a cancel/sell/reduce action.\n",
        encoding="utf-8")
    (args.output_dir / "DAILY_MONITOR_TEMPLATE.md").write_text(
        "# Daily v21 shadow check\n\n- UTC date:\n- Heartbeat uptime / stale events:\n- BTC recommended Risk-off transitions:\n- ETH recommended Risk-off transitions:\n- Duplicate event IDs:\n- Offline parity mismatches:\n- Counterfactual Grid PnL / drawdown / stops:\n- Operator notes:\n",
        encoding="utf-8")
    print(json.dumps(lock, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
