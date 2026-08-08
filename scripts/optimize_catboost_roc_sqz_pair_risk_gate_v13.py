#!/usr/bin/env python3
"""CatBoost retry of the BTC/ETH-independent ROC/SQZMOM Grid BUY gate.

The shared v8 research engine supplies the labels, walk-forward splits, state
machines and Grid accounting.  This module changes only the estimator and the
artifact contract.  CatBoost never emits a sell action.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from catboost import CatBoostClassifier
from sklearn.model_selection import ParameterSampler

import optimize_xgboost_roc_sqz_pair_risk_gate_v8 as engine
from tune_xgboost_momentum_stop_v2 import balanced_weights


SEED = 42


def canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))


def catboost_configurations() -> list[dict[str, Any]]:
    """Return two anchors plus 38 deterministic, unique sampled configs."""
    legacy_anchor = {
        "learning_rate": 0.04, "iterations": 500, "depth": 5,
        "l2_leaf_reg": 3.0, "random_strength": 1.0,
        "bagging_temperature": 1.0, "rsm": 0.85, "border_count": 128,
    }
    regularized_anchor = {
        "learning_rate": 0.03, "iterations": 800, "depth": 4,
        "l2_leaf_reg": 10.0, "random_strength": 2.0,
        "bagging_temperature": 2.0, "rsm": 0.80, "border_count": 254,
    }
    profiles = [(0.015, 1200), (0.025, 800), (0.04, 500), (0.06, 350), (0.08, 250)]
    space = {
        "learning_profile": list(range(len(profiles))),
        "depth": [3, 4, 5, 6, 7, 8],
        "l2_leaf_reg": [1.0, 3.0, 6.0, 10.0, 20.0],
        "random_strength": [0.0, 0.5, 1.0, 2.0, 5.0],
        "bagging_temperature": [0.0, 0.5, 1.0, 2.0, 5.0],
        "rsm": [0.65, 0.8, 0.9, 1.0],
        "border_count": [64, 128, 254],
    }
    items = [("legacy_anchor", legacy_anchor), ("regularized_anchor", regularized_anchor)]
    seen = {canonical(legacy_anchor), canonical(regularized_anchor)}
    for sampled in ParameterSampler(space, n_iter=64, random_state=SEED):
        sampled = dict(sampled)
        learning_rate, iterations = profiles[int(sampled.pop("learning_profile"))]
        params = {"learning_rate": learning_rate, "iterations": iterations, **sampled}
        key = canonical(params)
        if key in seen:
            continue
        seen.add(key)
        items.append(("sampled", params))
        if len(items) == 40:
            break
    if len(items) != 40:
        raise AssertionError(f"expected 40 CatBoost configurations, got {len(items)}")
    return [{
        "config_id": f"cat_{index:02d}", "order": index, "kind": kind,
        "uses_early_stopping": kind != "legacy_anchor", **params,
    } for index, (kind, params) in enumerate(items)]


def model_params(config: Mapping[str, Any], iterations: int | None = None) -> dict[str, Any]:
    return {
        "loss_function": "Logloss", "eval_metric": "Logloss",
        "iterations": int(iterations or config["iterations"]),
        "learning_rate": float(config["learning_rate"]),
        "depth": int(config["depth"]),
        "l2_leaf_reg": float(config["l2_leaf_reg"]),
        "random_strength": float(config["random_strength"]),
        "bootstrap_type": "Bayesian",
        "bagging_temperature": float(config["bagging_temperature"]),
        "rsm": float(config["rsm"]),
        "border_count": int(config["border_count"]),
        "grow_policy": "SymmetricTree", "boosting_type": "Plain",
        "random_seed": SEED, "thread_count": 2, "task_type": "CPU",
        "allow_writing_files": False, "verbose": False,
    }


def fit_catboost(
    config: Mapping[str, Any], features: list[str], train_part,
    core_part, validation_part,
) -> tuple[CatBoostClassifier, dict[str, Any]]:
    cap = int(config["iterations"])
    best_iterations, best_score = cap, math.nan
    if bool(config["uses_early_stopping"]):
        early = CatBoostClassifier(**model_params(config))
        early.fit(
            core_part[features], core_part.target.astype(int),
            sample_weight=balanced_weights(core_part),
            eval_set=(validation_part[features], validation_part.target.astype(int)),
            early_stopping_rounds=50, verbose=False,
        )
        best_iterations = max(1, min(cap, int(early.get_best_iteration()) + 1))
        best_score = float(early.get_best_score().get("validation", {}).get("Logloss", math.nan))
    final = CatBoostClassifier(**model_params(config, best_iterations))
    final.fit(
        train_part[features], train_part.target.astype(int),
        sample_weight=balanced_weights(train_part), verbose=False,
    )
    return final, {
        "tree_cap": cap, "best_tree_count": best_iterations,
        "early_stopping_used": bool(config["uses_early_stopping"]),
        "early_stopping_rounds": 50 if config["uses_early_stopping"] else 0,
        "best_validation_logloss": best_score,
        "task_type": "CPU", "model_threads": 2,
    }


engine.MODEL_VERSION = "catboost-roc-sqz-pair-risk-gate-v13"
engine.CONFIGURATION_FAMILY = "CatBoost"
engine.OUTPUT_DIR = Path("results/backtests/catboost_roc_sqz_pair_risk_gate_v13")
engine.MODEL_ARTIFACT_FILENAME = "catboost_roc_sqz_pair_risk_gate_v13.joblib"
engine.MODEL_SCHEMA = "catboost-roc-sqz-pair-risk-gate-v13-model-v1"
engine.LOCK_SCHEMA = "catboost-roc-sqz-pair-risk-gate-v13-lock-v1"
engine.SUMMARY_SCHEMA = "catboost-roc-sqz-pair-risk-gate-v13-summary-v1"
engine.PREDICTION_CACHE_SCHEMA = "catboost-roc-sqz-pair-v13-prediction-cache-v1"
engine.STRATEGY_LABEL = "CatBoost v13 ROC/SQZ pair-independent gate"
engine.PLOT_FILENAME = "catboost_v13_roc_sqz_pair_riskoff_plotly.html"
engine.PLOT_TITLE = "CatBoost v13：BTC/ETH独立ROC/SQZMOM Risk-off驱动Grid"
engine.FEATURE_NOTE = "特征仅限ROC/SQZMOM及其多周期、斜率、改善与阈值距离派生量"
engine.FEATURE_LIMITATION = "CatBoost inputs are intentionally restricted to ROC and SQZMOM derivatives."
engine.LONG_CHANNEL_LABEL = "长期ROC/SQZ风险"
engine.SHORT_CHANNEL_LABEL = "1h快速下跌"
engine.PARAMETERS_FILENAME = "catboost_40_parameters.csv"
engine.IMPORTANCE_FILENAME = "catboost_feature_importance.csv"
engine.CONFIGURATION_PROVIDER = lambda target, pair: catboost_configurations()
engine.fit_one_group = fit_catboost


if __name__ == "__main__":
    raise SystemExit(engine.main())
