#!/usr/bin/env python3
"""LightGBM retry of the independent long-regime/short-spike Grid BUY gate."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import ParameterSampler

import optimize_xgboost_roc_sqz_pair_risk_gate_v8 as engine
from tune_xgboost_momentum_stop_v2 import balanced_weights


SEED = 42
LONG_FEATURES = ("adx_14", "di_spread", "atr_pct", "btc_volatility_20")
SHORT_FEATURES = ("price_to_ema20_atr", "volume_zscore", "di_spread")


def canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))


def lightgbm_configurations() -> list[dict[str, Any]]:
    legacy = {
        "learning_rate": 0.04, "n_estimators": 240, "num_leaves": 31,
        "max_depth": -1, "min_child_samples": 60, "min_split_gain": 0.0,
        "subsample": 0.85, "colsample_bytree": 0.85,
        "reg_alpha": 0.0, "reg_lambda": 1.0, "max_bin": 255,
    }
    anchor = {
        "learning_rate": 0.03, "n_estimators": 800, "num_leaves": 15,
        "max_depth": 5, "min_child_samples": 100, "min_split_gain": 0.10,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.5, "reg_lambda": 8.0, "max_bin": 255,
    }
    profiles = [(0.015, 1200), (0.025, 800), (0.04, 500), (0.06, 350), (0.08, 250)]
    space = {
        "learning_profile": list(range(5)), "num_leaves": [7, 15, 31, 63, 127],
        "max_depth": [-1, 3, 5, 7, 9], "min_child_samples": [20, 40, 60, 100, 160],
        "min_split_gain": [0.0, 0.02, 0.05, 0.10, 0.25],
        "subsample": [0.65, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.65, 0.8, 0.9, 1.0],
        "reg_alpha": [0.0, 0.1, 0.5, 2.0, 5.0],
        "reg_lambda": [1.0, 3.0, 8.0, 20.0], "max_bin": [127, 255, 511],
    }
    items = [("legacy", legacy), ("regularized_anchor", anchor)]
    seen = {canonical(legacy), canonical(anchor)}
    for sampled in ParameterSampler(space, n_iter=64, random_state=SEED):
        sampled = dict(sampled)
        lr, trees = profiles[int(sampled.pop("learning_profile"))]
        params = {"learning_rate": lr, "n_estimators": trees, **sampled}
        key = canonical(params)
        if key in seen:
            continue
        seen.add(key); items.append(("sampled", params))
        if len(items) == 40:
            break
    if len(items) != 40:
        raise AssertionError(f"expected 40 LightGBM configurations, got {len(items)}")
    return [{
        "config_id": f"lgb_{index:02d}", "order": index, "kind": kind,
        "uses_early_stopping": kind != "legacy", **params,
    } for index, (kind, params) in enumerate(items)]


def model_params(config: Mapping[str, Any], trees: int | None = None) -> dict[str, Any]:
    return {
        "objective": "binary", "n_estimators": int(trees or config["n_estimators"]),
        "learning_rate": float(config["learning_rate"]), "num_leaves": int(config["num_leaves"]),
        "max_depth": int(config["max_depth"]), "min_child_samples": int(config["min_child_samples"]),
        "min_split_gain": float(config["min_split_gain"]), "subsample": float(config["subsample"]),
        "subsample_freq": 1, "colsample_bytree": float(config["colsample_bytree"]),
        "reg_alpha": float(config["reg_alpha"]), "reg_lambda": float(config["reg_lambda"]),
        "max_bin": int(config["max_bin"]), "random_state": SEED, "n_jobs": 4,
        "verbosity": -1, "deterministic": True, "force_col_wise": True,
        "importance_type": "gain",
    }


def fit_lightgbm(
    config: Mapping[str, Any], features: list[str], train_part,
    core_part, validation_part,
) -> tuple[LGBMClassifier, dict[str, Any]]:
    cap, best_trees, best_score = int(config["n_estimators"]), int(config["n_estimators"]), math.nan
    if bool(config["uses_early_stopping"]):
        early = LGBMClassifier(**model_params(config))
        early.fit(
            core_part[features], core_part.target.astype(int),
            sample_weight=balanced_weights(core_part),
            eval_set=[(validation_part[features], validation_part.target.astype(int))],
            eval_sample_weight=[balanced_weights(validation_part)],
            eval_metric="binary_logloss",
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        best_trees = max(1, min(cap, int(early.best_iteration_ or cap)))
        scores = early.best_score_.get("valid_0", {})
        best_score = float(scores.get("binary_logloss", math.nan))
    final = LGBMClassifier(**model_params(config, best_trees))
    final.fit(
        train_part[features], train_part.target.astype(int),
        sample_weight=balanced_weights(train_part), callbacks=[log_evaluation(0)],
    )
    return final, {
        "tree_cap": cap, "best_tree_count": best_trees,
        "early_stopping_used": bool(config["uses_early_stopping"]),
        "early_stopping_rounds": 50 if config["uses_early_stopping"] else 0,
        "best_validation_logloss": best_score,
    }


engine.MODEL_VERSION = "lightgbm-regime-spike-pair-risk-gate-v10"
engine.OUTPUT_DIR = Path("results/backtests/lightgbm_regime_spike_pair_risk_gate_v10")
engine.ROC_SQZ_FEATURES = tuple(dict.fromkeys((*LONG_FEATURES, *SHORT_FEATURES)))
engine.FEATURES_BY_TARGET = {
    "long_72h": LONG_FEATURES, "long_120h": LONG_FEATURES,
    "short_1h_6h": SHORT_FEATURES,
}
engine.MODEL_ARTIFACT_FILENAME = "lightgbm_regime_spike_pair_risk_gate_v10.joblib"
engine.MODEL_SCHEMA = "lightgbm-regime-spike-pair-risk-gate-v10-model-v1"
engine.LOCK_SCHEMA = "lightgbm-regime-spike-pair-risk-gate-v10-lock-v1"
engine.SUMMARY_SCHEMA = "lightgbm-regime-spike-pair-risk-gate-v10-summary-v1"
engine.PREDICTION_CACHE_SCHEMA = "lightgbm-regime-spike-pair-v10-prediction-cache-v1"
engine.STRATEGY_LABEL = "LightGBM v10 independent regime/spike BUY gate"
engine.PLOT_FILENAME = "lightgbm_v10_regime_spike_pair_riskoff_plotly.html"
engine.PLOT_TITLE = "LightGBM v10：BTC/ETH独立长期趋势与1h插针Risk-off驱动Grid"
engine.FEATURE_NOTE = "长期=ADX/DI spread/ATR%/BTC volatility；短期=EMA距离/成交量Z-score/DI spread"
engine.FEATURE_LIMITATION = "Long and short channels use separate compact feature contracts selected from prior model importance."
engine.LONG_CHANNEL_LABEL = "长期趋势风险"
engine.SHORT_CHANNEL_LABEL = "1h快速下跌"
engine.PARAMETERS_FILENAME = "lightgbm_40_parameters.csv"
engine.IMPORTANCE_FILENAME = "lightgbm_gain_feature_importance.csv"
engine.xgb_configurations = lightgbm_configurations
engine.fit_one_group = fit_lightgbm


if __name__ == "__main__":
    raise SystemExit(engine.main())
