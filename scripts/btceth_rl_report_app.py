"""Build the audited BTC/ETH RL report through the installed Data runtime.

No training, inference, PNL simulation or publication occurs here. Complete
172-run evidence and a matching final independent audit are mandatory before
any reviewed snapshot or HTML is written. Native and WSL modes are explicit.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.crypto_rl_reporting import clean_json

ALGORITHMS = ("recurrent_ppo", "tqc", "crossq", "sac")
BASELINES = ("cash", "btc_hold", "eth_hold", "equal_hold", "equal_rebalance", "sma200_trend", "momentum180")
SEEDS, PROFILES, SCENARIOS = tuple(range(5)), ("return", "balanced"), ("base", "stress")
SPLITS = {"validation": ("2025-01-01T00:00:00Z", "2025-10-23T00:00:00Z"),
          "test": ("2025-10-23T00:00:00Z", "2026-10-23T00:00:00Z")}
NAMES = {"recurrent_ppo": "RecurrentPPO（LSTM）", "tqc": "TQC", "crossq": "CrossQ", "sac": "SAC同环境对照",
         "cash": "USDT现金", "btc_hold": "BTC被动持有", "eth_hold": "ETH被动持有",
         "equal_hold": "BTC/ETH等权被动持有", "equal_rebalance": "BTC/ETH等权再平衡",
         "sma200_trend": "SMA200趋势对照", "momentum180": "180日动量对照"}
QUERY_IDS = ("strategy_summary", "equity_history", "monthly_returns", "seed_statistics",
             "validation_selection", "training_records", "research_targets", "audit_evidence")
NODE_DEFAULT = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe"
PLUGIN_DEFAULT = Path.home() / ".codex/plugins/cache/openai-curated-remote/data-analytics/1.1.1-exp-components.2/output_formats/html"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def safe_relative(root, name):
    name = str(name).replace("\\", "/")
    if name.startswith("/") or ":" in name or ".." in name.split("/"):
        raise ValueError("Evidence names must be root-relative")
    result = (Path(root).resolve() / name).resolve()
    result.relative_to(Path(root).resolve())
    return result


def audit_source_name(root, audit_path):
    """Keep an explicitly selected audit's identity; never redirect input roots."""
    root, audit_path = Path(root).resolve(), Path(audit_path).resolve()
    try:
        return audit_path.relative_to(root).as_posix()
    except ValueError:
        # Only the explicitly supplied audit may be external. Its input_sha256
        # entries are still resolved strictly against the original study root.
        return audit_path.as_posix()


def frame_rows(frame):
    return clean_json(frame.astype(object).where(pd.notna(frame), None).to_dict("records"))


def expected_runs(selection):
    result = {}
    for split in SPLITS:
        for scenario in SCENARIOS:
            for algorithm in ALGORITHMS:
                profiles = PROFILES if split == "validation" else (selection["selected_profiles"][algorithm],)
                for profile in profiles:
                    for key in (*map(str, SEEDS), "ensemble"):
                        result[f"{split}__{scenario}__{algorithm}__{profile}__{key}"] = (
                            split, scenario, algorithm, profile, None if key == "ensemble" else int(key), key == "ensemble")
            for baseline in BASELINES:
                result[f"{split}__{scenario}__{baseline}"] = (split, scenario, "baseline", "baseline", None, False)
    return result


def _verify_summary(summary, selection):
    expected = expected_runs(selection)
    if len(summary) != 172 or summary.run_id.duplicated().any() or set(summary.run_id) != set(expected):
        raise ValueError("Final report requires exactly 110 validation and 62 test runs")
    for row in summary.itertuples(index=False):
        split, scenario, algorithm, profile, seed, primary = expected[row.run_id]
        if (row.split, row.scenario, row.algorithm, row.profile) != (split, scenario, algorithm, profile):
            raise ValueError("Financial run identity mismatch: " + row.run_id)
        if bool(row.primary_policy) != primary or (not pd.isna(row.seed) if seed is None else row.seed != seed):
            raise ValueError("Seed or actual ensemble marker mismatch: " + row.run_id)
        if row.initial_capital != 10000.:
            raise ValueError("Every comparison requires one actual 10000 USDT ledger")
        if pd.Timestamp(row.start) != pd.Timestamp(SPLITS[split][0]) or pd.Timestamp(row.end_exclusive) != pd.Timestamp(SPLITS[split][1]):
            raise ValueError("Evaluation window mismatch")
        if row.drawdown_grain != "1h_close" or row.observed_hours != (pd.Timestamp(SPLITS[split][1]) - pd.Timestamp(SPLITS[split][0])).days * 24:
            raise ValueError("Complete hourly accounting is mandatory")
        if not np.isfinite([row.net_return, row.max_drawdown, row.fees, row.average_exposure]).all():
            raise ValueError("Missing primary financial metrics cannot become zero")
    return expected


def _verify_history(history, months, summary):
    if not {"run_id", "timestamp", "equity", "net_return", "drawdown"}.issubset(history.columns):
        raise ValueError("Daily curves require equity, net return and drawdown fields")
    expected_ids = set(summary.run_id)
    if set(history.run_id) != expected_ids or set(months.run_id) != expected_ids:
        raise ValueError("Curve/monthly runs do not match summary")
    for row in summary.itertuples(index=False):
        curve = history.loc[history.run_id.eq(row.run_id)]
        stamps = pd.DatetimeIndex(pd.to_datetime(curve.timestamp, utc=True))
        expected = pd.date_range(row.start, row.end_exclusive, freq="D")
        if not stamps.equals(expected) or curve.timestamp.duplicated().any():
            raise ValueError("Daily curves require the initial point and every daily boundary")
        if not np.isfinite(curve[["equity", "net_return", "drawdown"]].to_numpy(dtype=float)).all():
            raise ValueError("Daily curve values must be finite")
        if not np.isclose(float(curve.equity.iloc[0]), row.initial_capital, atol=1e-7, rtol=1e-12):
            raise ValueError("Curve initial equity differs from comparison capital")
        if not np.allclose(curve.net_return.to_numpy(), curve.equity.to_numpy() / row.initial_capital - 1, atol=1e-11, rtol=1e-11):
            raise ValueError("Plotted curve returns differ from ledger equity")
        if not np.isclose(float(curve.equity.iloc[-1]), row.final_equity, atol=1e-7, rtol=1e-12):
            raise ValueError("Curve final equity differs from summary")
        monthly = months.loc[months.run_id.eq(row.run_id)]
        needed = pd.date_range(row.start, row.end_exclusive, freq="D", inclusive="left").strftime("%Y-%m").unique().tolist()
        if monthly.month.tolist() != needed or monthly.month.duplicated().any():
            raise ValueError("Monthly window coverage mismatch")
        if not np.isclose(float(np.prod(1 + monthly["return"]) - 1), row.net_return, atol=1e-10, rtol=1e-10):
            raise ValueError("Monthly returns do not reconcile to net return")


def _research_targets(summary, selection, audit):
    leader = selection["primary_validation_leader"]
    rows = summary.loc[summary.split.eq("test") & summary.algorithm.eq(leader["algorithm"]) & summary.profile.eq(leader["profile"])]
    base = rows.loc[rows.scenario.eq("base") & rows.primary_policy.eq(True)].iloc[0]
    stress = rows.loc[rows.scenario.eq("stress") & rows.primary_policy.eq(True)].iloc[0]
    equal = summary.loc[summary.run_id.eq("test__base__equal_hold")].iloc[0]
    positive = int((rows.loc[rows.scenario.eq("base") & rows.seed.notna(), "net_return"] > 0).sum())
    gates = {
        "base_return_at_least_5pct": base.net_return >= .05,
        "base_max_drawdown_at_most_20pct": abs(base.max_drawdown) <= .20,
        "stress_return_nonnegative": stress.net_return >= 0.,
        "beats_equal_hold_return_and_drawdown": base.net_return > equal.net_return and base.max_drawdown > equal.max_drawdown,
        "at_least_three_positive_training_seeds": positive >= 3,
    }
    supplied = audit.get("research_targets", {})
    if supplied.get("primary_validation_leader") != leader or supplied.get("gates") != gates:
        raise ValueError("Research performance checks differ from independent audit")
    if supplied.get("all_predeclared_primary_targets_met") is not bool(all(gates.values())):
        raise ValueError("Aggregate research target flag contradicts the individual gates")
    for name, expected in (("base_net_return", base.net_return), ("base_max_drawdown", base.max_drawdown),
                           ("stress_net_return", stress.net_return)):
        actual = supplied.get(name)
        if not isinstance(actual, (int, float)) or not np.isfinite(actual) or not np.isclose(actual, expected, atol=1e-12, rtol=1e-12):
            raise ValueError("Research performance metric differs from independent audit: " + name)
    descriptions = {
        "base_return_at_least_5pct": ("主成本净收益至少5%", f"{base.net_return:.6%}", "5%"),
        "base_max_drawdown_at_most_20pct": ("小时最大回撤不超过20%", f"{base.max_drawdown:.6%}", "回撤幅度≤20%"),
        "stress_return_nonnegative": ("压力成本收益非负", f"{stress.net_return:.6%}", "≥0%"),
        "beats_equal_hold_return_and_drawdown": ("收益与回撤均优于等权被动持有", f"{equal.net_return:.6%} / {equal.max_drawdown:.6%}", "与同口径等权持有逐项比较"),
        "at_least_three_positive_training_seeds": ("至少三个训练种子测试收益为正", str(positive) + "/5", "≥3/5；不选择最好种子"),
    }
    result = [{"gate": name, "label": descriptions[name][0], "met": bool(value), "observed": descriptions[name][1],
               "criterion": descriptions[name][2], "category": "事前研究条件"} for name, value in gates.items()]
    controls = {}
    for name in ("sma200_trend", "momentum180"):
        other = summary.loc[summary.run_id.eq("test__base__" + name)].iloc[0]
        met = bool(base.net_return > other.net_return and base.max_drawdown > other.max_drawdown)
        controls[name] = met
        result.append({"gate": name, "label": "收益及回撤均优于" + NAMES[name], "met": met,
                       "observed": f"{other.net_return:.6%} / {other.max_drawdown:.6%}",
                       "criterion": "双方净收益更高且回撤幅度更小", "category": "简单规则对照"})
    if supplied.get("dominates_simple_trend_controls") != controls:
        raise ValueError("Simple trend comparison differs from independent audit")
    if supplied.get("stronger_claim_supported") is not bool(all(gates.values()) and all(controls.values())):
        raise ValueError("Stronger performance claim contradicts the gates and trend controls")
    return result


def _training_rows(root, spec, selection):
    rows, hashes = [], selection["training_source_sha256"]
    for name, expected in hashes.items():
        if sha(safe_relative(root, name)) != expected:
            raise ValueError("Training evidence differs from validation-time freeze: " + name)
    for algorithm in ALGORITHMS:
        for profile in PROFILES:
            for seed in SEEDS:
                relative = f"training/{algorithm}__{profile}__seed{seed}/record.json"
                record = read_json(root / relative)
                if (record["algorithm"], record["profile"], record["seed"]) != (algorithm, profile, seed):
                    raise ValueError("Training identity mismatch")
                if record["actual_timesteps"] != spec["timesteps_per_model"]:
                    raise ValueError("Training budget differs from protocol")
                quality, device = record["training_window_quality"], record["device"]
                rows.append({"job_id": f"{algorithm}__{profile}__seed{seed}", "algorithm": algorithm, "profile": profile,
                             "seed": seed, "timesteps": record["actual_timesteps"], "elapsed_seconds": record["elapsed_seconds"],
                             "model_sha256": record["model_sha256"], "parameter_sha256": record["parameter_sha256"],
                             "gpu_uuid": device.get("visible_gpu_uuid"), "gpu": device.get("name"),
                             "requested_train_days": quality.get("requested_train_days"), "episode_days": quality.get("episode_days"),
                             "covered_training_days": quality.get("covered_training_days"), "eligible_windows": quality.get("eligible_windows"),
                             "excluded_windows": quality.get("excluded_windows"), "source_record": relative})
    return rows


def make_reviewed_snapshot(study_root, audit_path=None, *, extension_root=None, extension_audit_path=None,
                           risk_reference_root=None, dataset_path=None, band_annual_root=None,
                           band_validation_root=None, band_audit_path=None, td3_root=None, td3_audit_path=None,
                           droq_root=None, droq_audit_path=None):
    if droq_audit_path is not None and droq_root is None:
        raise ValueError("An explicit DroQ study root is required for its separate audit")
    if td3_audit_path is not None and td3_root is None:
        raise ValueError("An explicit TD3 study root is required for its audit")
    if extension_audit_path is not None and extension_root is None:
        raise ValueError("An explicit extension root is required for its audit")
    if risk_reference_root is not None and dataset_path is None:
        raise ValueError("Risk-reference report requires an explicit frozen dataset path")
    if band_annual_root is None and (band_validation_root is not None or band_audit_path is not None):
        raise ValueError("A complete annual execution-band study is required for its report inputs")
    if band_annual_root is not None and (band_validation_root is None or extension_root is None):
        raise ValueError("Execution-band report requires explicit validation and original cap roots")
    root = Path(study_root).resolve()
    audit_path = Path(audit_path).resolve() if audit_path else root / "independent_audit.json"
    required = ("summary.csv", "daily_equity.csv", "monthly_returns.csv", "frozen_spec.json", "validation_selection.json",
                "training_completion.json", "validation_completion.json", "test_completion.json", "test_started.json")
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError("No final report before complete numerical evidence: " + name)
    audit = read_json(audit_path)
    if (audit.get("passed") is not True or audit.get("status") != "passed" or audit.get("test_complete") is not True
            or audit.get("failure_count") != 0 or audit.get("failures") != [] or audit.get("run_count", audit.get("runs_checked")) != 172):
        raise ValueError("A final passed 172-run independent audit is mandatory; validation-only audits cannot be used")
    inputs = audit.get("input_sha256")
    if not isinstance(inputs, dict) or not set(required).issubset(inputs):
        raise ValueError("Audit must bind every root input to exact bytes")
    for name, expected in inputs.items():
        if sha(safe_relative(root, name)) != expected:
            raise ValueError("Audited evidence changed: " + name)
    audit_source = audit_source_name(root, audit_path)
    direct_sources = {name: root / name for name in required}
    direct_sources[audit_source] = audit_path
    retained_audit = root / "independent_audit.json"
    retained_audit_source = None
    if retained_audit.is_file() and retained_audit.resolve() != audit_path:
        retained_audit_source = "independent_audit.json"
        direct_sources[retained_audit_source] = retained_audit
    if (root / "transfer_receipt.json").is_file():
        direct_sources["transfer_receipt.json"] = root / "transfer_receipt.json"
    initial_hashes = {name: sha(path) for name, path in direct_sources.items()}
    audit_identity = {"primary_audit_file": audit_source, "primary_audit_sha256": initial_hashes[audit_source],
                      "primary_audit_external_to_study": not audit_path.is_relative_to(root),
                      "retained_original_audit_file": retained_audit_source,
                      "retained_original_audit_sha256": initial_hashes.get(retained_audit_source),
                      "retained_original_role": "Preserved original audit/transfer evidence; not the audit used for this report" if retained_audit_source else None}
    spec, selection = read_json(root / "frozen_spec.json"), read_json(root / "validation_selection.json")
    spec_hash = sha(root / "frozen_spec.json")
    if audit.get("frozen_spec_sha256") != spec_hash or selection.get("frozen_spec_sha256") != spec_hash:
        raise ValueError("Protocol and selection do not match final audit")
    if spec["algorithms"] != list(ALGORITHMS) or spec["baseline_names"] != list(BASELINES) or spec["seeds"] != list(SEEDS):
        raise ValueError("Fixed BTC/ETH four-model protocol required; do not mix old Top10 studies")
    if set(selection["selected_profiles"]) != set(ALGORITHMS) or any(value not in PROFILES for value in selection["selected_profiles"].values()):
        raise ValueError("One validation-selected profile per algorithm required")
    if spec["capital"] != 10000. or spec["train_episode_days"] != 32:
        raise ValueError("Expected one 10000 USDT ledger and frozen 32-day training fragments")
    for split, boundaries in SPLITS.items():
        if [pd.Timestamp(x) for x in spec[split]] != [pd.Timestamp(x) for x in boundaries]:
            raise ValueError("Report window differs from fixed protocol")
    csv = lambda name: pd.read_csv(root / name, float_precision="round_trip")
    summary, history, months = csv("summary.csv"), csv("daily_equity.csv"), csv("monthly_returns.csv")
    expected = _verify_summary(summary, selection)
    if set(audit.get("runs", {})) != set(expected):
        raise ValueError("Audit run set differs from report run set")
    _verify_history(history, months, summary)
    labels, run_lookup = {}, {}
    for row in summary.itertuples(index=False):
        baseline = row.algorithm == "baseline"
        label = NAMES[row.strategy] if baseline else NAMES[row.algorithm] + ("五模型主组合" if row.primary_policy else f" seed {int(row.seed)}")
        labels[row.run_id] = label
        run_lookup[row.run_id] = {"algorithm": row.algorithm, "profile": row.profile, "primary_policy": bool(row.primary_policy),
                                  "policy_role": "baseline" if baseline else "primary" if row.primary_policy else "training_seed",
                                  "selected_profile": baseline or row.profile == selection["selected_profiles"][row.algorithm]}
    for frame in (summary, history, months):
        frame["label"] = frame.run_id.map(labels)
        for field in ("algorithm", "profile", "primary_policy", "policy_role", "selected_profile"):
            frame[field] = frame.run_id.map(lambda value: run_lookup[value][field])
    statistics = []
    for (split, scenario, algorithm, profile), frame in summary.loc[summary.seed.notna()].groupby(["split", "scenario", "algorithm", "profile"], sort=True):
        if len(frame) != 5 or set(frame.seed.astype(int)) != set(SEEDS):
            raise ValueError("Exactly five actual training seeds per displayed range required")
        statistics.append({"split": split, "scenario": scenario, "algorithm": algorithm, "profile": profile, "seeds": 5,
                           "best_return": float(frame.net_return.max()), "worst_return": float(frame.net_return.min()),
                           "median_return": float(frame.net_return.median()), "worst_drawdown": float(frame.max_drawdown.min()),
                           "best_drawdown": float(frame.max_drawdown.max()), "positive_seeds": int((frame.net_return > 0).sum()),
                           "selected_profile": profile == selection["selected_profiles"][algorithm]})
    candidates = []
    for item in selection["candidates"]:
        candidates.append({"algorithm": item["algorithm"], "profile": item["profile"], "score": item["score"],
                           "base_return": item["base"]["net_return"], "base_drawdown": item["base"]["max_drawdown"],
                           "stress_return": item["stress"]["net_return"], "stress_drawdown": item["stress"]["max_drawdown"],
                           "selected": selection["selected_profiles"][item["algorithm"]] == item["profile"],
                           "primary_leader": selection["primary_validation_leader"] == {"algorithm": item["algorithm"], "profile": item["profile"]}})
    training = _training_rows(root, spec, selection)
    gates = _research_targets(summary, selection, audit)
    leader = selection["primary_validation_leader"]
    leader_row = summary.loc[summary.split.eq("test") & summary.scenario.eq("base") & summary.algorithm.eq(leader["algorithm"]) & summary.profile.eq(leader["profile"]) & summary.primary_policy.eq(True)].iloc[0]
    met = bool(audit["research_targets"]["all_predeclared_primary_targets_met"])
    title = "BTC/ETH强化学习主方案达到本轮研究条件" if met else "BTC/ETH强化学习主方案仍未同时达到收益与回撤条件"
    period = {"start": SPLITS["validation"][0], "end": SPLITS["test"][1]}
    definitions = [
        {"label": "扣费净收益", "definition": "期末真实现金与持币估值相对10000 USDT的变化，已扣每腿费用与不利滑点；未强制期末清仓。", "formula": "final_equity/initial_capital-1"},
        {"label": "小时最大回撤", "definition": "完整小时收盘权益相对包含期初本金的历史峰值跌幅。日末曲线不是日频重新计算的最大回撤。"},
        {"label": "主组合", "definition": "五个独立训练模型在同一组合状态下给出目标权重，取均值后由一个真实10000 USDT账本成交；不是五份最终收益平均。"},
        {"label": "种子区间", "definition": "同一行情、同一算法和验证所选profile下五次独立训练的最差至最好结果，属于训练波动描述，不是独立市场置信区间。"},
    ]
    common_source = {"label": "固定BTCUSDT/ETHUSDT/USDT的独立现货模拟与最终逐笔审计", "classification": "modeled",
                     "timezone": "UTC", "period": period, "executedAt": audit["audit_completed_at"],
                     "auditEvidence": audit_identity,
                     "caveats": ["本年度已经用于此前研究，不是全新盲测样本。", "费用、滑点及小数币数量是研究假设，未重建历史盘口和数量过滤器。", "每日00小时卖、01小时买；不能与前两轮下一分钟策略口径混称。", "SAC为同环境算法对照；不与旧Top10结果混排。本轮未使用Kronos或Chronos。"],
                     "metricDefinitions": definitions,
                     "evidenceFlow": [{"title": "固定训练和验证边界", "detail": "训练2020-01-01至2025-01-01；只用2025年295天验证选择每算法profile及总主方案，全部五种子保留。"},
                                      {"title": "同一真实资金账本", "detail": "同一现货账本独立执行每种子、五模型平均目标组合与七个基准，base/stress分别扣费，小时权益覆盖连续完整窗口。"},
                                      {"title": "最终独立核对", "detail": "报告拒绝部分审计，逐项核对最终input SHA和172个run集合，并验证曲线及月度结果与汇总对账。"}]}
    queries = {}
    def query(identity, rows, files, methods):
        source = copy.deepcopy(common_source)
        source["files"] = list(dict.fromkeys([*files, "frozen_spec.json", "validation_selection.json", audit_source]))
        source["fileHashes"] = {name: initial_hashes[name] for name in source["files"] if name in initial_hashes}
        queries[identity] = {"rows": clean_json(rows), "source": source,
                             "methods": [{"language": "text", "code": methods}]}
    query("strategy_summary", frame_rows(summary), ["summary.csv"], "Root summary exact run IDs, chronological windows, capital and hourly coverage are checked against final audit. No result ranking changes the validation-chosen primary.")
    query("equity_history", frame_rows(history), ["daily_equity.csv"], "Saved daily display points retain initial10000 and every UTC day-end. Display-only percentage aliases multiply fractions by100; source summary DD uses all hourly closes.")
    query("monthly_returns", frame_rows(months), ["monthly_returns.csv"], "Monthly compound returns reconcile to the same independent saved ledger's final return. Partial first/last months use actual window only.")
    query("seed_statistics", statistics, ["summary.csv"], "Group exact five seed rows by split/scenario/algorithm/profile; report minimum,maximum,median and count positive. Never select the best test seed.")
    query("validation_selection", candidates, ["validation_selection.json"], "Frozen validation choices and candidate scores copied unchanged. Test ranking is descriptive only. Primary is the validation-selected overall algorithm/profile.")
    query("training_records", training, ["training_completion.json", *[r["source_record"] for r in training]], "All40 training records and model files match validation-time training_source_sha256. Training coverage is available complete-window union, not a claim every seed traversed all dates.")
    query("research_targets", gates, ["summary.csv", audit_source], "Recompute all five predeclared research gates for the validation leader's actual test ensemble, plus both simple trend controls; require exact agreement with the explicitly selected independent audit.")
    query("audit_evidence", [{"status": audit["status"], "passed": True, "audit_role":"原172例独立审计", "checks_total": audit.get("checks_total", audit.get("checks")),
                              "runs_checked": audit["runs_checked"], "failures": audit["failure_count"],
                              "audit_completed_at": audit["audit_completed_at"], "frozen_spec_sha256": spec_hash,
                              "dataset_sha256": spec["dataset_sha256"], "selection_sha256": sha(root / "validation_selection.json"),
                              **audit_identity}],
          [audit_source, *([retained_audit_source] if retained_audit_source else []),
           *(["transfer_receipt.json"] if "transfer_receipt.json" in direct_sources else [])],
          "Passed bookkeeping audit and performance-target attainment are separate conclusions. The explicitly selected primary audit and its SHA identify this report's audit. A retained original audit is separate transfer evidence and is never overwritten or substituted for the current audit.")
    snapshot = {"surface": "report", "status": "reviewed", "buildStatus": "complete", "title": title,
                "generatedAt": datetime.now(timezone.utc).isoformat(), "filters": [], "queries": queries,
                "report": {"asOf": "2026-10-22", "studyConfig": spec, "selection": selection,
                           "trainingCalendar": audit["training_calendar"], "researchTargets": audit["research_targets"],
                           "primaryLeader": leader, "primaryBaseReturn": float(leader_row.net_return),
                           "auditStatus": "passed", "auditEvidence": audit_identity, "scope": "fixed_btc_eth_usdt"}}
    for name, expected_sha in initial_hashes.items():
        actual = sha(direct_sources[name])
        if actual != expected_sha:
            raise AssertionError("Report source changed while assembling: " + name)
    evidence = {"input_sha256": inputs, "direct_input_sha256": initial_hashes,
                      "audit_path": str(audit_path), "audit_source_file": audit_source,
                      "retained_original_audit_source_file": retained_audit_source,
                      "numeric_results_changed": False, "runs": 172}
    if extension_root is not None:
        from scripts.btceth_rl_report_extension import augment_snapshot
        snapshot, evidence = augment_snapshot(snapshot, evidence, main_root=root, extension_root=extension_root,
            extension_audit_path=extension_audit_path, validate_history=_verify_history)
    if risk_reference_root is not None:
        from scripts.btceth_rl_report_extension import augment_risk_controls
        snapshot, evidence = augment_risk_controls(snapshot, evidence, reference_root=risk_reference_root,
            dataset_path=dataset_path, main_root=root, extension_root=extension_root)
    if band_annual_root is not None:
        from scripts.btceth_rl_report_execution_band import load_execution_band
        followup = load_execution_band(band_annual_root, band_validation_root, root, extension_root,
                                      audit_path=band_audit_path)
        if set(snapshot["queries"]) & set(followup["queries"]):
            raise ValueError("Independent execution-band queries cannot replace original study queries")
        snapshot["queries"].update(clean_json(followup["queries"]))
        snapshot["report"]["executionBand"] = clean_json({
            "targets": followup["targets"], "selection": followup["selection"],
            "audit": {key: followup["audit"].get(key) for key in
                      ("passed", "checks_total", "failure_count", "run_count", "repeat_count",
                       "reference_run_count", "baseline_run_count", "audited_at")},
            "queryRunCount": followup["query_run_count"], "additionalUniqueRuns": 12,
            "freshHoldoutSuccess": False, "originalGoalPreserved": True,
        })
        snapshot["title"] = "BTC/ETH强化学习：首轮与调仓门槛试验"
        evidence["execution_band_scope_runs"] = followup["query_run_count"]
        evidence["additional_unique_execution_band_runs"] = 12
        evidence["external_input_sha256"] = {**evidence.get("external_input_sha256", {}),
                                              **followup["external_input_sha256"]}
        from scripts.btceth_rl_report_extension import verify_external_inputs
        verify_external_inputs(evidence["external_input_sha256"])
    if td3_root is not None:
        from scripts.btceth_rl_report_td3 import load_td3
        followup = load_td3(td3_root, root, audit_path=td3_audit_path)
        if set(snapshot["queries"]) & set(followup["queries"]):
            raise ValueError("Independent TD3 queries cannot replace original study queries")
        snapshot["queries"].update(clean_json(followup["queries"]))
        snapshot["report"]["td3Followup"] = clean_json({
            "targets": followup["targets"], "selection": followup["selection"],
            "audit": {key: followup["audit"].get(key) for key in
                      ("passed", "checks_total", "failure_count", "run_count", "repeat_count",
                       "training_model_count", "validation_run_count", "test_run_count", "baseline_run_count", "audited_at")},
            "queryRunCount": 26, "additionalUniqueRuns": 12, "trainingModels": 10,
            "validationAccounts": 48, "trainingBudgetMultiple": 2,
            "freshHoldoutSuccess": False, "originalGoalPreserved": True,
        })
        snapshot["title"] = "BTC/ETH强化学习：收益与回撤比较"
        evidence["td3_scope_runs"] = 26
        evidence["additional_unique_td3_runs"] = 12
        evidence["td3_training_budget_multiple"] = 2
        external = evidence.get("external_input_sha256", {})
        for name, expected in followup["external_input_sha256"].items():
            if name in external and external[name] != expected:
                raise ValueError("TD3 evidence cannot replace an existing source identity: " + name)
        evidence["external_input_sha256"] = {**external, **followup["external_input_sha256"]}
        from scripts.btceth_rl_report_extension import verify_external_inputs
        verify_external_inputs(evidence["external_input_sha256"])
    if droq_root is not None:
        if (Path(droq_root) / "parallel_test_process.json").is_file():
            from scripts.btceth_rl_report_droq_parallel import load_droq
        else:
            from scripts.btceth_rl_report_droq import load_droq
        followup = load_droq(droq_root, root, audit_path=droq_audit_path)
        expected_queries = {"droq_summary", "droq_equity", "droq_monthly", "droq_targets",
                            "droq_validation_selection", "droq_training", "droq_audit"}
        if set(snapshot["queries"]) & set(followup["queries"]):
            raise ValueError("Independent DroQ queries cannot replace original study queries")
        if (set(followup["queries"]) != expected_queries or followup.get("query_run_count") != 26
                or followup.get("new_run_count") != 12 or followup.get("baseline_reference_run_count") != 14
                or followup.get("validation_run_count") != 24 or followup.get("training_model_count") != 5
                or followup.get("original_scopes_unchanged") is not True or followup.get("fresh_holdout_success") is not False
                or followup.get("environment_step_budget_multiple") != 1 or followup.get("compute_budget_comparable") is not False):
            raise ValueError("Complete DroQ annual26 with five models, prior24 validation and separate compute disclosure required")
        external = evidence.get("external_input_sha256", {})
        for name, expected in followup["external_input_sha256"].items():
            if name in external and external[name] != expected:
                raise ValueError("DroQ evidence cannot replace an existing source identity: " + name)
        snapshot["queries"].update(clean_json(followup["queries"]))
        snapshot["report"]["droqFollowup"] = clean_json({
            **followup["droqFollowup"], "targets": followup["targets"], "selection": followup["selection"],
            "chosen": followup["selection"]["candidate"],
            "audit": {key: followup["audit"].get(key) for key in
                      ("passed", "checks_total", "failure_count", "run_count", "repeat_count", "training_model_count",
                       "validation_run_count", "test_run_count", "baseline_run_count", "audited_at")},
            "queryRunCount": 26, "additionalUniqueRuns": 12, "trainingModels": 5, "validationAccounts": 24,
            "environmentStepBudgetMultiple": 1, "computeBudgetComparable": False,
            "freshHoldoutSuccess": False, "originalGoalPreserved": True,
        })
        snapshot["title"] = "BTC/ETH强化学习：收益与回撤比较"
        evidence.update(droq_scope_runs=26, additional_unique_droq_runs=12, droq_training_models=5,
                        droq_validation_accounts=24, droq_environment_step_budget_multiple=1,
                        droq_compute_budget_comparable=False)
        evidence["external_input_sha256"] = {**external, **followup["external_input_sha256"]}
        from scripts.btceth_rl_report_extension import verify_external_inputs
        verify_external_inputs(evidence["external_input_sha256"])
    return snapshot, evidence


def wsl_path(value):
    text = str(value)
    if text.startswith("/"):
        if "\\" in text:
            raise ValueError("Linux paths require forward separators")
        return text
    path = PureWindowsPath(text)
    if not path.is_absolute() or not path.drive.endswith(":"):
        raise ValueError("WSL requires a drive-absolute Windows path or Linux absolute path")
    return "/mnt/" + path.drive[0].lower() + "/" + "/".join(path.parts[1:])


def host_path(value):
    text = str(value)
    if os.name == "nt" and text.startswith("/mnt/"):
        tail = text[5:]
        if len(tail) == 1 and tail.isalpha():
            return Path(tail.upper() + ":/")
        if len(tail) >= 2 and tail[0].isalpha() and tail[1] == "/":
            return Path(tail[0].upper() + ":/" + tail[2:])
    return Path(text)


class DataRuntime:
    def __init__(self, *, node_path=None, plugin_root=None, build_mode="native", wsl_node=None, wsl_distro="Ubuntu-26.04"):
        self.node = Path(node_path) if node_path else NODE_DEFAULT
        self.plugin = Path(plugin_root) if plugin_root else PLUGIN_DEFAULT
        if not (self.plugin / "scripts/data-app.mjs").is_file() and (self.plugin / "output_formats/html/scripts/data-app.mjs").is_file():
            self.plugin /= "output_formats/html"
        if not (self.plugin / "scripts/data-app.mjs").is_file():
            raise FileNotFoundError("Installed official Data SDK required")
        if build_mode not in ("native", "wsl"):
            raise ValueError("Explicit native or wsl build mode required")
        if build_mode == "wsl" and not wsl_node:
            raise ValueError("WSL mode requires an explicitly verified Linux Node path; no automatic install")
        if build_mode == "wsl" and (not str(wsl_node).startswith("/") or "\\" in str(wsl_node)):
            raise ValueError("The explicit WSL Node executable must be a Linux absolute path")
        self.mode, self.wsl_node, self.distro = build_mode, wsl_node, wsl_distro

    def run(self, script, *arguments, cwd):
        script = self.plugin / "scripts" / script
        if self.mode == "wsl":
            converted = [wsl_path(value) if isinstance(value, (Path, PureWindowsPath)) or PureWindowsPath(str(value)).is_absolute()
                         else str(value) for value in arguments]
            command = ["wsl.exe", "--distribution", self.distro, "--exec", str(self.wsl_node), wsl_path(script), *converted]
        else:
            if not self.node.is_file():
                raise FileNotFoundError("Codex Node runtime required")
            command = [str(self.node), str(script), *map(str, arguments)]
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
                                **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
        if result.returncode:
            raise RuntimeError(f"Official Data SDK {script.name} failed; protected runtime remains unchanged.\n{result.stderr[-6000:]}\n{result.stdout[-1000:]}")
        for line in reversed(result.stdout.splitlines()):
            if line.strip().startswith("{"):
                return json.loads(line)
        raise RuntimeError("Official Data SDK returned no JSON result")


def build_report(study_root, *, output_dir=None, audit_path=None, node_path=None, plugin_root=None,
                 build_mode="native", wsl_node=None, wsl_distro="Ubuntu-26.04", extension_root=None,
                 extension_audit_path=None, risk_reference_root=None, dataset_path=None,
                 band_annual_root=None, band_validation_root=None, band_audit_path=None,
                 td3_root=None, td3_audit_path=None, droq_root=None, droq_audit_path=None):
    root = Path(study_root).resolve()
    snapshot, evidence = make_reviewed_snapshot(root, audit_path, extension_root=extension_root,
        extension_audit_path=extension_audit_path, risk_reference_root=risk_reference_root, dataset_path=dataset_path,
        band_annual_root=band_annual_root, band_validation_root=band_validation_root, band_audit_path=band_audit_path,
        td3_root=td3_root, td3_audit_path=td3_audit_path, droq_root=droq_root, droq_audit_path=droq_audit_path)
    if evidence.get("external_input_sha256"):
        from scripts.btceth_rl_report_extension import verify_external_inputs
        verify_external_inputs(evidence["external_input_sha256"])
    destination = Path(output_dir).resolve() if output_dir else root / "report"
    destination.mkdir(parents=True, exist_ok=True)
    reviewed_path = destination / "reviewed.json"
    write_json(reviewed_path, snapshot)
    write_json(destination / "report_input_manifest.json", evidence)
    runtime = DataRuntime(node_path=node_path, plugin_root=plugin_root, build_mode=build_mode,
                          wsl_node=wsl_node, wsl_distro=wsl_distro)
    project = destination / "report_app"
    created = not project.exists()
    if created:
        preparation = runtime.run("prepare-data-app.mjs", "--surface", "report", "--output", project, "--snapshot", reviewed_path, cwd=destination)
    else:
        preparation = runtime.run("data-app.mjs", "prepare", "--project-dir", project, cwd=destination)
    guide = (project / "AGENTS.md").read_text(encoding="utf-8")
    if "src/content/" not in guide:
        raise RuntimeError("Unexpected official Data ownership boundary")
    documentation = preparation["documentation"]
    reference = host_path(documentation["entryPoint"])
    reference.read_text(encoding="utf-8")
    component_root = host_path(documentation.get("referenceRoot", str(runtime.plugin / "docs/components")))
    for name in ("narrative/reports.md", "charts/charts.md", "tables/tables.md", "controls/controls.md", "sources/sources.md"):
        (component_root / name).read_text(encoding="utf-8")
    ownership = project / ".btceth-rl-authoring.json"
    prior = read_json(ownership) if ownership.exists() else {}
    if not created and not prior:
        raise FileExistsError("Preserve an existing app without BTC/ETH ownership evidence")
    for name, expected in prior.get("authored_sha256", {}).items():
        if sha(safe_relative(project, name)) != expected:
            raise RuntimeError("Preserving independent report edits: " + name)
    content = project / "src/content/report"
    sources = {Path(__file__).with_suffix(".jsx"): content / "ReportContent.jsx",
               Path(__file__).with_suffix(".css"): content / "btceth-rl-report.css",
               ROOT / "scripts/btceth_rl_execution_band_report.jsx": content / "BandFollowup.jsx",
               ROOT / "scripts/btceth_rl_td3_report.jsx": content / "TD3Followup.jsx",
               ROOT / "scripts/btceth_rl_droq_report.jsx": content / "DroQFollowup.jsx",
               ROOT / "scripts/btceth_rl_comparison.jsx": content / "DirectComparison.jsx",
               ROOT / "scripts/btceth_rl_comparison_data.mjs": content / "comparison-data.mjs"}
    for source, target in sources.items():
        shutil.copyfile(source, target)
    prepared = read_json(project / "src/data.json")
    snapshot["id"] = prepared["id"]
    # This user-requested comparison layout is authored revision 1; rebuilds preserve it.
    snapshot["authoredRevision"] = max(int(prepared.get("authoredRevision", 0)), 1)
    if "legacyPresentationTitle" in prepared:
        snapshot["legacyPresentationTitle"] = prepared["legacyPresentationTitle"]
    write_json(project / "src/data.json", snapshot)
    write_json(ownership, {"authored_sha256": {path.relative_to(project).as_posix(): sha(path) for path in sources.values()},
                           "reviewed_sha256": sha(reviewed_path)})
    build = runtime.run("data-app.mjs", "build", "--project-dir", project, "--separate-data", cwd=destination)
    exported_file = project / ".data-app-offline/exports/btceth-rl.html"
    exported = runtime.run("data-app.mjs", "export-html", "--project-dir", project, "--output", exported_file, cwd=destination)
    delivered = destination / "report.html"
    shutil.copyfile(exported_file, delivered)
    for name, expected in evidence["input_sha256"].items():
        if sha(safe_relative(root, name)) != expected:
            raise AssertionError("Numerical evidence changed during report build: " + name)
    for name, expected in evidence["direct_input_sha256"].items():
        path = Path(evidence["audit_path"]) if name == evidence["audit_source_file"] else safe_relative(root, name)
        if sha(path) != expected:
            raise AssertionError("Direct report evidence changed during build: " + name)
    if evidence.get("external_input_sha256"):
        verify_external_inputs(evidence["external_input_sha256"])
    result = {"project_dir": str(project), "html_path": str(delivered), "dist_dir": str(project / "dist"),
              "preview_path": str(project / "dist/index.html"), "offline_path": str(exported_file),
              "reviewed_sha256": sha(reviewed_path), "delivered_sha256": sha(delivered), "delivered_bytes": delivered.stat().st_size,
              "authoring_source_sha256": {path.name: sha(path) for path in (Path(__file__), *sources)},
              "runtime": build, "export": exported, "build_mode": build_mode, "published": False, "numeric_results_changed": False}
    write_json(destination / "report_build.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--audit-path", type=Path)
    parser.add_argument("--extension-root", type=Path)
    parser.add_argument("--extension-audit-path", type=Path)
    parser.add_argument("--risk-reference-root", type=Path)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--band-annual-root", type=Path)
    parser.add_argument("--band-validation-root", type=Path)
    parser.add_argument("--band-audit-path", type=Path)
    parser.add_argument("--td3-root", type=Path)
    parser.add_argument("--td3-audit-path", type=Path)
    parser.add_argument("--droq-root", type=Path)
    parser.add_argument("--droq-audit-path", type=Path)
    parser.add_argument("--node-path", type=Path)
    parser.add_argument("--plugin-root", type=Path)
    parser.add_argument("--build-mode", choices=("native", "wsl"), default="native")
    parser.add_argument("--wsl-node")
    parser.add_argument("--wsl-distro", default="Ubuntu-26.04")
    args = parser.parse_args()
    print(json.dumps(build_report(**vars(args)), ensure_ascii=False))


if __name__ == "__main__":
    main()
