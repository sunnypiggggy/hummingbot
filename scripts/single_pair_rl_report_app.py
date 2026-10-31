"""Read-only, audit-gated report for fresh independent BTC and ETH accounts.

No pickle loading, policy construction, inference, training or publication. The
old portfolio report is imported only for its official Data SDK transport.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backtest_single_pair_rl as runner
from scripts.crypto_rl_reporting import clean_json, summarize
from scripts.btceth_rl_report_app import DataRuntime, host_path

ALGORITHMS = ("sac", "tqc")
RESULT_FILES = ("equity_1h.csv.gz", "fills.csv", "weights.csv", "boundary_equity.csv", "metadata.json",
                "fingerprints.json", "actions.json", "summary.json", "daily_equity.csv", "monthly_returns.csv")
QUERY_IDS = ("account_summary", "equity_history", "monthly_returns", "seed_statistics", "validation_summary",
             "validation_selection", "training_records", "research_targets", "audit_evidence", "protocol")
SELECTION_SCORE_ABS_TOL = 1e-14


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def relative(root, name):
    name = str(name).replace("\\", "/")
    require(name and ":" not in name and not name.startswith("/") and ".." not in name.split("/"), "Unsafe evidence path")
    root = Path(root).resolve()
    path = (root / name).resolve()
    path.relative_to(root)
    require(path.is_file() and not path.is_symlink(), "Missing or unsafe evidence file: " + name)
    return path


def csv(path, *, indexed=False):
    result = pd.read_csv(path, float_precision="round_trip")
    if indexed:
        result["timestamp"] = pd.to_datetime(result.timestamp, utc=True)
        result = result.set_index("timestamp")
    return result


def equivalent(actual, expected, label):
    """CSV null/float decoding may differ, financial values and identities may not."""
    require(set(actual) == set(expected), "Changed fields: " + label)
    for key, value in expected.items():
        other = actual[key]
        if value is None:
            require(other is None or pd.isna(other), label + ": missing/null " + key)
        elif isinstance(value, (bool, str)):
            require(other == value, label + ": identity " + key)
        elif isinstance(value, (int, float)):
            require(isinstance(other, (int, float, np.number)) and math.isfinite(float(other))
                    and math.isclose(float(other), float(value), rel_tol=1e-10, abs_tol=1e-7), label + ": value " + key)
        else:
            require(other == value, label + ": field " + key)


def frame_equivalent(actual, expected, label):
    require(set(actual.columns) == set(expected.columns) and len(actual) == len(expected), "Changed CSV shape: " + label)
    for a, e in zip(actual[expected.columns].to_dict("records"), expected.to_dict("records")):
        equivalent(a, e, label)


def verify_selection_choice(choice, scored):
    """Permit only tiny platform math differences; preserve frozen identities."""
    require(set(choice) == {"algorithm", "seed", "all_candidates"}
            and choice["algorithm"] == scored[0]["algorithm"] and type(choice["seed"]) is int and choice["seed"] == 0,
            "Validation selected algorithm or fixed primary seed changed")
    actual = choice["all_candidates"]
    require(isinstance(actual, list) and len(actual) == len(scored), "Validation candidate population changed")
    maximum_error = 0.
    for frozen, local in zip(actual, scored):
        require(set(frozen) == {"algorithm", "score_median", "seed_scores"} and frozen["algorithm"] == local["algorithm"],
                "Validation candidate identity or order changed")
        require(isinstance(frozen["seed_scores"], list) and len(frozen["seed_scores"]) == len(runner.SEEDS), "Validation seed score population changed")
        for observed, expected in zip([frozen["score_median"], *frozen["seed_scores"]], [local["score_median"], *local["seed_scores"]]):
            require(type(observed) in (int, float) and math.isfinite(observed)
                    and math.isclose(observed, expected, rel_tol=0., abs_tol=SELECTION_SCORE_ABS_TOL),
                    "Validation selection score differs beyond platform math tolerance")
            maximum_error = max(maximum_error, abs(observed - expected))
    return maximum_error


def expected_cases(split):
    learned = [(s, a, seed, c) for s in runner.SYMBOLS for a in ALGORITHMS for seed in runner.SEEDS for c in runner.COSTS]
    baselines = [(s, a, None, c) for s in runner.SYMBOLS for a in runner.BASELINES for c in runner.COSTS]
    return {(f"{split}__{c}__{s}__{a}__{'baseline' if seed is None else seed}"):
            (s, a, seed, c, f"evaluation/{split}/{s}/{a}/{'baseline' if seed is None else 'seed' + str(seed)}/{c}")
            for s, a, seed, c in learned + baselines}


def verify_case(root, split, identity, row, audited_case):
    symbol, algorithm, seed, scenario, directory = identity
    require(audited_case.get("passed") is True and audited_case.get("failure_count") == 0
            and audited_case.get("relative_path") == directory, "Case audit failed or belongs to another account")
    require(set(audited_case.get("inputs_sha256", {})) == {f"{d}/{name}" for d in (directory, directory.replace("evaluation/", "repeat/", 1)) for name in RESULT_FILES}, "Complete case audit input set required")
    for key, expected in audited_case["inputs_sha256"].items():
        require(sha(relative(root, key)) == expected, "Changed audited case bytes: " + key)
    for name in RESULT_FILES:
        require(sha(relative(root, directory + "/" + name)) == sha(relative(root, directory.replace("evaluation/", "repeat/", 1) + "/" + name)), "Case repeat changed: " + directory)
    p = relative(root, directory + "/summary.json").parent
    actual = read(p / "summary.json")
    equivalent(row, actual, directory + " aggregate")
    require(actual == audited_case.get("summary"), "Audit summary no longer matches case JSON")
    require(actual["symbol"] == symbol and actual["algorithm"] == algorithm and actual["seed"] == seed
            and actual["scenario"] == scenario and actual["split"] == split and actual["ensemble"] is False
            and actual["initial_capital"] == 10000. and actual["primary_seed"] == (seed == 0)
            and actual["independent_capital"] is True, "Independent account identity changed")
    meta = read(p / "metadata.json")
    require(meta.get("pair") == symbol and meta.get("tradable_symbols") == [symbol] and meta.get("no_ensemble") is True
            and meta.get("independent_capital") == 10000. and meta.get("observation_dim") == 13 and meta.get("action_dim") == 1
            and set(meta.get("final_positions", {})) <= {symbol}, "Foreign coin, dimensions or ownership in case metadata")
    fingerprints = read(p / "fingerprints.json")
    require(set(fingerprints) == {"equity", "fills", "weights", "boundary_equity"}, "All four result fingerprints required")
    for key, name in (("equity", "equity_1h.csv.gz"), ("fills", "fills.csv"), ("weights", "weights.csv"), ("boundary_equity", "boundary_equity.csv")):
        payload = (p / name).read_bytes()
        if name.endswith(".gz"):
            payload = gzip.decompress(payload)
        require(hashlib.sha256(payload).hexdigest() == fingerprints[key], "CSV fingerprint changed: " + directory + "/" + key)
    result = {"equity": csv(p / "equity_1h.csv.gz", indexed=True), "fills": csv(p / "fills.csv"),
              "weights": csv(p / "weights.csv"), "boundary_equity": csv(p / "boundary_equity.csv", indexed=True), "metadata": meta}
    for frame in (result["fills"], result["weights"]):
        require(not len(frame) or set(frame.asset_key) == {symbol}, "Foreign inventory in independent account")
    bounds = runner.TEST if split == "test" else runner.VALIDATION
    re_summary, daily, months = summarize(result, run_id=actual["run_id"], strategy=algorithm, seed=seed, scenario=scenario,
        split=split, start=bounds[0], end=bounds[1], capital=10000., **runner.COSTS[scenario])
    equivalent({k: actual[k] for k in re_summary}, re_summary, directory + " serialized readback")
    daily["timestamp"] = daily.timestamp.map(str)
    saved_daily = csv(p / "daily_equity.csv")
    saved_daily["timestamp"] = pd.to_datetime(saved_daily.timestamp, utc=True).map(str)
    frame_equivalent(saved_daily, daily, directory + " daily")
    frame_equivalent(csv(p / "monthly_returns.csv"), months, directory + " monthly")
    for frame in (daily, months):
        frame["symbol"], frame["algorithm"] = symbol, algorithm
    return actual, daily.to_dict("records"), months.to_dict("records")


def make_reviewed_snapshot(study_root):
    root = Path(study_root).resolve()
    pins = {}
    def take(name):
        path = relative(root, name)
        pins[name] = sha(path)
        return read(path)
    spec = take("protocol.json")
    receipt = take("protocol_receipt.json")
    require(receipt.get("protocol_sha256") == pins["protocol.json"], "Protocol receipt mismatch")
    require(spec.get("symbols") == list(runner.SYMBOLS) and len(spec.get("algorithms", [])) == 2 and set(spec.get("algorithms", [])) == set(ALGORITHMS)
            and spec.get("seeds") == list(runner.SEEDS) and spec.get("primary_seed") == 0
            and spec.get("training_model_count") == 20 and spec.get("capital_per_account") == 10000.
            and spec.get("primary_seed_fixed_before_training") is True and spec.get("capital_shared_between_pairs") is False
            and spec.get("ensemble") is False and spec.get("test_year_already_observed") is True and spec.get("fresh_holdout") is False
            and spec.get("no_production_change") is True, "A complete 20-model independent-account protocol is required")
    require(spec.get("train") == list(runner.TRAIN) and spec.get("validation") == list(runner.VALIDATION)
            and spec.get("test") == list(runner.TEST) and spec.get("costs") == runner.COSTS
            and set(spec.get("sources", {})) == set(runner.SOURCE_FILES)
            and spec.get("timesteps_per_model") == {"sac": 32768, "tqc": 32768}, "Frozen sources, budgets, dates or costs changed")
    require(spec.get("research_gates_per_pair") == {"base_return_at_least": .05, "base_drawdown_magnitude_at_most": .20,
            "stress_return_at_least": 0., "positive_base_seeds_at_least": 3, "beats_same_coin_hold_return_and_drawdown": True}, "Research target contract changed")
    for key, value in {"observation_dim": 13, "action_dim": 1, "cash_floor": .05, "max_coin_target": .95,
                       "rebalance_band": .025, "drawdown_penalty": .25, "episode_days": 32}.items():
        require(spec["environment"].get(key) == value, "Single-coin environment contract changed: " + key)
    for name, expected in spec["sources"].items():
        pins["frozen_source/" + name] = sha(relative(root, "frozen_source/" + name))
        require(pins["frozen_source/" + name] == expected, "Frozen source changed: " + name)
        require(sha(relative(ROOT, name)) == expected, "Report readback helper differs from frozen source: " + name)
    training, training_original = [], []
    for symbol in runner.SYMBOLS:
        for algorithm in ALGORITHMS:
            for seed in runner.SEEDS:
                record = runner.validate_training_record(root, symbol, algorithm, seed, spec)
                name = f"training/{symbol}/{algorithm}/seed{seed}/record.json"
                take(name)
                directory = relative(root, name).parent
                actual_files = {p.relative_to(directory).as_posix() for p in (directory / "checkpoint").rglob("*") if p.is_file()}
                require(actual_files == set(record["files"]), "Incomplete checkpoint file inventory")
                for key, expected in record["files"].items():
                    filename = name.rsplit("/", 1)[0] + "/" + key
                    pins[filename] = sha(relative(root, filename))
                    require(pins[filename] == expected, "Training checkpoint bytes changed")
                training_original.append(record)
                training.append({"symbol": symbol, "algorithm": algorithm, "seed": seed, "capital": 10000., "observation_dim": 13,
                    "action_dim": 1, "ensemble": False, "actual_steps": record["training"]["actual_steps"],
                    "gradient_update_rounds": record["training"]["counters"]["gradient_update_rounds"],
                    "maximum_input_hour": record["maximum_input_hour"], "record_file": name,
                    "record_sha256": pins[name], "adapter_sha256": record["checkpoint"]["adapter_sha256"]})
    expected_records = {r["record_file"] for r in training}
    require({p.relative_to(root).as_posix() for p in (root / "training").rglob("record.json")} == expected_records, "Unexpected or incomplete trained model population")
    fit = take("fit_complete.json")
    require(fit.get("model_count") == 20 and len(fit.get("records", [])) == 20
            and sorted(fit["records"], key=lambda r: (r["symbol"], r["algorithm"], r["seed"])) == sorted(training_original, key=lambda r: (r["symbol"], r["algorithm"], r["seed"])), "Training completion does not cover all exact records")
    summaries, histories, monthly, audits = {}, [], [], []
    for split in ("validation", "test"):
        audit = take(split + "_audit.json")
        require(audit.get("phase") == split and audit.get("passed") is True and audit.get("failure_count") == 0
                and not audit.get("failures") and audit.get("checks_total", 0) > 0
                and all(audit.get(k) == 56 for k in ("expected_accounts", "accounts_checked", "repeats_checked", "accounts_count"))
                and audit.get("capital_per_account") == 10000. and audit.get("capital_shared_between_pairs") is False
                and audit.get("ensemble") is False and audit.get("primary_seed") == 0, "Complete passed financial audit required: " + split)
        require(audit.get("protocol_sha256") == pins["protocol.json"] and audit.get("summary_sha256") == sha(relative(root, split + "_summary.csv")), "Audit summary or protocol binding changed")
        for key, expected in audit.get("inputs_sha256", {}).items():
            pins[key] = sha(relative(root, key))
            require(pins[key] == expected, "Changed phase audit input: " + key)
        require(set(audit.get("inputs_sha256", {})) == {"protocol.json", split + "_summary.csv", split + "_complete.json"}, "Complete phase audit input binding required")
        frame = csv(relative(root, split + "_summary.csv"))
        expected = expected_cases(split)
        require(len(frame) == 56 and not frame.run_id.duplicated().any() and set(frame.run_id) == set(expected), "Exactly 56 distinct accounts required: " + split)
        observed_cases = {p.parent.relative_to(root).as_posix() for p in (root / "evaluation" / split).rglob("summary.json")}
        require(observed_cases == {i[-1] for i in expected.values()}, "Unexpected result folders")
        cases = {case.get("relative_path"): case for case in audit.get("cases", [])}
        require(len(audit.get("cases", [])) == 56 and set(cases) == observed_cases, "Every account needs its own passed oracle result")
        completion = take(split + "_complete.json")
        require(completion.get("accounts") == completion.get("repeats") == 56 and completion.get("repeat_equal") is True
                and len(completion.get("results", [])) == 56, "Incomplete execution completion")
        completed = {r["summary"]["run_id"]: r for r in completion["results"]}
        require(set(completed) == set(expected), "Completion account identities changed")
        rows = []
        for aggregate in frame.to_dict("records"):
            identity = expected[aggregate["run_id"]]
            case = cases[identity[-1]]
            row, daily, months = verify_case(root, split, identity, aggregate, case)
            item = completed[row["run_id"]]
            require(item.get("path") == identity[-1] and item.get("repeat_path") == identity[-1].replace("evaluation/", "repeat/", 1)
                    and item.get("repeat_equal") is True and item.get("summary") == row, "Wrong canonical/repeat completion path")
            pins.update(case["inputs_sha256"])
            rows.append(row)
            if split == "test":
                histories.extend(daily); monthly.extend(months)
        summaries[split] = rows
        audits.append({k: audit[k] for k in ("phase", "passed", "checks_total", "failure_count", "accounts_checked", "repeats_checked", "audited_at", "independent_oracle")})
    selection = take("validation_selection.json")
    require(selection.get("uses_test") is False and selection.get("seed_selection") is False
            and selection.get("protocol_sha256") == pins["protocol.json"]
            and selection.get("validation_audit_sha256") == pins["validation_audit.json"]
            and selection.get("validation_summary_sha256") == pins["validation_summary.csv"], "Selection does not bind audited validation-only evidence")
    candidates, selection_score_errors = [], {}
    # Match the frozen selector's original CSV decoding and operation order;
    # display readback elsewhere deliberately retains full JSON precision.
    score_frame = pd.read_csv(relative(root, "validation_summary.csv"))
    for symbol in runner.SYMBOLS:
        scored = []
        for algorithm in ALGORITHMS:
            group = score_frame.loc[(score_frame.symbol == symbol) & (score_frame.algorithm == algorithm)]
            scores = [float(np.mean([math.log1p(r.net_return) - .5 * abs(r.max_drawdown) - 2 * max(abs(r.max_drawdown) - .2, 0.)
                for r in group.loc[group.seed == seed].itertuples()])) for seed in runner.SEEDS]
            scored.append({"algorithm": algorithm, "score_median": float(np.median(scores)), "seed_scores": scores})
        scored.sort(key=lambda r: (-r["score_median"], r["algorithm"]))
        choice = selection["selection"][symbol]
        selection_score_errors[symbol] = verify_selection_choice(choice, scored)
        candidates.extend({"symbol": symbol, "algorithm": r["algorithm"], "score_median": r["score_median"], "selected": r["algorithm"] == choice["algorithm"],
                           "primary_seed": 0, **{f"seed{i}_score": value for i, value in enumerate(r["seed_scores"])}} for r in choice["all_candidates"])
    started = take("test_started.json")
    require(started.get("pins") == {"selection_sha256": pins["validation_selection.json"], "validation_audit_sha256": pins["validation_audit.json"],
            "protocol_sha256": pins["protocol.json"], "dataset_sha256": spec["dataset_sha256"]}, "Annual execution did not pin this selection before starting")
    require(pd.Timestamp(selection["at"]) >= pd.Timestamp(audits[0]["audited_at"])
            and pd.Timestamp(started["at"]) >= pd.Timestamp(selection["at"])
            and pd.Timestamp(read(relative(root, "test_complete.json"))["at"]) >= pd.Timestamp(started["at"])
            and pd.Timestamp(audits[1]["audited_at"]) >= pd.Timestamp(read(relative(root, "test_complete.json"))["at"]), "Validation, selection, annual start and final audit are out of chronological order")
    stats, gates = [], []
    for symbol in runner.SYMBOLS:
        for algorithm in ALGORITHMS:
            for scenario in runner.COSTS:
                group = [r for r in summaries["test"] if r["symbol"] == symbol and r["algorithm"] == algorithm and r["scenario"] == scenario]
                require(len(group) == 5 and {r["seed"] for r in group} == set(runner.SEEDS), "Five individual seeds required for statistics")
                returns = [r["net_return"] for r in group]
                stats.append({"symbol": symbol, "algorithm": algorithm, "scenario": scenario, "seeds": 5, "positive_seeds": sum(x > 0 for x in returns),
                    "median_return": float(np.median(returns)), "min_return": min(returns), "max_return": max(returns),
                    "worst_drawdown": min(r["max_drawdown"] for r in group)})
        algorithm = selection["selection"][symbol]["algorithm"]
        primary = {r["scenario"]: r for r in summaries["test"] if r["symbol"] == symbol and r["algorithm"] == algorithm and r["seed"] == 0}
        hold = next(r for r in summaries["test"] if r["symbol"] == symbol and r["algorithm"] == "buy_hold" and r["scenario"] == "base")
        positives = next(r["positive_seeds"] for r in stats if r["symbol"] == symbol and r["algorithm"] == algorithm and r["scenario"] == "base")
        measures = [("base_return", "主成本收益至少5%", primary["base"]["net_return"], .05, primary["base"]["net_return"] >= .05),
                    ("base_drawdown", "主成本回撤不超过20%", abs(primary["base"]["max_drawdown"]), .2, abs(primary["base"]["max_drawdown"]) <= .2),
                    ("stress_return", "压力成本收益不低于0", primary["stress"]["net_return"], 0., primary["stress"]["net_return"] >= 0),
                    ("positive_seeds", "主成本至少3/5种子盈利", positives, 3, positives >= 3),
                    ("beats_hold", "收益与回撤同时优于同币95%持有", None, None, primary["base"]["net_return"] > hold["net_return"] and abs(primary["base"]["max_drawdown"]) < abs(hold["max_drawdown"]))]
        gates.extend({"symbol": symbol, "algorithm": algorithm, "seed": 0, "gate": key, "label": label, "actual": value, "threshold": threshold, "passed": passed} for key, label, value, threshold, passed in measures)
    queries = {}
    common = {"label": "独立单币现货研究账户", "classification": "modeled", "timezone": "UTC", "period": {"start": runner.TEST[0], "end": runner.TEST[1]},
        "caveats": ["本年度已被其他研究观察，不是新的盲测样本。", "五随机种子反映同一行情上的训练波动，不是五份独立市场样本。", "最大回撤按完整小时收盘计算，图表为日末显示。", "手续费和滑点为研究假设，不含盘口队列与全部历史过滤器。", "每个账户独立10,000 USDT，无账户间资金、库存、特征或模型平均。"],
        "metricDefinitions": [{"label": "扣费收益", "definition": "期末真实现金加持币市值/10000-1；已扣模拟交易手续费与不利滑点。", "componentIds": ["single-primary", "single-seeds", "single-baselines", "single-details", "single-curve"]},
            {"label": "最大回撤", "definition": "包含期初10000的历史峰值，与完整小时收盘权益之比的最小值。表中显示跌幅绝对值，越小越好。"},
            {"label": "盈利种子", "definition": "同币同算法五份实际单模型主成本年度收益>0的数量，分母固定5。"},
            {"label": "净清算收益", "definition": "期末持仓按统一滑点和手续费假设退出后的权益/10000-1，仅估算，未强制成交。"}],
        "evidenceFlow": [{"title": "新账户与新训练", "detail": "每币/算法/种子独立训练，13维本币观测和1维动作；20模型原始记录与 checkpoint manifest 全量核对。"},
            {"title": "验证先于年度", "detail": "先完整审计验证56账户，再按预设五种子中位评分选算法；主seed0在训练前固定。年度两成本分别推理，无多模型平均。"},
            {"title": "真实输出读回", "detail": "逐个核对两阶段56账户与重复文件、原审计输入SHA、CSV fingerprints，并从保存小时账本重算摘要、每日与月度展示值。"}]}
    def query(key, rows, files, method, period=None):
        source = deepcopy(common)
        names = list(dict.fromkeys([*files, "protocol.json", "protocol_receipt.json", "validation_selection.json", "validation_audit.json", "test_audit.json"]))
        source.update(files=names, fileHashes={n: pins[n] for n in names}, executedAt=max(a["audited_at"] for a in audits))
        if period:
            source["period"] = {"start": period[0], "end": period[1]}
        queries[key] = {"rows": clean_json(rows), "source": source, "methods": [{"language": "text", "code": method}]}
    query("account_summary", summaries["test"], ["test_summary.csv"], "Exactly56 independent test accounts:20 learned models at2 costs plus4 fixed baselines per coin at2 costs. No cross-coin aggregation or target averaging.")
    query("equity_history", histories, [p for p in pins if p.startswith("evaluation/test/") and p.endswith("daily_equity.csv")], "Original daily boundary points per real account; all366 including initial10000. Plotted return/drawdown fractions are display-only multiplied by100. No median/averaged equity curve.")
    query("monthly_returns", monthly, [p for p in pins if p.startswith("evaluation/test/") and p.endswith("monthly_returns.csv")], "Actual partial first/last months per independent account; compound to each account's net return.")
    query("seed_statistics", stats, ["test_summary.csv"], "Five exact seeds0..4 per coin/algorithm/scenario. Median/min/max describe terminal outcomes only; they are not traded accounts.")
    query("validation_summary", summaries["validation"], ["validation_summary.csv"], "All56 validation accounts, separate from annual test; same capital/cost/ownership definitions.", runner.VALIDATION)
    query("validation_selection", candidates, ["validation_summary.csv", "validation_selection.json"], "Recompute per-seed mean base/stress log1p(return)-.5*abs(DD)-2*max(abs(DD)-.20,0); choose algorithm by median of5 scores and identifier tie-break. Always retain predeclared seed0.", runner.VALIDATION)
    query("training_records", training, ["fit_complete.json", *sorted(expected_records)], "Complete20 training manifests, payload hashes, exact32768 environment steps and32256 actual SAC/TQC update rounds. Maximum input hour precedes training end.", runner.TRAIN)
    query("research_targets", gates, ["test_summary.csv", "validation_selection.json"], "Five original per-coin research gates apply to validation-selected algorithm's seed0. Positive-seed denominator comes from all5 base accounts. Better hold means strictly greater return and strictly smaller DD magnitude.")
    query("audit_evidence", audits, ["validation_audit.json", "test_audit.json"], "Original independent closed-form and actual-fill financial audits must both pass all56 accounts; SHA-only rebinding of all inputs happens before report output. Audit success does not imply target attainment.")
    query("protocol", [{"capital_per_account": 10000., "training_models": 20, "test_accounts": 56, "seeds_per_algorithm": 5, "primary_seed": 0, "observation_dim": 13, "action_dim": 1,
        "cash_shared": False, "ensemble": False, "test_start": runner.TEST[0], "test_end_exclusive": runner.TEST[1], "test_year_already_observed": True,
        "dataset_sha256": spec["dataset_sha256"], "protocol_sha256": pins["protocol.json"]}], ["protocol.json"], "Frozen protocol bytes and receipt; source inputs are displayed as hashes only. This report never loads the original pickle or constructs a policy.")
    score_readback = {"scope": "validation selection scores only", "absolute_tolerance": SELECTION_SCORE_ABS_TOL, "relative_tolerance": 0.,
                      "maximum_absolute_error": max(selection_score_errors.values()), "per_symbol_maximum_absolute_error": selection_score_errors,
                      "frozen_display_values_preserved": True, "identities_and_candidate_order_exact": True}
    queries["validation_selection"]["source"]["selectionScoreReadback"] = score_readback
    snapshot = {"surface": "report", "status": "reviewed", "buildStatus": "complete", "title": "BTC 与 ETH 各自交易：独立账户收益与回撤", "generatedAt": datetime.now(timezone.utc).isoformat(),
        "filters": [], "queries": queries, "report": {"asOf": "2026-10-22", "scope": "independent_single_pair_accounts", "studyConfig": spec,
        "selection": selection, "auditStatus": "passed", "primarySeed": 0, "noEnsemble": True, "noPooledCapital": True,
        "selectionScoreReadback": score_readback}}
    evidence = {"input_sha256": pins, "accounts_per_phase": 56, "training_models": 20, "numeric_results_changed": False,
                "pickle_loaded": False, "policy_constructed": False, "selection_score_readback": score_readback,
                "source_sha256": {str(Path(__file__).relative_to(ROOT)): sha(__file__), "scripts/btceth_rl_report_app.py": sha(ROOT / "scripts/btceth_rl_report_app.py")}}
    verify_inputs(root, evidence)
    return snapshot, evidence


def verify_inputs(root, evidence):
    for name, expected in evidence["input_sha256"].items():
        require(sha(relative(root, name)) == expected, "Evidence changed during report preparation: " + name)


def build_report(study_root, *, output_dir=None, build_mode="native", wsl_node=None, wsl_distro="Ubuntu-26.04", node_path=None, plugin_root=None):
    root = Path(study_root).resolve()
    snapshot, evidence = make_reviewed_snapshot(root)
    destination = Path(output_dir).resolve() if output_dir else root / "report"
    project = destination / "report_app"
    ownership = project / ".single-pair-rl-authoring.json"
    prior = read(ownership) if ownership.is_file() else None
    if destination.exists() and any(destination.iterdir()):
        require(prior is not None, "Preserve nonempty report destination without single-pair ownership")
        require(prior.get("scope") == "independent_single_pair_accounts", "Preserve a different report identity")
        for name, expected in prior.get("authored_sha256", {}).items():
            require(sha(relative(project, name)) == expected, "Preserve user presentation edits: " + name)
        for name, expected in prior.get("delivery_sha256", {}).items():
            require(sha(relative(destination, name)) == expected, "Preserve changed delivered evidence: " + name)
    elif project.exists():
        raise FileExistsError("Unowned app project")
    destination.mkdir(parents=True, exist_ok=True)
    reviewed = destination / "reviewed.json"
    write(reviewed, snapshot)
    write(destination / "report_input_manifest.json", evidence)
    runtime = DataRuntime(node_path=node_path, plugin_root=plugin_root, build_mode=build_mode, wsl_node=wsl_node, wsl_distro=wsl_distro)
    if project.exists():
        preparation = runtime.run("data-app.mjs", "prepare", "--project-dir", project, cwd=destination)
    else:
        preparation = runtime.run("prepare-data-app.mjs", "--surface", "report", "--output", project, "--snapshot", reviewed, cwd=destination)
    require("src/content/" in (project / "AGENTS.md").read_text(encoding="utf-8"), "Unexpected official Data authoring boundary")
    documentation = preparation["documentation"]
    host_path(documentation["entryPoint"]).read_text(encoding="utf-8")
    component_root = host_path(documentation.get("referenceRoot", str(runtime.plugin / "docs/components")))
    for name in ("narrative/reports.md", "charts/charts.md", "tables/tables.md", "controls/controls.md", "sources/sources.md"):
        (component_root / name).read_text(encoding="utf-8")
    content = project / "src/content/report"
    sources = {Path(__file__).with_suffix(".jsx"): content / "ReportContent.jsx", Path(__file__).with_suffix(".css"): content / "single-pair-rl-report.css"}
    for source, target in sources.items():
        shutil.copyfile(source, target)
    prepared = read(project / "src/data.json")
    snapshot["id"] = prepared["id"]
    if "legacyPresentationTitle" in prepared:
        snapshot["legacyPresentationTitle"] = prepared["legacyPresentationTitle"]
    write(project / "src/data.json", snapshot)
    authored = {path.relative_to(project).as_posix(): sha(path) for path in sources.values()}
    write(ownership, {"scope": "independent_single_pair_accounts", "authored_sha256": authored})
    build = runtime.run("data-app.mjs", "build", "--project-dir", project, "--separate-data", cwd=destination)
    exported = project / ".data-app-offline/exports/single-pair-rl.html"
    exported_result = runtime.run("data-app.mjs", "export-html", "--project-dir", project, "--output", exported, cwd=destination)
    verify_inputs(root, evidence)
    delivered = destination / "report.html"
    shutil.copyfile(exported, delivered)
    result = {"scope": "independent_single_pair_accounts", "html_path": str(delivered), "preview_path": str(project / "dist/index.html"),
              "project_dir": str(project), "reviewed_sha256": sha(reviewed), "html_sha256": sha(delivered), "runtime": build, "export": exported_result,
              "build_mode": build_mode, "published": False, "numeric_results_changed": False,
              "authoring_source_sha256": {p.name: sha(p) for p in (Path(__file__), *sources)}}
    write(destination / "report_build.json", result)
    write(ownership, {"scope": "independent_single_pair_accounts", "authored_sha256": authored,
          "delivery_sha256": {p.name: sha(p) for p in (reviewed, destination / "report_input_manifest.json", delivered, destination / "report_build.json")}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--build-mode", choices=("native", "wsl"), default="native")
    parser.add_argument("--wsl-node")
    parser.add_argument("--wsl-distro", default="Ubuntu-26.04")
    parser.add_argument("--node-path")
    parser.add_argument("--plugin-root")
    args = parser.parse_args()
    print(json.dumps(build_report(args.study_root, output_dir=args.output_dir, build_mode=args.build_mode, wsl_node=args.wsl_node,
          wsl_distro=args.wsl_distro, node_path=args.node_path, plugin_root=args.plugin_root), ensure_ascii=False))


if __name__ == "__main__":
    main()
