"""Build an audit-gated report for independent, rolling BTC/ETH experiments.

The reader never trains or loads policies. Completed accounting and the exact
precommitted validation-only shortlist must exist before any snapshot is saved.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.btceth_rl_report_app import DataRuntime, host_path
from scripts.crypto_rl_reporting import clean_json

SYMBOLS = ("BTCUSDT", "ETHUSDT")
CADENCES = ("fixed", "every_4w", "every_week")
SCENARIOS = ("base", "stress")
SEEDS = tuple(range(5))
COSTS = {"base": {"fee": .001, "slippage": .0002}, "stress": {"fee": .002, "slippage": .0005}}
TEST = ("2025-10-23T00:00:00Z", "2026-10-23T00:00:00Z")
VALIDATION = ("2025-01-01T00:00:00Z", "2025-10-23T00:00:00Z")
EXPECTED = {"BTCUSDT": ("td3_default", "sac_default", "sac_gamma099"),
            "ETHUSDT": ("sac_default", "tqc_default", "sac_gamma099")}
INVENTORY_IDS = {"td3_default": "native_td3", "sac_default": "native_sac", "tqc_default": "native_tqc", "sac_gamma099": "sac_c02_gamma099"}
ALGORITHMS = {"td3_default": "td3", "sac_default": "sac", "tqc_default": "tqc", "sac_gamma099": "sac"}
INVENTORY_SHA256 = "0ed7baffe5f82680c5e873fa5231320a5766088b357024584ab37f40f029796b"
SCOPE = "all_algorithms_independent_rolling_comparison"
ACCOUNT_COUNT, MODEL_COUNT = 180, 1590
RESULT_FILES = ("equity_1h.csv.gz", "fills.csv", "weights.csv", "boundary_equity.csv", "metadata.json", "fingerprints.json", "actions.json", "summary.json", "daily_equity.csv", "monthly_returns.csv")
FINANCIAL_FIELDS = ("initial_capital", "final_equity", "net_pnl", "net_return", "max_drawdown", "fills", "fees", "turnover", "average_exposure", "final_cash", "final_base_value", "net_liquidation_equity", "net_liquidation_return", "slippage_cost", "observed_hours")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def root_path(value):
    path = Path(value).absolute()
    require(not any(p.is_symlink() or hasattr(p, "is_junction") and p.is_junction() for p in (path, *path.parents)), "Research roots cannot cross a link")
    return path.resolve()


def relative(root, name):
    name = str(name)
    parsed = PurePosixPath(name)
    require(name and parsed.as_posix() == name and not parsed.is_absolute() and ".." not in parsed.parts and ":" not in name and "\\" not in name, "Canonical relative evidence path required")
    result = root_path(root).joinpath(*parsed.parts)
    require(not any(p.is_symlink() or hasattr(p, "is_junction") and p.is_junction() for p in (result, *result.parents)), "Evidence cannot cross a link")
    result.resolve().relative_to(root_path(root))
    require(result.is_file(), "Missing evidence: " + name)
    return result


def frame_rows(frame):
    return clean_json(frame.astype(object).where(pd.notna(frame), None).to_dict("records"))


def expected_accounts():
    return {f"{symbol}__{recipe}__{cadence}__seed{seed}__{cost}": (symbol, recipe, cadence, seed, cost)
            for symbol in SYMBOLS for recipe in EXPECTED[symbol] for cadence in CADENCES for seed in SEEDS for cost in SCENARIOS}


def summarize_seeds(rows):
    result = []
    for symbol in SYMBOLS:
        for recipe in EXPECTED[symbol]:
            for cadence in CADENCES:
                for scenario in SCENARIOS:
                    group = [r for r in rows if (r["symbol"], r["recipe_id"], r["cadence"], r["scenario"]) == (symbol, recipe, cadence, scenario)]
                    require(len(group) == 5 and {r["seed"] for r in group} == set(SEEDS), "Exactly five complete final seeds per cell required")
                    returns = np.asarray([r["net_return"] for r in group], dtype=float)
                    dds = np.abs([r["max_drawdown"] for r in group])
                    require(np.isfinite(returns).all() and np.isfinite(dds).all(), "Missing outcomes cannot become zero")
                    qs = np.quantile(returns, [.1, .25, .5, .75, .9], method="linear")
                    result.append({"key": f"{symbol}:{recipe}:{cadence}:{scenario}", "symbol": symbol, "recipe_id": recipe, "algorithm": ALGORITHMS[recipe], "cadence": cadence, "scenario": scenario,
                        "seeds": 5, "mean_return": float(returns.mean()), "p10_return": float(qs[0]), "q1_return": float(qs[1]), "median_return": float(qs[2]), "q3_return": float(qs[3]), "p90_return": float(qs[4]),
                        "return_iqr": float(qs[3] - qs[1]), "worst_return": float(returns.min()), "best_return": float(returns.max()), "positive_seeds": int((returns > 0).sum()), "loss_seeds": int((returns < 0).sum()), "zero_seeds": int((returns == 0).sum()),
                        "median_drawdown": float(np.median(dds)), "p90_drawdown": float(np.quantile(dds, .9, method="linear")), "worst_drawdown": float(dds.max()),
                        "median_fees": float(np.median([r["fees"] for r in group])), "median_turnover": float(np.median([r["turnover"] for r in group])), "median_exposure": float(np.median([r["average_exposure"] for r in group])), "quantile_method": "linear"})
    return result


def paired_results(rows):
    keyed = {(r["symbol"], r["recipe_id"], r["cadence"], r["seed"], r["scenario"]): r for r in rows}
    require(len(keyed) == len(rows), "Duplicate account identity")
    result = []
    for symbol in SYMBOLS:
        for recipe in EXPECTED[symbol]:
            for seed in SEEDS:
                for scenario in SCENARIOS:
                    before = keyed[(symbol, recipe, "fixed", seed, scenario)]
                    for cadence in CADENCES[1:]:
                        after = keyed[(symbol, recipe, cadence, seed, scenario)]
                        result.append({"key": after["account_id"], "symbol": symbol, "recipe_id": recipe, "seed": seed, "scenario": scenario, "cadence": cadence,
                            "fixed_account_id": before["account_id"], "updated_account_id": after["account_id"], "fixed_return": before["net_return"], "updated_return": after["net_return"],
                            "return_difference": after["net_return"] - before["net_return"], "drawdown_difference": abs(after["max_drawdown"]) - abs(before["max_drawdown"]),
                            "fee_difference": after["fees"] - before["fees"], "turnover_difference": after["turnover"] - before["turnover"]})
    return result


def paired_statistics(rows):
    output = []
    for symbol in SYMBOLS:
        for recipe in EXPECTED[symbol]:
            for cadence in CADENCES[1:]:
                for scenario in SCENARIOS:
                    group = [r for r in rows if (r["symbol"], r["recipe_id"], r["cadence"], r["scenario"]) == (symbol, recipe, cadence, scenario)]
                    require(len(group) == 5 and {r["seed"] for r in group} == set(SEEDS), "Complete five-seed paired deltas required")
                    returns = np.asarray([r["return_difference"] for r in group])
                    dds = np.asarray([r["drawdown_difference"] for r in group])
                    output.append({"key": f"{symbol}:{recipe}:{cadence}:{scenario}", "symbol": symbol, "recipe_id": recipe, "cadence": cadence, "scenario": scenario, "seeds": 5,
                        "median_return_difference": float(np.median(returns)), "q1_return_difference": float(np.quantile(returns, .25, method="linear")), "q3_return_difference": float(np.quantile(returns, .75, method="linear")),
                        "median_drawdown_difference": float(np.median(dds)), "higher_return_seeds": int((returns > 0).sum()), "lower_drawdown_seeds": int((dds < 0).sum()), "both_better_seeds": int(((returns > 0) & (dds < 0)).sum()),
                        "median_fee_difference": float(np.median([r["fee_difference"] for r in group]))})
    return output


def verify_protocol_contract(protocol):
    require(protocol.get("selected") == {s: list(r) for s, r in EXPECTED.items()} and protocol.get("seeds") == list(SEEDS), "Exact two-coin five-seed shortlist protocol required")
    require(protocol.get("capital_per_account") == 10000 and protocol.get("capital_shared_between_pairs") is False and protocol.get("ensemble") is False, "Unchanged independent account funding required")
    require(protocol.get("costs") == COSTS and protocol.get("cadences") == {"fixed": None, "every_4w": 28, "every_week": 7}, "Unchanged two-cost three-cadence protocol required")
    require([pd.Timestamp(x) for x in protocol.get("test", [])] == [pd.Timestamp(x) for x in TEST], "Fixed365-day annual protocol required")
    require(protocol.get("expected_models") == MODEL_COUNT and protocol.get("expected_accounts") == ACCOUNT_COUNT and protocol.get("selection_uses_test") is False, "Exact frozen research population and validation-only selection required")
    require(protocol.get("train_days") == 1827 and protocol.get("lag_days") == 1 and protocol.get("preheat_days") == 200, "Physical rolling training window and information lag changed")
    require(protocol.get("environment") == {"observation_dim": 13, "action_dim": 1, "cash_floor": .05, "max_coin_target": .95, "rebalance_band": .025, "drawdown_penalty": .25, "episode_days": 32, "execution": "00UTC sells;01UTC buys;hourly close valuations"}, "Declared spot environment differs from the report method")
    return True


def bind_transport(top, protocol, gate, pins):
    expected = gate.get("external_inputs_sha256")
    require(isinstance(expected, dict) and expected, "Remote market inputs must remain explicitly bound")
    auditor = protocol["sources"]["scripts/audit_rolling_single_pair.py"]
    for path in sorted((top / "transport").glob("*.receipt.json")):
        relative(top, path.relative_to(top).as_posix())
        receipt = read(path)
        if (receipt.get("protocol_sha256") != sha(top / "protocol.json") or receipt.get("completion_gate_sha256") != sha(top / "completion_gate.json")
                or receipt.get("verifier_source_sha256") != auditor or receipt.get("external_inputs_sha256") != expected
                or receipt.get("external_verified_sha256") != expected or receipt.get("external_rechecked_after_archive") is not True):
            continue
        name = Path(str(receipt.get("archive", ""))).name
        if not name or name in (".", "..") or "\\" in name or ":" in name:
            continue
        archive_name = "transport/" + name
        archive = relative(top, archive_name)
        if sha(archive) != receipt.get("archive_sha256"):
            continue
        files = receipt.get("files")
        if (not isinstance(files, dict) or files.get("completion_gate.json") != sha(top / "completion_gate.json")
                or any(files.get(n) != h for n, h in gate["inputs_sha256"].items())):
            continue
        receipt_name = path.relative_to(top).as_posix()
        pins[receipt_name], pins[archive_name] = sha(path), receipt["archive_sha256"]
        return {"receipt": receipt_name, "receipt_sha256": pins[receipt_name], "archive": archive_name,
                "archive_sha256": receipt["archive_sha256"], "external_inputs_sha256": expected,
                "external_verified_sha256": receipt["external_verified_sha256"], "external_rechecked_after_archive": True,
                "verifier_source_sha256": auditor}
    raise ValueError("Successful current transport receipt with before/after remote market verification required")


def verify_completion_gate(top):
    top = root_path(top)
    require((top / "completion_gate.json").is_file(), "Completed rolling accounting and completion_gate.json required before report generation")
    gate = read(top / "completion_gate.json")
    require(gate.get("passed") is True and gate.get("failure_count") == 0 and gate.get("failures") == [], "Independent completion gate must pass without failures")
    for key in ("expected_accounts", "accounts_checked", "repeats_checked"):
        require(gate.get(key) == ACCOUNT_COUNT, "All180 accounts and exact replays must pass: " + key)
    for key in ("expected_models", "models_checked"):
        require(gate.get(key) == MODEL_COUNT, "All1590 actual unique trained models required: " + key)
    protocol = read(relative(top, "protocol.json"))
    verify_protocol_contract(protocol)
    require(gate.get("protocol_sha256") == sha(top / "protocol.json"), "Completed protocol changed")
    require(gate.get("summary_sha256") == sha(relative(top, "summary.csv")), "Completed account summary changed")
    pins = gate.get("inputs_sha256")
    require(isinstance(pins, dict) and pins and {"protocol.json", "protocol_receipt.json", "selection.json", "summary.csv", "accounts_manifest.json", "inventory.json"}.issubset(pins), "Completion must bind all mandatory evidence inputs")
    for name, digest in pins.items():
        require(sha(relative(top, name)) == digest, "Audited input changed: " + name)
    require(isinstance(protocol.get("sources"), dict) and protocol["sources"], "Exact training/auditor source identity required")
    require(gate.get("source_sha256") == protocol["sources"].get("scripts/audit_rolling_single_pair.py") and isinstance(gate.get("source_sha256"), str), "Completion must identify the exact frozen independent auditor")
    sources = {}
    for name, digest in protocol["sources"].items():
        source_name = "source/" + name
        path = relative(top, source_name)
        require(pins.get(source_name) == digest == sha(path), "Missing exact bound protocol source: " + name)
        sources[name] = {"path": str(path), "sha256": digest}
    require(protocol.get("inventory_sha256") == INVENTORY_SHA256 == sha(relative(top, "inventory.json")), "Precommitted experiment inventory changed")
    require(read(top / "protocol_receipt.json").get("protocol_sha256") == sha(top / "protocol.json"), "Frozen protocol receipt changed")
    require(protocol.get("selection_sha256") == sha(top / "selection.json") and read(top / "selection.json").get("uses_test") is False, "Frozen validation-only selection changed")
    return protocol, gate, dict(pins), sources


def validation_rows(inventory):
    rows = []
    for original in inventory["candidates"]:
        base, stress = original["costs"]["base"], original["costs"]["stress"]
        row = {"key": original["symbol"] + ":" + original["candidate_id"], "symbol": original["symbol"], "candidate_id": original["candidate_id"], "algorithm": original["algorithm"],
               "seed_set": ",".join(map(str, original["seed_set"])), "seeds": len(original["seed_set"]), "timesteps": original["timesteps"], "recipe_json": json.dumps(original["recipe"], ensure_ascii=False, sort_keys=True),
               "base_return": base["median_return"], "base_drawdown": base["median_drawdown"], "base_worst_drawdown": base["worst_drawdown"], "base_iqr": base["return_iqr"], "base_positive_seeds": base["positive_accounts"],
               "stress_return": stress["median_return"], "stress_drawdown": stress["median_drawdown"], "stress_p90_drawdown": stress["p90_drawdown"], "robust_score": original["robust_score"], "cashlike": original["cashlike_all_zero_fills"],
               "selected_for_experiment": original["candidate_id"] in [INVENTORY_IDS[x] for x in EXPECTED[original["symbol"]]], "original_acceptance_gates_json": json.dumps(original.get("original_sac_acceptance_gates"), sort_keys=True)}
        rows.append(row)
    require(len(rows) == 18 and len({r["key"] for r in rows}) == 18, "Nine distinct recipes per coin required")
    return rows


def historical_rows(inventory):
    output = []
    for family in inventory["families"]:
        for summary in family.get("summary_sets", []):
            for i, group in enumerate(summary["groups"]):
                d, s = group["dimensions"], group["statistics"]
                output.append({"key": f"{family['study']}:{summary['path']}:{i}", "study": family["study"], "scope": family["scope"], "source_path": summary["path"], "source_sha256": summary["sha256"],
                    "split": d.get("split", "test"), "scenario": d.get("scenario"), "algorithm": d.get("algorithm", d.get("strategy")), "symbol": d.get("symbol", d.get("pair", "共享组合")),
                    "dimensions_json": json.dumps(d, ensure_ascii=False, sort_keys=True), "accounts": s["accounts"], "seed_count": len(s["seeds"]), "capital_values": ",".join(map(str, s["initial_capital_values"])),
                    "median_return": s["median_return"], "median_drawdown": s["median_drawdown"], "worst_drawdown": s["worst_drawdown"], "return_iqr": s["return_iqr"], "positive_accounts": s["positive_accounts"],
                    "periods_json": json.dumps(s["periods"]), "cashlike_zero_fill_accounts": s["zero_fill_accounts"], "descriptive_only_not_selection": True})
    return output


def expanded_seed_rows(inventory):
    result = []
    for source in inventory["expanded_seed_descriptions"]:
        for group in source["groups"]:
            d, s = group["dimensions"], group["statistics"]
            if d.get("algorithm") not in ("sac", "td3"):
                continue
            require(s["accounts"] == 30 and s["seeds"] == list(range(30)), "Historical expanded studies need all30 prespecified seeds")
            result.append({"key": f"{source['study']}:{source['phase']}:{d['algorithm']}:{d['scenario']}", "study": source["study"], "symbol": d["symbol"], "algorithm": d["algorithm"], "split": source["phase"], "scenario": d["scenario"], "seeds": 30,
                "median_return": s["median_return"], "median_drawdown": s["median_drawdown"], "return_iqr": s["return_iqr"], "p10_return": s["p10_return"], "worst_drawdown": s["worst_drawdown"], "positive_seeds": s["positive_accounts"], "source_path": source["path"], "descriptive_only_not_selection": True})
    require(len(result) == 16, "All16 expanded coin/algorithm/cost/split groups required")
    return result


def query_source_metadata(common, key, period):
    """Keep mixed historical scopes distinct from the newly funded accounts."""
    source = deepcopy(common)
    source["period"] = {"start": period[0], "end": period[1]}
    if key in ("historical_results", "expanded_seed_results"):
        source["period"] = {"start": VALIDATION[0], "end": TEST[1]}
        source["caveats"] = [
            "这是已观察的历史验证与年度结果，逐行窗口分别保留，不跨窗口合并收益。",
            "旧固定模型与本轮新训练共同起始模型不同；历史成绩不参与本轮年度选参。",
            "不同seed描述同一行情上的训练随机性，不代表独立市场样本。",
        ]
    if key == "historical_results":
        source["label"] = "各原实验按原资金与原窗口保留的历史结果"
        source["metricDefinitions"] = [
            {"label": "扣费收益", "definition": "按各原账户实际initial_capital计算的期末净值变化；保留其原费用与滑点，不能统一改用10000作分母。"},
            {"label": "最大回撤", "definition": "各原实验按原权益采样粒度计算的相对历史峰值最大跌幅，显示绝对跌幅；原规则策略的粒度与本轮小时估值分别保留。"},
            {"label": "资金与账户身份", "definition": "每行保留原本金、单币或共享组合、算法配置、验证或年度窗口及seed数量；不能把共享组合收益当作独立单币收益。"},
        ]
        source["caveats"].append("原规则单币5000、独立单币10000及共享组合资金分别显示，不能跨资金与标的直接排名。")
    elif key == "expanded_seed_results":
        source["label"] = "原SAC与TD3每币30seed的验证及年度描述"
        source["metricDefinitions"] = [
            {"label": "扣费收益", "definition": "原独立账户期末现金加本币市值/10000−1，保留原费用、滑点及真实期末持仓估值。"},
            {"label": "最大回撤", "definition": "原账户完整小时收盘权益相对含期初10000的历史峰值最大跌幅；显示绝对跌幅。"},
            {"label": "seed分布", "definition": "SAC与TD3每币/算法/费用/窗口完整保留seed0—29，共30个；线性分位数仅描述训练随机性，不是本轮5seed频率账户。"},
        ]
    return source


def make_reviewed_snapshot(study_root):
    top = root_path(study_root)
    protocol, gate, pins, sources = verify_completion_gate(top)
    pins["completion_gate.json"] = sha(top / "completion_gate.json")
    transport = bind_transport(top, protocol, gate, pins)
    inventory = read(top / "inventory.json")
    require(inventory.get("candidate_selection_uses_annual") is False and inventory.get("training_or_inference_performed") is False and inventory.get("source_sha_unchanged_after_read") is True, "Validation-only historical selection evidence required")
    external = dict(inventory["source_sha256"])
    for name, digest in external.items():
        require(sha(relative(ROOT, name)) == digest, "Original experiment source changed: " + name)
    inventory_script = "results/analysis/all_experiments_rolling_20261031/summarize_existing_experiments_v2.py"
    require(sha(relative(ROOT, inventory_script)) == inventory["analysis_script_sha256"], "Historical aggregation source changed")
    external[inventory_script] = inventory["analysis_script_sha256"]
    framework_path = "results/backtests/rl_framework_comparison_20261024/framework_review_rows.json"
    provenance_path = "results/backtests/crypto_foundation_models_v100_20261024/model_provenance_evaluation.json"
    framework = read(relative(ROOT, framework_path))
    provenance = read(relative(ROOT, provenance_path))
    external[framework_path], external[provenance_path] = sha(ROOT / framework_path), sha(ROOT / provenance_path)
    framework_rows = [{k: r[k] for k in ("name", "task", "decision", "native_backtest_status", "commit", "repository") if k in r} for r in framework["rows"]]
    provenance_rows = [{k: r[k] for k in ("model_key", "model", "source_repository", "source_commit", "weights_revision", "weights_sha256", "first_usable_decision_utc", "local_fit", "pretraining_cutoff_independently_verified", "checkpoint_dates_note") if k in r} for r in provenance["rows"]]
    require(len(framework_rows) == 5 and len(provenance_rows) == 3, "Original framework/pretrained provenance evidence is incomplete")
    manifest = read(top / "accounts_manifest.json")
    entries = manifest.get("accounts")
    expected = expected_accounts()
    require(isinstance(entries, list) and len(entries) == ACCOUNT_COUNT and {e["account_id"] for e in entries} == set(expected), "Exact complete180-account manifest required")
    frame = pd.read_csv(top / "summary.csv", float_precision="round_trip")
    require(len(frame) == ACCOUNT_COUNT and not frame.account_id.duplicated().any() and set(frame.account_id) == set(expected), "Exact complete180 account summary required")
    records = frame_rows(frame)
    by_id = {r["account_id"]: r for r in records}
    history, months = [], []
    for entry in entries:
        aid = entry["account_id"]
        symbol, recipe, cadence, seed, scenario = expected[aid]
        row = by_id[aid]
        for obj in (row, entry):
            require((obj["symbol"], obj["recipe_id"], obj["cadence"], obj["seed"], obj["scenario"]) == (symbol, recipe, cadence, seed, scenario), "Account identity changed")
            require(obj["algorithm"] == ALGORITHMS[recipe], "Native algorithm must remain identified")
        require(row["split"] == "test" and row["initial_capital"] == 10000 and row["observed_hours"] == 8760 and row["drawdown_grain"] == "1h_close", "Unchanged independent10000 complete annual account required")
        require(pd.Timestamp(row["start"]) == pd.Timestamp(TEST[0]) and pd.Timestamp(row["end_exclusive"]) == pd.Timestamp(TEST[1]), "Annual account window changed")
        require(all(isinstance(row.get(field), (int, float)) and math.isfinite(row[field]) for field in FINANCIAL_FIELDS), "Complete finite financial fields required")
        require(row["fees"] >= 0 and row["final_cash"] >= -1e-8 and row["final_base_value"] >= -1e-8, "Spot account bounds changed")
        directory = entry["directory"]
        require(directory == row["evaluation_directory"] and directory == f"accounts/{symbol}/{recipe}/{cadence}/evaluation/test/{symbol}/{ALGORITHMS[recipe]}/seed{seed}/{scenario}", "Exact native account evidence directory required")
        files = entry["files_sha256"]
        require(set(RESULT_FILES).issubset(files), "All actual ledger/curve/monthly result files required")
        for name, digest in files.items():
            local = directory + "/" + name
            require(local in pins and pins[local] == digest == sha(relative(top, local)), "Account file is not bound to completed audit")
        ledger_summary = read(top / directory / "summary.json")
        for field in FINANCIAL_FIELDS:
            require(field in ledger_summary and math.isclose(row[field], ledger_summary[field], rel_tol=1e-12, abs_tol=1e-8), "Ledger/summary mismatch: " + field)
        daily = pd.read_csv(top / directory / "daily_equity.csv", float_precision="round_trip")
        require({"timestamp", "equity", "net_return", "drawdown"}.issubset(daily.columns), "Daily boundary data required")
        stamps = pd.DatetimeIndex(pd.to_datetime(daily.timestamp, utc=True))
        require(stamps.equals(pd.date_range(TEST[0], TEST[1], freq="D")) and len(daily) == 366, "Actual initial point and365 daily endpoints required")
        require(np.isfinite(daily[["equity", "net_return", "drawdown"]].to_numpy(dtype=float)).all(), "Missing daily values cannot be interpolated or filled")
        require(math.isclose(daily.iloc[0].equity, 10000, abs_tol=1e-8) and math.isclose(daily.iloc[-1].equity, row["final_equity"], rel_tol=1e-12, abs_tol=1e-8), "Daily curve terminal reconciliation failed")
        common = {"account_id": aid, "symbol": symbol, "recipe_id": recipe, "algorithm": ALGORITHMS[recipe], "cadence": cadence, "seed": seed, "scenario": scenario}
        history.extend({**r, **common} for r in frame_rows(daily))
        monthly = pd.read_csv(top / directory / "monthly_returns.csv", float_precision="round_trip")
        require(list(monthly.month) == list(pd.period_range("2025-10", "2026-10", freq="M").astype(str)) and len(monthly) == 13, "All13 actual partial/calendar months required")
        require(np.isfinite(monthly["return"].to_numpy(dtype=float)).all(), "Missing monthly return cannot become zero")
        require(math.isclose(float(np.prod(1 + monthly["return"])) - 1, row["net_return"], rel_tol=1e-10, abs_tol=1e-10), "Monthly returns do not compound to annual result")
        months.extend({**r, **common} for r in frame_rows(monthly))
    stats = summarize_seeds(records)
    paired = paired_results(records)
    pair_stats = paired_statistics(paired)
    candidates = validation_rows(inventory)
    selections = []
    for symbol in SYMBOLS:
        ranked = inventory["rankings"][symbol]["by_robust_score"][:3]
        require(ranked == [INVENTORY_IDS[r] for r in EXPECTED[symbol]], "Frozen validation-only shortlisted recipes changed")
        for rank, recipe in enumerate(EXPECTED[symbol], 1):
            c = next(r for r in candidates if r["symbol"] == symbol and r["candidate_id"] == INVENTORY_IDS[recipe])
            selections.append({**c, "recipe_id": recipe, "rank": rank, "selection_uses_test": False, "experiment_candidate_not_live_acceptance": True})
    families = [{"study": f["study"], "scope": f["scope"], "local_present": f["local_present"], "summary_sets": len(f.get("summary_sets", [])), "recorded_audits": len(f.get("audits", [])), "independent_account_recipe_source": f.get("eligible_as_independent_recipe_source", False)} for f in inventory["families"]]
    historical = historical_rows(inventory)
    training = []
    for name, digest in pins.items():
        if name.endswith("/rolling_model.json"):
            value = read(top / name)
            context, counters = value["rolling_context"], value["training"]["counters"]
            compact = {"rolling_context": context, "recipe": value["recipe"], "actual_steps": value["training"]["actual_steps"], "counters": counters, "parameter_sha256": value["parameter_sha256"], "checkpoint_files": value["checkpoint_files"]}
            training.append({"model_id": name, "manifest_sha256": digest, "symbol": value["symbol"], "recipe_id": value["recipe_id"], "seed": value["seed"], "cutoff": context["train_end"], "effective_day": context["effective_day"], "actual_steps": compact["actual_steps"], "record_json": json.dumps(compact, ensure_ascii=False, sort_keys=True)})
    require(len(training) == MODEL_COUNT, "All1590 model manifests must be auditable in Sources")
    common_source = {"classification": "modeled", "timezone": "UTC", "label": "完成审计的BTC/ETH独立账户重训频率实验", "period": {"start": TEST[0], "end": TEST[1]},
        "metricDefinitions": [
            {"label": "扣费收益", "definition": "每份真实独立账户期末现金加本币市值/10000−1，逐腿扣模拟手续费与不利滑点；真实期末持仓不强制清算。"},
            {"label": "最大回撤", "definition": "包含期初10000的全部8760小时收盘权益相对历史峰值的最大跌幅；显示绝对跌幅。"},
            {"label": "seed分布", "definition": "每币/配方/更新频率/费用各5个预定seed；线性分位数及同seed频率差值仅描述训练随机性。"},
            {"label": "真实曲线", "definition": "一条线对应同币、同配方、同seed、同费用的一份独立账户，不跨币或跨seed汇总，不把中位数收益当可交易组合。"}],
        "caveats": ["该365天行情此前已被研究观察，本轮不属于新的市场盲测。", "候选只用历史验证期筛选；旧年度成绩仅作历史描述，不参与本轮选择。", "旧Top10与BTC/ETH共享组合不等于本轮各币独立10000账户。", "固定/4周/每周均用新实验共同起始模型；旧固定年度成绩另列历史参考。", "重训频率固定，不按本年度收益临时改变超参数或挑seed。", "新增重训带来更多训练计算量，比较不能被解释为相同总算力下的因果优势。"],
        "executedAt": gate.get("audited_at_utc", gate.get("audited_at", gate.get("at"))), "transportExternalVerification": transport, "evidenceFlow": [{"title": "历史实验分清资金与任务", "detail": "18族历史研究保留资本、标的、seed数量、窗口和来源；只用同独立账户验证结果筛选9个不同配方。"}, {"title": "训练与更新规则预先固定", "detail": "两币各3候选、5seed、三种频率、两档费用；模型只使用各自当时可得数据，资金与库存跨更新连续。"}, {"title": "完成后才生成报告", "detail": "180账户及重复、1590实际唯一模型、金融账本、来源和所有当前文件哈希必须通过completion_gate。"}]}
    queries = {}
    anchors = ["protocol.json", "protocol_receipt.json", "selection.json", "inventory.json", "accounts_manifest.json", "completion_gate.json", transport["receipt"]]
    def query(key, values, files, method, period=TEST, external_files=()):
        source = query_source_metadata(common_source, key, period)
        names = list(dict.fromkeys([*files, *anchors]))
        source.update(files=[*names, *["workspace/" + n for n in external_files]], fileHashes={**{name: pins[name] for name in names}, **{"workspace/" + n: external[n] for n in external_files}})
        queries[key] = {"rows": clean_json(values), "source": source, "methods": [{"language": "text", "code": method}]}
    query("account_summary", records, ["summary.csv"], "180 fully funded actual accounts; exact unique coin/recipe/cadence/seed/cost identities; no pooled funding.")
    query("seed_statistics", stats, ["summary.csv"], "36 cells, exactly5 precommitted seeds each; linear quantiles of actual account outcomes, not median portfolios.")
    query("matched_seed_results", paired, ["summary.csv"], "Updated-minus-fixed outcomes for120 same-label seed/cost comparisons; fixed common initial models, descriptive differences only.")
    query("matched_statistics", pair_stats, ["summary.csv"], "24 cells of five paired differences; positive return delta improves return, negative drawdown delta reduces drawdown.")
    query("equity_history", history, [e["directory"] + "/daily_equity.csv" for e in entries], "180 real accounts×366 daily boundaries including initial10000; all hourly max drawdowns remain in summary.")
    query("monthly_returns", months, [e["directory"] + "/monthly_returns.csv" for e in entries], "180 accounts×13 actual month fragments; product reconciles to each annual after-cost return.")
    query("validation_candidates", candidates, ["inventory.json"], "All9 distinct recipe candidates per coin: native6 seeds0..4 and3 newSAC variants seeds100..104. Same5-size, not paired training seeds. Cashlike all-zero-fill policies excluded from tradable shortlist.", VALIDATION)
    query("candidate_selection", selections, ["inventory.json"], "Top3 tradable recipes by predeclared validation-only robust utility median−.25IQR. A research shortlist does not assert live acceptance; old replacement gates remain unchanged.", VALIDATION)
    query("sac_replication", inventory["sac_baseline_replication"], ["inventory.json"], "SAC default100..104 replication is shown separately; the same hyperparameter recipe cannot occupy another shortlist slot.", VALIDATION)
    query("experiment_families", families, ["inventory.json"], "18 historical families retain explicit capital/universe and recorded-audit status; paths are not proof of rerun success.")
    query("framework_reviews", framework_rows, [], "Five exact preexisting official-code adaptation reviews; native TensorTrade/TradeMaster/EarnHFT/MarketGAN/MacroHFT were not run as comparable backtests.", external_files=(framework_path,))
    query("foundation_provenance", provenance_rows, [], "Three existing pinned pretrained-model provenance records. Public availability evidence does not establish complete pretraining corpus cutoffs; no local fit is claimed.", external_files=(provenance_path,))
    query("historical_results", historical, ["inventory.json"], "Descriptive summaries of previous experiments, with each source path/hash, funding scope, seed count and period retained; no cross-family pooled ranking.")
    query("expanded_seed_results", expanded_seed_rows(inventory), ["inventory.json"], "PriorSAC/TD3 30-seed validation and annual descriptions are retained separately; no annual source is used to select recipes.")
    query("training_records", training, [r["model_id"] for r in training], "1590 actual unique fresh model manifests, including53 cutoffs×5seeds×6 coin/recipe slots;4-week cadence reuses exact scheduled weekly-date models.")
    queries["training_records"]["payloadColumns"] = ["record_json"]
    query("audit_evidence", [{"audit_id": "completion", "passed": gate["passed"], "checks_total": gate["checks_total"], "failure_count": gate["failure_count"], "accounts_checked": gate["accounts_checked"], "repeats_checked": gate["repeats_checked"], "models_checked": gate["models_checked"], "gate_sha256": pins["completion_gate.json"]}], ["completion_gate.json"], "Independent completed gate rehashed against current original sources, protocol and every bound model/account artifact.")
    query("protocol", [{"capital_per_account": 10000, "accounts": ACCOUNT_COUNT, "unique_models": MODEL_COUNT, "seeds": list(SEEDS), "cadences": list(CADENCES), "selected_recipe_slots": 6, "historical_families": len(families), "model_training_contract_json": json.dumps(protocol, ensure_ascii=False, sort_keys=True)}], ["protocol.json"], "Actual frozen research contract; no inference, training, economic simulation or production changes in the report builder.")
    queries["protocol"]["payloadColumns"] = ["model_training_contract_json"]
    counts = {"account_summary": 180, "seed_statistics": 36, "matched_seed_results": 120, "matched_statistics": 24, "equity_history": 65880, "monthly_returns": 2340, "validation_candidates": 18, "candidate_selection": 6, "sac_replication": 2, "expanded_seed_results": 16, "experiment_families": 18, "framework_reviews": 5, "foundation_provenance": 3, "training_records": 1590, "audit_evidence": 1, "protocol": 1}
    require(all(len(queries[k]["rows"]) == v for k, v in counts.items()), "Reviewed rolling report populations are incomplete")
    snapshot = {"surface": "report", "status": "reviewed", "buildStatus": "complete", "title": "BTC、ETH独立交易：全部算法与三种重训频率比较", "generatedAt": datetime.now(timezone.utc).isoformat(), "filters": [], "queries": queries,
        "report": {"asOf": "2026-10-22", "scope": SCOPE, "studyConfig": protocol, "auditStatus": "passed", "noEnsemble": True, "noPooledCapital": True}}
    evidence = {"input_sha256": pins, "external_input_sha256": external, "training_source_sha256": sources, "report_source_sha256": {str(p.relative_to(ROOT).as_posix()): sha(p) for p in (Path(__file__), Path(__file__).with_suffix(".jsx"), Path(__file__).with_suffix(".css"), ROOT / "scripts/btceth_rl_report_app.py", ROOT / "scripts/crypto_rl_reporting.py")}, "query_counts": {k: len(v["rows"]) for k, v in queries.items()}, "pickle_loaded": False, "policy_constructed": False, "financial_results_changed": False, "no_production_change": True}
    verify_inputs(top, evidence)
    return snapshot, evidence


def verify_inputs(top, evidence):
    for name, digest in evidence["input_sha256"].items():
        require(sha(relative(top, name)) == digest, "Input changed while building report: " + name)
    for name, digest in evidence["external_input_sha256"].items():
        require(sha(relative(ROOT, name)) == digest, "Original experiment changed while building report: " + name)
    for name, digest in evidence["report_source_sha256"].items():
        require(sha(relative(ROOT, name)) == digest, "Report source changed during build")


def build_report(study_root, *, output_dir=None, build_mode="native", wsl_node=None, wsl_distro="Ubuntu-26.04", node_path=None, plugin_root=None):
    top = root_path(study_root)
    snapshot, evidence = make_reviewed_snapshot(top)
    destination = root_path(output_dir) if output_dir else top / "report"
    project = destination / "report_app"
    owner = project / ".all-algorithms-rolling-authoring.json"
    previous = read(owner) if owner.is_file() else None
    if project.exists():
        require(previous and previous.get("scope") == SCOPE, "Preserve an unowned report app")
        for name, digest in previous.get("authored_sha256", {}).items():
            require(sha(relative(project, name)) == digest, "Preserve edited presentation")
        for name, digest in previous.get("delivery_sha256", {}).items():
            require(sha(relative(destination, name)) == digest, "Preserve changed delivered evidence")
    elif destination.exists():
        require(not any(p.name != "qa" for p in destination.iterdir()), "Preserve nonempty unowned report destination")
    destination.mkdir(parents=True, exist_ok=True)
    reviewed = destination / "reviewed.json"
    write(reviewed, snapshot)
    write(destination / "report_input_manifest.json", evidence)
    runtime = DataRuntime(node_path=node_path, plugin_root=plugin_root, build_mode=build_mode, wsl_node=wsl_node, wsl_distro=wsl_distro)
    prepared = runtime.run("data-app.mjs", "prepare", "--project-dir", project, cwd=destination) if project.exists() else runtime.run("prepare-data-app.mjs", "--surface", "report", "--output", project, "--snapshot", reviewed, cwd=destination)
    require("src/content/" in (project / "AGENTS.md").read_text(encoding="utf-8"), "Unexpected official authoring boundary")
    docs = prepared["documentation"]
    host_path(docs["entryPoint"]).read_text(encoding="utf-8")
    refs = host_path(docs.get("referenceRoot", str(runtime.plugin / "docs/components")))
    for name in ("narrative/reports.md", "charts/charts.md", "tables/tables.md", "controls/controls.md", "sources/sources.md"):
        (refs / name).read_text(encoding="utf-8")
    content = project / "src/content/report"
    targets = {Path(__file__).with_suffix(".jsx"): content / "ReportContent.jsx", Path(__file__).with_suffix(".css"): content / "all-algorithms-rolling.css"}
    for source, target in targets.items():
        shutil.copyfile(source, target)
    app = read(project / "src/data.json")
    snapshot["id"] = app["id"]
    if "legacyPresentationTitle" in app:
        snapshot["legacyPresentationTitle"] = app["legacyPresentationTitle"]
    write(project / "src/data.json", snapshot)
    authored = {p.relative_to(project).as_posix(): sha(p) for p in targets.values()}
    write(owner, {"scope": SCOPE, "authored_sha256": authored})
    built = runtime.run("data-app.mjs", "build", "--project-dir", project, "--separate-data", cwd=destination)
    exported = project / ".data-app-offline/exports/all-algorithms-rolling.html"
    exported_result = runtime.run("data-app.mjs", "export-html", "--project-dir", project, "--output", exported, cwd=destination)
    verify_inputs(top, evidence)
    delivered = destination / "report.html"
    shutil.copyfile(exported, delivered)
    result = {"scope": SCOPE, "html_path": str(delivered), "preview_path": str(project / "dist/index.html"), "project_dir": str(project), "reviewed_sha256": sha(reviewed), "html_sha256": sha(delivered), "runtime": built, "export": exported_result, "build_mode": build_mode, "published": False, "query_counts": evidence["query_counts"], "report_source_sha256": evidence["report_source_sha256"], "financial_results_changed": False}
    write(destination / "report_build.json", result)
    write(owner, {"scope": SCOPE, "authored_sha256": authored, "delivery_sha256": {p.name: sha(p) for p in (reviewed, destination / "report_input_manifest.json", delivered, destination / "report_build.json")}})
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
    print(json.dumps(build_report(args.study_root, output_dir=args.output_dir, build_mode=args.build_mode, wsl_node=args.wsl_node, wsl_distro=args.wsl_distro, node_path=args.node_path, plugin_root=args.plugin_root), ensure_ascii=False))


if __name__ == "__main__":
    main()
