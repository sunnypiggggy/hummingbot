"""Optional independently checked risk-control queries; no HTML or inference.

This reader opens annual prices only after completed six-case evidence and
the original test-dispatch contracts pass their SHA/time guards. It never
imports a goal selector, environment, model, or portfolio execution engine.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import importlib.util
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.collect_btceth_rl_results import inventory, safe_target, sha, read_json

CONTROLS = ("btc25_cash75_hold", "btceth25_cash75_hold", "vol20_inverse_vol30")
NAMES = {"btc25_cash75_hold": "25% BTC初始仓位持有", "btceth25_cash75_hold": "25% BTC/ETH等权初始仓位持有", "vol20_inverse_vol30": "20%目标波动率／30日逆波动分配"}
COSTS = {"base": {"fee": .001, "slippage": .0002}, "stress": {"fee": .002, "slippage": .0005}}
TEST = ("2025-10-23T00:00:00Z", "2026-10-23T00:00:00Z")
GATE_FILES = ("frozen_spec.json", "test_started.json", "validation_selection.json", "goal_candidate_protocol.json", "goal_candidate_selection.json", "goal_candidate_selection_receipt.json")
FINANCIAL = ("scripts/backtest_btceth_rl.py", "scripts/btceth_rl_env.py", "scripts/crypto_rl_portfolio_env.py", "scripts/crypto_rl_reporting.py")


def utc(value):
    value = pd.Timestamp(value)
    if value.tzinfo is None:
        raise ValueError("Risk evidence timestamps need an explicit timezone")
    return value.tz_convert("UTC")


def verify_pins(root, values):
    if not isinstance(values, dict) or not values:
        raise ValueError("Risk evidence needs nonempty SHA sources")
    for name, expected in values.items():
        if sha(safe_target(root, name)) != expected:
            raise ValueError("Risk evidence SHA changed: " + name)


def risk_manifest(target):
    """The run manifest excludes itself; every other test file is exact."""
    target = Path(target)
    manifest = read_json(safe_target(target, "file_manifest.json"))
    if not isinstance(manifest, list):
        raise ValueError("Risk test manifest must list the original file bytes")
    names, entries = set(), []
    for item in manifest:
        name = item["path"]
        safe_target(target, name)
        if name in names or name == "file_manifest.json" or type(item.get("size")) is not int:
            raise ValueError("Duplicate or malformed risk manifest")
        names.add(name); entries.append(dict(path=name, size=item["size"], sha256=item["sha256"]))
    actual = [f for f in inventory(target) if f["path"] not in ("file_manifest.json", "independent_audit.local.json", "transfer_verification.local.json")]
    if sorted(entries, key=lambda i: i["path"]) != actual:
        raise ValueError("Risk test files differ from the original manifest")
    return {i["path"]: i["sha256"] for i in entries}


def verify_adapter_provenance(reference, main, source_root=ROOT):
    """Verify shape-only adapter sidecars; never import or execute the adapter."""
    directory = Path(reference) / "adapter_attempts/compat_20261030"
    p, prep, execution = (read_json(safe_target(directory, n)) for n in ("protocol.json", "preparation.json", "execution.json"))
    marker_sha = sha(safe_target(main, "test_started.json"))
    expected_sources = {"scripts/btceth_rl_risk_controls.py", "scripts/btceth_rl_goal_candidate.py", "scripts/btceth_rl_risk_controls_adapter.py"}
    if (p.get("financial_rules_changed") is not False or p.get("annual_data_decoded_during_preparation") is not False
            or set(p.get("source_sha256", {})) != expected_sources
            or p.get("original_risk_sha256") != p["source_sha256"]["scripts/btceth_rl_risk_controls.py"]
            or p.get("original_goal_sha256") != p["source_sha256"]["scripts/btceth_rl_goal_candidate.py"]
            or p.get("target_absent_at_preparation") is not True or p.get("main_marker_sha256") != marker_sha
            or p.get("risk_protocol_sha256") != sha(Path(reference) / "test_protocol.json")
            or p.get("risk_receipt_sha256") != sha(Path(reference) / "test_protocol_receipt.json")
            or prep.get("passed") is not True or prep.get("test_data_loaded") is not False
            or prep.get("protocol_sha256") != sha(directory / "protocol.json")
            or execution.get("passed") is not True or execution.get("exit_code") != 0
            or execution.get("protocol_sha256") != sha(directory / "protocol.json")
            or execution.get("main_marker_sha256") != marker_sha
            or execution.get("original_validator_restored") is not True or execution.get("original_validation_unchanged") is not True
            or execution.get("cuda_initialized") is not False or execution.get("source_sha256") != p.get("source_sha256")
            or execution.get("result", {}).get("passed") is not True
            or execution.get("result", {}).get("formal_rankings_modified") is not False
            or execution.get("result", {}).get("cases") != 6 or execution.get("result", {}).get("full_repeats") != 6):
        raise ValueError("Risk adapter provenance must preserve original gates and all financial rules")
    verify_pins(source_root, p["source_sha256"])
    verify_pins(directory / "sources", p["source_sha256"])
    if (sha(directory / "original_failure/run.log") != p.get("failed_log_sha256")
            or sha(directory / "original_failure/process.json") != p.get("failed_process_sha256")):
        raise ValueError("Original pre-price adapter failure bytes were not preserved")
    values = {safe_target(directory, item["path"]).resolve().as_posix(): item["sha256"] for item in inventory(directory)}
    for item in p["original_reference_files"]:
        path = safe_target(reference, item["path"])
        if path.stat().st_size != item["size"] or sha(path) != item["sha256"]:
            raise ValueError("Risk adapter modified original reference bytes: " + item["path"])
        values[path.resolve().as_posix()] = item["sha256"]
    values.update({safe_target(source_root, name).resolve().as_posix(): digest for name, digest in p["source_sha256"].items()})
    return values


def verify_contracts(reference, dataset, main, cap_output=None, source_root=ROOT):
    reference, dataset, main = map(Path, (reference, dataset, main))
    if not main.is_dir():
        raise ValueError("Completed original main study is required for risk sources")
    target = reference / "test"
    completed = read_json(safe_target(target, "completion.json"))
    if (completed.get("passed") is not True or completed.get("cases") != 6 or completed.get("full_repeats") != 6
            or completed.get("hours_reconstructed") != 6 * 8760 or completed.get("failures") != 0
            or completed.get("independent_checks", 0) <= 0 or completed.get("formal_rankings_modified") is not False):
        raise ValueError("Require all six separate risk tests, repeats and accounting checks")
    pins = risk_manifest(target)
    protocol, receipt, dispatch = (read_json(safe_target(reference, n)) for n in ("test_protocol.json", "test_protocol_receipt.json", "test/test_dispatch.json"))
    protocol_sha = sha(safe_target(reference, "test_protocol.json"))
    if (receipt.get("protocol_file") != "test_protocol.json" or receipt.get("protocol_sha256") != protocol_sha
            or receipt.get("entrypoint_sha256") != protocol.get("entrypoint_sha256")
            or receipt.get("original_frozen_spec_sha256") != protocol.get("original_frozen_spec_sha256")
            or protocol.get("capital") != 10000. or protocol.get("controls") != list(CONTROLS)
            or protocol.get("costs") != COSTS or protocol.get("dates_utc", {}).get("test") != list(TEST)
            or dispatch.get("passed") is not True or dispatch.get("test_pnl_read") is not False
            or dispatch.get("holdout_protocol_sha256") != protocol_sha
            or dispatch.get("holdout_protocol_receipt_sha256") != sha(safe_target(reference, "test_protocol_receipt.json"))
            or completed.get("holdout_protocol_sha256") != protocol_sha
            or completed.get("main_contract_sha256") != dispatch.get("main_contract_sha256")
            or set(dispatch.get("main_contract_sha256", {})) != set(GATE_FILES)):
        raise ValueError("Risk protocol, receipt, dispatch or original gates differ")
    verify_pins(main, dispatch["main_contract_sha256"])
    verify_pins(reference, protocol["validation_source_sha256"])
    verify_pins(reference / "holdout_sources", protocol["snapshot_sha256"])
    verify_pins(source_root, protocol["financial_code_sha256"])
    if (set(protocol["financial_code_sha256"]) != set(FINANCIAL)
            or sha(safe_target(source_root, "scripts/btceth_rl_risk_controls.py")) != protocol["entrypoint_sha256"]
            or sha(dataset) != protocol["dataset_sha256"] or sha(safe_target(dataset.parent, "manifest.json")) != protocol["dataset_manifest_sha256"]
            or sha(safe_target(main, "frozen_spec.json")) != protocol["original_frozen_spec_sha256"]):
        raise ValueError("Risk financial source, data or original study changed")
    original, marker, goal_selection, goal_receipt, goal_protocol = (read_json(safe_target(main, n)) for n in ("validation_selection.json", "test_started.json", "goal_candidate_selection.json", "goal_candidate_selection_receipt.json", "goal_candidate_protocol.json"))
    if (marker.get("validation_selection_sha256") != sha(safe_target(main, "validation_selection.json"))
            or marker.get("frozen_spec_sha256") != protocol["original_frozen_spec_sha256"]
            or goal_receipt.get("selection_sha256") != sha(safe_target(main, "goal_candidate_selection.json"))
            or goal_receipt.get("declared_protocol_sha256") != sha(safe_target(main, "goal_candidate_protocol.json"))
            or goal_selection.get("original_validation_selection_sha256") != marker["validation_selection_sha256"]
            or not utc(protocol["prepared_at"]) <= utc(receipt["sealed_at"]) < utc(marker["at"])
            or not utc(goal_protocol["declared_at"]) <= utc(goal_selection["selected_at"]) <= utc(goal_receipt["sealed_at"]) < utc(marker["at"])
            or utc(original["frozen_at"]) > utc(marker["at"]) or utc(dispatch["main_test_started_at"]) != utc(marker["at"])):
        raise ValueError("Risk source choices must have been sealed before the main test")
    verify_pins(source_root, goal_protocol["source_sha256"])
    # Compare the frozen byte maps, without platform-sensitive score recompute.
    for field in ("training_source_sha256", "validation_source_sha256"):
        verify_pins(main, original[field])
        if goal_selection.get(field) != original[field]:
            raise ValueError("Original new-model selection source differs")
    if cap_output is not None:
        cap_output = Path(cap_output)
        selection, seal = read_json(safe_target(cap_output, "selection.json")), read_json(safe_target(cap_output, "selection_receipt.json"))
        if (seal.get("selection_sha256") != sha(safe_target(cap_output, "selection.json"))
                or seal.get("protocol_sha256") != sha(safe_target(cap_output, "protocol.json"))
                or len(selection.get("candidates", [])) != 8 or selection.get("selection_uses_test") is not False
                or not utc(selection["selected_at"]) <= utc(seal["sealed_at"]) < utc(marker["at"])):
            raise ValueError("The eight-candidate extension was not sealed before main test")
    main_audit_name = "independent_audit.local.json" if (main / "independent_audit.local.json").is_file() else "independent_audit.json"
    audit = read_json(safe_target(main, main_audit_name))
    if (audit.get("passed") is not True or audit.get("status") != "passed" or audit.get("test_complete") is not True
            or audit.get("run_count") != 172 or audit.get("failures") != []):
        raise ValueError("Risk report needs the completed original172 accounting audit")
    verify_pins(main, audit["input_sha256"])
    adapter_inputs = verify_adapter_provenance(reference, main, source_root)
    external = {dataset.resolve().as_posix(): protocol["dataset_sha256"],
                (dataset.parent / "manifest.json").resolve().as_posix(): protocol["dataset_manifest_sha256"],
                (Path(__file__).resolve()).as_posix(): sha(Path(__file__)),
                (reference / "test_protocol.json").resolve().as_posix(): protocol_sha,
                (reference / "test_protocol_receipt.json").resolve().as_posix(): sha(reference / "test_protocol_receipt.json"),
                (target / "file_manifest.json").resolve().as_posix(): sha(target / "file_manifest.json"),
                (main / main_audit_name).resolve().as_posix(): sha(main / main_audit_name)}
    for root, values in ((target, pins), (reference, protocol["validation_source_sha256"]),
                         (reference / "holdout_sources", protocol["snapshot_sha256"]),
                         (Path(source_root), protocol["financial_code_sha256"]),
                         (Path(source_root), goal_protocol["source_sha256"]),
                         (main, dispatch["main_contract_sha256"]), (main, audit["input_sha256"]),
                         (main, original["training_source_sha256"]), (main, original["validation_source_sha256"])):
        external.update({safe_target(root, n).resolve().as_posix(): value for n, value in values.items()})
    external[safe_target(source_root, "scripts/btceth_rl_risk_controls.py").resolve().as_posix()] = protocol["entrypoint_sha256"]
    external.update(adapter_inputs)
    if cap_output is not None:
        for name in ("selection.json", "selection_receipt.json", "protocol.json"):
            external[safe_target(cap_output, name).resolve().as_posix()] = sha(safe_target(cap_output, name))
    return dict(protocol=protocol, completion=completed, file_hashes=pins,
        external_input_sha256=external,
        main_audit_name=main_audit_name, main_audit_sha256=sha(main / main_audit_name), main_audit_inputs=audit["input_sha256"],
        contract_hashes=dispatch["main_contract_sha256"], protocol_sha256=protocol_sha,
        protocol_receipt_sha256=sha(reference / "test_protocol_receipt.json"), manifest_sha256=sha(target / "file_manifest.json"))


def independent_vol_targets(data, days):
    """Independent causal covariance arithmetic, used only as an audit."""
    closing = data["hourly_close"]
    valid = pd.DataFrame(True, index=closing.index, columns=closing.columns)
    for field in ("open", "high", "low", "close", "volume"):
        frame = data["hourly_" + field]
        valid &= np.isfinite(frame) & (frame.ge(0) if field == "volume" else frame.gt(0))
    complete = valid.resample("D").sum().eq(24).all(axis=1)
    closes = closing.loc[closing.index.hour == 23, ["BTCUSDT", "ETHUSDT"]].copy()
    closes.index = closes.index.normalize()
    rows = []
    for day in days:
        lookback = pd.date_range(day - pd.Timedelta(days=31), day, freq="D", inclusive="left")
        if not complete.reindex(lookback, fill_value=False).all():
            raise ValueError("Risk covariance needs31 preceding complete days; no fill or interpolation")
        values = closes.reindex(lookback).to_numpy(dtype=float)
        returns = values[1:] / values[:-1] - 1
        centered = returns - returns.mean(axis=0)
        c00, c01, c11 = (float(np.sum(centered[:, i] * centered[:, j]) * 365 / 29) for i, j in ((0, 0), (0, 1), (1, 1)))
        if not np.isfinite([c00, c01, c11]).all() or min(c00, c11) <= 0:
            raise ValueError("Risk covariance requires finite nonzero observed variance")
        inverse = 1 / np.sqrt([c00, c11]); weights = inverse / inverse.sum()
        risk = float(np.sqrt(weights[0] ** 2 * c00 + 2 * weights[0] * weights[1] * c01 + weights[1] ** 2 * c11))
        if not np.isfinite(risk) or risk <= 0:
            raise ValueError("Observed portfolio volatility must be positive")
        gross = min(.95, .20 / risk)
        rows.append([weights[0] * gross, weights[1] * gross, 1 - gross])
    return np.asarray(rows)


def load_auditor(path):
    name = "_risk_report_audit_" + sha(path)[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous_bytecode_setting = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
    return module


def verify_curve_months(history, monthly, row, rebuilt):
    curve = history.loc[history.run_id.eq(row.run_id)]
    days = pd.DatetimeIndex(pd.to_datetime(curve.timestamp, utc=True))
    expected = pd.date_range(TEST[0], TEST[1], freq="D")
    daily = rebuilt["equity"].equity.resample("D").last()
    nav = np.r_[10000., daily.to_numpy()]
    dd = np.r_[0., rebuilt["drawdown"].resample("D").last().to_numpy()]
    if (not days.equals(expected) or not np.allclose(curve.equity, nav, atol=1e-6, rtol=1e-10)
            or not np.allclose(curve.net_return, nav / 10000 - 1, atol=1e-10, rtol=1e-10)
            or not np.allclose(curve.drawdown, dd, atol=1e-10, rtol=1e-10)):
        raise ValueError("Risk display curve differs from independently reconstructed ledger")
    month = monthly.loc[monthly.run_id.eq(row.run_id)]
    if month.month.duplicated().any() or set(month.month) != set(rebuilt["monthly"]):
        raise ValueError("Risk monthly coverage differs from the actual365 days")
    for name, value in month[["month", "return"]].itertuples(index=False, name=None):
        if not np.isclose(value, rebuilt["monthly"][name], atol=1e-10, rtol=1e-10):
            raise ValueError("Risk monthly return differs from ledger")
    if not np.isclose(np.prod(1 + month["return"]) - 1, row.net_return, atol=1e-10, rtol=1e-10):
        raise ValueError("Risk monthly product differs from final net return")


def json_rows(frame):
    return json.loads(frame.to_json(orient="records", date_format="iso", double_precision=15))


def load_risk_controls(reference_root, *, dataset_path, main_output=None, cap_output=None, source_root=ROOT):
    reference, dataset = Path(reference_root).resolve(), Path(dataset_path).resolve()
    if main_output is None:
        raise ValueError("Risk queries require an explicit completed main_output")
    proof = verify_contracts(reference, dataset, main_output, cap_output, source_root)
    target = reference / "test"
    summary, history, monthly = [pd.read_csv(safe_target(target, name), float_precision="round_trip") for name in ("summary.csv", "daily_equity.csv", "monthly_returns.csv")]
    expected = {f"test__{scenario}__{control}" for scenario in COSTS for control in CONTROLS}
    if len(summary) != 6 or summary.run_id.duplicated().any() or set(summary.run_id) != expected:
        raise ValueError("Exactly six separate risk-budget tests are required")
    if set(history.run_id) != expected or set(monthly.run_id) != expected:
        raise ValueError("Risk curves and months need all six run identities")
    for row in summary.itertuples(index=False):
        if (row.run_id != f"test__{row.scenario}__{row.strategy}" or row.strategy not in CONTROLS or row.scenario not in COSTS
                or row.initial_capital != 10000. or row.split != "test" or utc(row.start) != utc(TEST[0]) or utc(row.end_exclusive) != utc(TEST[1])
                or row.observed_hours != 8760 or row.drawdown_grain != "1h_close"
                or not isinstance(row.formal_baseline, (bool, np.bool_)) or bool(row.formal_baseline)
                or not isinstance(row.repeat_exact, (bool, np.bool_)) or not bool(row.repeat_exact)):
            raise ValueError("Risk strategy/window/capital/repeat identity changed")
    # No price deserialization occurs until all file/gate/source checks above.
    with gzip.open(dataset, "rb") as stream:
        data = pickle.load(stream)
    data = {k: (v.loc[v.index < utc(TEST[1])].copy() if k.startswith("hourly_") else v) for k, v in data.items()}
    auditor = load_auditor(reference / "holdout_sources/independent_audit_source_snapshot.py")
    decision_days = pd.date_range(*TEST, freq="D", inclusive="left")
    volatility_targets = independent_vol_targets(data, decision_days)
    total_checks, checksums = 0, {}
    for row in summary.itertuples(index=False):
        directory = target / row.scenario / row.strategy
        checks = auditor.Checks()
        result = auditor.reconstruct_fills(data, auditor.read_csv(directory / "fills.csv"), auditor.read_equity(directory / "equity_1h.csv.gz"),
            auditor.read_json(directory / "metadata.json"), start=TEST[0], end=TEST[1], checks=checks, label=row.run_id, **COSTS[row.scenario])
        auditor.verify_saved_fingerprints(directory, checks, row.run_id)
        if result is None or checks.failed:
            raise ValueError("Independent risk fills reconstruction failed: " + repr(checks.failures))
        saved = read_json(directory / "independent_reconstruction.json")
        if saved.get("passed") is not True or saved.get("failed") != 0 or saved.get("hours") != 8760:
            raise ValueError("Saved independent risk proof is incomplete")
        for key in ("net_return", "max_drawdown", "fees", "slippage_cost", "average_exposure", "turnover", "net_liquidation_return", "final_equity"):
            value = result["metrics"][key]
            if not np.isclose(getattr(row, key), value, atol=1e-8, rtol=1e-10) or not np.isclose(saved["metrics"][key], value, atol=1e-8, rtol=1e-10):
                raise ValueError("Risk reported metrics differ from independently replayed fills: " + key)
        verify_curve_months(history, monthly, row, result)
        trace = pd.read_csv(directory / "decision_targets_actual_weights.csv", float_precision="round_trip")
        if not pd.DatetimeIndex(pd.to_datetime(trace.date, utc=True)).equals(decision_days) or not trace.control.eq(row.strategy).all():
            raise ValueError("Risk decision trace must cover every exact test day")
        targets = trace[["target_btc", "target_eth", "target_cash"]].to_numpy(dtype=float)
        if row.strategy == "vol20_inverse_vol30":
            if not np.allclose(targets, volatility_targets, atol=1e-10, rtol=1e-10):
                raise ValueError("Risk volatility targets differ from independent causal covariance")
        else:
            first = [.25, 0, .75] if row.strategy == "btc25_cash75_hold" else [.125, .125, .75]
            if not np.allclose(targets[0], first, atol=1e-15) or not np.isnan(targets[1:]).all() or not trace.action_kind.iloc[1:].eq("hold_unchanged_units").all():
                raise ValueError("Low-position reference must retain its original inventory after first entry")
        daily_nav = result["equity"].equity.resample("D").last().to_numpy()
        quantity = result["quantity"].resample("D").last().to_numpy()
        closing = data["hourly_close"].loc[decision_days + pd.Timedelta(hours=23), ["BTCUSDT", "ETHUSDT"]].to_numpy()
        actual_after = np.column_stack([quantity * closing / daily_nav[:, None], result["equity"].cash.resample("D").last().to_numpy() / daily_nav])
        actual_before = np.vstack([[0., 0., 1.], actual_after[:-1]])
        if (not np.allclose(trace[["actual_btc_before", "actual_eth_before", "actual_cash_before"]], actual_before, atol=1e-10, rtol=1e-10)
                or not np.allclose(trace[["actual_btc_after", "actual_eth_after", "actual_cash_after"]], actual_after, atol=1e-10, rtol=1e-10)):
            raise ValueError("Risk decision observations differ from actual reconstructed holdings")
        expected_rebalance = np.max(np.abs(targets - actual_before), axis=1) >= .025
        if not np.array_equal(trace.rebalanced.to_numpy(dtype=bool), expected_rebalance):
            raise ValueError("Risk rebalance decisions differ from the frozen2.5pp band")
        weights = pd.read_csv(directory / "weights.csv", float_precision="round_trip")
        weights["date"] = pd.to_datetime(weights.date, utc=True)
        for column, values in (("actual_weight", actual_after[:, :2]), ("target_weight", targets[:, :2])):
            pivot = weights.pivot(index="date", columns="asset_key", values=column)
            if (set(pivot.columns) != {"BTCUSDT", "ETHUSDT"} or not pivot.index.equals(decision_days)
                    or not np.allclose(pivot[["BTCUSDT", "ETHUSDT"]], values, atol=1e-10, rtol=1e-10, equal_nan=True)):
                raise ValueError("Risk stored target/actual weights differ from the same funded ledger")
        total_checks += checks.count
        checksums[row.run_id] = dict(checks=checks.count, hours=8760, reconstructed=True)
    if risk_manifest(target) != proof["file_hashes"]:
        raise ValueError("Risk evidence changed while constructing report queries")
    verify_pins(main_output, proof["main_audit_inputs"])
    verify_pins(main_output, proof["contract_hashes"])
    if sha(dataset) != proof["protocol"]["dataset_sha256"] or sha(Path(main_output) / proof["main_audit_name"]) != proof["main_audit_sha256"]:
        raise ValueError("Original dataset or final main audit changed during risk inspection")
    for name, expected_sha in proof["external_input_sha256"].items():
        if sha(Path(name)) != expected_sha:
            raise ValueError("External risk source changed during inspection: " + name)
    for frame in (summary, history, monthly):
        frame["name"] = frame.strategy.map(NAMES)
        frame["scope"] = "independent_risk_budget_reference"
    summary["display_return_percent"], summary["display_drawdown_percent"] = summary.net_return * 100, summary.max_drawdown * 100
    history["display_return_percent"] = history.net_return * 100
    monthly["display_return_percent"] = monthly["return"] * 100
    stamp = datetime.now(timezone.utc).isoformat()
    source = dict(label="独立风险预算附加对照：逐笔重建核验", classification="modeled", timezone="UTC", period={"start": TEST[0], "endExclusive": TEST[1]},
        executedAt=stamp, files=[(reference / n).resolve().as_posix() for n in ("test_protocol.json", "test_protocol_receipt.json", "test/file_manifest.json", "test/completion.json")],
        caveats=["独立附加对照，不加入原172例或扩展36例模型排名。", "25%为初始仓位，持有期间随价格漂移。20%为历史目标波动率，不是实际风险保证。", "每日00UTC减仓、01UTC买入；扣每腿费用与不利滑点，最大回撤使用全小时权益。", "年度窗口已被此前研究看过，不能称为新盲测。"],
        fileHashes=proof["external_input_sha256"])
    queries = {}
    for identity, frame, filename in (("risk_budget_summary", summary, "summary.csv"), ("risk_budget_equity", history, "daily_equity.csv"), ("risk_budget_monthly", monthly, "monthly_returns.csv")):
        evidence = dict(source); evidence["files"] = [*source["files"], (target / filename).resolve().as_posix()]
        queries[identity] = dict(rows=json_rows(frame), source=evidence,
            methods=[{"language": "text", "code": "Require six completed immutable test cases and repeats; verify source/data/dispatch/receipt SHA and chronological boundaries; independently replay exported fills at every hour and recompute net return, drawdown, costs, causal covariance targets, daily display curve and monthly compound returns. No model/selector/environment import or inference."}])
    return dict(queries=queries, source=source, external_input_sha256=proof["external_input_sha256"], audit=dict(passed=True, cases=6, repeats=6, full_repeats=6, independent_checks=total_checks, runs=checksums,
        formal_rankings_modified=False, protocol_sha256=proof["protocol_sha256"], manifest_sha256=proof["manifest_sha256"],
        main_audit_sha256=proof["main_audit_sha256"], source_sha256=sha(Path(__file__))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--risk-reference-root", type=Path, required=True)
    parser.add_argument("--main-output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cap-output", type=Path)
    args = parser.parse_args()
    result = load_risk_controls(args.risk_reference_root, dataset_path=args.dataset, main_output=args.main_output, cap_output=args.cap_output)
    print(json.dumps(result["audit"], ensure_ascii=False))
