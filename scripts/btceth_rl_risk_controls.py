"""Prepare and gate separate, fixed BTC/ETH risk-budget holdout controls.

Preparation hashes data without loading prices. Test evaluation is unavailable
until both validation selections are sealed before the original test dispatch.
The completed validation runner and its financial sources are never rewritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

REFERENCE = ROOT / "results/backtests/btceth_rl_pretest_reference_20261029/risk_budget_controls"
MAIN = ROOT / "results/backtests/btceth_rl_20261029"
DATASET = ROOT / "data/crypto_rl_btceth_20261029/dataset.pkl.gz"
TRAIN = ("2020-01-01T00:00:00Z", "2025-01-01T00:00:00Z")
VALIDATION = (TRAIN[1], "2025-10-23T00:00:00Z")
TEST = (VALIDATION[1], "2026-10-23T00:00:00Z")
CONTROLS = ("btc25_cash75_hold", "btceth25_cash75_hold", "vol20_inverse_vol30")
COSTS = {"base": {"fee": .001, "slippage": .0002},
         "stress": {"fee": .002, "slippage": .0005}}
FINANCIAL = ("scripts/backtest_btceth_rl.py", "scripts/btceth_rl_env.py",
             "scripts/crypto_rl_portfolio_env.py", "scripts/crypto_rl_reporting.py")
REQUIRED_GATE = ("frozen_spec.json", "test_started.json", "validation_selection.json",
                 "goal_candidate_protocol.json", "goal_candidate_selection.json",
                 "goal_candidate_selection_receipt.json")


class GateClosed(ValueError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def stamp(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise GateClosed("Contract timestamps must have an explicit UTC offset")
    return result.astimezone(timezone.utc)


def sha(path):
    path = Path(path)
    if path.is_symlink():
        raise GateClosed("Pinned files cannot be symbolic links")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def json_default(value):
    if hasattr(value, "dtype") and hasattr(value, "item"):
        return value.item()
    raise TypeError("Unsupported evidence value: " + type(value).__name__)


def write_new(path, value):
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=json_default) + "\n"
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def relative(root, name):
    text = str(name).replace("\\", "/")
    parts = PurePosixPath(text)
    if not text or parts.is_absolute() or ":" in text or ".." in parts.parts:
        raise GateClosed("Pinned file paths must stay relative to their declared root")
    root = Path(root).resolve()
    p = root
    for part in parts.parts:
        p = p / part
        if p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction()):
            raise GateClosed("Pinned file paths cannot cross links or junctions")
    p.resolve().relative_to(root)
    return p


def prepare(reference=REFERENCE, main=MAIN, dataset=DATASET, *, source_root=ROOT):
    """No pickle, market-data frame, or test signal is loaded in this phase."""
    reference, main, dataset, source_root = map(Path, (reference, main, dataset, source_root))
    target = reference / "test_protocol.json"
    receipt_path = reference / "test_protocol_receipt.json"
    sources = reference / "holdout_sources"
    if target.exists() or receipt_path.exists() or sources.exists():
        raise FileExistsError("Preserve the previously prepared holdout protocol and byte snapshots")
    if (main / "test_started.json").exists() or (reference / "test").exists():
        raise GateClosed("Risk-control holdout must be declared before any test dispatch")
    protocol = read_json(reference / "protocol.json")
    completion = read_json(reference / "completion.json")
    spec = read_json(main / "frozen_spec.json")
    if (protocol["controls"] != list(CONTROLS) or protocol["costs"] != COSTS
            or protocol["capital"] != 10000 or protocol["initial_targets"] != {
                "btc25_cash75_hold": [.25, 0, .75], "btceth25_cash75_hold": [.125, .125, .75]}
            or completion.get("passed") is not True or completion.get("cases") != 6
            or completion.get("failures") != 0):
        raise GateClosed("Require the unchanged, completed six-case validation controls")
    if (tuple(spec["train"]) != TRAIN or tuple(spec["validation"]) != VALIDATION
            or tuple(spec["test"]) != TEST or spec["costs"] != COSTS
            or set(spec["code_sha256"]) != set(FINANCIAL)):
        raise GateClosed("Original financial protocol or chronological split changed")
    if sha(dataset) != spec["dataset_sha256"] or sha(dataset.parent / "manifest.json") != spec["dataset_manifest_sha256"]:
        raise GateClosed("Dataset bytes do not match the original frozen experiment")
    snapshots = {
        "btceth_rl_risk_controls.py": Path(__file__).resolve(),
        "run_risk_budget_controls.py": reference / "run_risk_budget_controls.py",
        "independent_audit_source_snapshot.py": reference / "independent_audit_source_snapshot.py",
        "validation_protocol.json": reference / "protocol.json",
    }
    for name in FINANCIAL:
        path = relative(source_root, name)
        if sha(path) != spec["code_sha256"][name]:
            raise GateClosed("Frozen financial source changed: " + name)
        snapshots["financial_sources/" + Path(name).name] = path
    snapshot_hashes = {name: sha(path) for name, path in snapshots.items()}
    pins = {name: sha(reference / name) for name in (
        "run_risk_budget_controls.py", "independent_audit_source_snapshot.py", "protocol.json",
        "completion.json", "serialized_verification.json", "summary.csv", "monthly_returns.csv")}
    result = {
        "schema_version": 1, "prepared_at": now(), "scope": "Independent fixed risk-budget holdout references; never a formal RL baseline or ranking input",
        "phase": "prepared_only", "test_evaluations_at_preparation": 0,
        "dates_utc": {"train": list(TRAIN), "validation": list(VALIDATION), "test": list(TEST)},
        "dataset_sha256": spec["dataset_sha256"], "dataset_manifest_sha256": spec["dataset_manifest_sha256"],
        "original_frozen_spec_sha256": sha(main / "frozen_spec.json"),
        "entrypoint_sha256": sha(Path(__file__).resolve()), "financial_code_sha256": spec["code_sha256"],
        "validation_source_sha256": pins, "snapshot_sha256": snapshot_hashes,
        "controls": list(CONTROLS), "capital": 10000., "costs": COSTS,
        "initial_targets": protocol["initial_targets"], "volatility_rule": protocol["volatility_rule"],
        "execution": protocol["execution"], "cash_floor_target": .05, "single_coin_cap_target": .95,
        "rebalance_band": .025, "output_subdirectory": "test",
        "signal_cutoff": "Decision-day 00UTC uses only preceding actual completed 23UTC closes; 31 complete OHLCV days for 30 return vectors. Earlier completed test days are causal inputs; no current/future day.",
        "covariance_implementation": protocol["covariance_implementation"],
        "runtime_context": "Run original byte-snapshotted functions with explicit test dates/output binding; never edit validation source or results.",
        "required_gate_files": list(REQUIRED_GATE), "goal_validator": "scripts.btceth_rl_goal_candidate.validate_frozen_candidate(require_test_started=True)",
        "gate_time_order": "risk.prepared_at < main.test_started.at; main.validation_selection.frozen_at <= test_started.at; goal.declared_at <= selected_at <= receipt.sealed_at < test_started.at",
        "goal_source_pin": "The officially declared goal_candidate_protocol.source_sha256 pins the goal-helper implementation; no draft helper version is silently accepted.",
        "repeats_per_case": 1, "cases": 6, "independent_accounting": "Replay exported fills using the original independent audit byte snapshot; reconcile hourly NAV, cash, holdings, fees, exposure, daily volatility and monthly returns.",
        "interpretation": "25% initial weight is not constant exposure or an exact model-risk match; past 20% volatility target is not a realized-risk guarantee; known annual research window is not a fresh blind holdout.",
    }
    sources.mkdir()
    for name, original in snapshots.items():
        dest = relative(sources, name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as stream:
            stream.write(original.read_bytes())
        if sha(dest) != snapshot_hashes[name]:
            raise GateClosed("Byte snapshot mismatch")
    write_new(target, result)
    write_new(receipt_path, {"schema_version": 1, "sealed_at": now(),
        "protocol_file": target.name, "protocol_sha256": sha(target),
        "entrypoint_sha256": result["entrypoint_sha256"],
        "original_frozen_spec_sha256": result["original_frozen_spec_sha256"]})
    return result


def verify_protocol(reference, dataset, *, source_root=ROOT):
    reference, dataset = Path(reference), Path(dataset)
    p = read_json(reference / "test_protocol.json")
    receipt = read_json(reference / "test_protocol_receipt.json")
    if (receipt.get("protocol_file") != "test_protocol.json"
            or receipt.get("protocol_sha256") != sha(reference / "test_protocol.json")
            or receipt.get("entrypoint_sha256") != p.get("entrypoint_sha256")
            or receipt.get("original_frozen_spec_sha256") != p.get("original_frozen_spec_sha256")
            or stamp(receipt["sealed_at"]) < stamp(p["prepared_at"])):
        raise GateClosed("Prepared holdout receipt no longer matches its original protocol")
    if (p.get("entrypoint_sha256") != sha(Path(__file__).resolve()) or p.get("controls") != list(CONTROLS)
            or p.get("costs") != COSTS or p.get("capital") != 10000.
            or p.get("dates_utc") != {"train": list(TRAIN), "validation": list(VALIDATION), "test": list(TEST)}
            or p.get("output_subdirectory") != "test" or p.get("cases") != 6):
        raise GateClosed("Prepared holdout contract or entrypoint changed")
    original = read_json(reference / "protocol.json")
    if (p.get("initial_targets") != original["initial_targets"]
            or p.get("volatility_rule") != original["volatility_rule"]
            or p.get("execution") != original["execution"]
            or p.get("rebalance_band") != .025 or p.get("cash_floor_target") != .05
            or p.get("single_coin_cap_target") != .95 or p.get("repeats_per_case") != 1):
        raise GateClosed("Prepared fixed control rules differ from their validation source")
    for name, expected in p["validation_source_sha256"].items():
        if sha(relative(reference, name)) != expected:
            raise GateClosed("Original validation evidence changed: " + name)
    for name, expected in p["snapshot_sha256"].items():
        if sha(relative(reference / "holdout_sources", name)) != expected:
            raise GateClosed("Prepared source snapshot changed: " + name)
    for name, expected in p["financial_code_sha256"].items():
        if sha(relative(source_root, name)) != expected:
            raise GateClosed("Original financial source changed: " + name)
    if (sha(dataset) != p["dataset_sha256"] or sha(dataset.parent / "manifest.json") != p["dataset_manifest_sha256"]):
        raise GateClosed("Prepared market-data bytes changed")
    return p


def goal_validator(main):
    from scripts.btceth_rl_goal_candidate import validate_frozen_candidate
    return validate_frozen_candidate(main, require_test_started=True)


def validate_test_gate(reference=REFERENCE, main=MAIN, dataset=DATASET, *, source_root=ROOT):
    reference, main, dataset = map(Path, (reference, main, dataset))
    missing = [name for name in REQUIRED_GATE if not (main / name).is_file()]
    if missing:
        raise GateClosed("No test data was loaded; missing sealed main contracts: " + ", ".join(missing))
    p = verify_protocol(reference, dataset, source_root=source_root)
    if sha(main / "frozen_spec.json") != p["original_frozen_spec_sha256"]:
        raise GateClosed("Main test belongs to another frozen experiment")
    context = goal_validator(main)  # Validation-only evidence; this API never reads test PNL.
    marker = context["test_started"]
    risk_receipt = read_json(reference / "test_protocol_receipt.json")
    if (context.get("test_pnl_read") is not False or marker is None
            or not stamp(p["prepared_at"]) < stamp(marker["at"])
            or not stamp(risk_receipt["sealed_at"]) < stamp(marker["at"])
            or not stamp(context["original_selection"]["frozen_at"]) <= stamp(marker["at"])):
        raise GateClosed("Risk controls and both validation choices must be fixed before main test dispatch")
    if (marker.get("validation_selection_sha256") != sha(main / "validation_selection.json")
            or marker.get("frozen_spec_sha256") != p["original_frozen_spec_sha256"]):
        raise GateClosed("Main test-dispatch SHA mismatch")
    return {"passed": True, "checked_at": now(), "test_pnl_read": False,
            "holdout_protocol_sha256": sha(reference / "test_protocol.json"),
            "holdout_protocol_receipt_sha256": sha(reference / "test_protocol_receipt.json"),
            "main_contract_sha256": {name: sha(main / name) for name in REQUIRED_GATE},
            "main_validation_leader": context["original_selection"]["primary_validation_leader"],
            "new_model_validation_leader": context["new_model_validation_leader"],
            "main_test_started_at": marker["at"]}


def load_core(reference, p, *, source_root=ROOT):
    """Original bytes remain intact; context bindings select a separate run."""
    path = relative(Path(reference) / "holdout_sources", "run_risk_budget_controls.py")
    if sha(path) != p["snapshot_sha256"]["run_risk_budget_controls.py"]:
        raise GateClosed("Risk-control formula source changed")
    name = "_btceth_risk_core_" + sha(path)[:16]
    # Resolve the pinned real package before the historical script derives its
    # Windows validation-directory context from its byte-snapshot location.
    for financial in FINANCIAL:
        module_name = financial[:-3].replace("/", ".")
        loaded = __import__(module_name, fromlist=["__file__"])
        if sha(Path(loaded.__file__)) != p["financial_code_sha256"][financial]:
            raise GateClosed("Imported financial module differs from its frozen source")
    spec = importlib.util.spec_from_file_location(name, path)
    core = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    try:
        spec.loader.exec_module(core)
    finally:
        sys.path[:] = old_path
    core.ROOT = Path(source_root).resolve()
    return core


def formula_sample_check(reference=REFERENCE, *, output=None, source_root=ROOT, dataset=DATASET):
    """Synthetic prices only; no original dataset is deserialized."""
    import numpy as np
    import pandas as pd
    reference = Path(reference)
    p = verify_protocol(reference, dataset, source_root=source_root)
    core = load_core(reference, p, source_root=source_root)
    output = Path(output) if output is not None else reference / "holdout_prepare_checks"
    output.mkdir(exist_ok=False)
    core.OUT = output
    core.hand_calculation_check()
    cutoff = pd.Timestamp("2024-02-01T00:00:00Z")
    index = pd.date_range("2023-12-31T00:00:00Z", "2024-02-03T00:00:00Z", freq="h", inclusive="left")
    n = np.arange(len(index), dtype=float)
    close = pd.DataFrame({"BTCUSDT": 100 + n * .05 + 3 * np.sin(n / 50),
                          "ETHUSDT": 50 + n * .025 + 2 * np.sin(n / 35)}, index=index)
    data = {"hourly_" + field: close * scale for field, scale in (
        ("open", .999), ("high", 1.002), ("low", .998), ("close", 1.))}
    data["hourly_volume"] = close * 0 + 100
    core.VALIDATION = (str(cutoff), str(cutoff + pd.Timedelta(days=1)))
    variants = {"full_synthetic_future": data,
                "future_truncated": {k: v.loc[v.index < cutoff].copy() for k, v in data.items()},
                "future_changed": {k: v.copy() for k, v in data.items()}}
    for frame in variants["future_changed"].values():
        frame.loc[frame.index >= cutoff] *= 1000
    targets = {}
    for key, values in variants.items():
        core.OUT = output / key
        core.OUT.mkdir()
        signals = core.prepare_signals(values)
        targets[key] = signals.loc[cutoff, ["target_btc", "target_eth", "target_cash"]].to_numpy(dtype=float)
    original = targets["full_synthetic_future"]
    if any(not np.array_equal(original, value) for value in targets.values()):
        raise AssertionError("Synthetic past signals changed after future truncation/mutation")
    result = {"passed": True, "synthetic_only": True, "original_dataset_loaded": False,
              "hand_calculation_passed": True, "future_truncation_exact": True, "future_mutation_exact": True,
              "decision": str(cutoff), "target": original.tolist()}
    write_new(output / "causality_checks.json", result)
    return result


def independent_case(core, directory, data, row, monthly, dates, costs):
    import numpy as np
    import pandas as pd
    audit = core.AUDIT
    checks = audit.Checks()
    equity = audit.read_equity(directory / "equity_1h.csv.gz")
    rebuilt = audit.reconstruct_fills(data, audit.read_csv(directory / "fills.csv"), equity,
        audit.read_json(directory / "metadata.json"), start=dates[0], end=dates[1], **costs,
        checks=checks, label=row["run_id"])
    audit.verify_saved_fingerprints(directory, checks, row["run_id"])
    if rebuilt is None or checks.failed:
        raise AssertionError("Independent control accounting failed: " + repr(checks.failures))
    for key in ("net_return", "max_drawdown", "fees", "slippage_cost", "average_exposure", "turnover", "net_liquidation_return"):
        if abs(float(row[key]) - float(rebuilt["metrics"][key])) > 1e-8:
            raise AssertionError("Control summary differs from independent fills: " + key)
    for month, value in monthly[["month", "return"]].itertuples(index=False, name=None):
        if abs(value - rebuilt["monthly"][month]) > 1e-12:
            raise AssertionError("Monthly control returns differ from independent fills")
    daily = rebuilt["equity"].equity.resample("D").last()
    ret = daily.pct_change(fill_method=None)
    ret.iloc[0] = daily.iloc[0] / 10000 - 1
    deviation = float(ret.std(ddof=1))
    pd.DataFrame({"date": daily.index, "equity": daily.to_numpy(), "net_daily_return": ret.to_numpy()}).to_csv(
        directory / "daily_returns_independent.csv", index=False, float_format="%.17g")
    rebuilt["quantity"].to_csv(directory / "reconstructed_quantities.csv.gz", index_label="timestamp", float_format="%.17g")
    result = {"passed": True, "checks": checks.count, "failed": checks.failed, "maximum_errors": checks.maximum_errors,
              "metrics": rebuilt["metrics"], "daily_return_std_ddof1": deviation,
              "realized_annualized_volatility": deviation * np.sqrt(365),
              "positive_months": int(sum(value > 0 for value in rebuilt["monthly"].values())),
              "observed_months": len(rebuilt["monthly"]), "hours": len(equity),
              "partial_months": ["2025-10 [23,31]", "2026-10 [01,23)"]}
    write_new(directory / "independent_reconstruction.json", result)
    return result


def run_test(reference=REFERENCE, main=MAIN, dataset=DATASET, *, source_root=ROOT):
    """The gate runs before target creation, pickle loading, or test signals."""
    reference, main, dataset = map(Path, (reference, main, dataset))
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    gate = validate_test_gate(reference, main, dataset, source_root=source_root)
    target = reference / "test"
    if target.exists():
        raise FileExistsError("Preserve every previous holdout-control attempt")
    p = verify_protocol(reference, dataset, source_root=source_root)
    core = load_core(reference, p, source_root=source_root)
    import importlib.metadata
    import pandas as pd
    for name, expected in (("numpy", "2.4.6"), ("pandas", "3.0.3")):
        if importlib.metadata.version(name) != expected:
            raise GateClosed("Control runtime differs from the validated numeric environment: " + name)
    target.mkdir()
    write_new(target / "test_dispatch.json", gate)
    core.OUT, core.VALIDATION = target, TEST
    data = dict(core.load_dataset(dataset))
    cutoff = pd.Timestamp(TEST[1])
    for field in ("open", "high", "low", "close", "volume"):
        frame = data["hourly_" + field]
        data["hourly_" + field] = frame.loc[frame.index < cutoff].copy()
    signals = core.prepare_signals(data)
    rows, histories, months, proofs = [], [], [], []
    for scenario, costs in COSTS.items():
        for control in CONTROLS:
            result, trace = core.execute(data, control, costs, signals)
            result["metadata"].update(scope="Separately gated risk-budget research test; outside formal172 rows",
                holdout_protocol_sha256=gate["holdout_protocol_sha256"])
            directory = target / scenario / control
            fingerprints = core.save_result(directory, result)
            trace.to_csv(directory / "decision_targets_actual_weights.csv", index=False, float_format="%.17g")
            row, history, monthly = core.summarize(result, run_id=f"test__{scenario}__{control}", strategy=control,
                scenario=scenario, split="test", start=TEST[0], end=TEST[1], capital=10000., **costs)
            proof = independent_case(core, directory, data, row, monthly, TEST, costs)
            repeat, repeat_trace = core.execute(data, control, costs, signals)
            if fingerprints != core.result_fingerprints(repeat) or not trace.equals(repeat_trace):
                raise AssertionError("Fixed control complete repeat differed")
            row.update(control_family="risk_budget_holdout_reference", formal_baseline=False, repeat_exact=True,
                daily_return_std_ddof1=proof["daily_return_std_ddof1"],
                realized_annualized_volatility=proof["realized_annualized_volatility"],
                positive_months=proof["positive_months"], observed_months=proof["observed_months"],
                rebalanced_days=int(trace.rebalanced.sum()))
            rows.append(row)
            histories.append(history)
            months.append(monthly)
            proofs.append(proof)
    # Re-check every original gate and pin before declaring completion.
    after = validate_test_gate(reference, main, dataset, source_root=source_root)
    if (after["main_contract_sha256"] != gate["main_contract_sha256"]
            or after["holdout_protocol_sha256"] != gate["holdout_protocol_sha256"]
            or after["holdout_protocol_receipt_sha256"] != gate["holdout_protocol_receipt_sha256"]):
        raise GateClosed("Main contracts changed during the separate control evaluation")
    pd.DataFrame(rows).to_csv(target / "summary.csv", index=False, float_format="%.17g")
    pd.concat(histories, ignore_index=True).to_csv(target / "daily_equity.csv", index=False, float_format="%.17g")
    pd.concat(months, ignore_index=True).to_csv(target / "monthly_returns.csv", index=False, float_format="%.17g")
    result = {"passed": True, "completed_at": now(), "cases": 6, "full_repeats": 6,
              "hours_reconstructed": sum(item["hours"] for item in proofs),
              "independent_checks": sum(item["checks"] for item in proofs), "failures": 0,
              "formal_rankings_modified": False, "main_contract_sha256": gate["main_contract_sha256"],
              "holdout_protocol_sha256": gate["holdout_protocol_sha256"]}
    write_new(target / "completion.json", result)
    files = [{"path": path.relative_to(target).as_posix(), "size": path.stat().st_size, "sha256": sha(path)}
             for path in sorted(target.rglob("*")) if path.is_file()]
    write_new(target / "file_manifest.json", files)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "check-test-gate", "self-check", "test"), required=True)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE)
    parser.add_argument("--main-output-dir", type=Path, default=MAIN)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    try:
        if args.phase == "prepare":
            result = prepare(args.reference_root, args.main_output_dir, args.dataset)
        elif args.phase == "check-test-gate":
            result = validate_test_gate(args.reference_root, args.main_output_dir, args.dataset)
        elif args.phase == "self-check":
            result = formula_sample_check(args.reference_root, dataset=args.dataset)
        else:
            result = run_test(args.reference_root, args.main_output_dir, args.dataset)
        print(json.dumps(result, ensure_ascii=False, default=json_default))
    except (ValueError, FileNotFoundError, FileExistsError, AssertionError) as error:
        print(json.dumps({"passed": False, "phase": args.phase, "test_data_loaded": False if args.phase != "test" else "gate precedes loading", "error": str(error)}, ensure_ascii=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
