"""Shape-only adapter for the frozen risk helper's test-dispatch context.

The original validator still validates every SHA/time condition. No strategy,
fee, capital, market calculation, or original frozen source is changed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import btceth_rl_risk_controls as risk
from scripts.collect_btceth_rl_results import safe_target, sha, inventory

RISK_SHA = "412ab9e00b8d868b9bd8d02b8cbb86ee99d942fa8ed379482da691f2bf95b1dc"
GOAL_SHA = "911034a4861d79dbc9ca9933302e177d645d783f99d89e33ac01438dfa107109"
PROTOCOL_SHA = "b8b15b88a50d7faf8204bd6fc94b76ef28e12521a0774c58f14ac045c6e2c411"
RECEIPT_SHA = "408b18752531f49472ebfe55e15911b510382a5b1c45a4e002c5d1333f5120b9"
FAILED_LOG_SHA = "c608849be28c0a11fe885737823791cbf91fd42303a4eaaa1c9afe6b164a45a2"


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    risk.write_new(Path(path), value)


def adapted_validator(original, expected_marker_sha):
    def validate(main):
        main = Path(main)
        marker_path = safe_target(main, "test_started.json")
        if sha(marker_path) != expected_marker_sha:
            raise ValueError("Main marker changed before original goal validation")
        context = original(main)
        if context.get("test_started") is not True or context.get("test_pnl_read") is not False:
            raise ValueError("Original validator must confirm test_started True without test PNL reads")
        marker = risk.read_json(marker_path)
        if (sha(marker_path) != expected_marker_sha
                or marker.get("validation_selection_sha256") != sha(safe_target(main, "validation_selection.json"))
                or marker.get("frozen_spec_sha256") != sha(safe_target(main, "frozen_spec.json"))):
            raise ValueError("Validated actual marker/selection/spec SHA mismatch")
        risk.stamp(marker["at"])
        result = dict(context)
        result["test_started"] = marker
        return result
    return validate


@contextmanager
def compatible_validator(expected_marker_sha):
    original = risk.goal_validator
    risk.goal_validator = adapted_validator(original, expected_marker_sha)
    try:
        yield
    finally:
        risk.goal_validator = original


def source_pins(source_root=ROOT):
    values = {"scripts/btceth_rl_risk_controls.py": RISK_SHA, "scripts/btceth_rl_goal_candidate.py": GOAL_SHA}
    for name, expected in values.items():
        if sha(safe_target(source_root, name)) != expected:
            raise ValueError("Original frozen API source changed: " + name)
    return {**values, "scripts/btceth_rl_risk_controls_adapter.py": sha(Path(__file__))}


def prepare(reference, main, dataset, adapter_root, failed_attempt, *, source_root=ROOT):
    reference, main, dataset, adapter_root, failed_attempt = map(Path, (reference, main, dataset, adapter_root, failed_attempt))
    if adapter_root.exists() or (reference / "test").exists():
        raise FileExistsError("Preserve prior adapter or risk test attempt")
    adapter_root.resolve().relative_to((reference / "adapter_attempts").resolve())
    sources = source_pins(source_root)
    if sha(reference / "test_protocol.json") != PROTOCOL_SHA or sha(reference / "test_protocol_receipt.json") != RECEIPT_SHA:
        raise ValueError("Original risk protocol/receipt changed")
    failure = risk.read_json(failed_attempt / "process.json")
    if failure.get("exit_code") != 1 or failure.get("status") != "failed" or sha(failed_attempt / "run.log") != FAILED_LOG_SHA:
        raise ValueError("Preserve and identify the original pre-price interface failure")
    marker_sha = sha(safe_target(main, "test_started.json"))
    # This calls only the original read-only gate. It cannot decode prices.
    with compatible_validator(marker_sha):
        checked = risk.validate_test_gate(reference, main, dataset, source_root=source_root)
    if checked.get("passed") is not True or checked.get("test_pnl_read") is not False or sha(main / "test_started.json") != marker_sha:
        raise ValueError("Original risk gate or marker changed during adapter preparation")
    original_files = [item for item in inventory(reference) if not item["path"].startswith("adapter_attempts/")]
    protocol = dict(schema_version=1, declared_at=now(), source_sha256=sources,
        purpose="Convert the already validated context.test_started boolean to its actual SHA-verified marker object; no financial-rule changes",
        risk_protocol_sha256=PROTOCOL_SHA, risk_receipt_sha256=RECEIPT_SHA,
        main_marker_sha256=marker_sha, original_risk_sha256=RISK_SHA, original_goal_sha256=GOAL_SHA,
        failed_log_sha256=FAILED_LOG_SHA, failed_process_sha256=sha(failed_attempt / "process.json"),
        failed_attempt_root=str(failed_attempt.resolve()), financial_rules_changed=False,
        annual_data_decoded_during_preparation=False, target_absent_at_preparation=True,
        original_reference_files=original_files, main_contract_sha256=checked["main_contract_sha256"],
        source_adaptation="temporary risk.goal_validator only; original goal validator executes first, finally restore")
    adapter_root.mkdir(parents=True)
    for name in sources:
        destination = adapter_root / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe_target(source_root, name), destination)
    (adapter_root / "original_failure").mkdir()
    for name in ("process.json", "run.log"):
        shutil.copyfile(failed_attempt / name, adapter_root / "original_failure" / name)
    write_new(adapter_root / "protocol.json", protocol)
    write_new(adapter_root / "preparation.json", dict(at=now(), passed=True, gate=checked,
        protocol_sha256=sha(adapter_root / "protocol.json"), original_goal_function_restored=True, test_data_loaded=False))
    return protocol


def run(reference, main, dataset, adapter_root, *, source_root=ROOT):
    reference, main, dataset, adapter_root = map(Path, (reference, main, dataset, adapter_root))
    p = risk.read_json(adapter_root / "protocol.json")
    prep = risk.read_json(adapter_root / "preparation.json")
    if ((reference / "test").exists() or any((adapter_root / n).exists() for n in ("execution.json", "execution_started.json", "execution_failed.json"))
            or p.get("source_sha256") != source_pins(source_root)
            or p.get("financial_rules_changed") is not False or prep.get("passed") is not True
            or prep.get("protocol_sha256") != sha(adapter_root / "protocol.json")
            or not (sha(reference / "test_protocol.json") == p.get("risk_protocol_sha256") == PROTOCOL_SHA)
            or not (sha(reference / "test_protocol_receipt.json") == p.get("risk_receipt_sha256") == RECEIPT_SHA)
            or sha(main / "test_started.json") != p.get("main_marker_sha256")):
        raise ValueError("Adapter source/protocol/preparation changed or a test was already attempted")
    def protected():
        for name, expected in p["source_sha256"].items():
            if sha(safe_target(adapter_root / "sources", name)) != expected:
                raise ValueError("Adapter source byte snapshot changed: " + name)
        for item in p["original_reference_files"]:
            path = safe_target(reference, item["path"])
            if path.stat().st_size != item["size"] or sha(path) != item["sha256"]:
                raise ValueError("Original risk validation bytes changed: " + item["path"])
        if sha(main / "test_started.json") != p["main_marker_sha256"]:
            raise ValueError("Main marker changed during adapter execution")
    protected()
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    write_new(adapter_root / "execution_started.json", dict(at=now(), pid=os.getpid(),
        protocol_sha256=sha(adapter_root / "protocol.json"), source_sha256=p["source_sha256"], main_marker_sha256=p["main_marker_sha256"]))
    original = risk.goal_validator
    try:
        with compatible_validator(p["main_marker_sha256"]):
            # The original gate is again executed inside the original run_test,
            # before target creation or pickle loading; the formulas are reused.
            result = risk.run_test(reference, main, dataset, source_root=source_root)
        protected()
        if risk.goal_validator is not original:
            raise AssertionError("Original validator was not restored")
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            raise AssertionError("CPU-only risk controls initialized CUDA")
        evidence = dict(at=now(), passed=True, exit_code=0, result=result, protocol_sha256=sha(adapter_root / "protocol.json"),
            main_marker_sha256=p["main_marker_sha256"], original_validator_restored=True,
            original_validation_unchanged=True, cuda_initialized=False, source_sha256=p["source_sha256"])
    except BaseException as error:
        write_new(adapter_root / "execution_failed.json", dict(at=now(), passed=False, error_type=type(error).__name__, error=str(error),
            original_validator_restored=risk.goal_validator is original))
        raise
    write_new(adapter_root / "execution.json", evidence)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("prepare", "test"))
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--main-output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--failed-attempt-root", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        if args.failed_attempt_root is None:
            parser.error("prepare requires --failed-attempt-root")
        result = prepare(args.reference_root, args.main_output, args.dataset, args.adapter_root, args.failed_attempt_root)
    else:
        result = run(args.reference_root, args.main_output, args.dataset, args.adapter_root)
    print(json.dumps(result, ensure_ascii=False, default=risk.json_default))
