"""SHA-only transport for the separately audited TQC exposure extension.

Reuses the original collector's archive, path, extraction and conflict checks.
No financial module, model or score is imported or evaluated. Original remote
audits are preserved; a subsequent local audit must use a .local.json filename.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1] if globals().get("__file__") not in (None, "<stdin>") else Path.cwd()
# The SSH stdin program receives an explicit repository path before import.
_bootstrap = Path(sys.argv[sys.argv.index("--source-root") + 1]) if "--source-root" in sys.argv else ROOT
if str(_bootstrap) not in sys.path:
    sys.path.insert(0, str(_bootstrap))
from scripts import collect_btceth_rl_results as transport

DEFAULT_OUTPUT = ROOT / "results/backtests/btceth_rl_tqc_exposure_cap_20261030"
DEFAULT_MAIN = ROOT / "results/backtests/btceth_rl_20261029"
REMOTE_SOURCE = "/home/sunny/research/btceth_rl_20261029"
REMOTE_ROOT = REMOTE_SOURCE + "/exposure_cap_075"
REMOTE_MAIN = REMOTE_SOURCE + "/experiment"
REMOTE_DATASET = REMOTE_SOURCE + "/data/crypto_rl_btceth_20261029/dataset.pkl.gz"
CASE_FILES = ("equity_1h.csv.gz", "fills.csv", "weights.csv", "boundary_equity.csv", "metadata.json", "fingerprints.json", "member_targets.csv", "replay.json")
GROUP_FILES = ("summary.csv", "daily_equity.csv", "monthly_returns.csv", "completion.json")
FINANCIAL = {"scripts/backtest_btceth_rl.py", "scripts/btceth_rl_env.py", "scripts/crypto_rl_portfolio_env.py", "scripts/crypto_rl_reporting.py"}
SOURCE_NAMES = FINANCIAL | {"scripts/btceth_rl_exposure_cap.py", "scripts/btceth_rl_goal_candidate.py"}
DEFAULT_VALIDATION_AUDIT = "independent_audit.validation.json"
sha, read_json, safe_target = transport.sha, transport.read_json, transport.safe_target


def hashes(root, mapping, label):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(label + " requires nonempty source pins")
    for name, expected in mapping.items():
        if not re.fullmatch(r"[0-9a-f]{64}", str(expected)) or sha(safe_target(root, name)) != expected:
            raise ValueError(label + " changed: " + str(name))


def identities(root, main, dataset, source_root):
    """Verify bytes and identities, without importing the frozen cap module."""
    root, main, dataset, source_root = map(Path, (root, main, dataset, source_root))
    if root.resolve() == main.resolve() or root.resolve().is_relative_to(main.resolve()) or main.resolve().is_relative_to(root.resolve()):
        raise ValueError("Extension and original studies must remain separate")
    protocol = read_json(safe_target(root, "protocol.json"))
    receipt = read_json(safe_target(root, "protocol_receipt.json"))
    spec = read_json(safe_target(main, "frozen_spec.json"))
    protocol_hash = sha(safe_target(root, "protocol.json"))
    if (receipt.get("protocol_file") != "protocol.json" or receipt.get("protocol_sha256") != protocol_hash
            or protocol.get("original_frozen_spec_sha256") != sha(safe_target(main, "frozen_spec.json"))
            or protocol.get("original_goal_protocol_sha256") != sha(safe_target(main, "goal_candidate_protocol.json"))
            or protocol.get("multiplier") != .75 or protocol.get("capital") != 10000.
            or protocol.get("profiles") != ["return", "balanced"] or protocol.get("seeds") != list(range(5))
            or protocol.get("dataset_sha256") != spec.get("dataset_sha256")
            or protocol.get("dataset_manifest_sha256") != spec.get("dataset_manifest_sha256")
            or set(protocol.get("source_sha256", {})) != SOURCE_NAMES
            or set(spec.get("code_sha256", {})) != FINANCIAL
            or any(protocol["source_sha256"][n] != spec["code_sha256"][n] for n in FINANCIAL)):
        raise ValueError("Extension protocol, receipt or original study identity differs")
    if transport.is_link(dataset) or sha(dataset) != protocol["dataset_sha256"] or sha(safe_target(dataset.parent, "manifest.json")) != protocol["dataset_manifest_sha256"]:
        raise ValueError("Frozen dataset or data manifest changed")
    hashes(source_root, protocol["source_sha256"], "Current frozen source")
    hashes(root / "sources", protocol["source_sha256"], "Original source snapshot")
    hashes(main, protocol["training_source_sha256"], "Frozen TQC checkpoints")
    expected = {f"training/tqc__{p}__seed{s}/{n}" for p in ("return", "balanced") for s in range(5) for n in ("record.json", "model.zip")}
    if set(protocol["training_source_sha256"]) != expected or len(protocol.get("models", [])) != 10:
        raise ValueError("Exactly ten original TQC models and twenty files are required")
    found = set()
    for model in protocol["models"]:
        identity = model["profile"], model["seed"]
        folder = f"training/tqc__{model['profile']}__seed{model['seed']}"
        record = read_json(safe_target(main, folder + "/record.json"))
        if (identity in found or identity not in {(p, s) for p in ("return", "balanced") for s in range(5)}
                or model.get("record_file") != folder + "/record.json" or model.get("model_file") != folder + "/model.zip"
                or (record.get("algorithm"), record.get("profile"), record.get("seed")) != ("tqc", *identity)
                or record.get("model_relative_path") != model["model_file"]
                or model.get("record_sha256") != sha(safe_target(main, model["record_file"]))
                or model.get("model_sha256") != sha(safe_target(main, model["model_file"]))
                or model["model_sha256"] != record.get("model_sha256")
                or model.get("parameter_sha256") != record.get("parameter_sha256")
                or record.get("frozen_spec_sha256") != protocol["original_frozen_spec_sha256"]):
            raise ValueError("Original TQC checkpoint/record identity changed")
        found.add(identity)
    return protocol, {"protocol_sha256": protocol_hash, "frozen_spec_sha256": protocol["original_frozen_spec_sha256"],
        "dataset_sha256": protocol["dataset_sha256"], "dataset_manifest_sha256": protocol["dataset_manifest_sha256"],
        "source_sha256": protocol["source_sha256"], "training_source_sha256": protocol["training_source_sha256"]}


def group_names(split, profiles, *, capped=True, algorithm="tqc"):
    prefix = lambda p: f"{split}/{p}" if capped else f"evaluation/{split}/{algorithm}/{p}"
    return {*(f"{prefix(p)}/{f}" for p in profiles for f in GROUP_FILES),
        *(f"{prefix(p)}/{c}/{k}/{f}" for p in profiles for c in ("base", "stress")
          for k in (*map(str, range(5)), "ensemble") for f in (CASE_FILES if capped else CASE_FILES[:6]))}


def audit_filename(validation_only, validation_audit_name):
    name = transport.safe_name(validation_audit_name)
    if not name.endswith(".json"):
        raise ValueError("Validation audit identity must be a safe relative JSON filename")
    if not validation_only and name != DEFAULT_VALIDATION_AUDIT:
        raise ValueError("An explicit validation audit cannot replace the full36 audit")
    return name if validation_only else "independent_audit.json"


def terminal_guard(root, *, main, dataset, source_root=ROOT, validation_only=False,
                   process_root=None, process_main=None, process_dataset=None,
                   validation_audit_name=DEFAULT_VALIDATION_AUDIT):
    root, main = Path(root), Path(main)
    protocol, identity = identities(root, main, dataset, source_root)
    phase = "validate" if validation_only else "test"
    split = "validation" if validation_only else "test"
    audit_name = audit_filename(validation_only, validation_audit_name)
    process = read_json(safe_target(root, phase + "_process.json"))
    log_name = phase + ".log"
    if (process.get("phase") != phase or process.get("status") != "completed"
            or type(process.get("exit_code")) is not int or process["exit_code"] != 0
            or not process.get("completed_at") or not process.get("started_at")
            or process.get("protocol_sha256") != identity["protocol_sha256"]
            or process.get("runner_sha256") != protocol["source_sha256"]["scripts/btceth_rl_exposure_cap.py"]
            or process.get("log_relative_path") != log_name or process.get("log_sha256") != sha(safe_target(root, log_name))):
        raise ValueError("Overlay process must be completed0 with the exact protocol/runner/log")
    command = process.get("command", [])
    bindings = {"--phase": phase, "--output-dir": str(process_root or root.resolve()),
                "--main-output": str(process_main or main.resolve()), "--dataset": str(process_dataset or Path(dataset).resolve())}
    for option, expected in bindings.items():
        if command.count(option) != 1 or command.index(option) + 1 >= len(command) or command[command.index(option) + 1] != expected:
            raise ValueError("Terminal process command belongs to another study: " + option)
    if datetime.fromisoformat(process["completed_at"]) < datetime.fromisoformat(process["started_at"]):
        raise ValueError("Terminal process timestamps are inconsistent")
    if any(type(process.get(k)) is not int or process[k] <= 0 for k in ("child_pid", "supervisor_pid")):
        raise ValueError("Terminal process proof needs actual PID identities")
    if sys.platform == "linux":
        for pid in (process.get("child_pid"), process.get("supervisor_pid")):
            if type(pid) is not int or pid <= 0:
                raise ValueError("Terminal process proof needs actual PID identities")
            proc = Path("/proc") / str(pid)
            if proc.exists():
                try:
                    state = (proc / "stat").read_text().rsplit(")", 1)[1].strip().split()[0]
                    text = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
                except (FileNotFoundError, ProcessLookupError):
                    continue
                if state != "Z" and "btceth_rl_exposure_cap" in text:
                    raise ValueError("Overlay process is still live")
    audited = read_json(safe_target(root, audit_name))
    count = 24 if validation_only else 36
    expected_status = "passed_validation_only" if validation_only else "passed"
    if (audited.get("passed") is not True or audited.get("status") != expected_status
            or audited.get("test_complete") is not (not validation_only) or audited.get("run_count") != count
            or audited.get("failure_count") != 0 or audited.get("failures") != []
            or type(audited.get("checks")) is not int or audited["checks"] <= 0
            or audited.get("dataset_sha256") != identity["dataset_sha256"]):
        raise ValueError("Independent overlay audit does not cover the requested complete scope")
    if (set(audited.get("source_sha256", {})) != SOURCE_NAMES | {"scripts/audit_btceth_rl_exposure_cap.py", "scripts/audit_btceth_rl.py"}
            or audited.get("test_financial_reads_enabled") is not (not validation_only)
            or audited.get("research_targets", {}).get("evaluated") is not (not validation_only)):
        raise ValueError("Independent audit source or declared financial-read scope differs")
    expected_ids = {f"validation__{c}__cap075__tqc__{p}__{k}" for p in ("return", "balanced") for c in ("base", "stress") for k in (*map(str, range(5)), "ensemble")}
    required = {"protocol.json", "protocol_receipt.json", "validation_completion.json", "summary.csv", "daily_equity.csv", "monthly_returns.csv"} | group_names("validation", ("return", "balanced"))
    completed = read_json(safe_target(root, split + "_completion.json"))
    if completed.get("protocol_sha256") != identity["protocol_sha256"] or completed.get("case_count") != (24 if validation_only else 12) or completed.get("repeat_count") != (24 if validation_only else 12):
        raise ValueError("Overlay completion does not match the protocol/scope")
    if not validation_only:
        selected = read_json(safe_target(root, "selection.json"))["candidate"]
        family, algorithm, profile = selected.get("family"), selected.get("algorithm"), selected.get("profile")
        if family not in ("raw", "cap075") or algorithm not in ("recurrent_ppo", "tqc", "crossq") or profile not in ("return", "balanced") or (family == "cap075" and algorithm != "tqc"):
            raise ValueError("Transport cannot accept a different candidate population")
        expected_ids.update(f"test__{c}__{'cap075__' if family == 'cap075' else ''}{algorithm}__{profile}__{k}" for c in ("base", "stress") for k in (*map(str, range(5)), "ensemble"))
        required |= {"selection.json", "selection_receipt.json", "test_started.json", "test_completion.json"}
        if family == "cap075":
            required |= group_names("test", (profile,))
        else:
            raw_names = group_names("test", (profile,), capped=False, algorithm=algorithm)
            if set(completed.get("raw_source_sha256", {})) != raw_names or not raw_names.issubset(audited.get("main_input_sha256", {})):
                raise ValueError("Raw test references need all76 original byte sources")
        main_audit = audited.get("main_final_audit_path")
        if not isinstance(main_audit, str) or main_audit not in audited.get("main_input_sha256", {}):
            raise ValueError("Final overlay audit must pin the original172 audit")
        original = read_json(safe_target(main, main_audit))
        if original.get("passed") is not True or original.get("status") != "passed" or original.get("test_complete") is not True or original.get("run_count") != 172 or original.get("failures") != []:
            raise ValueError("Original complete passed172 audit required")
    if not required.issubset(audited.get("input_sha256", {})):
        raise ValueError("Audit lacks required immutable extension files")
    runs = audited.get("runs", {})
    if set(runs) != expected_ids or any(proof.get("reconstructed") is not True or proof.get("hours") != (7080 if key.startswith("validation__") else 8760) for key, proof in runs.items()):
        raise ValueError("Audit must reconstruct every exact24/36 funded ledger")
    hashes(root, audited["input_sha256"], "Audited extension input")
    hashes(main, audited.get("main_input_sha256"), "Audited original study input")
    hashes(source_root, audited.get("source_sha256"), "Independent audit source")
    if any(audited["source_sha256"].get(n) != v for n, v in protocol["source_sha256"].items()):
        raise ValueError("Audit and protocol frozen source pins differ")
    protected = {n: sha(safe_target(root, n)) for n in ("protocol.json", "protocol_receipt.json", phase + "_process.json", log_name, audit_name)}
    return dict(mode="validation_only" if validation_only else "full", **identity,
        audit_file=audit_name, audit_sha256=sha(safe_target(root, audit_name)),
        terminal_proof_sha256=protected, audit_input_sha256=audited["input_sha256"], main_input_sha256=audited["main_input_sha256"], audit_source_sha256=audited["source_sha256"])


@contextmanager
def scoped_guard(callback):
    """Adapt the reused export routine in this process only; no disk edits."""
    original = transport.terminal_guard
    transport.terminal_guard = callback
    try:
        yield
    finally:
        transport.terminal_guard = original


def export_snapshot(root, *, main, dataset, source_root=ROOT, validation_only=False, export_parent=None,
                    validation_audit_name=DEFAULT_VALIDATION_AUDIT):
    audit_filename(validation_only, validation_audit_name)
    parent = export_parent or str(Path(root).parent.parent / "btceth_rl_exposure_cap_exports")
    if Path(parent).resolve() == Path(main).resolve() or Path(parent).resolve().is_relative_to(Path(main).resolve()):
        raise ValueError("Export files must remain outside the original study")
    dependency_hash = sha(Path(transport.__file__))
    callback = lambda path, validation_only=False: terminal_guard(path, main=main, dataset=dataset, source_root=source_root, validation_only=validation_only, validation_audit_name=validation_audit_name)
    with scoped_guard(callback):
        response = transport.export_snapshot(root, validation_only=validation_only, export_parent=parent)
    response["protocol_sha256"] = sha(safe_target(root, "protocol.json"))
    if sha(Path(transport.__file__)) != dependency_hash:
        raise RuntimeError("Reused transport source changed during export")
    response["reused_transport_sha256"] = dependency_hash
    proof = read_json(response["manifest_path"])["terminal_guard"]
    response.update(audit_file=proof["audit_file"], audit_sha256=proof["audit_sha256"])
    return response


def collect(*, output, reference_output=DEFAULT_OUTPUT, main_output=DEFAULT_MAIN,
            dataset_path=transport.DEFAULT_DATASET, source_root=ROOT, alias=transport.DEFAULT_ALIAS,
            remote_python=transport.DEFAULT_REMOTE_PYTHON, remote_root=REMOTE_ROOT,
            remote_main=REMOTE_MAIN, remote_dataset=REMOTE_DATASET, remote_source=REMOTE_SOURCE,
            export_parent=None, staging=None, validation_only=False,
            validation_audit_name=DEFAULT_VALIDATION_AUDIT):
    expected_audit = audit_filename(validation_only, validation_audit_name)
    output, reference_output = Path(output).resolve(), Path(reference_output).resolve()
    original_root = Path(main_output).resolve()
    if output == original_root or output.is_relative_to(original_root) or original_root.is_relative_to(output):
        raise ValueError("Collected extension must remain separate from the original study")
    if validation_only and (output.exists() or output == reference_output or output.is_relative_to(reference_output) or reference_output.is_relative_to(output)):
        raise ValueError("Validation-only collection needs a new separate snapshot directory")
    _, local_proof = identities(reference_output, main_output, dataset_path, source_root)
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex
    staging = Path(staging or ROOT / "results/backtests/btceth_rl_exposure_cap_transfers" / token)
    if (transport.is_link(staging) or staging.exists()
            or any(staging.resolve() == bound or staging.resolve().is_relative_to(bound) or bound.is_relative_to(staging.resolve())
                   for bound in (output, reference_output, original_root))):
        raise ValueError("Collection requires a new staging tree separate from output")
    staging.mkdir(parents=True)
    source = Path(__file__).read_bytes()
    dependency_hash = sha(Path(transport.__file__))
    parent = export_parent or str(PurePosixPath(remote_root).parent.parent / "btceth_rl_exposure_cap_exports")
    common = ["--remote-root", str(remote_root), "--remote-export-parent", str(parent), "--main-output", str(remote_main), "--dataset", str(remote_dataset), "--source-root", str(remote_source), "--validation-audit-name", validation_audit_name]
    args = [*common, "--remote-export-worker"] + (["--validation-only"] if validation_only else [])
    response = transport._remote_call(alias, remote_python, args, source)
    transport._validate_remote_export(response, remote_root, parent, validation_only)
    if response.get("protocol_sha256") != local_proof["protocol_sha256"] or response.get("frozen_spec_sha256") != local_proof["frozen_spec_sha256"]:
        raise ValueError("Exported overlay protocol differs from local prepared bytes")
    if response.get("reused_transport_sha256") != dependency_hash:
        raise ValueError("Remote safe-transport implementation differs from the local reviewed dependency")
    if response.get("audit_file") != expected_audit or not re.fullmatch(r"[0-9a-f]{64}", str(response.get("audit_sha256"))):
        raise ValueError("Remote export returned a different validation/full audit identity")
    transport.write_new_json(staging / "remote_export_receipt.json", response)
    manifest_path, archive_path = staging / "snapshot_manifest.json", staging / "experiment_snapshot.tar.gz"
    transport._remote_call(alias, remote_python, [*common, "--remote-read-file", response["manifest_path"]], source, destination=manifest_path)
    if sha(manifest_path) != response["manifest_sha256"]:
        raise ValueError("Downloaded manifest SHA changed")
    manifest = read_json(manifest_path)
    if any(manifest.get(n) != response[n] for n in ("mode", "remote_root", "archive_sha256", "archive_size")):
        raise ValueError("Downloaded manifest disagrees with the unique export receipt")
    if manifest.get("terminal_guard", {}).get("audit_file") != expected_audit or manifest["terminal_guard"].get("audit_sha256") != response["audit_sha256"]:
        raise ValueError("Downloaded manifest identifies another audit file/SHA")
    transport.manifest_entries(manifest)
    transport._remote_call(alias, remote_python, [*common, "--remote-read-file", response["archive_path"]], source, destination=archive_path)
    extracted = staging / "verified_files"
    entries = transport.verify_extract(archive_path, manifest, extracted)
    proof = terminal_guard(extracted, main=main_output, dataset=dataset_path, source_root=source_root,
        validation_only=validation_only, process_root=remote_root, process_main=remote_main, process_dataset=remote_dataset,
        validation_audit_name=validation_audit_name)
    if proof != manifest["terminal_guard"] or proof["protocol_sha256"] != local_proof["protocol_sha256"]:
        raise ValueError("Extracted terminal evidence differs from the frozen export")
    if "transfer_receipt.json" in entries:
        raise ValueError("Preserve the source transfer receipt; do not nest imported studies")
    if identities(reference_output, main_output, dataset_path, source_root)[1] != local_proof:
        raise RuntimeError("Local prepared inputs changed during collection")
    if sha(Path(transport.__file__)) != dependency_hash:
        raise RuntimeError("Reused transport source changed during collection")
    hashes(main_output, proof["main_input_sha256"], "Original audited inputs before merge")
    hashes(source_root, proof["audit_source_sha256"], "Reviewed audit source before merge")
    # Only separately named local re-audits may accompany remote original bytes.
    original_allowed = transport._local_extra_allowed
    transport._local_extra_allowed = lambda n: original_allowed(n) or n in ("independent_audit.local.json", "independent_audit.validation.local.json")
    try:
        merged = transport.merge_verified_files(extracted, entries, output)
    finally:
        transport._local_extra_allowed = original_allowed
    receipt = dict(schema_version=1, collected_at=transport.stamp(), mode=proof["mode"],
        remote_alias=alias, remote_root=str(remote_root), remote_export_directory=response["export_directory"],
        local_staging_directory=str(staging), archive_sha256=manifest["archive_sha256"], archive_size=manifest["archive_size"],
        manifest_sha256=response["manifest_sha256"], collector_source_sha256=hashlib.sha256(source).hexdigest(),
        reused_transport_sha256=sha(Path(transport.__file__)), protocol_sha256=local_proof["protocol_sha256"],
        audit_file=proof["audit_file"], audit_sha256=proof["audit_sha256"],
        terminal_guard=proof, exactfiles=[{"path": n, "sha256": item["sha256"]} for n, item in sorted(entries.items())],
        file_count=len(entries), total_bytes=manifest["total_bytes"],
        local_reaudit_policy="Keep original independent_audit*.json bytes; write separate independent_audit.local.json or independent_audit.validation.local.json", **merged)
    transport.write_new_json(staging / "collection_receipt.json", receipt)
    if not merged["receipt_reused"]:
        transport.write_new_json(output / "transfer_receipt.json", receipt)
    return dict(status="collected", mode=proof["mode"], output_directory=str(output), staging_directory=str(staging), file_count=len(entries))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--main-output", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--dataset", type=Path, default=transport.DEFAULT_DATASET)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--ssh-alias", default=transport.DEFAULT_ALIAS)
    parser.add_argument("--remote-python", default=transport.DEFAULT_REMOTE_PYTHON)
    parser.add_argument("--remote-root", default=REMOTE_ROOT)
    parser.add_argument("--remote-main-output", default=REMOTE_MAIN)
    parser.add_argument("--remote-dataset", default=REMOTE_DATASET)
    parser.add_argument("--remote-source-root", default=REMOTE_SOURCE)
    parser.add_argument("--remote-export-parent")
    parser.add_argument("--staging-dir", type=Path)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--validation-audit-name", default=DEFAULT_VALIDATION_AUDIT)
    parser.add_argument("--remote-export-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--remote-read-file", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.remote_read_file:
        # Reuse the original collector's restricted generated-artifact reader.
        sys.argv = [sys.argv[0], "--remote-root", args.remote_root, "--remote-read-file", args.remote_read_file]
        if args.remote_export_parent:
            sys.argv += ["--remote-export-parent", args.remote_export_parent]
        transport.main()
        return
    if args.remote_export_worker:
        result = export_snapshot(args.remote_root, main=args.main_output, dataset=args.dataset,
            source_root=args.source_root, validation_only=args.validation_only, export_parent=args.remote_export_parent,
            validation_audit_name=args.validation_audit_name)
    else:
        result = collect(output=args.output_dir, reference_output=args.reference_output_dir, main_output=args.main_output,
            dataset_path=args.dataset, source_root=args.source_root, alias=args.ssh_alias, remote_python=args.remote_python,
            remote_root=args.remote_root, remote_main=args.remote_main_output, remote_dataset=args.remote_dataset,
            remote_source=args.remote_source_root, export_parent=args.remote_export_parent, staging=args.staging_dir,
            validation_only=args.validation_only, validation_audit_name=args.validation_audit_name)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
