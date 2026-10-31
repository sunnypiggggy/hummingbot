"""Collect a terminal, audited BTC/ETH study without overwriting evidence.

The same standard-library-only file runs through SSH stdin as the remote
snapshot worker. Downloads and extraction remain in a new staging directory.
Original experiment, source, checkpoint and record bytes are never rewritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import uuid

ROOT = Path(__file__).resolve().parents[1] if globals().get("__file__") not in (None, "<stdin>") else Path.cwd()
DEFAULT_OUTPUT = ROOT / "results/backtests/btceth_rl_20261029"
DEFAULT_DATASET = ROOT / "data/crypto_rl_btceth_20261029/dataset.pkl.gz"
DEFAULT_ALIAS = "v100-192-168-8-120"
DEFAULT_REMOTE_PYTHON = "/home/sunny/research-envs/btceth-rl-20261029/bin/python"
DEFAULT_REMOTE_ROOT = "/home/sunny/research/btceth_rl_20261029/experiment"
DEVICE_NAMES = {"CON", "PRN", "AUX", "NUL", *["COM" + str(i) for i in range(1, 10)], *["LPT" + str(i) for i in range(1, 10)]}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    if Path(path).is_symlink():
        raise ValueError("Refuse to read a symbolic link: " + str(path))
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Expected an ordinary file")
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise RuntimeError("File changed during hashing: " + str(path))
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def safe_name(value):
    name = str(value)
    if not name or "\\" in name or any(char in name for char in '\x00<>:"|?*'):
        raise ValueError("Unsafe portable evidence member name: " + repr(name))
    posix = PurePosixPath(name)
    if posix.is_absolute() or posix.as_posix() != name or any(part in ("", ".", "..") for part in posix.parts):
        raise ValueError("Evidence member must be canonical and relative")
    for part in posix.parts:
        if part.endswith((" ", ".")) or part.split(".")[0].upper() in DEVICE_NAMES:
            raise ValueError("Unsafe Windows member name: " + repr(name))
    return name


def is_link(path):
    path = Path(path)
    if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
        return True
    if path.exists():
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return False


def safe_target(root, name):
    root = Path(root)
    name = safe_name(name)
    if is_link(root):
        raise ValueError("Evidence root cannot be a symlink or junction")
    root = root.resolve()
    target = root
    for component in PurePosixPath(name).parts:
        target /= component
        if is_link(target):
            raise ValueError("Evidence path crosses a symlink or junction")
    target.resolve().relative_to(root)
    return target


def inventory(root):
    root = Path(root)
    if is_link(root):
        raise ValueError("Experiment root cannot be a symlink")
    root = root.resolve()
    if not root.is_dir():
        raise ValueError("Experiment root does not exist")
    found, folded = [], set()
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in directories:
            if is_link(Path(parent) / name):
                raise ValueError("Refuse a linked directory inside the experiment")
        for name in files:
            path = Path(parent) / name
            relative = safe_name(path.relative_to(root).as_posix())
            if relative.casefold() in folded:
                raise ValueError("Case-colliding evidence files are not portable")
            folded.add(relative.casefold())
            path = safe_target(root, relative)
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError("Experiment contains a link or nonregular file")
            found.append({"path": relative, "size": path.stat().st_size, "sha256": sha(path)})
    return sorted(found, key=lambda item: item["path"])


def terminal_guard(root, *, validation_only=False):
    """Fail before export unless process exit and finished audit agree."""
    if is_link(root):
        raise ValueError("Experiment root cannot be a linked directory")
    root = Path(root).resolve()
    phase = "fit" if validation_only else "test"
    split = "validation" if validation_only else "test"
    audit_name = "validation_audit.json" if validation_only else "independent_audit.json"
    required = ("frozen_spec.json", split + "_completion.json", audit_name, phase + "_process.json")
    for name in required:
        if not safe_target(root, name).is_file():
            raise ValueError("Terminal collection requires " + name)
    spec_hash = sha(root / "frozen_spec.json")
    spec = read_json(root / "frozen_spec.json")
    process = read_json(root / (phase + "_process.json"))
    if process.get("phase") != phase or process.get("status") != "completed" or type(process.get("exit_code")) is not int or process["exit_code"] != 0:
        raise ValueError("Research process is not verified completed with exit zero")
    if process.get("frozen_spec_sha256") != spec_hash:
        raise ValueError("Process proof belongs to a different frozen protocol")
    log_name = process.get("log_relative_path")
    if log_name != phase + ".log" or sha(safe_target(root, log_name)) != process.get("log_sha256"):
        raise ValueError("Terminal process log does not match its proof")
    completion = read_json(root / (split + "_completion.json"))
    if completion.get("frozen_spec_sha256") != spec_hash:
        raise ValueError("Completion belongs to a different frozen protocol")
    audited = read_json(root / audit_name)
    expected_status = "partial_test_pending" if validation_only else "passed"
    expected_count = 110 if validation_only else 172
    if (audited.get("passed") is not True or audited.get("status") != expected_status
            or audited.get("test_complete") is not (not validation_only) or audited.get("failures") != []
            or audited.get("run_count", audited.get("runs_checked")) != expected_count
            or audited.get("checks_total", 0) <= 0):
        raise ValueError("Finished independent audit does not cover the requested terminal scope")
    if audited.get("frozen_spec_sha256") != spec_hash or audited.get("dataset_sha256") != spec.get("dataset_sha256"):
        raise ValueError("Audit protocol/data identity mismatch")
    if audited.get("source_sha256") != spec.get("code_sha256"):
        raise ValueError("Audit and protocol source hashes differ")
    audited_inputs = audited.get("input_sha256")
    required_inputs = {"frozen_spec.json", "validation_selection.json", "training_completion.json",
                       "summary.csv", "daily_equity.csv", "monthly_returns.csv", "validation_completion.json"}
    if not validation_only:
        required_inputs |= {"test_started.json", "test_completion.json"}
    if not isinstance(audited_inputs, dict) or not required_inputs.issubset(audited_inputs):
        raise ValueError("Audit does not bind all required immutable input files")
    for name, expected in audited_inputs.items():
        if not re.fullmatch(r"[0-9a-f]{64}", str(expected)) or sha(safe_target(root, name)) != expected:
            raise ValueError("Audited input changed: " + name)
    protected = {name: sha(root / name) for name in required}
    protected[log_name] = sha(root / log_name)
    return {"mode": "validation_only" if validation_only else "full", "frozen_spec_sha256": spec_hash,
            "terminal_proof_sha256": protected, "audit_input_sha256": dict(audited_inputs)}


def export_snapshot(root, *, validation_only=False, export_parent=None):
    if is_link(root):
        raise ValueError("Experiment root cannot be a linked directory")
    root = Path(root).resolve()
    first_guard = terminal_guard(root, validation_only=validation_only)
    before = inventory(root)
    parent = Path(export_parent) if export_parent else root.parent.parent / "btceth_rl_exports"
    if is_link(parent):
        raise ValueError("Export parent cannot be a linked directory")
    parent = parent.resolve()
    if parent == root or parent.is_relative_to(root):
        raise ValueError("Export must be outside the experiment directory")
    parent.mkdir(parents=True, exist_ok=True)
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex
    destination = safe_target(parent, ("validation-" if validation_only else "full-") + "/" + token)
    destination.parent.mkdir(exist_ok=True)
    destination.mkdir(exist_ok=False)
    archive = destination / "experiment_snapshot.tar.gz"
    try:
        with archive.open("xb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(mode="w", fileobj=compressed, format=tarfile.PAX_FORMAT) as tar:
                for item in before:
                    path = safe_target(root, item["path"])
                    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
                    with os.fdopen(os.open(path, flags), "rb") as stream:
                        details = os.fstat(stream.fileno())
                        if not stat.S_ISREG(details.st_mode) or details.st_size != item["size"]:
                            raise RuntimeError("Source file changed before snapshot")
                        info = tarfile.TarInfo(item["path"])
                        info.size, info.mode, info.mtime = item["size"], 0o600, 0
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        tar.addfile(info, stream)
        if inventory(root) != before or terminal_guard(root, validation_only=validation_only) != first_guard:
            raise RuntimeError("Experiment changed during snapshot; preserve failed export and retry only after diagnosis")
        manifest = {"schema_version": 1, "created_at": stamp(), "mode": first_guard["mode"],
                    "remote_root": str(root), "file_count": len(before), "total_bytes": sum(item["size"] for item in before),
                    "files": before, "terminal_guard": first_guard, "archive_name": archive.name,
                    "archive_sha256": sha(archive), "archive_size": archive.stat().st_size}
        manifest_path = destination / "snapshot_manifest.json"
        write_new_json(manifest_path, manifest)
        return {"status": "complete", "mode": first_guard["mode"], "remote_root": str(root),
                "export_directory": str(destination), "archive_path": str(archive),
                "manifest_path": str(manifest_path), "archive_sha256": manifest["archive_sha256"],
                "archive_size": manifest["archive_size"], "manifest_sha256": sha(manifest_path),
                "frozen_spec_sha256": first_guard["frozen_spec_sha256"]}
    except BaseException as error:
        write_new_json(destination / "export_failure.json", {"at": stamp(), "error_type": type(error).__name__, "error": str(error)})
        raise


def manifest_entries(manifest, *, max_total_bytes=20 * 1024**3):
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list):
        raise ValueError("Unsupported snapshot manifest")
    if len(manifest["files"]) > 100000:
        raise ValueError("Snapshot contains too many files")
    entries, folded = {}, set()
    for item in manifest["files"]:
        name = safe_name(item["path"])
        if name in entries or name.casefold() in folded:
            raise ValueError("Duplicate or case-colliding snapshot member")
        if type(item.get("size")) is not int or item["size"] < 0 or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256"))):
            raise ValueError("Invalid file size or SHA256 in snapshot manifest")
        entries[name] = dict(item)
        folded.add(name.casefold())
    total = sum(item["size"] for item in entries.values())
    if manifest.get("file_count") != len(entries) or manifest.get("total_bytes") != total or total > max_total_bytes:
        raise ValueError("Snapshot manifest size/count mismatch or expansion limit exceeded")
    return entries


def verify_extract(archive_path, manifest, destination, *, max_total_bytes=20 * 1024**3):
    archive_path, destination = Path(archive_path), Path(destination)
    entries = manifest_entries(manifest, max_total_bytes=max_total_bytes)
    if sha(archive_path) != manifest.get("archive_sha256") or archive_path.stat().st_size != manifest.get("archive_size"):
        raise ValueError("Snapshot archive SHA256 or size mismatch")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Extraction requires a new destination")
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        names, folded = set(), set()
        for member in members:
            name = safe_name(member.name)
            if (member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE) or member.sparse is not None
                    or name in names or name.casefold() in folded):
                raise ValueError("Archive has a linked, sparse, nonregular or duplicate member")
            names.add(name)
            folded.add(name.casefold())
            if name not in entries or member.size != entries[name]["size"]:
                raise ValueError("Archive member is absent from manifest or has wrong size")
        if names != set(entries):
            raise ValueError("Archive is missing manifest members")
        for member in members:
            target = safe_target(destination, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            digest, count = hashlib.sha256(), 0
            source = tar.extractfile(member)
            if source is None:
                raise ValueError("Archive regular member has no readable body")
            with source, target.open("xb") as sink:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    count += len(block)
                    if count > entries[member.name]["size"]:
                        raise ValueError("Archive member exceeds declared size")
                    sink.write(block)
                    digest.update(block)
            if count != entries[member.name]["size"] or digest.hexdigest() != entries[member.name]["sha256"]:
                raise ValueError("Extracted member SHA256 or size mismatch: " + member.name)
    if inventory(destination) != sorted(entries.values(), key=lambda item: item["path"]):
        raise ValueError("Verified extraction does not exactly match its manifest")
    return entries


def verify_local_inputs(reference_output, dataset_path, *, source_root=ROOT):
    reference_output, dataset_path, source_root = Path(reference_output).resolve(), Path(dataset_path).resolve(), Path(source_root).resolve()
    spec = read_json(reference_output / "frozen_spec.json")
    manifest_path = dataset_path.parent / "manifest.json"
    if sha(dataset_path) != spec["dataset_sha256"] or sha(manifest_path) != spec["dataset_manifest_sha256"]:
        raise ValueError("Existing local prepared dataset no longer matches the frozen protocol")
    for name, expected in spec["code_sha256"].items():
        if sha(safe_target(source_root, name)) != expected:
            raise ValueError("Existing local frozen source changed: " + name)
    return {"frozen_spec_sha256": sha(reference_output / "frozen_spec.json"),
            "dataset_sha256": spec["dataset_sha256"], "dataset_manifest_sha256": spec["dataset_manifest_sha256"],
            "source_sha256": dict(spec["code_sha256"])}


def _local_extra_allowed(name):
    parts = PurePosixPath(name).parts
    return (name in ("original_inputs_protection.json", "work_status.json", "preflight.json", "backend_preflight.json", "transfer_receipt.json")
            or parts[0] == "preflight"
            or (len(parts) == 1 and parts[0].startswith("preflight_") and parts[0].endswith(".json")))


def _receipt_map(receipt):
    files = receipt.get("exactfiles")
    if isinstance(files, list):
        result = {safe_name(item.get("path", item.get("relative_path"))): item["sha256"] for item in files}
        if len(result) != len(files):
            raise ValueError("Existing receipt contains duplicate evidence paths")
        return result
    if isinstance(files, dict):
        return {safe_name(name): value if isinstance(value, str) else value["sha256"] for name, value in files.items()}
    raise ValueError("Existing transfer receipt has an unsupported exactfiles schema")


def merge_verified_files(extracted, entries, output):
    """Precheck the complete merge before creating any destination result file."""
    extracted, output = Path(extracted).resolve(), Path(output)
    expected = sorted(entries.values(), key=lambda item: item["path"])
    if inventory(extracted) != expected:
        raise ValueError("Staging files changed after archive verification")
    if is_link(output) or (output.exists() and not output.is_dir()):
        raise ValueError("Local output must be an ordinary directory")
    output = output.resolve()
    current = inventory(output) if output.exists() else []
    for item in current:
        name = item["path"]
        if name in entries:
            if item != entries[name]:
                raise FileExistsError("Different existing result would be overwritten: " + name)
        elif not _local_extra_allowed(name):
            raise FileExistsError("Unannounced existing result would mix studies: " + name)
    old_receipt = output / "transfer_receipt.json"
    if old_receipt.is_file():
        if _receipt_map(read_json(old_receipt)) != {name: item["sha256"] for name, item in entries.items()}:
            raise FileExistsError("An existing transfer receipt binds a different snapshot; preserve it and choose a new output")
    for name in entries:
        target = safe_target(output, name)
        if target.exists() and not target.is_file():
            raise FileExistsError("Destination file path conflicts with an existing directory")
        parent = target.parent
        while parent != output:
            if parent.exists() and not parent.is_dir():
                raise FileExistsError("Destination parent conflicts with an existing file")
            parent = parent.parent
    output.mkdir(parents=True, exist_ok=True)
    copied, identical = [], []
    for name, item in sorted(entries.items()):
        source, target = safe_target(extracted, name), safe_target(output, name)
        if target.exists():
            identical.append(name)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as incoming, target.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        if target.stat().st_size != item["size"] or sha(target) != item["sha256"]:
            raise RuntimeError("Copied result does not match verified source bytes: " + name)
        copied.append(name)
    for name, item in entries.items():
        target = safe_target(output, name)
        if target.stat().st_size != item["size"] or sha(target) != item["sha256"]:
            raise RuntimeError("Final output byte check failed: " + name)
    return {"copied_files": copied, "identical_files_preserved": identical, "receipt_reused": old_receipt.is_file()}


def ssh_command(alias, python, arguments):
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@:-]*", str(alias)):
        raise ValueError("Invalid SSH alias")
    if not PurePosixPath(str(python)).is_absolute():
        raise ValueError("Remote Python must be an explicit absolute path")
    # OpenSSH passes one command through the remote user's shell. Every element
    # is shell-quoted individually; no JSON or path is composed as shell code.
    command = shlex.join([str(python), "-", *map(str, arguments)])
    return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "--", str(alias), command]


def _remote_call(alias, python, arguments, source, *, destination=None):
    command = ssh_command(alias, python, arguments)
    if destination is None:
        response = subprocess.run(command, input=source, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  check=False, timeout=3600)
    else:
        with Path(destination).open("xb") as stream:
            response = subprocess.run(command, input=source, stdout=stream, stderr=subprocess.PIPE,
                                      check=False, timeout=3600)
    if response.returncode != 0:
        detail = response.stderr.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError("SSH snapshot operation failed: " + detail)
    return json.loads(response.stdout.decode("utf-8")) if destination is None else None


def _validate_remote_export(response, remote_root, export_parent, validation_only):
    root = PurePosixPath(str(remote_root))
    parent = PurePosixPath(str(export_parent)) if export_parent else root.parent.parent / "btceth_rl_exports"
    if response.get("status") != "complete" or response.get("mode") != ("validation_only" if validation_only else "full"):
        raise ValueError("Remote worker did not return a completed snapshot of requested scope")
    if response.get("remote_root") != str(root):
        raise ValueError("Remote worker resolved a different experiment")
    export = PurePosixPath(response["export_directory"])
    prefix = "validation-" if validation_only else "full-"
    if export.parent != parent / prefix or ".." in export.parts or not export.is_absolute():
        raise ValueError("Remote snapshot escaped the expected dedicated export parent")
    if not re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{32}", export.name):
        raise ValueError("Remote snapshot lacks a unique generated directory")
    if (response.get("archive_path") != str(export / "experiment_snapshot.tar.gz")
            or response.get("manifest_path") != str(export / "snapshot_manifest.json")):
        raise ValueError("Unexpected remote snapshot artifact paths")


def collect(*, output, reference_output, dataset_path=DEFAULT_DATASET, source_root=ROOT,
            alias=DEFAULT_ALIAS, remote_python=DEFAULT_REMOTE_PYTHON, remote_root=DEFAULT_REMOTE_ROOT,
            export_parent=None, staging=None, validation_only=False):
    output, reference_output = Path(output).resolve(), Path(reference_output).resolve()
    if validation_only and (output == reference_output or output.is_relative_to(reference_output) or reference_output.is_relative_to(output)):
        raise ValueError("Validation snapshot needs a separate, non-nested output directory")
    local_proof = verify_local_inputs(reference_output, dataset_path, source_root=source_root)
    if staging is None:
        token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex
        staging = ROOT / "results/backtests/btceth_rl_transfers" / token
    staging = Path(staging)
    if is_link(staging) or staging.exists():
        raise FileExistsError("Collection requires a new staging directory")
    staging = staging.resolve()
    if staging == output or staging.is_relative_to(output) or output.is_relative_to(staging):
        raise ValueError("Staging and result output must be separate directories")
    staging.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes()
    arguments = ["--remote-export-worker", "--remote-root", str(remote_root)]
    if export_parent:
        arguments += ["--remote-export-parent", str(export_parent)]
    if validation_only:
        arguments.append("--validation-only")
    response = _remote_call(alias, remote_python, arguments, source)
    _validate_remote_export(response, remote_root, export_parent, validation_only)
    if response.get("frozen_spec_sha256") != local_proof["frozen_spec_sha256"]:
        raise ValueError("Remote snapshot protocol differs from existing local frozen inputs")
    write_new_json(staging / "remote_export_receipt.json", response)
    manifest_path, archive_path = staging / "snapshot_manifest.json", staging / "experiment_snapshot.tar.gz"
    read_arguments = ["--remote-root", str(remote_root)]
    if export_parent:
        read_arguments += ["--remote-export-parent", str(export_parent)]
    _remote_call(alias, remote_python, [*read_arguments, "--remote-read-file", response["manifest_path"]], source, destination=manifest_path)
    if sha(manifest_path) != response["manifest_sha256"]:
        raise ValueError("Downloaded manifest SHA256 mismatch")
    manifest = read_json(manifest_path)
    if (manifest.get("mode") != response["mode"] or manifest.get("remote_root") != response["remote_root"]
            or manifest.get("archive_sha256") != response["archive_sha256"]
            or manifest.get("archive_size") != response["archive_size"]):
        raise ValueError("Downloaded manifest disagrees with remote export proof")
    _remote_call(alias, remote_python, [*read_arguments, "--remote-read-file", response["archive_path"]], source, destination=archive_path)
    extracted = staging / "verified_files"
    entries = verify_extract(archive_path, manifest, extracted)
    proof = terminal_guard(extracted, validation_only=validation_only)
    if proof != manifest["terminal_guard"] or proof["frozen_spec_sha256"] != local_proof["frozen_spec_sha256"]:
        raise ValueError("Extracted terminal proofs disagree with original export")
    if "transfer_receipt.json" in entries:
        raise ValueError("Source already contains a transfer receipt; preserve it and use a separate export design")
    # Recheck local sources immediately before importing any result bytes.
    if verify_local_inputs(reference_output, dataset_path, source_root=source_root) != local_proof:
        raise RuntimeError("Local frozen inputs changed during download")
    merged = merge_verified_files(extracted, entries, output)
    receipt = {"schema_version": 1, "collected_at": stamp(), "mode": proof["mode"],
               "remote_alias": alias, "remote_root": str(remote_root), "remote_export_directory": response["export_directory"],
               "local_staging_directory": str(staging), "archive_sha256": manifest["archive_sha256"],
               "archive_size": manifest["archive_size"], "manifest_sha256": response["manifest_sha256"],
               "collector_source_sha256": hashlib.sha256(source).hexdigest(),
               "frozen_spec_sha256": local_proof["frozen_spec_sha256"], "dataset_sha256": local_proof["dataset_sha256"],
               "source_sha256": local_proof["source_sha256"], "terminal_guard": proof,
               "exactfiles": [{"path": name, "sha256": item["sha256"]} for name, item in sorted(entries.items())],
               "file_count": len(entries), "total_bytes": manifest["total_bytes"], **merged}
    write_new_json(staging / "collection_receipt.json", receipt)
    if not merged["receipt_reused"]:
        write_new_json(output / "transfer_receipt.json", receipt)
    return {"status": "collected", "output_directory": str(output), "staging_directory": str(staging),
            "receipt_path": str(output / "transfer_receipt.json"), "file_count": len(entries),
            "copied_files": len(merged["copied_files"]), "identical_files_preserved": len(merged["identical_files_preserved"]),
            "mode": proof["mode"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--ssh-alias", default=DEFAULT_ALIAS)
    parser.add_argument("--remote-python", default=DEFAULT_REMOTE_PYTHON)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--remote-export-parent")
    parser.add_argument("--staging-dir", type=Path)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--remote-export-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--remote-read-file", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.remote_read_file:
        path = Path(args.remote_read_file)
        if is_link(path) or not path.is_file():
            raise ValueError("Expected an ordinary generated export file")
        # Only fixed artifact filenames in generated export directories can be
        # streamed; the stdin program cannot become an arbitrary file reader.
        if path.name not in ("snapshot_manifest.json", "experiment_snapshot.tar.gz") or not re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{32}", path.parent.name):
            raise ValueError("Only generated snapshot artifacts may be downloaded")
        parent = Path(args.remote_export_parent) if args.remote_export_parent else Path(args.remote_root).parent.parent / "btceth_rl_exports"
        if path.parent.parent.parent.resolve() != parent.resolve() or path.parent.parent.name not in ("full-", "validation-"):
            raise ValueError("Download escaped the dedicated export parent")
        path = safe_target(parent, path.relative_to(parent).as_posix())
        with path.open("rb") as stream:
            shutil.copyfileobj(stream, sys.stdout.buffer, length=1024 * 1024)
        return
    if args.remote_export_worker:
        result = export_snapshot(args.remote_root, validation_only=args.validation_only, export_parent=args.remote_export_parent)
    else:
        result = collect(output=args.output_dir, reference_output=args.reference_output_dir, dataset_path=args.dataset,
                         source_root=args.source_root, alias=args.ssh_alias, remote_python=args.remote_python,
                         remote_root=args.remote_root, export_parent=args.remote_export_parent, staging=args.staging_dir,
                         validation_only=args.validation_only)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
