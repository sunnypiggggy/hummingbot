"""Fake byte fixtures only: no SSH, data prices, model or financial import."""
from copy import deepcopy
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

import pytest

from scripts import collect_btceth_rl_td3_validation as c


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def fill(root, names):
    for name in names:
        path = root / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("SYNTHETIC_NOT_A_RESULT:" + name).encode())


def mapping(root):
    return {p.relative_to(root).as_posix(): c.sha(p) for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    root, main, source, dataset = tmp_path / "td3", tmp_path / "main", tmp_path / "repo", tmp_path / "data/dataset.pkl.gz"
    for directory in (root, main, source, dataset.parent):
        directory.mkdir(parents=True)
    dataset.write_bytes(b"SYNTHETIC_NOT_MARKET_DATA")
    dump(dataset.parent / "manifest.json", {"synthetic_fixture_only": True})
    fill(source, c.SOURCE_NAMES)
    source_sha = mapping(source)
    monkeypatch.setattr(c, "RUNNER_SHA256", source_sha["scripts/btceth_rl_td3.py"])
    monkeypatch.setattr(c, "AUDITOR_SHA256", source_sha["scripts/audit_btceth_rl_td3.py"])
    old = dict(dataset_sha256=c.sha(dataset), dataset_manifest_sha256=c.sha(dataset.parent / "manifest.json"), code_sha256={n: source_sha[n] for n in c.FINANCIAL})
    dump(main / "frozen_spec.json", old); dump(main / "runtime.json", {"synthetic_fixture_only": True}); dump(main / "independent_audit.json", {"synthetic_fixture_only": True})
    for name in source_sha:
        path = root / "sources" / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes((source / name).read_bytes())
    fill(root / "upstream", {"LICENSE", "verified_manifest.json"})
    p = dict(algorithm="td3", model_count=10, timesteps_per_model=65536, old_timesteps_per_model=32768, training_budget_multiple=2,
        train_episode_days=32, validation_cases=48, validation_repeats=48, seeds=list(range(5)), profiles={profile: {} for profile in c.PROFILES},
        budgets={"raw": 1., "cap075": .75}, capital=10000., seed_selection=False, selection_uses_test=False,
        annual_requires_separate_authorization=True, recipe_is_pure_official_default=False,
        original_frozen_spec_sha256=c.sha(main / "frozen_spec.json"), dataset_sha256=old["dataset_sha256"], dataset_manifest_sha256=old["dataset_manifest_sha256"],
        source_sha256=source_sha, original_input_sha256=mapping(main), upstream_evidence_sha256=mapping(root / "upstream"), prepared_at="2026-10-30T00:01:00Z", **{flag: True for flag in c.FLAGS})
    dump(root / "protocol.json", p); psha = c.sha(root / "protocol.json")
    dump(root / "protocol_receipt.json", dict(protocol_file="protocol.json", protocol_sha256=psha, sealed_at="2026-10-30T00:02:00Z"))
    for phase, pre, start, complete, end in (("fit", "03", "04", "06", "07"), ("validate", "08", "09", "11", "12")):
        (root / (phase + ".log")).write_bytes(b"SYNTHETIC_TERMINAL_LOG")
        command = ["python", "-u", str(PurePosixPath(str(source.resolve())) / "scripts/btceth_rl_td3.py"), "--phase", "_" + phase,
            "--reference-main", str(main.resolve()), "--dataset", str(dataset.resolve()), "--output-dir", str(root.resolve())]
        dump(root / (phase + "_process.json"), dict(phase=phase, status="completed", exit_code=0, source_sha256=source_sha["scripts/btceth_rl_td3.py"],
            protocol_sha256=psha, log_relative_path=phase + ".log", log_sha256=c.sha(root / (phase + ".log")), command=command,
            supervisor_pid=2147483600, child_pid=2147483601, started_at="2026-10-30T00:" + start + ":00Z", completed_at="2026-10-30T00:" + end + ":00Z"))
        value = dict(protocol_sha256=psha, at="2026-10-30T00:" + complete + ":00Z")
        value.update(model_count=10) if phase == "fit" else value.update(case_count=48, repeat_count=48)
        dump(root / (phase + "_completion.json"), value)
        dump(root / "preflight" / (phase + ".json"), dict(at="2026-10-30T00:" + pre + ":00Z"))
    fill(root, c.validation_files() | c.training_files())
    records = []
    for profile in c.PROFILES:
        for seed in c.SEEDS:
            folder = f"training/td3__{profile}__seed{seed}"
            record = dict(algorithm="td3", profile=profile, seed=seed, actual_timesteps=65536, actual_n_updates=65024, actor_update_count=32512,
                protocol_sha256=psha, model_relative_path=folder + "/model.zip", model_sha256=c.sha(root / folder / "model.zip"),
                timeout_trace_relative_path=folder + "/training_timeout_transitions.jsonl", timeout_trace_sha256=c.sha(root / folder / "training_timeout_transitions.jsonl"))
            dump(root / folder / "record.json", record); records.append(record)
    dump(root / "training_completion.json", dict(protocol_sha256=psha, model_count=10, records=records))
    dump(root / "validation_completion.json", dict(protocol_sha256=psha, case_count=48, repeat_count=48))
    run_ids = {f"validation__{cost}__{family}__td3__{profile}__{key}" for family in c.FAMILIES for profile in c.PROFILES for cost in ("base", "stress") for key in (*map(str, c.SEEDS), "ensemble")}
    audit = dict(passed=True, status="partial_test_pending", phase="validation", validation_complete=True, test_complete=False,
        training_model_count=10, validation_run_count=48, run_count=48, test_run_count=0, baseline_run_count=0, repeat_count=48,
        failure_count=0, failures=[], research_targets=None, training_budget_multiple=2, fresh_holdout_success=False,
        checks_total=100, dataset_sha256=old["dataset_sha256"], source_sha256=source_sha, input_sha256=mapping(root), main_input_sha256=mapping(main),
        runs={n: dict(reconstructed=True, hours=7080, repeat_exact=True) for n in run_ids}, audited_at="2026-10-30T00:13:00Z", **{flag: True for flag in c.FLAGS})
    dump(root / c.AUDIT_FILE, audit)
    pool = [dict(algorithm="td3", family=family, profile=profile, score=1.) for family in c.FAMILIES for profile in sorted(c.PROFILES)]
    pool.sort(key=lambda r: (-r["score"], 0 if r["family"] == "raw" else 1, r["profile"]))
    selected = dict(protocol_sha256=psha, audit_sha256=c.sha(root / c.AUDIT_FILE), candidates=pool, candidate=pool[0], selected_at="2026-10-30T00:14:00Z",
        seed_selection=False, selection_uses_test=False, test_results_available_to_selector=False,
        training_source_sha256={n: c.sha(root / n) for n in c.training_files(include_started=False)},
        validation_source_sha256={n: c.sha(root / n) for n in c.validation_files()}, **{flag: True for flag in c.FLAGS})
    dump(root / "selection.json", selected)
    dump(root / "selection_receipt.json", dict(selection_file="selection.json", selection_sha256=c.sha(root / "selection.json"), protocol_sha256=psha, sealed_at="2026-10-30T00:15:00Z"))
    return root, main, dataset, source


def guard(evidence):
    root, main, data, source = evidence
    return c.terminal_guard(root, main=main, dataset=data, source_root=source)


def test_full48_sealed_validation_has_no_annual_or_financial_capability(evidence):
    before = c.transport.inventory(evidence[0])
    proof = guard(evidence)
    assert proof["mode"] == "validation_only" and proof["training_models"] == 10 and proof["validation_cases"] == proof["validation_repeats"] == 48
    assert proof["annual_reads_enabled"] is False and proof["financial_recomputation"] is False
    assert before == c.transport.inventory(evidence[0])


@pytest.mark.parametrize("damage", ["42", "partial", "failed", "no_repeat", "unreconstructed"])
def test_partial_or_smaller_audit_never_exports(evidence, damage):
    root = evidence[0]; path = root / c.AUDIT_FILE; value = c.read_json(path)
    if damage == "unreconstructed":
        value["runs"][next(iter(value["runs"]))]["reconstructed"] = False
    else:
        value.update({"run_count": 42} if damage == "42" else {"status": "running"} if damage == "partial" else {"passed": False} if damage == "failed" else {"repeat_count": 47})
    dump(path, value)
    with pytest.raises(ValueError):
        guard(evidence)


@pytest.mark.parametrize("name", ["test_started.json", "annual_authorization.json", "test/raw/return/summary.csv", "independent_audit.json", "goal_candidate_evidence.json"])
def test_even_annual_marker_or_scope_refuses_before_reading_file_content(evidence, name, monkeypatch):
    root = evidence[0]; path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"MUST_NOT_BE_READ")
    original = c.sha
    def reject(target):
        assert Path(target).resolve() != path.resolve(), "annual content must not be hashed or read"
        return original(target)
    monkeypatch.setattr(c, "sha", reject)
    with pytest.raises(ValueError, match="annual/test"):
        guard(evidence)


def test_missing_choice_seal_does_not_qualify_as_a_finished_validation(evidence):
    (evidence[0] / "selection_receipt.json").unlink()
    with pytest.raises(FileNotFoundError):
        guard(evidence)


@pytest.mark.parametrize("kind", ["source", "snapshot", "old", "model", "log", "input", "data"])
def test_changed_frozen_sources_old_dependencies_model_log_or_input_are_rejected(evidence, kind):
    root, main, data, source = evidence
    target = {"source": source / "scripts/btceth_rl_td3.py", "snapshot": root / "sources/scripts/btceth_rl_td3.py",
        "old": main / "runtime.json", "model": root / "training/td3__return__seed0/model.zip", "log": root / "fit.log",
        "input": root / "validation/raw/return/base/ensemble/member_targets.csv", "data": data}[kind]
    target.write_bytes(b"CHANGED")
    with pytest.raises(ValueError):
        guard(evidence)


def test_export_extract_and_merge_preserve_prepared_same_bytes_and_local_preflight(evidence, tmp_path):
    root, main, data, source = evidence
    before, original_guard = c.transport.inventory(root), c.transport.terminal_guard
    response = c.export_snapshot(root, main=main, dataset=data, source_root=source, export_parent=tmp_path / "exports")
    assert c.transport.terminal_guard is original_guard and c.transport.inventory(root) == before
    manifest = c.read_json(response["manifest_path"]); extracted = tmp_path / "extract"
    entries = c.transport.verify_extract(response["archive_path"], manifest, extracted)
    local = tmp_path / "local"
    for name in ("protocol.json", "protocol_receipt.json"):
        path = local / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes((root / name).read_bytes())
    dump(local / "preflight/local_seal_receipt.json", {"synthetic_local_provenance": True})
    result = c.transport.merge_verified_files(extracted, entries, local)
    assert result["identical_files_preserved"] == ["protocol.json", "protocol_receipt.json"]
    assert c.sha(local / c.AUDIT_FILE) == c.sha(root / c.AUDIT_FILE)
    assert (local / "preflight/local_seal_receipt.json").exists()


def test_changed_source_during_archive_is_detected_without_any_global_patch(evidence, tmp_path, monkeypatch):
    root, main, data, source = evidence
    original, old_guard = c.transport.inventory, c.transport.terminal_guard
    calls = 0
    def inventory(where):
        nonlocal calls
        calls += 1
        if calls == 2:
            (root / "validation/summary.csv").write_bytes(b"CHANGED_DURING_SNAPSHOT")
        assert c.transport.terminal_guard is old_guard
        return original(where)
    monkeypatch.setattr(c.transport, "inventory", inventory)
    with pytest.raises(RuntimeError, match="changed during"):
        c.export_snapshot(root, main=main, dataset=data, source_root=source, export_parent=tmp_path / "exports")
    assert list((tmp_path / "exports").rglob("export_failure.json"))
    assert c.transport.terminal_guard is old_guard


def test_different_existing_sha_is_never_overwritten(evidence, tmp_path):
    root, main, data, source = evidence
    response = c.export_snapshot(root, main=main, dataset=data, source_root=source, export_parent=tmp_path / "exports")
    manifest = c.read_json(response["manifest_path"]); extracted = tmp_path / "extract"
    entries = c.transport.verify_extract(response["archive_path"], manifest, extracted)
    output = tmp_path / "local"; output.mkdir(); (output / "protocol.json").write_bytes(b"DIFFERENT_EXISTING_PROTOCOL")
    before = c.transport.inventory(output)
    with pytest.raises(FileExistsError, match="overwritten"):
        c.transport.merge_verified_files(extracted, entries, output)
    assert c.transport.inventory(output) == before


@pytest.mark.parametrize("member,kind", [("../escape", "file"), ("symlink", "link")])
def test_old_safe_extractor_rejects_traversal_and_links(tmp_path, member, kind):
    archive = tmp_path / "malicious.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo(member)
        if kind == "link":
            info.type = tarfile.SYMTYPE; info.linkname = "../escape"; tar.addfile(info)
        else:
            info.size = 1; tar.addfile(info, io.BytesIO(b"x"))
    manifest = dict(schema_version=1, files=[], file_count=0, total_bytes=0, archive_sha256=c.sha(archive), archive_size=archive.stat().st_size)
    with pytest.raises(ValueError):
        c.transport.verify_extract(archive, manifest, tmp_path / "extract")


def test_collection_path_guard_blocks_old_study_before_any_ssh(evidence, tmp_path, monkeypatch):
    root, main, data, source = evidence
    monkeypatch.setattr(c.transport, "_remote_call", lambda *a, **k: pytest.fail("SSH must never execute in fixtures"))
    with pytest.raises(ValueError, match="separate"):
        c.collect(output=main, reference_output=root, main_output=main, dataset_path=data, source_root=source)


def test_modified_safe_transport_is_refused_without_execution(tmp_path):
    path = tmp_path / "collector.py"; path.write_text("raise AssertionError('must not execute')")
    with pytest.raises(ValueError, match="reviewed"):
        c.load_transport(path)


@pytest.mark.parametrize("kind", ["root", "main"])
def test_original_link_identity_is_rejected_before_resolve_without_side_effects(evidence, tmp_path, monkeypatch, kind):
    root, main, data, source = evidence
    before = c.transport.inventory(root), c.transport.inventory(main)
    link = tmp_path / "mock-linked-root"
    original = c.transport.is_link
    monkeypatch.setattr(c.transport, "is_link", lambda path: Path(path) == link or original(path))
    with pytest.raises(ValueError, match="links or junctions"):
        c.terminal_guard(link if kind == "root" else root, main=link if kind == "main" else main, dataset=data, source_root=source)
    assert not link.exists() and before == (c.transport.inventory(root), c.transport.inventory(main))


@pytest.mark.parametrize("damage", [None, "manifest_sha", "manifest_audit_receipt"])
def test_collect_three_remote_calls_and_receipt_boundaries_are_fully_local(evidence, tmp_path, monkeypatch, damage):
    root, main, data, source = evidence
    real = c.export_snapshot(root, main=main, dataset=data, source_root=source, export_parent=tmp_path / "exports")
    proof = deepcopy(c.read_json(real["manifest_path"])["terminal_guard"])
    token = "20261030T010203Z-" + "a" * 32
    remote_root, remote_parent = "/remote/td3", "/remote/exports"
    remote_dir = remote_parent + "/validation-/" + token
    response = {**real, "remote_root": remote_root, "export_directory": remote_dir,
        "archive_path": remote_dir + "/experiment_snapshot.tar.gz", "manifest_path": remote_dir + "/snapshot_manifest.json"}
    manifest = c.read_json(real["manifest_path"]); manifest["remote_root"] = remote_root
    if damage == "manifest_audit_receipt":
        manifest["terminal_guard"]["audit_sha256"] = "b" * 64
    manifest_bytes = json.dumps(manifest).encode()
    response["manifest_sha256"] = c.hashlib.sha256(manifest_bytes).hexdigest() if damage != "manifest_sha" else "0" * 64
    local = tmp_path / "prepared"
    for name in ("protocol.json", "protocol_receipt.json"):
        target = local / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes((root / name).read_bytes())
    dump(local / "preflight/local_seal_receipt.json", {"fixture_local_only": True})
    before = c.transport.inventory(local)
    calls, checked_process_binding = [], []
    def remote_call(alias, python, args, payload, *, destination=None):
        calls.append((list(args), destination))
        assert payload == Path(c.__file__).read_bytes()
        for option, expected in {"--remote-root": remote_root, "--remote-export-parent": remote_parent,
            "--main-output": "/remote/main", "--dataset": "/remote/data/dataset.pkl.gz", "--source-root": "/remote/formal_source",
            "--transport-source": "/remote/old/scripts/collect_btceth_rl_results.py"}.items():
            assert args.count(option) == 1 and args[args.index(option) + 1] == expected
        if destination is None:
            assert args[-1] == "--remote-export-worker"
            return deepcopy(response)
        requested = args[args.index("--remote-read-file") + 1]
        if requested == response["manifest_path"]:
            Path(destination).write_bytes(manifest_bytes)
        else:
            assert requested == response["archive_path"]
            shutil.copyfile(real["archive_path"], destination)
    def extracted_guard(*args, **kwargs):
        checked_process_binding.append(kwargs)
        assert kwargs["process_root"] == remote_root and kwargs["process_main"] == "/remote/main"
        assert kwargs["process_dataset"] == "/remote/data/dataset.pkl.gz" and kwargs["process_source"] == "/remote/formal_source"
        return deepcopy(proof)
    monkeypatch.setattr(c.transport, "_remote_call", remote_call)
    # Actual terminal_guard is covered above; this fixture isolates the remote
    # argument/download/extract/merge path without pretending to launch SSH.
    monkeypatch.setattr(c, "terminal_guard", extracted_guard)
    options = dict(output=local, reference_output=local, main_output=main, dataset_path=data, source_root=source,
        remote_root=remote_root, remote_main="/remote/main", remote_dataset="/remote/data/dataset.pkl.gz", remote_source="/remote/formal_source",
        remote_transport="/remote/old/scripts/collect_btceth_rl_results.py", export_parent=remote_parent, staging=tmp_path / "staging")
    if damage:
        with pytest.raises(ValueError, match="manifest|Manifest"):
            c.collect(**options)
        assert c.transport.inventory(local) == before and len(calls) == 2 and not checked_process_binding
        return
    result = c.collect(**options)
    assert len(calls) == 3 and len(checked_process_binding) == 1 and result["status"] == "collected"
    receipt = c.read_json(local / "transfer_receipt.json")
    assert receipt["mode"] == "validation_only" and receipt["audit_sha256"] == proof["audit_sha256"]
    assert receipt["collector_source_sha256"] == c.sha(c.__file__) and receipt["manifest_sha256"] == response["manifest_sha256"]
    assert receipt["no_annual_capability"] is True and receipt["no_financial_recomputation"] is True
    assert receipt["identical_files_preserved"] == ["protocol.json", "protocol_receipt.json"]
    assert (local / "preflight/local_seal_receipt.json").exists()
