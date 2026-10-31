"""Synthetic full26 transport only; no SSH, annual prices, inference or PNL."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

import pytest

from scripts import collect_btceth_rl_td3_annual as a
from test.test_collect_btceth_rl_td3_validation import evidence as original_fixture, fill, mapping, dump


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    # Reuse only fake bytes and phase metadata from the reviewed validation
    # fixture. This does not import any research engine or actual study file.
    root, main, dataset, source = original_fixture.__wrapped__(tmp_path, monkeypatch)
    p = a.read_json(root / "protocol.json"); old_protocol_sha = a.sha(root / "protocol.json")
    monkeypatch.setattr(a.validation, "RUNNER_SHA256", p["source_sha256"]["scripts/btceth_rl_td3.py"])
    monkeypatch.setattr(a.validation, "AUDITOR_SHA256", p["source_sha256"]["scripts/audit_btceth_rl_td3.py"])
    old = a.read_json(main / "frozen_spec.json"); old["research_targets"] = a.TARGETS; dump(main / "frozen_spec.json", old)
    fill(main, a.baseline_files()); dump(main / "test_completion.json", {"synthetic_original172_completion": True})
    baseline_ids = {f"test__{cost}__{name}" for cost in ("base", "stress") for name in a.BASELINES}
    fingerprints = {key: {"equity": hashlib.sha256(key.encode()).hexdigest()} for key in baseline_ids}
    main_audit = dict(passed=True, status="passed", test_complete=True, run_count=172, failures=[],
        input_sha256={k: v for k, v in mapping(main).items() if k != "independent_audit.json"},
        runs={key: dict(reconstructed=True, hours=8760, fingerprints=value) for key, value in fingerprints.items()})
    dump(main / "independent_audit.json", main_audit)
    p.update(original_input_sha256=mapping(main), original_frozen_spec_sha256=a.sha(main / "frozen_spec.json"), research_targets=a.TARGETS)
    dump(root / "protocol.json", p); protocol_sha = a.sha(root / "protocol.json")
    def replace(value):
        if isinstance(value, dict):
            return {k: replace(v) for k, v in value.items()}
        if isinstance(value, list):
            return [replace(v) for v in value]
        return protocol_sha if value == old_protocol_sha else value
    for path in root.rglob("*.json"):
        if path.name == "protocol.json":
            continue
        try:
            value = a.read_json(path)
        except (UnicodeError, json.JSONDecodeError):
            continue
        dump(path, replace(value))
    partial = a.read_json(root / a.validation.AUDIT_FILE)
    partial.update(declared_research_targets=a.TARGETS, accounting_passage_is_not_goal_success=True,
        input_sha256={name: a.sha(root / name) for name in partial["input_sha256"]}, main_input_sha256=mapping(main))
    dump(root / a.validation.AUDIT_FILE, partial)
    selection = a.read_json(root / "selection.json")
    selection.update(audit_sha256=a.sha(root / a.validation.AUDIT_FILE),
        training_source_sha256={name: a.sha(root / name) for name in selection["training_source_sha256"]},
        validation_source_sha256={name: a.sha(root / name) for name in selection["validation_source_sha256"]})
    dump(root / "selection.json", selection)
    seal = a.read_json(root / "selection_receipt.json"); seal["selection_sha256"] = a.sha(root / "selection.json"); dump(root / "selection_receipt.json", seal)
    auth = dict(annual_evaluation_authorized=True, no_live_trading=True, financial_parameters_fixed=True, selection_sha256=a.sha(root / "selection.json"),
        authorized_at="2026-10-30T00:16:00Z", **{flag: True for flag in a.FLAGS})
    dump(root / "annual_authorization.json", auth)
    dump(root / "test_started.json", dict(at="2026-10-30T00:17:00Z", protocol_sha256=protocol_sha, selection_sha256=a.sha(root / "selection.json"), authorization_sha256=a.sha(root / "annual_authorization.json")))
    chosen = selection["candidate"]; fill(root, a.group_files(chosen["family"], chosen["profile"]))
    (root / "test.log").write_bytes(b"SYNTHETIC_ANNUAL_TERMINAL_LOG")
    command = ["python", "-u", str(PurePosixPath(str(source.resolve())) / "scripts/btceth_rl_td3.py"), "--phase", "_test", "--reference-main", str(main.resolve()), "--dataset", str(dataset.resolve()), "--output-dir", str(root.resolve())]
    dump(root / "test_process.json", dict(phase="test", status="completed", exit_code=0, source_sha256=p["source_sha256"]["scripts/btceth_rl_td3.py"], protocol_sha256=protocol_sha,
        log_relative_path="test.log", log_sha256=a.sha(root / "test.log"), command=command, supervisor_pid=2147483600, child_pid=2147483601,
        started_at="2026-10-30T00:19:00Z", completed_at="2026-10-30T00:21:00Z"))
    dump(root / "preflight/test.json", dict(at="2026-10-30T00:18:00Z"))
    dump(root / "test_completion.json", dict(protocol_sha256=protocol_sha, at="2026-10-30T00:20:00Z", case_count=12, repeat_count=12,
        candidate=chosen, selection_sha256=a.sha(root / "selection.json")))
    ids = {f"test__{cost}__{chosen['family']}__td3__{chosen['profile']}__{key}" for cost in ("base", "stress") for key in (*map(str, range(5)), "ensemble")}
    gates = {key: key != "at_least_three_positive_training_seeds" for key in a.GATES}
    research = dict(candidate={k: chosen[k] for k in ("algorithm", "family", "profile")}, gates=gates, goal_performance_achieved=False, all_five_diagnostic_targets_met=False,
        base_net_return=.12, base_max_drawdown=-.18, stress_net_return=.05, stress_max_drawdown=-.19, base_fees_usdt=500., stress_fees_usdt=1000.,
        positive_base_training_seeds=2, base_seed_returns=[dict(seed=s, net_return=.10 if s in (1, 3) else -.01) for s in range(5)],
        beats_simple_trend_controls={"sma200_trend": True, "momentum180": True}, stronger_control_comparison_met=True,
        fresh_holdout_success=False, training_budget_multiple=2, recipe_is_pure_official_default=False, accounting_passage_is_not_goal_success=True, **{flag: True for flag in a.FLAGS})
    audit = dict(passed=True, status="passed", phase="test", validation_complete=True, test_complete=True, training_model_count=10, validation_run_count=48,
        test_run_count=12, baseline_run_count=14, run_count=26, repeat_count=12, failure_count=0, failures=[], checks_total=100,
        fresh_holdout_success=False, training_budget_multiple=2, accounting_passage_is_not_goal_success=True,
        declared_research_targets=a.TARGETS, research_targets=research, dataset_sha256=p["dataset_sha256"], source_sha256=p["source_sha256"],
        runs={key: dict(reconstructed=True, hours=8760, repeat_exact=True) for key in ids},
        baseline_runs={key: dict(reconstructed=True, hours=8760, fingerprints=fingerprints[key]) for key in baseline_ids},
        audited_at="2026-10-30T00:22:00Z", **{flag: True for flag in a.FLAGS})
    def reseal():
        audit.update(input_sha256={name: value for name, value in mapping(root).items() if name not in ("independent_audit.json", "goal_candidate_evidence.json")}, main_input_sha256=mapping(main))
        dump(root / "independent_audit.json", audit)
    def goal():
        dump(root / "goal_candidate_evidence.json", dict(at="2026-10-30T00:23:00Z", protocol_sha256=protocol_sha, selection_sha256=a.sha(root / "selection.json"),
            audit_sha256=a.sha(root / "independent_audit.json"), accounting_audit_passed=True, **research))
    reseal(); goal()
    return root, main, dataset, source, audit, reseal, goal


def guard(evidence):
    root, main, dataset, source = evidence[:4]
    return a.terminal_guard(root, main=main, dataset=dataset, source_root=source)


def test_full26_and_original48_can_be_transported_with_failed_goal(evidence):
    before = a.transport.inventory(evidence[0])
    proof = guard(evidence)
    assert proof["mode"] == "full" and proof["new_test_accounts"] == 12 and proof["baseline_reference_accounts"] == 14
    assert proof["original_validation_accounts"] == 48 and proof["training_models"] == 10 and proof["test_repeats"] == 12
    assert proof["goal_performance_achieved"] is False and proof["financial_recomputation"] is False and proof["fresh_holdout_success"] is False
    assert before == a.transport.inventory(evidence[0])


@pytest.mark.parametrize("damage", ["partial48", "25", "12", "missing_ref", "no_repeat"])
def test_validation_or_smaller_scope_cannot_masquerade_as_final26(evidence, damage):
    root, _, _, _, audit, reseal, _ = evidence
    if damage == "partial48":
        audit.update(status="partial_test_pending", phase="validation", run_count=48, test_complete=False)
    elif damage == "missing_ref":
        audit["baseline_runs"].pop(next(iter(audit["baseline_runs"])))
    else:
        audit.update({"run_count": int(damage)} if damage in ("25", "12") else {"repeat_count": 11})
    reseal()
    with pytest.raises(ValueError):
        guard(evidence)


@pytest.mark.parametrize("damage", ["missing_auth", "early_auth", "late_marker", "wrong_selection"])
def test_annual_auth_and_sealed_choice_time_order_are_required(evidence, damage):
    root = evidence[0]
    path = root / ("test_started.json" if damage == "late_marker" else "annual_authorization.json")
    if damage == "missing_auth":
        path.unlink()
    else:
        value = a.read_json(path)
        value.update({"authorized_at": "2026-10-30T00:14:00Z"} if damage == "early_auth" else {"at": "2026-10-30T00:20:00Z"} if damage == "late_marker" else {"selection_sha256": "0" * 64})
        dump(path, value)
    with pytest.raises((ValueError, FileNotFoundError)):
        guard(evidence)


@pytest.mark.parametrize("kind", ["baseline", "old_audit", "selection", "model", "source", "log", "trace"])
def test_changed_references_selection_model_source_or_byte_input_are_rejected(evidence, kind):
    root, main, _, source = evidence[:4]
    paths = dict(baseline=main / "evaluation/test/baselines/base/btc_hold/fills.csv", old_audit=main / "independent_audit.json", selection=root / "selection.json",
        model=root / "training/td3__return__seed0/model.zip", source=source / "scripts/btceth_rl_td3.py", log=root / "test.log",
        trace=root / "test/raw/balanced/base/ensemble/member_targets.csv")
    paths[kind].write_bytes(b"CHANGED")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        guard(evidence)


def test_goal_identity_or_value_cannot_be_replaced_by_a_success_label(evidence):
    path = evidence[0] / "goal_candidate_evidence.json"; value = a.read_json(path); value["goal_performance_achieved"] = True; dump(path, value)
    with pytest.raises(ValueError, match="goal evidence"):
        guard(evidence)


def test_export_and_extract_preserve_remote_audits_and_old_receipts(evidence, tmp_path):
    root, main, dataset, source = evidence[:4]
    before = a.transport.inventory(root); old_guard = a.transport.terminal_guard
    result = a.export_snapshot(root, main=main, dataset=dataset, source_root=source, export_parent=tmp_path / "exports")
    manifest = a.read_json(result["manifest_path"]); extracted = tmp_path / "extracted"
    entries = a.transport.verify_extract(result["archive_path"], manifest, extracted)
    assert a.transport.inventory(root) == before and a.transport.terminal_guard is old_guard
    assert a.sha(extracted / "independent_audit.json") == a.sha(root / "independent_audit.json")
    assert a.sha(extracted / "independent_audit.validation.json") == a.sha(root / "independent_audit.validation.json")
    assert len(entries) == len(before)


def test_annual_target_cannot_reuse_or_nest_the_validation_receipt_tree(evidence, tmp_path, monkeypatch):
    root, main, dataset, source = evidence[:4]
    monkeypatch.setattr(a.transport, "_remote_call", lambda *args, **kwargs: pytest.fail("no SSH"))
    for output in (root, root / "nested", main):
        with pytest.raises(ValueError, match="sibling"):
            a.collect(output=output, reference_output=root, main_output=main, dataset_path=dataset, source_root=source)
    target = tmp_path / "annual"; target.mkdir(); dump(target / "transfer_receipt.json", {"mode": "validation_only"})
    with pytest.raises(ValueError, match="validation transfer receipt"):
        a.collect(output=target, reference_output=root, main_output=main, dataset_path=dataset, source_root=source)


@pytest.mark.parametrize("kind", ["root", "main"])
def test_original_link_root_is_rejected_before_resolution(evidence, tmp_path, monkeypatch, kind):
    root, main, dataset, source = evidence[:4]; linked = tmp_path / "mock-linked-root"
    original = a.transport.is_link; monkeypatch.setattr(a.transport, "is_link", lambda p: Path(p) == linked or original(p))
    with pytest.raises(ValueError, match="links or junctions"):
        a.terminal_guard(linked if kind == "root" else root, main=linked if kind == "main" else main, dataset=dataset, source_root=source)
    assert not linked.exists()


@pytest.mark.parametrize("member,kind", [("../escape", "file"), ("link", "symlink")])
def test_full_archive_cannot_cross_path_or_link_boundaries(tmp_path, member, kind):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo(member)
        if kind == "symlink":
            info.type = tarfile.SYMTYPE; info.linkname = "../escape"; tar.addfile(info)
        else:
            info.size = 1; tar.addfile(info, io.BytesIO(b"x"))
    manifest = dict(schema_version=1, files=[], file_count=0, total_bytes=0, archive_sha256=a.sha(archive), archive_size=archive.stat().st_size)
    with pytest.raises(ValueError):
        a.transport.verify_extract(archive, manifest, tmp_path / "extract")


@pytest.mark.parametrize("damage", [None, "manifest_sha", "validation_receipt"])
def test_full_collect_is_three_local_simulations_and_preserves_validation_receipt(evidence, tmp_path, monkeypatch, damage):
    root, main, dataset, source = evidence[:4]
    exported = a.export_snapshot(root, main=main, dataset=dataset, source_root=source, export_parent=tmp_path / "exports")
    manifest = a.read_json(exported["manifest_path"]); proof = deepcopy(manifest["terminal_guard"])
    remote_root, remote_parent = "/remote/td3", "/remote/annual_exports"
    remote_dir = remote_parent + "/full-/20261030T010203Z-" + "a" * 32
    response = {**exported, "remote_root": remote_root, "export_directory": remote_dir, "archive_path": remote_dir + "/experiment_snapshot.tar.gz", "manifest_path": remote_dir + "/snapshot_manifest.json"}
    manifest["remote_root"] = remote_root; manifest_bytes = json.dumps(manifest).encode()
    response["manifest_sha256"] = "0" * 64 if damage == "manifest_sha" else hashlib.sha256(manifest_bytes).hexdigest()
    reference = tmp_path / "prepared_validation"
    for name in ("protocol.json", "protocol_receipt.json"):
        target = reference / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes((root / name).read_bytes())
    dump(reference / "transfer_receipt.json", dict(mode="validation_only", protocol_sha256=proof["protocol_sha256"], audit_file=a.validation.AUDIT_FILE,
        audit_sha256="0" * 64 if damage == "validation_receipt" else proof["validation_audit_sha256"], synthetic_fixture_only=True))
    dump(reference / "preflight/local_partial_audit.json", {"synthetic_original_local_audit": True})
    before = a.transport.inventory(reference); calls = []
    def remote_call(alias, python, args, payload, *, destination=None):
        calls.append(list(args)); assert payload == Path(a.__file__).read_bytes()
        for key, expected in {"--remote-root": remote_root, "--remote-export-parent": remote_parent, "--main-output": "/remote/main", "--dataset": "/remote/data/dataset.pkl.gz",
            "--source-root": "/remote/formal", "--transport-source": "/remote/old54ef.py", "--validation-collector-source": "/remote/deps/validation0dea.py"}.items():
            assert args.count(key) == 1 and args[args.index(key) + 1] == expected
        if destination is None:
            assert args[-1] == "--remote-export-worker"; return deepcopy(response)
        requested = args[args.index("--remote-read-file") + 1]
        if requested == response["manifest_path"]:
            Path(destination).write_bytes(manifest_bytes)
        else:
            assert requested == response["archive_path"]; shutil.copyfile(exported["archive_path"], destination)
    def extracted_guard(*args, **kwargs):
        assert kwargs["process_root"] == remote_root and kwargs["process_main"] == "/remote/main"
        assert kwargs["process_dataset"] == "/remote/data/dataset.pkl.gz" and kwargs["process_source"] == "/remote/formal"
        return deepcopy(proof)
    monkeypatch.setattr(a.transport, "_remote_call", remote_call); monkeypatch.setattr(a, "terminal_guard", extracted_guard)
    output, staging = tmp_path / "annual_snapshot", tmp_path / "staging"
    options = dict(output=output, reference_output=reference, main_output=main, dataset_path=dataset, source_root=source, remote_root=remote_root,
        remote_main="/remote/main", remote_dataset="/remote/data/dataset.pkl.gz", remote_source="/remote/formal", remote_transport="/remote/old54ef.py",
        remote_validation_helpers="/remote/deps/validation0dea.py", export_parent=remote_parent, staging=staging)
    if damage:
        with pytest.raises(ValueError, match="manifest SHA|Retained validation receipt"):
            a.collect(**options)
        assert len(calls) == (2 if damage == "manifest_sha" else 3) and not output.exists()
    else:
        result = a.collect(**options); receipt = a.read_json(output / "transfer_receipt.json")
        assert len(calls) == 3 and result["goal_performance_achieved"] is False and receipt["mode"] == "full"
        assert receipt["new_test_accounts"] == 12 and receipt["baseline_reference_accounts"] == 14 and receipt["original_validation_accounts"] == 48
        assert receipt["retained_validation_transfer_receipt_sha256"] == a.sha(reference / "transfer_receipt.json")
        assert (staging / "retained_validation_transfer_receipt.json").read_bytes() == (reference / "transfer_receipt.json").read_bytes()
        assert a.sha(output / "independent_audit.json") == a.sha(root / "independent_audit.json")
    assert a.transport.inventory(reference) == before


def test_changed_validation_dependency_is_not_executed(tmp_path):
    source = tmp_path / "helper.py"; source.write_text("raise AssertionError('must not execute')")
    with pytest.raises(ValueError, match="reviewed0dea"):
        a.load_validation_helpers(source)
