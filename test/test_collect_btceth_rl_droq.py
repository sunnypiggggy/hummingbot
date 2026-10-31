"""Pure fake-byte transport fixtures; no SSH, market, GPU or Torch imports."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

import pytest

from scripts import collect_btceth_rl_droq as c


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def fill(root, names):
    for name in names:
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(("SYNTHETIC_METADATA_FIXTURE:" + name).encode())


def mapping(root):
    return {v["path"]: v["sha256"] for v in c.transport.inventory(root)}


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    root, main, source, data_dir = (tmp_path / name for name in ("droq", "main", "source", "data"))
    for path in (root, main, source, data_dir):
        path.mkdir()
    fill(source, c.SOURCE_HASHES); monkeypatch.setattr(c, "SOURCE_HASHES", mapping(source))
    data = data_dir / "dataset.pkl.gz"; data.write_bytes(b"Not a pickle: no dataset can be loaded")
    dump(data_dir / "manifest.json", {"synthetic_only": True})
    fill(main, c.baseline_files()); dump(main / "test_completion.json", {"synthetic_only": True})
    old = dict(dataset_sha256=c.sha(data), dataset_manifest_sha256=c.sha(data_dir / "manifest.json"),
        train=["2020-01-01T00:00:00Z", "2025-01-01T00:00:00Z"], validation=["2025-01-01T00:00:00Z", "2025-10-23T00:00:00Z"],
        test=["2025-10-23T00:00:00Z", "2026-10-23T00:00:00Z"], research_targets=c.TARGETS,
        code_sha256={k: c.SOURCE_HASHES[k] for k in c.FINANCIAL})
    dump(main / "frozen_spec.json", old); dump(main / "runtime.json", {"synthetic_only": True})
    baseline_ids = {f"test__{cost}__{key}" for cost in c.COSTS for key in c.BASELINES}
    fingerprints = {key: {"equity": hashlib.sha256(key.encode()).hexdigest()} for key in baseline_ids}
    dump(main / "independent_audit.json", dict(passed=True, status="passed", test_complete=True, run_count=172, failures=[],
        input_sha256=mapping(main), runs={key: dict(reconstructed=True, hours=8760, fingerprints=v) for key, v in fingerprints.items()}))
    for name in c.SOURCE_HASHES:
        target = root / "sources" / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source / name, target)
    upstream = root / "upstream"; authored = ["LICENSE", *(f"fake_author_source{i}.py" for i in range(26))]
    fill(upstream, authored)
    manifest = dict(commit=c.UPSTREAM_COMMIT, license="MIT", file_count=27, all_git_blob_sha1_verified=True,
        files=[dict(path=name, sha256=c.sha(upstream / name)) for name in authored])
    dump(upstream / "verified_manifest.json", manifest); dump(upstream / "github_commit.json", {"synthetic_only": True}); dump(upstream / "github_tree.json", {"synthetic_only": True})
    monkeypatch.setattr(c, "UPSTREAM_MANIFEST_SHA256", c.sha(upstream / "verified_manifest.json"))
    dump(root / "effective_config_contract.json", dict(effective_constructor_kwargs=c.RECIPE, official_commit=c.UPSTREAM_COMMIT))
    monkeypatch.setattr(c, "CONFIG_SHA256", c.sha(root / "effective_config_contract.json"))
    auth = dict(authorized_at="2026-10-30T00:00:00Z", fit_validation_authorized=True, no_live_trading=True,
        reviewed_source_sha256=c.SOURCE_HASHES, hyperparameters=c.RECIPE, seeds=list(c.SEEDS), timesteps_per_model=32768,
        budgets=dict(raw=1., cap075=.75), **{flag: True for flag in c.FLAGS})
    dump(root / "preflight/fit_validation_authorization.json", auth)
    p = dict(algorithm="droq", profiles={"balanced": {"drawdown_penalty": .25, "rebalance_band": .025}}, seeds=list(c.SEEDS),
        budgets=dict(raw=1., cap075=.75), capital=10000., timesteps_per_model=32768, train_episode_days=32, model_count=5,
        validation_cases=24, validation_repeats=24, test_cases=12, test_repeats=12, cash_floor=.05, max_asset_weight=.95,
        hyperparameters=c.RECIPE, expected_counters=c.COUNTS, training_budget_multiple=1, environment_step_budget_multiple=1,
        compute_budget_comparable=False, recipe_is_pure_official_default=False, seed_selection=False, selection_uses_test=False,
        new_recipe_search=False, annual_requires_separate_authorization=True, research_targets=c.TARGETS,
        source_sha256=c.SOURCE_HASHES, prepared_at="2026-10-30T00:01:00Z", original_input_sha256=mapping(main),
        original_frozen_spec_sha256=c.sha(main / "frozen_spec.json"), upstream_evidence_sha256=mapping(upstream),
        upstream_commit=c.UPSTREAM_COMMIT, upstream_manifest_sha256=c.UPSTREAM_MANIFEST_SHA256, effective_config_contract_sha256=c.CONFIG_SHA256,
        fit_validation_authorization_sha256=c.sha(root / "preflight/fit_validation_authorization.json"), **{key: old[key] for key in ("dataset_sha256", "dataset_manifest_sha256", "train", "validation", "test")},
        **{flag: True for flag in c.FLAGS})
    dump(root / "protocol.json", p); psha = c.sha(root / "protocol.json")
    dump(root / "protocol_receipt.json", dict(protocol_file="protocol.json", protocol_sha256=psha, sealed_at="2026-10-30T00:02:00Z"))
    def phase(name, minute, **counts):
        (root / (name + ".log")).write_bytes(b"SYNTHETIC_TERMINAL_ONLY")
        command = ["python", "-u", str(PurePosixPath(str(source.resolve())) / "scripts/btceth_rl_droq.py"), "--phase", "_" + name,
            "--reference-main", str(main.resolve()), "--dataset", str(data.resolve()), "--output-dir", str(root.resolve())]
        dump(root / (name + "_process.json"), dict(phase=name, status="completed", exit_code=0, source_sha256=c.SOURCE_HASHES["scripts/btceth_rl_droq.py"],
            protocol_sha256=psha, command=command, supervisor_pid=2147483600, child_pid=2147483601, log_relative_path=name + ".log", log_sha256=c.sha(root / (name + ".log")),
            started_at=f"2026-10-30T00:{minute+1:02}:00Z", completed_at=f"2026-10-30T00:{minute+3:02}:00Z"))
        dump(root / (name + "_completion.json"), dict(at=f"2026-10-30T00:{minute+2:02}:00Z", protocol_sha256=psha, **counts))
        dump(root / f"preflight/{name}.json", dict(at=f"2026-10-30T00:{minute:02}:00Z"))
    phase("fit", 3, model_count=5); phase("validate", 7, case_count=24, repeat_count=24)
    fill(root, c.training_files() | c.validation_files())
    records = []
    for seed in c.SEEDS:
        folder = f"training/droq__balanced__seed{seed}"
        tensor = folder + "/model_tensors.pt"; model = folder + "/model.json"; trace = folder + "/training_timeout_transitions.jsonl"
        dump(root / model, dict(parameter_sha256=hashlib.sha256(str(seed).encode()).hexdigest(), tensor_sha256=c.sha(root / tensor), counters=c.COUNTS))
        record = dict(algorithm="droq", profile="balanced", seed=seed, actual_timesteps=32768, actual_counters=c.COUNTS,
            hyperparameters=c.RECIPE, protocol_sha256=psha, model_tensor_file=tensor, model_tensor_sha256=c.sha(root / tensor),
            model_manifest_file=model, model_manifest_sha256=c.sha(root / model), timeout_trace_relative_path=trace, timeout_trace_sha256=c.sha(root / trace),
            parameter_sha256=hashlib.sha256(str(seed).encode()).hexdigest())
        dump(root / (folder + "/record.json"), record); records.append(record)
    dump(root / "training_completion.json", dict(model_count=5, protocol_sha256=psha, records=records))
    dump(root / "validation_completion.json", dict(case_count=24, repeat_count=24, protocol_sha256=psha))
    ids = {f"validation__{cost}__{family}__droq__balanced__{key}" for family in c.FAMILIES for cost in c.COSTS for key in (*map(str, c.SEEDS), "ensemble")}
    partial = dict(passed=True, status="partial_test_pending", phase="validation", validation_complete=True, test_complete=False,
        training_model_count=5, validation_run_count=24, test_run_count=0, baseline_run_count=0, run_count=24, repeat_count=24,
        failure_count=0, failures=[], checks_total=123, fresh_holdout_success=False, training_budget_multiple=1, recipe_is_pure_official_default=False,
        accounting_passage_is_not_goal_success=True, declared_research_targets=c.TARGETS, research_targets=None, source_sha256=c.SOURCE_HASHES,
        dataset_sha256=p["dataset_sha256"], audited_at="2026-10-30T00:11:00Z", input_sha256=mapping(root), main_input_sha256=mapping(main),
        runs={key: dict(reconstructed=True, hours=7080, repeat_exact=True) for key in ids}, **{flag: True for flag in c.FLAGS})
    dump(root / c.VALIDATION_AUDIT, partial)
    pool = [dict(algorithm="droq", family=f, profile="balanced", score=.1, rebalance_band=.025, exposure_multiplier=1. if f == "raw" else .75) for f in c.FAMILIES]
    selection = dict(protocol_sha256=psha, audit_sha256=c.sha(root / c.VALIDATION_AUDIT), selected_at="2026-10-30T00:12:00Z",
        candidates=pool, candidate=pool[0], seed_selection=False, selection_uses_test=False, test_results_available_to_selector=False,
        training_source_sha256={n: c.sha(root / n) for n in c.training_files(started=False)}, validation_source_sha256={n: c.sha(root / n) for n in c.validation_files()}, **{flag: True for flag in c.FLAGS})
    dump(root / "selection.json", selection)
    dump(root / "selection_receipt.json", dict(selection_file="selection.json", selection_sha256=c.sha(root / "selection.json"), protocol_sha256=psha, sealed_at="2026-10-30T00:13:00Z"))
    return dict(root=root, main=main, dataset=data, source=source, protocol=p, phase=phase, validation_inventory=mapping(root), fingerprints=fingerprints)


@pytest.fixture
def full(evidence):
    e = evidence; root, p = e["root"], e["protocol"]; psha = c.sha(root / "protocol.json")
    selection = c.read_json(root / "selection.json"); chosen = selection["candidate"]
    dump(root / "annual_authorization.json", dict(authorized_at="2026-10-30T00:14:00Z", annual_evaluation_authorized=True, no_live_trading=True,
        financial_parameters_fixed=True, selection_sha256=c.sha(root / "selection.json"), **{flag: True for flag in c.FLAGS}))
    dump(root / "test_started.json", dict(at="2026-10-30T00:15:00Z", protocol_sha256=psha, selection_sha256=c.sha(root / "selection.json"), authorization_sha256=c.sha(root / "annual_authorization.json")))
    e["phase"]("test", 16, case_count=12, repeat_count=12, candidate=chosen, selection_sha256=c.sha(root / "selection.json"))
    fill(root, c.group_files("test", chosen["family"]))
    gates = {k: k != "at_least_three_positive_training_seeds" for k in c.GATES}
    research = dict(candidate={k: chosen[k] for k in ("algorithm", "family", "profile")}, gates=gates,
        goal_performance_achieved=False, all_five_diagnostic_targets_met=False, fresh_holdout_success=False, training_budget_multiple=1,
        recipe_is_pure_official_default=False, accounting_passage_is_not_goal_success=True,
        base_net_return=.1, base_max_drawdown=-.18, stress_net_return=.01, stress_max_drawdown=-.19, base_fees_usdt=500., stress_fees_usdt=1000.,
        positive_base_training_seeds=2, base_seed_returns=[dict(seed=s, net_return=.1 if s in (1, 3) else -.01) for s in c.SEEDS],
        beats_simple_trend_controls={"sma200_trend": True, "momentum180": False}, stronger_control_comparison_met=False, **{flag: True for flag in c.FLAGS})
    ids = {f"test__{cost}__{chosen['family']}__droq__balanced__{key}" for cost in c.COSTS for key in (*map(str, c.SEEDS), "ensemble")}
    audit = dict(passed=True, status="passed", phase="test", validation_complete=True, test_complete=True,
        training_model_count=5, validation_run_count=24, test_run_count=12, baseline_run_count=14, run_count=26, repeat_count=12, failure_count=0, failures=[], checks_total=456,
        fresh_holdout_success=False, training_budget_multiple=1, recipe_is_pure_official_default=False, accounting_passage_is_not_goal_success=True,
        declared_research_targets=c.TARGETS, source_sha256=c.SOURCE_HASHES, dataset_sha256=p["dataset_sha256"], research_targets=research,
        audited_at="2026-10-30T00:20:00Z", runs={key: dict(reconstructed=True, hours=8760, repeat_exact=True) for key in ids},
        baseline_runs={key: dict(reconstructed=True, hours=8760, fingerprints=value) for key, value in e["fingerprints"].items()}, **{flag: True for flag in c.FLAGS})
    def reseal():
        audit.update(input_sha256={name: digest for name, digest in mapping(root).items() if name not in ("independent_audit.json", "goal_candidate_evidence.json")}, main_input_sha256=mapping(e["main"]))
        dump(root / "independent_audit.json", audit)
    def goal():
        dump(root / "goal_candidate_evidence.json", dict(at="2026-10-30T00:21:00Z", protocol_sha256=psha, selection_sha256=c.sha(root / "selection.json"),
            audit_sha256=c.sha(root / "independent_audit.json"), accounting_audit_passed=True, **research))
    reseal(); goal(); e.update(audit=audit, reseal=reseal, goal=goal)
    return e


def guard(e, mode="validation_only"):
    return c.terminal_guard(e["root"], main=e["main"], dataset=e["dataset"], source_root=e["source"], mode=mode)


def test_validation24_is_sha_only_complete_and_cannot_claim_annual(evidence):
    before = c.transport.inventory(evidence["root"]); result = guard(evidence)
    assert result["training_models"] == 5 and result["original_validation_accounts"] == result["validation_repeats"] == 24
    assert result["financial_recomputation"] is False and result["audit_file"] == c.VALIDATION_AUDIT
    assert c.transport.inventory(evidence["root"]) == before
    with pytest.raises(FileNotFoundError):
        guard(evidence, "full")


def test_full26_preserves_false_goal_and_old_validation24(full):
    result = guard(full, "full")
    assert result["new_test_accounts"] == 12 and result["baseline_reference_accounts"] == 14
    assert result["goal_performance_achieved"] is False and result["fresh_holdout_success"] is False
    with pytest.raises(ValueError, match="annual/test"):
        guard(full)


@pytest.mark.parametrize("damage", ["partial", "smaller", "model_count", "repeat", "bool_count"])
def test_partial_or_small_scope_cannot_masquerade_as_full26(full, damage):
    audit = full["audit"]
    audit.update({"status": "partial_test_pending", "test_complete": False} if damage == "partial" else {"run_count": 12} if damage == "smaller"
        else {"training_model_count": 4} if damage == "model_count" else {"repeat_count": 11} if damage == "repeat" else {"training_budget_multiple": True})
    full["reseal"]()
    with pytest.raises(ValueError, match="scope"):
        guard(full, "full")


@pytest.mark.parametrize("kind", ["model", "manifest", "timeout", "source", "config", "upstream", "selection", "baseline", "fit_auth"])
def test_every_model_source_reference_and_contract_dependency_is_bound(evidence, kind):
    e = evidence
    file = dict(model=e["root"] / "training/droq__balanced__seed0/model_tensors.pt", manifest=e["root"] / "training/droq__balanced__seed0/model.json",
        timeout=e["root"] / "training/droq__balanced__seed0/training_timeout_transitions.jsonl", source=e["source"] / "scripts/btceth_rl_droq.py",
        config=e["root"] / "effective_config_contract.json", upstream=e["root"] / "upstream/LICENSE", selection=e["root"] / "selection.json",
        baseline=e["main"] / "evaluation/test/baselines/base/btc_hold/fills.csv", fit_auth=e["root"] / "preflight/fit_validation_authorization.json")[kind]
    file.write_bytes(b"TAMPERED")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        guard(e)


@pytest.mark.parametrize("kind", ["missing", "early", "wrong_selection"])
def test_annual_authorization_is_independent_and_after_selection(full, kind):
    path = full["root"] / "annual_authorization.json"
    if kind == "missing":
        path.unlink()
    else:
        value = c.read_json(path); value.update({"authorized_at": "2026-10-30T00:12:00Z"} if kind == "early" else {"selection_sha256": "0" * 64}); dump(path, value)
    with pytest.raises((ValueError, FileNotFoundError)):
        guard(full, "full")


def test_original_goal_outcome_cannot_be_relabelled(full):
    path = full["root"] / "goal_candidate_evidence.json"; value = c.read_json(path); value["goal_performance_achieved"] = True; dump(path, value)
    with pytest.raises(ValueError, match="goal evidence"):
        guard(full, "full")


@pytest.mark.parametrize("mode", ["validation_only", "full"])
def test_export_extract_and_conflict_precheck_preserve_raw_bytes(evidence, full, tmp_path, mode):
    # full fixture has added year files; validation export uses a clean copy of
    # the original validation inventory without changing any source bytes.
    e = full
    if mode == "validation_only":
        copy = tmp_path / "val_copy"
        for name in e["validation_inventory"]:
            target = copy / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(e["root"] / name, target)
        # The process metadata still binds its original root explicitly.
        original = c.terminal_guard
        def wrapper(root, **kw):
            return original(root, **kw, process_root=e["root"])
        # Export's top-level entry has no process-root parameter. This fixture
        # uses annual mode separately; validation actual export is exercised by
        # test_collect_three_legs below using its unmodified original fixture.
        proof = original(copy, main=e["main"], dataset=e["dataset"], source_root=e["source"], mode=mode, process_root=e["root"])
        assert proof["mode"] == "validation_only"
        return
    before = c.transport.inventory(e["root"])
    result = c.export_snapshot(e["root"], main=e["main"], dataset=e["dataset"], source_root=e["source"], mode=mode, export_parent=tmp_path / "exports")
    manifest = c.read_json(result["manifest_path"]); extracted = tmp_path / "extracted"
    entries = c.transport.verify_extract(result["archive_path"], manifest, extracted)
    assert c.transport.inventory(e["root"]) == before and c.sha(extracted / "independent_audit.json") == c.sha(e["root"] / "independent_audit.json")
    output = tmp_path / "conflict"; output.mkdir(); (output / "protocol.json").write_bytes(b"different existing source")
    with pytest.raises(FileExistsError):
        c.transport.merge_verified_files(extracted, entries, output)
    assert sorted(p.name for p in output.iterdir()) == ["protocol.json"]


@pytest.mark.parametrize("member,kind", [("../escape", "file"), ("link", "symlink")])
def test_archive_traversal_and_symlinks_are_rejected(tmp_path, member, kind):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo(member)
        if kind == "symlink":
            info.type = tarfile.SYMTYPE; info.linkname = "../escape"; tar.addfile(info)
        else:
            info.size = 1; tar.addfile(info, io.BytesIO(b"x"))
    manifest = dict(schema_version=1, files=[], file_count=0, total_bytes=0, archive_sha256=c.sha(archive), archive_size=archive.stat().st_size)
    with pytest.raises(ValueError):
        c.transport.verify_extract(archive, manifest, tmp_path / "extracted")


@pytest.mark.parametrize("field", ["root", "main", "source"])
def test_original_links_are_refused_before_resolution(evidence, tmp_path, monkeypatch, field):
    linked = tmp_path / "mock_link"; original = c.transport.is_link
    monkeypatch.setattr(c.transport, "is_link", lambda path: Path(path) == linked or original(path))
    values = dict(root=evidence["root"], main=evidence["main"], dataset=evidence["dataset"], source_root=evidence["source"], mode="validation_only")
    values[{"source": "source_root"}.get(field, field)] = linked
    with pytest.raises(ValueError, match="links or junctions"):
        c.terminal_guard(**values)
    assert not linked.exists()


@pytest.mark.parametrize("mode,damage", [("validation_only", None), ("full", None), ("full", "manifest"), ("full", "old_receipt")])
def test_collect_three_legs_receipts_same_bytes_and_remote_options(evidence, tmp_path, monkeypatch, mode, damage):
    e = evidence
    if mode == "full":
        e = full.__wrapped__(e)
    exported = c.export_snapshot(e["root"], main=e["main"], dataset=e["dataset"], source_root=e["source"], mode=mode, export_parent=tmp_path / "exports")
    manifest = c.read_json(exported["manifest_path"]); proof = deepcopy(manifest["terminal_guard"])
    remote_root, remote_parent = "/remote/droq", "/remote/droq_exports"
    prefix = "validation-" if mode == "validation_only" else "full-"
    directory = remote_parent + "/" + prefix + "/20261030T010203Z-" + "a" * 32
    response = {**exported, "remote_root": remote_root, "export_directory": directory, "manifest_path": directory + "/snapshot_manifest.json", "archive_path": directory + "/experiment_snapshot.tar.gz"}
    manifest["remote_root"] = remote_root; manifest_bytes = json.dumps(manifest).encode()
    response["manifest_sha256"] = "0" * 64 if damage == "manifest" else hashlib.sha256(manifest_bytes).hexdigest()
    reference = tmp_path / "retained_validation"
    for name in e["validation_inventory"]:
        target = reference / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(e["root"] / name, target)
    if mode == "full":
        dump(reference / "transfer_receipt.json", dict(mode="validation_only", protocol_sha256=proof["protocol_sha256"], audit_file=c.VALIDATION_AUDIT,
            audit_sha256="0" * 64 if damage == "old_receipt" else proof["validation_audit_sha256"], exactfiles=[dict(path=n, sha256=s) for n, s in e["validation_inventory"].items()]))
    before = c.transport.inventory(reference); calls = []
    def remote(alias, python, argv, payload, *, destination=None):
        calls.append(argv); assert payload == Path(c.__file__).read_bytes()
        for key, expected in {"--mode": mode, "--remote-root": remote_root, "--remote-export-parent": remote_parent,
                "--main-output": "/remote/main", "--dataset": "/remote/data/dataset.pkl.gz", "--source-root": "/remote/source", "--transport-source": "/remote/helpers/54ef.py"}.items():
            assert argv.count(key) == 1 and argv[argv.index(key)+1] == expected
        if destination is None:
            assert argv[-1] == "--remote-export-worker"; return deepcopy(response)
        name = argv[argv.index("--remote-read-file")+1]
        if name == response["manifest_path"]:
            Path(destination).write_bytes(manifest_bytes)
        else:
            assert name == response["archive_path"]; shutil.copyfile(exported["archive_path"], destination)
    original_guard = c.terminal_guard
    def extracted_guard(root, **kw):
        assert kw["process_root"] == remote_root and kw["process_main"] == "/remote/main"
        assert kw["process_dataset"] == "/remote/data/dataset.pkl.gz" and kw["process_source"] == "/remote/source"
        # Actually rerun every metadata/hash gate over the extracted files,
        # using the fixture's honest process command identities.
        kw.update(process_root=e["root"], process_main=e["main"], process_dataset=e["dataset"], process_source=e["source"])
        return original_guard(root, **kw)
    monkeypatch.setattr(c.transport, "_remote_call", remote); monkeypatch.setattr(c, "terminal_guard", extracted_guard)
    output = tmp_path / "snapshot"; output.mkdir(); shutil.copyfile(e["root"] / "protocol.json", output / "protocol.json")
    options = dict(output=output, reference_output=reference, main_output=e["main"], dataset_path=e["dataset"], source_root=e["source"], mode=mode,
        remote_root=remote_root, remote_main="/remote/main", remote_dataset="/remote/data/dataset.pkl.gz", remote_source="/remote/source", remote_transport="/remote/helpers/54ef.py", export_parent=remote_parent, staging=tmp_path / "staging")
    if damage:
        with pytest.raises(ValueError, match="manifest SHA|Retained validation receipt"):
            c.collect(**options)
        assert not (output / "transfer_receipt.json").exists()
    else:
        result = c.collect(**options)
        assert len(calls) == 3 and result["status"] == "collected" and result["mode"] == mode
        receipt = c.read_json(output / "transfer_receipt.json")
        assert receipt["manifest_sha256"] == response["manifest_sha256"] and receipt["collector_source_sha256"] == c.sha(Path(c.__file__))
        assert "protocol.json" in receipt["identical_files_preserved"] and receipt["remote_audits_preserved"] is True
        assert c.sha(output / c.VALIDATION_AUDIT) == c.sha(e["root"] / c.VALIDATION_AUDIT)
        if mode == "full":
            assert result["goal_performance_achieved"] is False and c.sha(output / "independent_audit.json") == c.sha(e["root"] / "independent_audit.json")
    assert c.transport.inventory(reference) == before


def test_existing_validation_receipt_and_nested_output_are_never_promoted(evidence, tmp_path, monkeypatch):
    e = evidence; monkeypatch.setattr(c.transport, "_remote_call", lambda *a, **k: pytest.fail("no SSH"))
    options = dict(reference_output=e["root"], main_output=e["main"], dataset_path=e["dataset"], source_root=e["source"], mode="full")
    for output in (e["root"], e["root"] / "nested", e["main"]):
        with pytest.raises(ValueError, match="sibling"):
            c.collect(output=output, **options)
    output = tmp_path / "annual"; output.mkdir(); dump(output / "transfer_receipt.json", dict(mode="validation_only"))
    with pytest.raises(ValueError, match="cannot become"):
        c.collect(output=output, **options)


def test_reviewed_safe_transport_dependency_cannot_be_modified(tmp_path):
    path = tmp_path / "fake54ef.py"; path.write_text("# changed")
    with pytest.raises(ValueError, match="reviewed54ef"):
        c.load_transport(path)
