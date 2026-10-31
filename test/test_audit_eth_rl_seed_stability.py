"""Seed-extension audit gates; never trains, infers, or unpickles market data.

Metadata comes from a real immutable ETH account. Contract-only fixtures use
explicit non-market bytes and original source hashes. Financial-oracle calls are
small labelled stubs, not fabricated backtest evidence.
"""
from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from scripts import audit_eth_rl_seed_stability as audit


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results/backtests/single_pair_rl_six_20261030"


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def metadata_tree(tmp_path):
    meta = audit.read(OLD / "evaluation/test/ETHUSDT/sac/seed0/base/metadata.json")
    paths = []
    cases = [(a, seed, cost) for a in audit.ALGORITHMS for seed in audit.SEEDS for cost in audit.COSTS]
    cases += [(a, None, cost) for a in audit.BASELINES for cost in audit.COSTS]
    for kind in ("evaluation", "repeat"):
        for algorithm, seed, cost in cases:
            path = tmp_path / kind / "test/ETHUSDT" / algorithm / ("baseline" if seed is None else f"seed{seed}") / cost / "metadata.json"
            dump(path, meta)
            paths.append(path)
    return SimpleNamespace(root=tmp_path, paths=paths, meta=meta)


def test_real_eth_features_cover_exact_256_metadata_files(metadata_tree):
    checks, pins = audit.Checks(), {}
    audit.metadata_checks(metadata_tree.root, "test", checks, pins)
    assert not checks.failures
    assert len(pins) == len(metadata_tree.paths) == 256
    assert metadata_tree.meta["observation_features"] == audit.FEATURES


@pytest.mark.parametrize("mutation", ["btc_feature", "foreign_inventory", "foreign_pair", "dimensions", "capital"])
def test_metadata_changes_are_rejected(metadata_tree, mutation):
    changed = deepcopy(metadata_tree.meta)
    if mutation == "btc_feature":
        changed["observation_features"][0] = changed["observation_features"][0].replace("ETHUSDT", "BTCUSDT")
    elif mutation == "foreign_inventory":
        changed["final_positions"] = {"BTCUSDT": 1}
    elif mutation == "foreign_pair":
        changed["tradable_symbols"] = ["ETHUSDT", "BTCUSDT"]
    elif mutation == "dimensions":
        changed["action_dim"] = 2
    else:
        changed["independent_capital"] = 5000
    dump(metadata_tree.paths[-1], changed)
    checks = audit.Checks()
    audit.metadata_checks(metadata_tree.root, "test", checks, {})
    assert checks.failures


def test_missing_seed_metadata_is_not_silently_skipped(metadata_tree):
    metadata_tree.paths[-1].unlink()
    with pytest.raises(FileNotFoundError):
        audit.metadata_checks(metadata_tree.root, "test", audit.Checks(), {})


def test_extra_coin_metadata_is_not_accepted_as_expected_population(metadata_tree):
    dump(metadata_tree.root / "evaluation/test/BTCUSDT/sac/seed0/base/metadata.json", metadata_tree.meta)
    checks = audit.Checks()
    audit.metadata_checks(metadata_tree.root, "test", checks, {})
    assert "exact full account population:evaluation" in checks.failures


@pytest.fixture
def contract_tree(tmp_path):
    output, core = tmp_path / "study", tmp_path / "core"
    dataset = tmp_path / "data/dataset.pkl.gz"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(b"EXPLICIT SYNTHETIC CONTRACT BYTES: never unpickle")
    dump(dataset.parent / "manifest.json", {"synthetic_contract_only": True})
    retained = audit.read(OLD / "protocol.json")
    retained.update(dataset_sha256=audit.sha(dataset), dataset_manifest_sha256=audit.sha(dataset.parent / "manifest.json"))
    spec = deepcopy(retained)
    spec.update(symbols=["ETHUSDT"], algorithms=["sac", "td3"], seeds=list(range(30)), training_model_count=60,
                recipes={a: spec["recipes"][a] for a in audit.ALGORITHMS},
                timesteps_per_model={"sac": 32768, "td3": 65536},
                training_window_quality={"ETHUSDT": retained["training_window_quality"]["ETHUSDT"]},
                runtime_overlay=deepcopy(audit.RUNTIME_OVERLAY), seed_extension=deepcopy(audit.SEED_EXTENSION),
                validation_selection={"mode": "fixed_algorithm_comparison", "seed_selection": False,
                                      "algorithm_selection": False, "compatibility_anchor_algorithm": "sac"})
    for name in spec["sources"]:
        for base in (core, output / "frozen_source"):
            path = base / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, path)
    extensions = {}
    for name in ("scripts/backtest_eth_rl_seed_stability.py", "scripts/audit_eth_rl_seed_stability.py"):
        path = output / "extension_source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Path(audit.__file__).read_bytes() if name.endswith("/audit_eth_rl_seed_stability.py")
                         else b"# SYNTHETIC CONTRACT FIXTURE: never executed\n")
        extensions[name] = audit.sha(path)
    dump(output / "retained_protocol.json", retained)
    retained_models, counters = [], []
    for algorithm in audit.ALGORITHMS:
        steps, updates = (32768, 32256) if algorithm == "sac" else (65536, 65024)
        optimizers = {"actor": 32256, "critic": 32256, "entropy": 32256} if algorithm == "sac" else {"actor": 32512, "critic": 65024}
        native = {"environment_steps": steps, "gradient_update_rounds": updates, "optimizer_steps": optimizers,
                  "replay_add_calls": steps, "replay_transitions": steps}
        for seed in range(30):
            directory = f"training/ETHUSDT/{algorithm}/seed{seed}"
            record = {"symbol": "ETHUSDT", "algorithm": algorithm, "seed": seed, "synthetic_fixture_only": True,
                      "training": {"actual_steps": steps, "counters": deepcopy(native)}}
            dump(output / directory / "record.json", record)
            digest = audit.sha(output / directory / "record.json")
            counters.append({"symbol": "ETHUSDT", "algorithm": algorithm, "seed": seed,
                             "record_sha256": digest, "counters": deepcopy(native)})
            if seed < 5:
                payload = output / directory / "checkpoint/synthetic_not_a_model.txt"
                payload.parent.mkdir(parents=True, exist_ok=True)
                payload.write_text("SYNTHETIC NONMODEL CONTRACT FIXTURE\n", encoding="utf-8")
                retained_models.append({"symbol": "ETHUSDT", "algorithm": algorithm, "seed": seed,
                    "source_directory": directory, "destination_directory": directory, "source_record_sha256": digest,
                    "file_sha256": {"record.json": digest, "checkpoint/synthetic_not_a_model.txt": audit.sha(payload)}})
    imported = {"schema_version": 1, "source_protocol_sha256": audit.sha(output / "retained_protocol.json"),
        "compatibility_fields": list(audit.FINANCIAL_FIELDS), "retained_model_count": 10, "new_model_count": 50,
        "retained_models": retained_models,
        "new_models": [{"symbol": "ETHUSDT", "algorithm": a, "seed": seed} for a in audit.ALGORITHMS for seed in range(5, 30)],
        "dataset_sha256": spec["dataset_sha256"], "dataset_manifest_sha256": spec["dataset_manifest_sha256"],
        "evaluation_results_reused": False, "old_test_year_already_observed": True, "synthetic_fixture_only": True}
    extension = {"extension_sources": extensions, "runtime_overlay": deepcopy(audit.RUNTIME_OVERLAY),
        "models": 60, "retained_models": 10, "fresh_models": 50, "accounts_per_phase": 128,
        "repeat_all_cases": True, "both_costs_each_model": True}
    counter = {"passed": True, "failure_count": 0, "failures": [], "models_checked": 60, "models": counters,
               "extension_source_sha256": extensions["scripts/backtest_eth_rl_seed_stability.py"]}
    tree = SimpleNamespace(output=output, core=core, dataset=dataset, spec=spec, extension=extension,
                           imported=imported, counter=counter)
    reseal_protocol(tree)
    return tree


def reseal_protocol(tree):
    dump(tree.output / "protocol.json", tree.spec)
    digest = audit.sha(tree.output / "protocol.json")
    dump(tree.output / "protocol_receipt.json", {"protocol_sha256": digest})
    tree.imported["destination_protocol_sha256"] = digest
    dump(tree.output / "training_import.json", tree.imported)
    import_digest = audit.sha(tree.output / "training_import.json")
    dump(tree.output / "training_import_receipt.json", {"training_import_sha256": import_digest})
    tree.extension.update(core_protocol_sha256=digest, training_import_sha256=import_digest)
    dump(tree.output / "extension_manifest.json", tree.extension)
    tree.counter["core_protocol_sha256"] = digest
    dump(tree.output / "training_counter_audit.json", tree.counter)


def test_valid_contract_is_same_finance_with_extended_seed_population(contract_tree):
    _, checks, pins, _ = audit.contract_checks(contract_tree.output, contract_tree.dataset, "validation", core_root=contract_tree.core)
    assert not checks.failures
    assert "frozen_source/scripts/audit_single_pair_rl.py" in pins


@pytest.mark.parametrize("mutation", ["seed", "capital", "cost", "date", "steps", "source", "extension_source"])
def test_contract_tampering_is_rejected(contract_tree, mutation):
    t = contract_tree
    if mutation == "seed":
        t.spec["seeds"] = list(range(29))
    elif mutation == "capital":
        t.spec["capital_per_account"] = 5000
    elif mutation == "cost":
        t.spec["costs"]["base"]["fee"] = 0
    elif mutation == "date":
        t.spec["test"][1] = "2026-10-24T00:00:00Z"
    elif mutation == "steps":
        t.spec["timesteps_per_model"]["td3"] = 32768
    elif mutation == "source":
        (t.output / "frozen_source/scripts/audit_single_pair_rl.py").write_text("changed source", encoding="utf-8")
    else:
        (t.output / "extension_source/scripts/audit_eth_rl_seed_stability.py").write_text("changed wrapper", encoding="utf-8")
    reseal_protocol(t)
    _, checks, _, _ = audit.contract_checks(t.output, t.dataset, "validation", core_root=t.core)
    assert checks.failures


@pytest.mark.parametrize("mutation", ["recipe", "runtime"])
def test_recipe_or_runtime_changes_rejected_even_after_receipts_resealed(contract_tree, mutation):
    t = contract_tree
    if mutation == "recipe":
        t.spec["recipes"]["td3"]["policy_delay"] = 3
    else:
        t.spec["runtime_packages"]["torch"] = "SYNTHETIC WRONG RUNTIME"
    reseal_protocol(t)
    _, checks, _, _ = audit.contract_checks(t.output, t.dataset, "validation", core_root=t.core)
    assert checks.failures
    assert any("retained" in value or "recipes" in value for value in checks.failures)


@pytest.mark.parametrize("mutation", ["import_field", "provenance_hash"])
def test_import_fields_and_provenance_hash_cannot_be_relabeled(contract_tree, mutation):
    t = contract_tree
    if mutation == "import_field":
        t.imported["new_models"][-1]["seed"] = 28
    else:
        t.imported["source_protocol_sha256"] = "0" * 64
    reseal_protocol(t)
    _, checks, _, _ = audit.contract_checks(t.output, t.dataset, "validation", core_root=t.core)
    assert checks.failures


def test_resealed_counter_record_cannot_hide_wrong_td3_update_budget(contract_tree):
    t = contract_tree
    path = t.output / "training/ETHUSDT/td3/seed29/record.json"
    record = audit.read(path)
    record["training"]["counters"]["optimizer_steps"]["actor"] += 1
    dump(path, record)
    row = next(m for m in t.counter["models"] if m["algorithm"] == "td3" and m["seed"] == 29)
    row.update(record_sha256=audit.sha(path), counters=record["training"]["counters"])
    reseal_protocol(t)
    _, checks, _, _ = audit.contract_checks(t.output, t.dataset, "validation", core_root=t.core)
    assert any("independently recalculated native update budget" in value for value in checks.failures)


def stub_boundary(tmp_path, monkeypatch, *, financial_passed=True, wrong_oracle=False, failed_contract=False):
    output, core = tmp_path / "study", tmp_path / "core"
    output.mkdir()
    source = core / "scripts/audit_single_pair_rl.py"
    source.parent.mkdir(parents=True)
    source.write_text("# SYNTHETIC ORACLE BOUNDARY; never imported\n", encoding="utf-8")
    dump(output / "protocol.json", {"fixture_only": True})
    calls = []
    checks = audit.Checks()
    if failed_contract:
        checks.check(False, "invalid fixture contract")
    monkeypatch.setattr(audit, "contract_checks", lambda *args, **kwargs: ({}, checks, {"protocol.json": audit.sha(output / "protocol.json")}, core))
    monkeypatch.setattr(audit, "metadata_checks", lambda *args: None)
    oracle = SimpleNamespace(__file__=str(source if not wrong_oracle else core / "wrong.py"))
    def phase(actual_output, dataset, split, *, workers=None):
        calls.append((actual_output, split, workers))
        assert oracle.SYMBOLS == ("ETHUSDT",)
        assert oracle.SEEDS == tuple(range(30))
        value = {"passed": financial_passed, "failure_count": int(not financial_passed), "accounts_checked": 128, "repeats_checked": 128, "synthetic_fixture_only": True}
        dump(output / (split + "_audit.json"), value)
        return value
    oracle.audit_phase = phase
    monkeypatch.setattr(audit.importlib, "import_module", lambda name: oracle)
    return SimpleNamespace(output=output, core=core, calls=calls)


def test_oracle_runtime_overlay_and_exact_file_are_applied(tmp_path, monkeypatch):
    t = stub_boundary(tmp_path, monkeypatch)
    result = audit.audit(t.output, tmp_path / "unused", "validation", core_root=t.core, workers=3)
    assert result["passed"] is True
    assert len(t.calls) == 1 and t.calls[0][2] == 3
    assert result["runtime_overlay"] == {"oracle.SYMBOLS": ["ETHUSDT"], "oracle.SEEDS": list(range(30))}


@pytest.mark.parametrize("reason", ["failed_contract", "wrong_oracle", "financial_failure"])
def test_failed_gates_never_claim_passed(tmp_path, monkeypatch, reason):
    t = stub_boundary(tmp_path, monkeypatch, financial_passed=reason != "financial_failure",
                      wrong_oracle=reason == "wrong_oracle", failed_contract=reason == "failed_contract")
    result = audit.audit(t.output, tmp_path / "unused", "validation", core_root=t.core)
    assert result["passed"] is False and result["failure_count"] > 0
    if reason != "financial_failure":
        assert t.calls == []


@pytest.mark.parametrize("existing", ["validation_audit.json", "validation_seed_contract_audit.json"])
def test_existing_audit_evidence_is_not_overwritten_or_recomputed(tmp_path, monkeypatch, existing):
    t = stub_boundary(tmp_path, monkeypatch)
    path = t.output / existing
    original = b'{"preserve_original_audit":true}\n'
    path.write_bytes(original)
    with pytest.raises(FileExistsError):
        audit.audit(t.output, tmp_path / "unused", "validation", core_root=t.core)
    assert t.calls == []
    assert path.read_bytes() == original
