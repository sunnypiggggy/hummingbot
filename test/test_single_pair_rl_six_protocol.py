"""Six-algorithm protocol/dispatch proof without a GPU or formal training.

Market preprocessing is real. Fake checkpoint bytes are explicitly labelled
synthetic fixtures; the pool is replaced to test the exact 20-retained/40-new
task population. These tests do not certify optimizer counters or native model
payloads for the four newly activated algorithms.
"""
from copy import deepcopy
from pathlib import Path
import shutil
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import audit_single_pair_rl as audit
from scripts import backtest_btceth_rl as old_loader
from scripts import backtest_single_pair_rl as runner
from scripts import single_pair_rl_models as models
from scripts import extend_single_pair_rl as extension
from test.test_backtest_single_pair_rl import synthetic_market


RETAINED = ("sac", "tqc")
NEW = ("crossq", "recurrent_ppo", "td3", "droq")


@pytest.fixture
def studies(tmp_path, monkeypatch, synthetic_market):
    source = tmp_path / "source"
    names = ("scripts/backtest_single_pair_rl.py", "scripts/single_pair_rl_env.py", "scripts/single_pair_rl_models.py")
    for name in names:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Synthetic protocol fixture source: " + name, encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", source)
    monkeypatch.setattr(runner, "SOURCE_FILES", names)
    monkeypatch.setattr(old_loader, "load_dataset", lambda path: synthetic_market)
    monkeypatch.setattr(models, "recipe", lambda algorithm: {"synthetic_recipe_only": algorithm, "observation_dim": 13, "action_dim": 1})
    monkeypatch.setattr(models, "runtime_versions", lambda: {"synthetic_runtime_only": "pinned"})
    dataset = tmp_path / "data/dataset.pkl.gz"
    dataset.parent.mkdir()
    dataset.write_bytes(b"Synthetic dispatch fixture only; never unpickled")
    runner.write(dataset.parent / "manifest.json", {"synthetic_fixture_only": True})
    old, new = tmp_path / "old_twenty", tmp_path / "new_sixty"
    old_spec = runner.prepare(old, dataset, RETAINED)
    new_spec = runner.prepare(new, dataset, runner.ALGORITHMS)
    return SimpleNamespace(old=old, new=new, dataset=dataset, old_spec=old_spec, new_spec=new_spec)


def fake_completed_record(root, symbol, algorithm, seed, spec):
    """Only a manifest/dispatch fixture; payload is deliberately NOT a model."""
    directory = root / "training" / symbol / algorithm / f"seed{seed}"
    checkpoint = directory / "checkpoint"
    checkpoint.mkdir(parents=True, exist_ok=False)
    (checkpoint / "model.zip").write_bytes(f"SYNTHETIC NONMODEL {symbol} {algorithm} {seed}".encode())
    steps = spec["timesteps_per_model"][algorithm]
    counters = {"environment_steps": steps}
    if algorithm in RETAINED:
        updates = steps - 512
        counters.update(gradient_update_rounds=updates, replay_transitions=steps, replay_add_calls=steps,
                        optimizer_steps={"actor": updates, "critic": updates, "entropy": updates})
    training = {"actual_steps": steps, "counters": counters, "synthetic_fixture_only": True}
    manifest = {"symbol": symbol, "algorithm": algorithm, "seed": seed, "observation_shape": [13], "action_shape": [1],
                "independent_account": True, "multi_model_target_average": False,
                "adapter_sha256": spec["sources"]["scripts/single_pair_rl_models.py"],
                "recipe": spec["recipes"][algorithm], "training": training,
                "checkpoint_files": {"model.zip": runner.sha(checkpoint / "model.zip")}, "synthetic_fixture_only": True}
    runner.write(checkpoint / "single_pair_model.json", manifest)
    record = {"symbol": symbol, "algorithm": algorithm, "seed": seed, "capital": 10000., "ensemble": False,
              "primary_seed": seed == 0, "observation_shape": [13], "action_shape": [1],
              "training": training, "checkpoint": manifest, "maximum_input_hour": "2024-12-31T23:00:00Z",
              "files": {path.relative_to(directory).as_posix(): runner.sha(path) for path in checkpoint.iterdir()},
              "synthetic_fixture_only": True}
    runner.write(directory / "record.json", record)
    return record


def test_retained_twenty_are_compatible_only_because_financial_and_learning_inputs_match(studies):
    old, new = studies.old_spec, studies.new_spec
    assert old["training_model_count"] == 20 and new["training_model_count"] == 60
    assert old["algorithms"] == list(RETAINED) and new["algorithms"] == list(runner.ALGORITHMS)
    for key in ("train", "validation", "test", "symbols", "seeds", "primary_seed", "primary_seed_fixed_before_training",
                "capital_per_account", "capital_shared_between_pairs", "ensemble", "environment", "costs", "sources",
                "dataset_sha256", "dataset_manifest_sha256", "runtime_packages", "python", "training_window_quality"):
        assert old[key] == new[key], key
    for algorithm in RETAINED:
        assert old["recipes"][algorithm] == new["recipes"][algorithm]
        assert old["timesteps_per_model"][algorithm] == new["timesteps_per_model"][algorithm] == 32768
    assert old["test_year_already_observed"] is new["test_year_already_observed"] is True
    assert old["fresh_holdout"] is new["fresh_holdout"] is False
    assert runner.sha(studies.old / "protocol.json") != runner.sha(studies.new / "protocol.json")
    for directory in (studies.old, studies.new):
        assert runner.read(directory / "protocol_receipt.json")["protocol_sha256"] == runner.sha(directory / "protocol.json")


def test_importing_twenty_unchanged_directories_leaves_exactly_forty_pending_tasks(studies, monkeypatch):
    before = {}
    for symbol in runner.SYMBOLS:
        for algorithm in RETAINED:
            for seed in runner.SEEDS:
                record = fake_completed_record(studies.old, symbol, algorithm, seed, studies.old_spec)
                original = studies.old / "training" / symbol / algorithm / f"seed{seed}"
                copied = studies.new / "training" / symbol / algorithm / f"seed{seed}"
                shutil.copytree(original, copied)
                assert runner.validate_training_record(studies.new, symbol, algorithm, seed, studies.new_spec) == record
                before.update({path.relative_to(studies.new).as_posix(): runner.sha(path) for path in copied.rglob("*") if path.is_file()})
    dispatched = []
    def synthetic_pool(output, tasks, function, phase, max_workers=None):
        assert phase == "fit" and function is runner.train_one
        dispatched.extend(tasks)
        return [fake_completed_record(Path(output), task["symbol"], task["algorithm"], task["seed"], studies.new_spec) for task in tasks]
    monkeypatch.setattr(runner, "run_parallel", synthetic_pool)
    runner.fit(studies.new, studies.dataset)
    expected = {(symbol, algorithm, seed) for symbol in runner.SYMBOLS for algorithm in NEW for seed in runner.SEEDS}
    assert len(dispatched) == len(expected) == 40
    assert {(task["symbol"], task["algorithm"], task["seed"]) for task in dispatched} == expected
    assert not any(task["algorithm"] in RETAINED for task in dispatched)
    assert all(runner.sha(studies.new / name) == value for name, value in before.items())
    completion = runner.read(studies.new / "fit_complete.json")
    assert completion["model_count"] == len(completion["records"]) == 60
    assert {(row["symbol"], row["algorithm"], row["seed"]) for row in completion["records"]} == {
        (symbol, algorithm, seed) for symbol in runner.SYMBOLS for algorithm in runner.ALGORITHMS for seed in runner.SEEDS}


def test_six_algorithm_validation_and_annual_dispatch_all_136_real_account_identities(studies, monkeypatch):
    runner.write(studies.new / "fit_complete.json", {"synthetic_dispatch_fixture_only": True})
    seen = {}
    def synthetic_pool(output, tasks, function, phase, max_workers=None):
        assert function is runner.evaluate_case
        seen[phase] = deepcopy(tasks)
        results = []
        for task in tasks:
            symbol, algorithm, seed, scenario = (task[key] for key in ("symbol", "algorithm", "seed", "scenario"))
            row = {"run_id": f"{phase}__{scenario}__{symbol}__{algorithm}__{'baseline' if seed is None else seed}",
                   "symbol": symbol, "algorithm": algorithm, "seed": seed, "scenario": scenario, "split": phase,
                   "initial_capital": 10000., "ensemble": False, "primary_seed": seed == 0}
            canonical = audit.case_relative(phase, symbol, algorithm, seed, scenario).as_posix()
            repeat = audit.case_relative(phase, symbol, algorithm, seed, scenario, True).as_posix()
            results.append({"summary": row, "path": canonical, "repeat_path": repeat, "repeat_equal": True})
        return results
    monkeypatch.setattr(runner, "run_parallel", synthetic_pool)
    validation = runner.evaluate(studies.new, studies.dataset, "validation")
    runner.write(studies.new / "validation_audit.json", {"passed": True, "synthetic_dispatch_fixture_only": True})
    runner.write(studies.new / "validation_selection.json", {"uses_test": False, "seed_selection": False,
        "validation_audit_sha256": runner.sha(studies.new / "validation_audit.json"),
        "validation_summary_sha256": runner.sha(studies.new / "validation_summary.csv"),
        "protocol_sha256": runner.sha(studies.new / "protocol.json"),
        "selection": {symbol: {"algorithm": "sac", "seed": 0} for symbol in runner.SYMBOLS},
        "synthetic_dispatch_fixture_only": True})
    annual = runner.evaluate(studies.new, studies.dataset, "test")
    for phase, frame in (("validation", validation), ("test", annual)):
        tasks = seen[phase]
        assert len(tasks) == len(frame) == 136
        identities = {(task["symbol"], task["algorithm"], task["seed"], task["scenario"]) for task in tasks}
        assert identities == set(audit.expected_cases(runner.ALGORITHMS))
        assert len(identities) == 136
        assert sum(task["seed"] is not None for task in tasks) == 120
        assert sum(task["seed"] is None for task in tasks) == 16
        assert sum(task["seed"] == 0 for task in tasks) == 24
        assert sum(task["symbol"] == "BTCUSDT" for task in tasks) == sum(task["symbol"] == "ETHUSDT" for task in tasks) == 68
        assert not frame.run_id.duplicated().any()
        completion = runner.read(studies.new / f"{phase}_complete.json")
        assert completion["accounts"] == completion["repeats"] == 136
        assert len({item["path"] for item in completion["results"]}) == len({item["repeat_path"] for item in completion["results"]}) == 136
        assert all(item["path"].startswith(f"evaluation/{phase}/") and item["repeat_path"].startswith(f"repeat/{phase}/") for item in completion["results"])
    pins = runner.read(studies.new / "test_started.json")["pins"]
    assert pins["selection_sha256"] == runner.sha(studies.new / "validation_selection.json")


def test_all_four_new_recipes_are_dimension_adapted_without_hidden_budget_changes():
    assert models.STEPS["td3"] == 65536
    assert all(models.STEPS[algorithm] == 32768 for algorithm in NEW if algorithm != "td3")
    td3, recurrent, droq = (models.recipe(algorithm) for algorithm in ("td3", "recurrent_ppo", "droq"))
    assert td3["action_noise"]["mean"] == [0.] and td3["action_noise"]["sigma"] == [.1]
    assert td3["policy_delay"] == 2 and td3["learning_starts"] == 512
    assert recurrent["n_steps"] == recurrent["batch_size"] == 128 and recurrent["n_epochs"] == 5
    assert (droq["obs_dim"], droq["act_dim"], droq["effective_target_entropy"]) == (13, 1, -1)
    assert droq["start_steps"] == 5000 and droq["utd_ratio"] == 20 and droq["num_Q"] == 2
    assert droq["target_drop_rate"] == .01 and droq["layer_norm"] is True


@pytest.mark.parametrize("symbol", runner.SYMBOLS)
def test_retained_record_cannot_be_relabelled_to_other_pair_or_seed(studies, symbol):
    fake_completed_record(studies.new, symbol, "sac", 0, studies.new_spec)
    other = next(value for value in runner.SYMBOLS if value != symbol)
    original = studies.new / "training" / symbol / "sac/seed0"
    copied = studies.new / "training" / other / "sac/seed1"
    shutil.copytree(original, copied)
    with pytest.raises(ValueError, match="exact pair/algorithm/seed"):
        runner.validate_training_record(studies.new, other, "sac", 1, studies.new_spec)


@pytest.mark.parametrize("damage", ["cost", "recipe", "timesteps", "capital", "cash_floor", "band", "reward", "dataset", "manifest", "cutoff", "core_source"])
def test_actual_extension_compatibility_rejects_changed_retained_economics_or_learning_contract(studies, damage):
    old, new = deepcopy(studies.old_spec), deepcopy(studies.new_spec)
    extension.compatible(old, new)
    if damage == "cost": new["costs"]["base"]["fee"] = .0005
    elif damage == "recipe": new["recipes"]["sac"]["changed_learning_rate"] = .1
    elif damage == "timesteps": new["timesteps_per_model"]["tqc"] = 65536
    elif damage == "capital": new["capital_per_account"] = 5000.
    elif damage == "cash_floor": new["environment"]["cash_floor"] = .1
    elif damage == "band": new["environment"]["rebalance_band"] = .05
    elif damage == "reward": new["environment"]["drawdown_penalty"] = 0.
    elif damage == "dataset": new["dataset_sha256"] = "0" * 64
    elif damage == "manifest": new["dataset_manifest_sha256"] = "0" * 64
    elif damage == "cutoff": new["train"][1] = "2025-02-01T00:00:00Z"
    else: new["sources"]["scripts/single_pair_rl_models.py"] = "0" * 64
    with pytest.raises(ValueError, match="differs"):
        extension.compatible(old, new)


@pytest.fixture
def counter_evidence(tmp_path):
    """Synthetic observed-counter records, no policy construction or training."""
    root = tmp_path / "counter_evidence"
    root.mkdir()
    spec = {"algorithms": list(extension.SIX_ALGORITHMS), "timesteps_per_model": dict(models.STEPS)}
    runner.write(root / "protocol.json", spec)
    for symbol in runner.SYMBOLS:
        for algorithm in extension.SIX_ALGORITHMS:
            steps = models.STEPS[algorithm]
            if algorithm in ("sac", "tqc", "crossq", "td3"):
                updates = steps - 512
                delay = 3 if algorithm == "crossq" else 2 if algorithm == "td3" else 1
                optimizers = {"actor": updates // delay, "critic": updates}
                if algorithm != "td3": optimizers["entropy"] = updates // delay
                counters = {"environment_steps": steps, "gradient_update_rounds": updates, "optimizer_steps": optimizers,
                            "replay_transitions": steps, "replay_add_calls": steps}
            elif algorithm == "recurrent_ppo":
                counters = {"environment_steps": steps, "gradient_update_rounds": 1280, "optimizer_steps": {"policy": 1280}}
            else:
                counters = {"environment_steps": steps, "q_inner_rounds": 555360, "q_optimizer_steps": [555360, 555360],
                    "total_q_optimizer_steps": 1110720, "policy_optimizer_steps": 27768, "alpha_optimizer_steps": 27768,
                    "target_updates_per_network": [555360, 555360], "learning_train_calls": 27768,
                    "replay_adds": 32768, "random_exploration_actions": 5001}
            completed = [{"start": "2020-01-01 00:00:00+00:00", "end_exclusive": "2020-02-02 00:00:00+00:00",
                          "completed_until": "2020-02-02 00:00:00+00:00", "completed_steps": 32,
                          "terminated": False, "truncated": True} for _ in range(steps // 32)]
            if algorithm != "droq":
                completed.append({"start": "2020-01-01 00:00:00+00:00", "completed_steps": 0, "terminated": False, "truncated": False})
            effective = {"obs_dim": 13, "act_dim": 1, "target_entropy": -1., "utd_ratio": 20,
                         "num_Q": 2, "num_min": 2, "policy_update_delay": 20} if algorithm == "droq" else {}
            for seed in runner.SEEDS:
                runner.write(root / f"training/{symbol}/{algorithm}/seed{seed}/record.json", {
                    "symbol": symbol, "algorithm": algorithm, "seed": seed, "synthetic_counter_fixture_only": True,
                    "training": {"counters": counters, "effective_parameters": effective, "training_episode_history": completed}})
    core = SimpleNamespace(TRAIN=runner.TRAIN, write=runner.write,
        validate_training_record=lambda output, symbol, algorithm, seed, spec: runner.read(Path(output) / f"training/{symbol}/{algorithm}/seed{seed}/record.json"))
    return root, core


def test_actual_counter_companion_accepts_distinct_native_update_formulas(counter_evidence):
    root, core = counter_evidence
    result = extension.training_counter_audit(root, core)
    assert result["passed"] is True and result["models_checked"] == 60
    assert sum(row["retained"] for row in result["models"]) == 20
    indexed = {row["algorithm"]: row["counters"] for row in result["models"]}
    assert indexed["recurrent_ppo"]["optimizer_steps"] == {"policy": 1280}
    assert indexed["td3"]["gradient_update_rounds"] == 65024
    assert indexed["td3"]["optimizer_steps"] == {"actor": 32512, "critic": 65024}
    assert indexed["droq"]["q_optimizer_steps"] == [555360, 555360]
    assert indexed["crossq"]["optimizer_steps"] == {"actor": 10752, "critic": 32256, "entropy": 10752}


@pytest.mark.parametrize("algorithm,damage", [("recurrent_ppo", "policy"), ("td3", "actor"), ("crossq", "entropy"), ("droq", "q"), ("droq", "entropy_dimension")])
def test_actual_counter_companion_rejects_wrong_delayed_or_recurrent_counts(counter_evidence, algorithm, damage):
    root, core = counter_evidence
    path = root / f"training/BTCUSDT/{algorithm}/seed0/record.json"
    record = runner.read(path)
    if damage == "q": record["training"]["counters"]["q_optimizer_steps"][0] -= 1
    elif damage == "entropy_dimension": record["training"]["effective_parameters"]["target_entropy"] = -2.
    else: record["training"]["counters"]["optimizer_steps"][damage] -= 1
    runner.write(path, record)
    with pytest.raises(ValueError, match="counter audit failed"):
        extension.training_counter_audit(root, core)
    evidence = runner.read(root / "training_counter_audit.json")
    assert evidence["passed"] is False and evidence["failure_count"] == 1
    assert evidence["failures"][0]["algorithm"] == algorithm


def test_real_twenty_checkpoints_copy_and_import_receipt_roundtrip(tmp_path, monkeypatch):
    """Exercise actual import-copy/hash code on completed local model bytes.

    Only the NEW protocol-preprocessing boundary is a fixture, avoiding a
    learner/runtime dependency on the audit machine. The original twenty real
    records/checkpoints/audits and pinned author source are read and copied.
    """
    source_root = Path(__file__).resolve().parents[1] / "results/backtests/single_pair_rl_20261030"
    dataset = Path(__file__).resolve().parents[1] / "data/crypto_rl_btceth_20261029/dataset.pkl.gz"
    upstream = Path(__file__).resolve().parents[1] / "results/backtests/btceth_rl_droq_20261030_preflight/upstream/DroQ"
    if not all(path.exists() for path in (source_root / "test_audit.json", dataset, upstream / "verified_manifest.json")):
        pytest.skip("The completed original twenty-model evidence is not available on this host")
    source_spec = runner.read(source_root / "protocol.json")
    def frozen_source_check(root, data):
        assert Path(root) == source_root and Path(data) == dataset
        assert runner.sha(data) == source_spec["dataset_sha256"]
        assert runner.sha(data.parent / "manifest.json") == source_spec["dataset_manifest_sha256"]
        assert runner.read(root / "protocol_receipt.json")["protocol_sha256"] == runner.sha(root / "protocol.json")
        return source_spec
    def new_protocol_fixture(output, data, algorithms):
        output = Path(output)
        output.mkdir()
        value = deepcopy(source_spec)
        value.update(algorithms=list(algorithms), training_model_count=60, frozen_at="synthetic preprocessing boundary only",
                     recipes={algorithm: models.recipe(algorithm) for algorithm in algorithms}, timesteps_per_model={algorithm: models.STEPS[algorithm] for algorithm in algorithms})
        for name in value["sources"]:
            target = output / "frozen_source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_root / "frozen_source" / name, target)
        runner.write(output / "protocol.json", value)
        runner.write(output / "protocol_receipt.json", {"protocol_sha256": runner.sha(output / "protocol.json")})
        return value
    core = SimpleNamespace(check_protocol=frozen_source_check, prepare=new_protocol_fixture,
                           validate_training_record=runner.validate_training_record)
    monkeypatch.setattr(extension, "core_module", lambda root=None: core)
    target = tmp_path / "real_import_only"
    before = {path.relative_to(source_root).as_posix(): runner.sha(path) for path in (source_root / "training").rglob("*") if path.is_file()}
    assert extension.prepare_extension(source_root, target, dataset, upstream_root=upstream) == {"retained": 20, "pending": 40, "accounts_per_phase": 136}
    imported = extension.validate_import(target, source_root)
    assert len(imported["retained_models"]) == 20 and len(imported["new_models"]) == 40
    assert imported["evaluation_results_reused"] is False
    assert not (target / "evaluation").exists() and not (target / "validation_selection.json").exists()
    assert all(runner.sha(source_root / name) == expected for name, expected in before.items())
    copied = {path.relative_to(target).as_posix(): runner.sha(path) for path in (target / "training").rglob("*") if path.is_file()}
    assert copied == before
    receipt = runner.read(target / "training_import_receipt.json")
    assert receipt["training_import_sha256"] == runner.sha(target / "training_import.json")
    # A content edit in the copied record must fail without changing the source.
    copied_record = target / "training/BTCUSDT/sac/seed0/record.json"
    copied_record.write_bytes(copied_record.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="bytes changed"):
        extension.validate_import(target, source_root)
    assert all(runner.sha(source_root / name) == expected for name, expected in before.items())
