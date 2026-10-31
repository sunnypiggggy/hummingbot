"""Independent validation-selection and provenance gates; no learner or training.

Tiny JSON/ZIP fixtures below are labelled synthetic, nonfinancial evidence.
Real immutable metadata supplies the two independent 13-feature contracts.
"""
from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
import pytest

from scripts import audit_sac_validation_optimization_v2 as audit

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "results/backtests/sac_validation_optimization_20261031_preflight/search_plan.json"
OLD = ROOT / "results/backtests/single_pair_rl_six_20261030"


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def copy_plan(top):
    top.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PLAN, top / "search_plan.json")
    shutil.copyfile(PLAN.with_name("search_plan_receipt.json"), top / "search_plan_receipt.json")
    return audit.load_plan(top)


def metric(score=0., scores=None):
    return {"seed_scores": scores or [score] * 5, "score_median": score, "score_iqr": 0., "robust_score": score,
        "base_positive_seeds": 5, "stress_median_return": .05, "stress_p90_drawdown": .10,
        "subwindow_median_scores": [score] * 3}


def population():
    return {c: {s: metric() for s in audit.SYMBOLS} for c in audit.CANDIDATE_ORDER}


def test_frozen_search_plan_rejects_changed_plan_or_receipt(tmp_path):
    copy_plan(tmp_path)
    assert audit.sha(tmp_path / "search_plan.json") == audit.PLAN_SHA256
    receipt = audit.read(tmp_path / "search_plan_receipt.json"); receipt["sha256"] = "0" * 64
    dump(tmp_path / "search_plan_receipt.json", receipt)
    with pytest.raises(ValueError, match="original plan"):
        audit.load_plan(tmp_path)
    copy_plan(tmp_path)
    with (tmp_path / "search_plan.json").open("a") as stream: stream.write(" ")
    with pytest.raises(ValueError, match="original plan"):
        audit.load_plan(tmp_path)


def test_selection_is_per_coin_all_gates_and_sorted_candidate_tie_break():
    plan = audit.read(PLAN); values = population()
    values["c01_lr1e4"]["BTCUSDT"] = metric(.02)
    values["c02_gamma099"]["BTCUSDT"] = metric(.03)
    values["c03_lr1e4_gamma099"]["BTCUSDT"] = metric(.03)
    selected = audit.compute_expected_selection(values, plan)
    assert selected["BTCUSDT"]["candidate_id"] == "c02_gamma099"
    assert selected["BTCUSDT"]["eligible_candidates"] == ["c02_gamma099", "c03_lr1e4_gamma099", "c01_lr1e4", "c00_baseline"]
    assert selected["ETHUSDT"]["candidate_id"] == "c00_baseline"
    assert selected["ETHUSDT"]["fallback_to_baseline"] is True
    assert selected["BTCUSDT"]["selected_recipe"]["gamma"] == .99


@pytest.mark.parametrize("gate", list(audit.read(PLAN)["acceptance"]))
def test_each_acceptance_gate_can_force_original_baseline_fallback(gate):
    values = population(); changed = metric(.02)
    if gate == "score_improvement_at_least": changed["robust_score"] = .009999
    elif gate == "paired_better_seed_count_at_least": changed["seed_scores"] = [.1, .1, 0., 0., 0.]
    elif gate == "positive_base_seed_count_at_least": changed["base_positive_seeds"] = 2
    elif gate == "stress_return_median_at_least": changed["stress_median_return"] = -.0001
    elif gate == "stress_p90_drawdown_above_baseline_at_most": changed["stress_p90_drawdown"] = .110001
    else: changed["subwindow_median_scores"] = [.1, -.001, -.001]
    values["c01_lr1e4"]["BTCUSDT"] = changed
    chosen = audit.compute_expected_selection(values, audit.read(PLAN))["BTCUSDT"]
    assert chosen["gates_by_candidate"]["c01_lr1e4"][gate] is False
    assert chosen["candidate_id"] == "c00_baseline"


def test_linear_quantiles_and_utility_use_both_costs_and_penalize_large_drawdown():
    assert audit.quantile([4, 0, 3, 1, 2], .9) == pytest.approx(3.6)
    assert audit.utility(.1, -.3) == pytest.approx(np.log1p(.1) - .15 - .2)
    with pytest.raises(ValueError): audit.utility(-1., 0)


def test_continuous_subwindow_preserves_prior_nav_first_hour_cost_and_gap_gate():
    times = pd.date_range("2025-03-31T23:00:00Z", periods=4, freq="h")
    hourly = pd.DataFrame({"timestamp": times, "equity": [12000., 11880., 11940., 12120.]})
    ret, dd = audit.continuous_window_metrics(hourly, "2025-04-01T00:00:00Z", "2025-04-01T03:00:00Z")
    assert ret == pytest.approx(.01)
    assert dd == pytest.approx(-.01)
    with pytest.raises(ValueError, match="Every subwindow hour"):
        audit.continuous_window_metrics(hourly.drop(index=2), "2025-04-01T00:00:00Z", "2025-04-01T03:00:00Z")
    first = pd.DataFrame({"timestamp": ["2025-01-01T00:00:00Z"], "equity": [9980.]})
    assert audit.continuous_window_metrics(first, "2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z") == pytest.approx((-.002, -.002))


@pytest.fixture
def validation_tree(tmp_path):
    plan = copy_plan(tmp_path); child = tmp_path / "tuning/c00_baseline"
    times = pd.date_range(*plan["validation"], freq="h", inclusive="left")
    hourly = pd.DataFrame({"timestamp": times, "equity": np.linspace(10000., 11000., len(times))})
    rows = []
    for symbol in audit.SYMBOLS:
        for algorithm, seed in [("sac", n) for n in audit.TUNING_SEEDS] + [(a, None) for a in audit.BASELINES]:
            for cost in audit.COSTS:
                rows.append({"symbol": symbol, "algorithm": algorithm, "seed": seed, "scenario": cost,
                    "split": "validation", "initial_capital": 10000., "net_return": .1, "max_drawdown": 0.})
                if seed is not None:
                    p = child / f"evaluation/validation/{symbol}/sac/seed{seed}/{cost}/equity_1h.csv.gz"
                    p.parent.mkdir(parents=True, exist_ok=True); hourly.to_csv(p, index=False)
    pd.DataFrame(rows).to_csv(child / "validation_summary.csv", index=False)
    return SimpleNamespace(top=tmp_path, child=child, plan=plan)


def test_validation_metrics_never_open_annual_or_final_accounts(validation_tree, monkeypatch):
    actual_read = pd.read_csv
    opened = []
    def guarded(path, *args, **kwargs):
        name = Path(path).as_posix(); opened.append(name)
        assert "/test/" not in name and "/final/" not in name
        return actual_read(path, *args, **kwargs)
    monkeypatch.setattr(audit.pd, "read_csv", guarded)
    before = audit.compute_candidate_metrics(validation_tree.child, validation_tree.plan)
    dump(validation_tree.top / "final/evaluation/test/poison.json", {"annual_return": 999999})
    after = audit.compute_candidate_metrics(validation_tree.child, validation_tree.plan)
    assert before == after and len(opened) == 42
    for result in before.values():
        assert result["seed_scores"] == pytest.approx([np.log1p(.1)] * 5)
        assert result["score_iqr"] == 0 and result["base_positive_seeds"] == 5


def test_incomplete_tuning_seed_or_wrong_split_is_rejected(validation_tree):
    p = validation_tree.child / "validation_summary.csv"; frame = pd.read_csv(p)
    frame.loc[0, "seed"] = 0
    frame.to_csv(p, index=False)
    with pytest.raises(ValueError, match="Exactly36"):
        audit.compute_candidate_metrics(validation_tree.child, validation_tree.plan)


@pytest.fixture
def contract_tree(tmp_path):
    plan = copy_plan(tmp_path)
    spec = {k: deepcopy(plan[k]) for k in ("symbols", "capital_per_account", "ensemble", "train", "validation", "test", "costs", "environment", "dataset_sha256", "runtime_packages")}
    recipes = {r["candidate_id"]: r["native_recipe"] for r in plan["candidates"]}
    spec.update(sources=plan["frozen_core_sources"], tuning_seeds=list(audit.TUNING_SEEDS), final_seeds=list(audit.FINAL_SEEDS),
        candidate_order=list(audit.CANDIDATE_ORDER), candidate_recipes=recipes, search_plan_sha256=audit.PLAN_SHA256,
        steps_per_model=32768, algorithm="sac", capital_shared_between_pairs=False, primary_seed=0,
        primary_seed_fixed_before_training=True, base_recipe=recipes["c00_baseline"],
        frozen_at="2026-10-31T00:00:00+00:00",
        selection_contract={k: deepcopy(plan[k]) for k in ("score", "acceptance", "subwindows", "subwindow_metric", "tie_break", "fallback")})
    refs, files = {}, {}
    for symbol in audit.SYMBOLS:
        oldroot = ROOT / f"results/backtests/{symbol[:3].lower()}_rl_seed_stability_20261031"
        source = oldroot / "protocol.json"; target = tmp_path / f"references/{symbol}/protocol.json"
        target.parent.mkdir(parents=True); shutil.copyfile(source, target)
        refs[symbol] = {"source_root": str(oldroot), "protocol_sha256": audit.sha(source),
            "validation_audit_sha256": audit.sha(oldroot / "validation_audit.json"), "test_audit_sha256": audit.sha(oldroot / "test_audit.json")}
        files[target.relative_to(tmp_path).as_posix()] = audit.sha(target)
    spec["reference_sources"] = refs
    for name in plan["frozen_core_sources"]:
        target = tmp_path / "frozen_source" / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, target)
    extensions = {}
    for name in audit.EXTENSION_FILES:
        target = tmp_path / "extension_source" / name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes() if name == audit.EXTENSION_FILES[2] else b"# SYNTHETIC NONFINANCIAL NEVER EXECUTED\n")
        extensions[name] = audit.sha(target)
    dump(tmp_path / "protocol.json", spec)
    dump(tmp_path / "protocol_receipt.json", {"protocol_sha256": audit.sha(tmp_path / "protocol.json")})
    dump(tmp_path / "extension_manifest.json", {"protocol_sha256": audit.sha(tmp_path / "protocol.json"), "search_plan_sha256": audit.PLAN_SHA256, "extension_sources": extensions})
    dump(tmp_path / "reference_manifest.json", {"sources": refs, "files_sha256": files})
    dump(tmp_path / "reference_receipt.json", {"reference_manifest_sha256": audit.sha(tmp_path / "reference_manifest.json")})
    dump(tmp_path / "pipeline_launch.json", {"synthetic_nonfinancial": True, "pid": 99999})
    revision_copy = tmp_path / audit.REVISION_AUDIT_PATH; revision_copy.parent.mkdir(parents=True, exist_ok=True); revision_copy.write_bytes(Path(audit.__file__).read_bytes())
    controller = tmp_path / audit.REVISION_CONTROLLER_PATH; controller.write_text("# SYNTHETIC NONFINANCIAL CONTROLLER; NEVER EXECUTED\n")
    revision = {"schema_version":1,"revision_id":audit.REVISION_ID,"reason_code":audit.REVISION_REASON,"created_at_utc":"2026-10-31T00:01:00+00:00",
        "original_protocol_sha256":audit.sha(tmp_path / "protocol.json"),"search_plan_sha256":audit.PLAN_SHA256,
        "original_extension_manifest_sha256":audit.sha(tmp_path / "extension_manifest.json"),"original_audit_source_sha256":audit.ORIGINAL_AUDITOR_SHA256,
        "revised_audit_source_path":audit.REVISION_AUDIT_PATH,"revised_audit_source_sha256":audit.sha(revision_copy),
        "controller_source_path":audit.REVISION_CONTROLLER_PATH,"controller_source_sha256":audit.sha(controller),
        "original_pipeline_launch_sha256":audit.sha(tmp_path / "pipeline_launch.json"),
        "financial_oracle_source_sha256":plan["frozen_core_sources"]["scripts/audit_single_pair_rl.py"],
        "permitted_metadata_normalization":audit.REVISION_NORMALIZATION,"financial_math_changed":False,"model_recipe_changed":False,"training_restarted":False}
    dump(tmp_path / "audit_revision.json", revision)
    dump(tmp_path / "audit_revision_receipt.json", {"sha256":audit.sha(tmp_path / "audit_revision.json"),"at_utc":"2026-10-31T00:01:01+00:00"})
    return SimpleNamespace(top=tmp_path, plan=plan, spec=spec, extensions=extensions)


def test_real_frozen_sources_and_original_two_coin_protocols_pass(contract_tree):
    checks, pins = audit.Checks(), {}
    audit._base_contract(contract_tree.top, checks, pins)
    assert not checks.failures and len(pins) >= 23
    originals = audit._reference_external_inputs(contract_tree.top, checks)
    assert len(originals) == 2 and not checks.failures


@pytest.mark.parametrize("field,value", [("final_seeds", [0]), ("capital_per_account", 20000), ("costs", {}),
    ("candidate_recipes", {}), ("selection_contract", {}), ("sources", {})])
def test_resealed_top_contract_cannot_change_plan(field, value, contract_tree):
    contract_tree.spec[field] = value
    dump(contract_tree.top / "protocol.json", contract_tree.spec)
    digest = audit.sha(contract_tree.top / "protocol.json")
    dump(contract_tree.top / "protocol_receipt.json", {"protocol_sha256": digest})
    extension = audit.read(contract_tree.top / "extension_manifest.json"); extension["protocol_sha256"] = digest
    dump(contract_tree.top / "extension_manifest.json", extension)
    checks = audit.Checks(); audit._base_contract(contract_tree.top, checks, {})
    assert checks.failures


@pytest.fixture
def native_tree(contract_tree):
    tree = contract_tree; child = tree.top / "tuning/c01_lr1e4"; child.mkdir(parents=True)
    dump(child / "protocol.json", {"synthetic_contract_only": True})
    recipe = tree.spec["candidate_recipes"]["c01_lr1e4"]
    counters = {"environment_steps": 32768, "replay_add_calls": 32768, "replay_transitions": 32768,
        "gradient_update_rounds": 32256, "optimizer_steps": {"actor": 32256, "critic": 32256, "entropy": 32256}}
    episode = {"start": "2024-01-01T00:00:00+00:00", "end_exclusive": "2024-02-02T00:00:00+00:00", "completed_until": "2024-02-02T00:00:00+00:00",
        "completed_steps": 32, "terminated": False, "truncated": True}
    records, rows = [], []
    for symbol in audit.SYMBOLS:
        for seed in audit.TUNING_SEEDS:
            training = {"actual_steps": 32768, "counters": deepcopy(counters), "effective_parameters": {k: deepcopy(v) for k, v in recipe.items() if k not in ("policy", "policy_kwargs")},
                "training_episode_history": [deepcopy(episode) for _ in range(1024)]}
            training["effective_parameters"].update(native_class="SAC", train_freq=[1, "step"], observation_shape=[13], action_shape=[1])
            directory = child / f"training/{symbol}/sac/seed{seed}"; cp = directory / "checkpoint"; cp.mkdir(parents=True)
            actual = deepcopy(recipe); actual["use_sde"] = False; actual["policy_kwargs"]["use_sde"] = False
            with zipfile.ZipFile(cp / "model.zip", "w") as archive: archive.writestr("data", json.dumps(actual))
            manifest = {"schema_version": 2, "symbol": symbol, "algorithm": "sac", "seed": seed, "observation_shape": [13], "action_shape": [1],
                "independent_account": True, "multi_model_target_average": False, "nonfinancial_preflight": False,
                "recipe": deepcopy(recipe), "training": training, "checkpoint_files": {"model.zip": audit.sha(cp / "model.zip")},
                "base_adapter_sha256": tree.plan["frozen_core_sources"]["scripts/single_pair_rl_models.py"], "adapter_sha256": tree.extensions[audit.EXTENSION_FILES[1]],
                "model_context": {"stage": "tuning", "symbol": symbol, "seed": seed, "candidate_id": "c01_lr1e4", "parent_protocol_sha256": audit.sha(tree.top / "protocol.json")}}
            dump(cp / "single_pair_model.json", manifest)
            record = {"symbol": symbol, "algorithm": "sac", "seed": seed, "capital": 10000., "ensemble": False, "observation_shape": [13], "action_shape": [1],
                "maximum_input_hour": "2024-12-31T23:00:00Z", "training": training, "checkpoint": manifest,
                "files": {"checkpoint/" + p.name: audit.sha(p) for p in cp.iterdir()}}
            dump(directory / "record.json", record); records.append(record)
            rows.append({"symbol": symbol, "seed": seed, "record_sha256": audit.sha(directory / "record.json"), "counters": counters})
    dump(child / "fit_complete.json", {"model_count": 10, "records": records})
    dump(child / "training_counter_audit.json", {"passed": True, "failure_count": 0, "models_checked": 10, "models": rows,
        "core_protocol_sha256": audit.sha(child / "protocol.json"), "parent_protocol_sha256": audit.sha(tree.top / "protocol.json"), "extension_source_sha256": tree.extensions[audit.EXTENSION_FILES[0]]})
    tree.child = child; tree.recipes = {s: recipe for s in audit.SYMBOLS}
    return tree


def run_native_checks(tree):
    checks = audit.Checks(); prefix = tree.child.relative_to(tree.top).as_posix()
    pins = {"protocol.json": audit.sha(tree.top / "protocol.json"), prefix + "/protocol.json": audit.sha(tree.child / "protocol.json")}
    audit._training_checks(tree.top, tree.child, tree.plan, tree.recipes, checks, pins, candidate_id="c01_lr1e4")
    return checks


def test_actual_zip_recipe_native_counters_and_timeout_train_cutoff_pass(native_tree):
    assert not run_native_checks(native_tree).failures


@pytest.mark.parametrize("mutation", ["zip_lr", "actor_steps", "future_training", "counter_sha", "manifest_recipe"])
def test_actual_native_evidence_corruption_is_rejected(native_tree, mutation):
    p = native_tree.child / "training/BTCUSDT/sac/seed100/record.json"
    record = audit.read(p)
    if mutation == "zip_lr":
        recipe = deepcopy(native_tree.recipes["BTCUSDT"]); recipe["learning_rate"] = .0003; recipe["use_sde"] = False; recipe["policy_kwargs"]["use_sde"] = False
        with zipfile.ZipFile(p.parent / "checkpoint/model.zip", "w") as archive: archive.writestr("data", json.dumps(recipe))
    elif mutation == "actor_steps": record["training"]["counters"]["optimizer_steps"]["actor"] -= 1; dump(p, record)
    elif mutation == "future_training": record["maximum_input_hour"] = "2025-01-01T00:00:00Z"; dump(p, record)
    elif mutation == "manifest_recipe": record["checkpoint"]["recipe"]["gamma"] = .99; dump(p, record)
    else:
        c = audit.read(native_tree.child / "training_counter_audit.json"); c["models"][0]["record_sha256"] = "0" * 64
        dump(native_tree.child / "training_counter_audit.json", c)
    assert run_native_checks(native_tree).failures


@pytest.fixture
def metadata_tree(tmp_path):
    child = tmp_path / "tuning/c00_baseline"; paths = []
    for symbol in audit.SYMBOLS:
        meta = audit.read(OLD / f"evaluation/test/{symbol}/sac/seed0/base/metadata.json")
        for algorithm, seed in [("sac", n) for n in audit.TUNING_SEEDS] + [(a, None) for a in audit.BASELINES]:
            for kind in ("evaluation", "repeat"):
                for cost in audit.COSTS:
                    p = child / f"{kind}/validation/{symbol}/{algorithm}/{'baseline' if seed is None else 'seed'+str(seed)}/{cost}/metadata.json"
                    dump(p, meta); paths.append(p)
    return SimpleNamespace(top=tmp_path, child=child, paths=paths)


def test_real_btc_eth_own_features_and_complete72_metadata_pass(metadata_tree):
    checks, pins = audit.Checks(), {}
    cases = audit._case_population(metadata_tree.top, metadata_tree.child, "validation", audit.TUNING_SEEDS, checks, pins)
    assert not checks.failures and len(cases) == 36 and len(pins) == 72


@pytest.mark.parametrize("mutation", ["foreign_feature", "foreign_inventory", "pooled_capital", "missing_seed"])
def test_coin_isolation_or_missing_seed_metadata_is_rejected(metadata_tree, mutation):
    p = metadata_tree.paths[0]; meta = audit.read(p)
    if mutation == "foreign_feature": meta["observation_features"][0] = "ETHUSDT:foreign"
    elif mutation == "foreign_inventory": meta["final_positions"] = {"ETHUSDT": 1.}
    elif mutation == "pooled_capital": meta["independent_capital"] = 20000.
    else:
        p.unlink()
        with pytest.raises(FileNotFoundError): audit._case_population(metadata_tree.top, metadata_tree.child, "validation", audit.TUNING_SEEDS, audit.Checks(), {})
        return
    dump(p, meta); checks = audit.Checks()
    audit._case_population(metadata_tree.top, metadata_tree.child, "validation", audit.TUNING_SEEDS, checks, {})
    assert checks.failures


def test_existing_audit_is_never_overwritten_or_original_oracle_called(tmp_path, monkeypatch):
    p = tmp_path / "tuning/c00_baseline/validation_optimization_audit.json"; dump(p, {"preserve": "original"}); before = p.read_bytes()
    monkeypatch.setattr(audit.importlib, "import_module", lambda *a: pytest.fail("Must not import financial oracle"))
    with pytest.raises(FileExistsError): audit.audit_phase(tmp_path, tmp_path / "missing", "tuning-validation", candidate="c00_baseline")
    assert p.read_bytes() == before


def test_invalid_stage_gate_cannot_run_financial_oracle(contract_tree, monkeypatch):
    monkeypatch.setattr(audit.importlib, "import_module", lambda *a: pytest.fail("Invalid provenance must stop before oracle import"))
    result = audit.audit_phase(contract_tree.top, contract_tree.top / "absent_dataset", "tuning-validation", candidate="c00_baseline")
    assert result["passed"] is False and result["core_financial_audit_passed"] is False
    assert (contract_tree.top / "tuning/c00_baseline/validation_optimization_audit.json").is_file()


def test_directory_import_rejects_missing_population_and_wrong_coin(tmp_path):
    selection = {"at": "2026-10-31T00:00:00+00:00", "selection": {s: {"candidate_id": "c00_baseline"} for s in audit.SYMBOLS}}
    dump(tmp_path / "selection.json", selection)
    for name in ("references/ETHUSDT/training/ETHUSDT/sac/seed0/synthetic.txt", "final/training/BTCUSDT/sac/seed0/synthetic.txt"):
        p = tmp_path / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("SYNTHETIC NONFINANCIAL")
    dump(tmp_path / "final/import_manifest.json", {"at": "2026-10-31T01:00:00+00:00", "selection_sha256": audit.sha(tmp_path / "selection.json"), "entries": [{
        "origin": "retained_baseline_fallback", "source_directory": "references/ETHUSDT/training/ETHUSDT/sac/seed0", "destination_directory": "final/training/BTCUSDT/sac/seed0",
        "file_sha256": {"synthetic.txt": audit.sha(tmp_path / "final/training/BTCUSDT/sac/seed0/synthetic.txt")}}]})
    checks = audit.Checks(); audit._import_checks(tmp_path, "final/import_manifest.json", audit.sha(tmp_path / "selection.json"), checks, {})
    assert any("population" in v for v in checks.failures)
    assert any("same coin/seed" in v for v in checks.failures)


@pytest.fixture
def selection_tree(contract_tree, monkeypatch):
    tree = contract_tree; values = population(); evidence = {}
    for cid in audit.CANDIDATE_ORDER:
        root = tree.top / "tuning" / cid; root.mkdir(parents=True)
        (root / "validation_summary.csv").write_text("synthetic nonfinancial summary\n")
        dump(root / "validation_audit.json", {"passed": True, "failure_count": 0, "accounts_checked": 36, "repeats_checked": 36, "audited_at": "2026-10-31T01:00:00+00:00"})
        dump(root / "validation_optimization_audit.json", {"passed": True, "failure_count": 0, "audited_at": "2026-10-31T01:01:00+00:00",
            "protocol_sha256": audit.sha(tree.top / "protocol.json"), "audit_source_sha256": audit.sha(audit.__file__), "inputs_sha256": {}})
        evidence[cid] = {"summary": audit.sha(root / "validation_summary.csv"), "financial_audit": audit.sha(root / "validation_audit.json"), "contract_audit": audit.sha(root / "validation_optimization_audit.json")}
    chosen = {"at": "2026-10-31T02:00:00+00:00", "uses_test": False, "uses_final_validation": False, "seed_selection": False, "algorithm_selection": False,
        "tuning_seeds": list(audit.TUNING_SEEDS), "final_seeds": list(audit.FINAL_SEEDS), "selection": deepcopy(audit.compute_expected_selection(values, tree.plan)),
        "protocol_sha256": audit.sha(tree.top / "protocol.json"), "search_plan_sha256": audit.PLAN_SHA256, "validation_evidence_sha256": evidence}
    dump(tree.top / "selection.json", chosen); dump(tree.top / "selection_receipt.json", {"sha256": audit.sha(tree.top / "selection.json")})
    monkeypatch.setattr(audit, "compute_candidate_metrics", lambda child, plan: values[Path(child).name])
    tree.chosen = chosen
    return tree


def test_selection_recomputed_without_any_seed_or_annual_selection(selection_tree):
    t = selection_tree; checks = audit.Checks(); pins = {"protocol.json": audit.sha(t.top / "protocol.json")}
    audit._selection_contract(t.top, t.plan, t.spec, checks, pins)
    assert not checks.failures
    assert all(f"tuning/{cid}/validation_summary.csv" in pins for cid in audit.CANDIDATE_ORDER)


@pytest.mark.parametrize("mutation", ["uses_test", "uses_final_validation", "seed_selection", "wrong_selected_recipe", "future_audit", "evidence_sha"])
def test_resealed_selection_cannot_leak_or_change_chosen_recipe(selection_tree, mutation):
    t = selection_tree; value = t.chosen
    if mutation in ("uses_test", "uses_final_validation", "seed_selection"): value[mutation] = True
    elif mutation == "wrong_selected_recipe": value["selection"]["BTCUSDT"]["selected_recipe"]["learning_rate"] = 1.
    elif mutation == "future_audit": value["at"] = "2026-10-31T00:00:00+00:00"
    else: value["validation_evidence_sha256"]["c00_baseline"]["summary"] = "0" * 64
    dump(t.top / "selection.json", value); dump(t.top / "selection_receipt.json", {"sha256": audit.sha(t.top / "selection.json")})
    checks = audit.Checks(); audit._selection_contract(t.top, t.plan, t.spec, checks, {"protocol.json": audit.sha(t.top / "protocol.json")})
    assert checks.failures


@pytest.fixture
def annual_gate_tree(tmp_path):
    for name in ("protocol.json", "selection.json", "final/protocol.json", "final/validation_audit.json"):
        dump(tmp_path / name, {"synthetic_nonfinancial": name})
    dump(tmp_path / "selection_audit.json", {"passed": True, "failure_count": 0, "failures": [], "inputs_sha256": {}})
    dump(tmp_path / "final/validation_optimization_audit.json", {"passed": True, "failure_count": 0, "failures": [], "inputs_sha256": {}, "audited_at": "2026-10-31T02:00:00+00:00"})
    values = {"selection_sha256": audit.sha(tmp_path / "selection.json"), "selection_audit_sha256": audit.sha(tmp_path / "selection_audit.json"),
        "validation_audit_sha256": audit.sha(tmp_path / "final/validation_audit.json"), "validation_optimization_audit_sha256": audit.sha(tmp_path / "final/validation_optimization_audit.json"),
        "protocol_sha256": audit.sha(tmp_path / "protocol.json"), "final_protocol_sha256": audit.sha(tmp_path / "final/protocol.json")}
    dump(tmp_path / "final/validation_selection.json", {"at": "2026-10-31T02:01:00+00:00", "uses_test": False, "seed_selection": False, "algorithm_selection": False,
        "mode": "fixed_sac_recipe_comparison", "selection": {s: {"algorithm": "sac", "seed": 0} for s in audit.SYMBOLS},
        "top_selection_sha256": values["selection_sha256"], "protocol_sha256": values["final_protocol_sha256"],
        "validation_audit_sha256": values["validation_audit_sha256"], "validation_optimization_audit_sha256": values["validation_optimization_audit_sha256"]})
    dump(tmp_path / "final/test_started.json", {"at": "2026-10-31T02:02:00+00:00", "pins": values, "oracle_compatibility_selection_sha256": audit.sha(tmp_path / "final/validation_selection.json")})
    return tmp_path


def annual_checks(top):
    pins = {n: audit.sha(top / n) for n in ("protocol.json", "selection.json", "final/protocol.json")}
    checks = audit.Checks(); audit._annual_gate(top, checks, pins)
    return checks


def test_annual_gate_binds_passed_final_validation_and_fixed_seed_compatibility(annual_gate_tree):
    assert not annual_checks(annual_gate_tree).failures


@pytest.mark.parametrize("mutation", ["start_early", "changed_selection_pin", "failed_validation", "compat_seed"])
def test_annual_gate_rejects_leakage_or_stale_bindings(annual_gate_tree, mutation):
    top = annual_gate_tree
    if mutation == "compat_seed":
        p = top / "final/validation_selection.json"; value = audit.read(p); value["selection"]["BTCUSDT"]["seed"] = 29
    elif mutation == "failed_validation":
        p = top / "final/validation_optimization_audit.json"; value = audit.read(p); value["passed"] = False
    else:
        p = top / "final/test_started.json"; value = audit.read(p)
        if mutation == "start_early": value["at"] = "2026-10-31T01:00:00+00:00"
        else: value["pins"]["selection_sha256"] = "0" * 64
    dump(p, value)
    assert annual_checks(top).failures


def test_completion_is_read_only_currently_bound_and_excludes_mutable_status(tmp_path, monkeypatch):
    # Scope this fixture to read-only binding composition, never financial evidence.
    for name in ("protocol.json", "selection.json"):
        dump(tmp_path / name, {"at": "2026-10-31T02:00:00+00:00", "synthetic_nonfinancial": True})
    leaf = tmp_path / "synthetic_bound_input.json"; dump(leaf, {"immutable": True})
    names = [f"tuning/{cid}/validation_optimization_audit.json" for cid in audit.CANDIDATE_ORDER]
    names += ["selection_audit.json", "final/validation_optimization_audit.json", "final/test_optimization_audit.json"]
    for name in names:
        dump(tmp_path / name, {"passed": True, "checks_total": 1, "failure_count": 0, "failures": [],
            "protocol_sha256": audit.sha(tmp_path / "protocol.json"), "audit_source_sha256": audit.sha(audit.__file__),
            "inputs_sha256": {leaf.name: audit.sha(leaf)}, "external_inputs_sha256": {}})
    for name in ("final/fit_complete.json", "final/training_counter_audit.json", "final/import_manifest.json", "final/validation_import_manifest.json", "final/test_import_manifest.json"):
        dump(tmp_path / name, {"synthetic_nonfinancial": True})
    dump(tmp_path / "reference_annual_import.json", {"at": "2026-10-31T03:00:00+00:00", "selection_sha256": audit.sha(tmp_path / "selection.json"), "files_sha256": {}})
    dump(tmp_path / "pipeline_status.json", {"still_mutable": True}); dump(tmp_path / "completion_contract_audit.json", {"preserve": True})
    before = {p.relative_to(tmp_path).as_posix(): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    def base(top, checks, pins):
        pins["protocol.json"] = audit.sha(top / "protocol.json")
        return {}, {}, {}
    def selected(top, plan, spec, checks, pins):
        pins["selection.json"] = audit.sha(top / "selection.json"); return audit.read(top / "selection.json")
    monkeypatch.setattr(audit, "_base_contract", base); monkeypatch.setattr(audit, "_selection_contract", selected)
    monkeypatch.setattr(audit, "_reference_external_inputs", lambda *a, **kw: {})
    monkeypatch.setattr(audit, "_case_population", lambda *a: set()); monkeypatch.setattr(audit, "_financial_bindings", lambda *a: None)
    result = audit.verify_completed_study(tmp_path)
    assert result["passed"] is True and result["financial_arithmetic_rerun"] is False
    assert result["expected_tuning_models"] == 40 and result["final_models"] == 60
    assert "pipeline_status.json" not in result["inputs_sha256"] and "completion_contract_audit.json" not in result["inputs_sha256"]
    assert before == {p.relative_to(tmp_path).as_posix(): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    dump(leaf, {"immutable": False})
    changed = audit.verify_completed_study(tmp_path)
    assert changed["passed"] is False and any("completed stage current inputs" in v for v in changed["failures"])


@pytest.mark.parametrize("symbol",audit.SYMBOLS)
def test_real_original_native_zip_expands_only_boolean_false_sde_default(symbol):
    root=ROOT / f"results/backtests/{symbol[:3].lower()}_rl_seed_stability_20261031"
    directory=root / f"training/{symbol}/sac/seed0"
    record=audit.read(directory / "record.json")
    path=directory / "checkpoint/model.zip"
    assert audit.sha(path)==record["files"]["checkpoint/model.zip"]
    with zipfile.ZipFile(path) as archive: actual=json.loads(archive.read("data"))
    declared=audit.read(PLAN)["candidates"][0]["native_recipe"]["policy_kwargs"]
    assert actual["policy_kwargs"]=={"net_arch":[128,128],"use_sde":False}
    assert actual["use_sde"] is False
    assert audit.native_policy_kwargs_match(actual["policy_kwargs"],declared)
    assert actual["policy_kwargs"]!=declared
    assert audit.sha(ROOT / "scripts/audit_sac_validation_optimization.py")==audit.ORIGINAL_AUDITOR_SHA256


@pytest.mark.parametrize("actual,okay",[
    ({"net_arch":[128,128]},True),({"net_arch":[128,128],"use_sde":False},True),
    ({"net_arch":[128,128],"use_sde":True},False),({"net_arch":[128,128],"use_sde":0},False),
    ({"net_arch":[128,128],"use_sde":None},False),({"net_arch":[128,128],"activation_fn":"other"},False),
    ({"net_arch":[64,64],"use_sde":False},False),({},False),([],False)])
def test_native_default_normalization_is_narrow_and_strict(actual,okay):
    assert audit.native_policy_kwargs_match(actual,{"net_arch":[128,128]}) is okay


@pytest.mark.parametrize("mutation",["old_source","protocol","plan","oracle","controller_sha","model_recipe","restart","reason","revision_copy","receipt"])
def test_resealed_revision_cannot_replace_original_contract_or_widen_correction(contract_tree,mutation):
    top=contract_tree.top;value=audit.read(top / "audit_revision.json")
    key={"old_source":"original_audit_source_sha256","protocol":"original_protocol_sha256","plan":"search_plan_sha256","oracle":"financial_oracle_source_sha256","controller_sha":"controller_source_sha256"}.get(mutation)
    if key:value[key]="0"*64
    elif mutation=="model_recipe":value["model_recipe_changed"]=True
    elif mutation=="restart":value["training_restarted"]=True
    elif mutation=="reason":value["permitted_metadata_normalization"]="allow arbitrary policy kwargs"
    elif mutation=="revision_copy":
        with (top / audit.REVISION_AUDIT_PATH).open("a") as stream:stream.write("\n# changed copy\n")
    dump(top / "audit_revision.json",value)
    dump(top / "audit_revision_receipt.json",{"sha256":audit.sha(top / "audit_revision.json") if mutation!="receipt" else "0"*64,"at_utc":"2026-10-31T00:01:01+00:00"})
    checks=audit.Checks();audit._base_contract(top,checks,{})
    assert checks.failures
