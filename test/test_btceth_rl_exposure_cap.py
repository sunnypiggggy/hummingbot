"""Synthetic contract/causal ledger checks; never train or export a report."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import btceth_rl_exposure_cap as cap
from scripts import btceth_rl_goal_candidate as goal


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def row(spec, algorithm, profile, scenario, seed, ret=.1, dd=-.1, family="raw"):
    key = "ensemble" if seed is None else str(seed)
    prefix = "__cap075" if family == "cap075" else ""
    return dict(run_id=f"validation__{scenario}{prefix}__{algorithm}__{profile}__{key}",
        algorithm=algorithm, profile=profile, scenario=scenario, seed=seed, primary_policy=seed is None,
        split="validation", strategy=algorithm, initial_capital=10000., start=spec["validation"][0],
        end_exclusive=spec["validation"][1], net_return=ret, final_equity=10000 * (1 + ret),
        max_drawdown=dd, fees=10., family=family, exposure_multiplier=.75 if family == "cap075" else 1.)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    main, output, diag = tmp_path / "main", tmp_path / "extension", tmp_path / "diag"
    main.mkdir(); diag.mkdir()
    spec = json.loads((cap.ROOT / "results/backtests/btceth_rl_20261029/frozen_spec.json").read_text(encoding="utf-8"))
    spec["frozen_at"] = "2026-10-29T12:00:00Z"
    dataset = tmp_path / "dataset.pkl.gz"
    dataset.write_bytes(b"TEST_ONLY_NOT_MARKET_DATA")
    write(tmp_path / "manifest.json", {"test_only": True})
    spec["dataset_sha256"], spec["dataset_manifest_sha256"] = cap.sha(dataset), cap.sha(tmp_path / "manifest.json")
    write(main / "frozen_spec.json", spec)
    monkeypatch.setattr(cap.formal, "check_frozen", lambda m, d: deepcopy(spec))
    monkeypatch.setattr(cap, "now", lambda: "2026-10-29T17:00:00Z")
    monkeypatch.setattr(goal, "now", lambda: "2026-10-29T16:00:00Z")
    goal.declare_protocol(main)
    models, training = [], {}
    for algorithm in goal.ALGORITHMS:
        for profile in cap.PROFILES:
            for seed in cap.formal.SEEDS:
                folder = f"training/{cap.formal.job_id(algorithm, profile, seed)}"
                model_file = folder + "/model.zip"
                path = main / model_file
                path.parent.mkdir(parents=True)
                path.write_bytes(f"TEST_ONLY {algorithm} {profile} {seed}".encode())
                record = dict(algorithm=algorithm, profile=profile, seed=seed, model_relative_path=model_file,
                    model_sha256=cap.sha(path), parameter_sha256=str(seed) * 64,
                    frozen_spec_sha256=cap.sha(main / "frozen_spec.json"))
                write(main / folder / "record.json", record)
                models.append(record)
                for filename in ("model.zip", "record.json"):
                    name = folder + "/" + filename
                    training[name] = cap.sha(main / name)
    write(main / "training_completion.json", dict(records=models, model_count=40, frozen_spec_sha256=cap.sha(main / "frozen_spec.json")))
    training["training_completion.json"] = cap.sha(main / "training_completion.json")
    write(main / "runtime.json", {"physical_gpus": [{"uuid": "GPU-test0", "name": "Tesla V100", "index": 0}, {"uuid": "GPU-test1", "name": "Tesla V100", "index": 1}]})
    (diag / "summary.csv").write_bytes(b"TEST_ONLY")
    monkeypatch.setattr(cap, "DIAGNOSTIC_SHA", cap.sha(diag / "summary.csv"))
    write(diag / "protocol.json", {"test_only": True})
    write(diag / "completion.json", {"passed": True, "cases": 4, "test_evaluations": 0})
    cap.prepare(main, dataset, output, diag)
    return main, dataset, output, spec, diag


def validation_ready(prepared, monkeypatch, *, raw_ret=.10, capped_ret=.15, sac_ret=.50):
    main, dataset, output, spec, _ = prepared
    raw_candidates, hashes = [], {}
    for algorithm in goal.ALGORITHMS:
        for profile in cap.PROFILES:
            ret = sac_ret if algorithm == "sac" else raw_ret
            ret -= .01 if profile == "return" else 0
            frame = pd.DataFrame([row(spec, algorithm, profile, c, s, ret=ret) for c in cap.formal.COSTS for s in (*cap.formal.SEEDS, None)])
            path = main / f"evaluation/validation/{algorithm}/{profile}/summary.csv"
            path.parent.mkdir(parents=True)
            frame.to_csv(path, index=False, float_format="%.17g")
            hashes[path.relative_to(main).as_posix()] = cap.sha(path)
            raw_candidates.append(goal.candidate_row(frame, algorithm, profile))
    ordered = goal.rank(raw_candidates)
    original = dict(frozen_at="2026-10-29T17:01:00Z", frozen_spec_sha256=cap.sha(main / "frozen_spec.json"),
        selected_profiles={a: "balanced" for a in goal.ALGORITHMS},
        primary_validation_leader={k: ordered[0][k] for k in ("algorithm", "profile")}, candidates=raw_candidates,
        validation_source_sha256=hashes, training_source_sha256=cap.pin(main, goal.training_names()),
        selection_uses_test=False, test_results_available_to_selector=False)
    write(main / "validation_selection.json", original)
    write(main / "validation_completion.json", dict(model_groups=8, baselines=7, frozen_spec_sha256=cap.sha(main / "frozen_spec.json")))
    monkeypatch.setattr(goal, "now", lambda: "2026-10-29T17:02:00Z")
    goal.select_candidate(main)
    protocol = cap.read_json(output / "protocol.json")
    for profile in cap.PROFILES:
        frame = pd.DataFrame([row(spec, "tqc", profile, c, s, ret=capped_ret - (.01 if profile == "return" else 0), family="cap075") for c in cap.formal.COSTS for s in (*cap.formal.SEEDS, None)])
        group = output / "validation" / profile
        group.mkdir(parents=True)
        frame.to_csv(group / "summary.csv", index=False, float_format="%.17g")
        pd.DataFrame({"synthetic_test_only": []}).to_csv(group / "daily_equity.csv", index=False)
        pd.DataFrame({"synthetic_test_only": []}).to_csv(group / "monthly_returns.csv", index=False)
        proofs = []
        for c in cap.formal.COSTS:
            for key in (*map(str, cap.formal.SEEDS), "ensemble"):
                folder = group / c / key
                folder.mkdir(parents=True)
                selected = [r for r in protocol["models"] if r["profile"] == profile and (key == "ensemble" or r["seed"] == int(key))]
                for name in cap.CASE_FILES[:6]:
                    (folder / name).write_bytes(b"TEST_ONLY")
                (folder / "member_targets.csv").write_bytes(b"TEST_ONLY")
                write(folder / "fingerprints.json", {"equity": "testonly"})
                proof = dict(run_id=f"validation__{c}__cap075__tqc__{profile}__{key}", passed=True, repeat_exact=True,
                    fingerprints={"equity": "testonly"}, repeated_fingerprints={"equity": "testonly"},
                    member_targets_sha256=cap.sha(folder / "member_targets.csv"), repeated_member_targets_sha256=cap.sha(folder / "member_targets.csv"),
                    model_records=selected, parameter_sha256_before=[r["parameter_sha256"] for r in selected],
                    parameter_sha256_after=[r["parameter_sha256"] for r in selected], device={"visible_gpu_uuid": "GPU-test0", "device": "cuda:0"})
                write(folder / "replay.json", proof)
                proofs.append(proof)
        write(group / "completion.json", dict(case_count=12, repeats=proofs, protocol_sha256=cap.sha(output / "protocol.json")))
    write(output / "validation_completion.json", dict(case_count=24, repeat_count=24, protocol_sha256=cap.sha(output / "protocol.json")))
    monkeypatch.setattr(cap, "now", lambda: "2026-10-29T17:03:00Z")
    return prepared


def test_prepare_is_separate_pins_exact_ten_models_and_preserves_main(prepared):
    main, dataset, output, spec, diag = prepared
    p = cap.verify_protocol(main, dataset, output)
    assert p["multiplier"] == .75 and len(p["models"]) == 10
    assert len(p["training_source_sha256"]) == 20 and p["new_training"] == 0
    assert p["designed_after_validation"] and not p["selection_uses_test"]
    assert not (main / "test_started.json").exists()
    with pytest.raises(FileExistsError):
        cap.prepare(main, dataset, output, diag)


@pytest.mark.parametrize("field,value", [("multiplier", .8), ("capital", 5000), ("rebalance_band", .03), ("costs", {})])
def test_changed_financial_contract_refused_even_if_receipt_rehashed(prepared, field, value):
    main, dataset, output, _, _ = prepared
    p = cap.read_json(output / "protocol.json"); p[field] = value
    write(output / "protocol.json", p)
    receipt = cap.read_json(output / "protocol_receipt.json"); receipt["protocol_sha256"] = cap.sha(output / "protocol.json")
    write(output / "protocol_receipt.json", receipt)
    with pytest.raises(ValueError, match="protocol changed"):
        cap.verify_protocol(main, dataset, output)


def test_changed_checkpoint_refused(prepared):
    main, dataset, output, _, _ = prepared
    model = main / "training/tqc__return__seed0/model.zip"
    model.write_bytes(b"changed")
    with pytest.raises(ValueError, match="input changed"):
        cap.verify_protocol(main, dataset, output)


def test_nested_output_refused(prepared):
    main, _, _, _, _ = prepared
    with pytest.raises(ValueError, match="separate"):
        cap.separate(main, main / "nested")


def test_cap_mapping_and_real_costs_not_equity_scaling(monkeypatch):
    index = pd.date_range("2024-01-01T00:00Z", periods=370 * 24, freq="h")
    frame = pd.DataFrame({"BTCUSDT": 100., "ETHUSDT": 200.}, index=index)
    data = {"hourly_" + f: frame.copy() for f in ("open", "high", "low", "close")}
    data["hourly_volume"] = frame * 0 + 100
    monkeypatch.setattr(cap.formal, "VALIDATION", ("2025-01-01T00:00:00Z", "2025-01-03T00:00:00Z"))
    monkeypatch.setattr(cap.formal, "parameter_digest", lambda p: "unchanged")
    class Policy:
        policy = object()
        def __init__(self): self.observed = []
        def predict(self, obs, deterministic=True):
            self.observed.append(obs.copy())
            return np.array([1., -1.]), None
    models = [Policy() for _ in range(5)]
    result, trace = cap.evaluate_cap(data, models, list(range(5)), "return", "validation", cap.formal.COSTS["base"])
    assert result["metadata"]["initial_capital"] == 10000 and result["metadata"]["model_count"] == 5
    cost = 1.0002 * 1.001 - 1
    expected = 10000 / (1 + .7125 * cost)
    assert result["metadata"]["final_equity"] == pytest.approx(expected)
    assert result["metadata"]["final_cash"] == pytest.approx(.2875 * expected)
    assert result["metadata"]["fees"] == pytest.approx(.7125 * expected * 1.0002 * .001)
    assert models[0].observed[0][-4:].tolist() == [0, 0, 1, 0]
    assert models[0].observed[1][-4] == pytest.approx(.7125)
    for model in models[1:]:
        np.testing.assert_array_equal(model.observed, models[0].observed)
    assert trace.groupby("day").observation_sha256.nunique().eq(1).all()
    assert len(trace) == 10
    np.testing.assert_allclose(trace.capped_btc, .7125, atol=1e-15)
    np.testing.assert_allclose(trace.target_btc, .7125, atol=1e-15)
    np.testing.assert_allclose(trace.target_cash, .2875, atol=1e-15)
    assert result["fills"].side.tolist() == ["BUY"]


def test_mean_target_mapping_is_before_execution_and_no_cash_averaging_artifact(monkeypatch):
    assert cap.cap_target([.475, .475, .05]) == pytest.approx([.35625, .35625, .2875])
    assert cap.cap_target([0, 0, 1]) == pytest.approx([0, 0, 1])
    with pytest.raises(ValueError):
        cap.cap_target([1, 0, 0])


def test_select_eight_candidates_excludes_sac_and_pins_only_immutable_inputs(prepared, monkeypatch):
    main, dataset, output, _, _ = validation_ready(prepared, monkeypatch)
    selected = cap.select(main, dataset, output)
    assert selected["candidate"]["family"] == "cap075"
    assert selected["candidate"]["profile"] == "balanced"
    assert selected["original_global_validation_leader"]["algorithm"] == "sac"
    assert len(selected["candidates"]) == 8 and all(r["algorithm"] != "sac" for r in selected["candidates"])
    assert len(selected["training_source_sha256"]) == 81 and len(selected["validation_source_sha256"]) == 8
    assert "summary.csv" not in selected["extension_source_sha256"]
    assert "summary.csv" not in selected["validation_source_sha256"]
    # Root assembled tables are legitimately appended when tests finish.
    (main / "summary.csv").write_text("legitimately changed root aggregate")
    (output / "summary.csv").write_text("legitimately changed extension aggregate")
    assert cap.validate_selection(main, dataset, output)[1]["candidate"] == selected["candidate"]


def test_exact_score_tie_prefers_raw(prepared, monkeypatch):
    main, dataset, output, _, _ = validation_ready(prepared, monkeypatch, capped_ret=.10)
    selected = cap.select(main, dataset, output)
    assert selected["candidate"]["family"] == "raw"
    assert selected["candidate"]["profile"] == "balanced"
    assert selected["candidate"]["algorithm"] == "crossq"


def test_after_test_selection_refused_before_any_annual_price_loading(prepared, monkeypatch):
    main, dataset, output, _, _ = validation_ready(prepared, monkeypatch)
    write(main / "test_started.json", {"at": "2026-10-29T17:04:00Z"})
    monkeypatch.setattr(cap.formal, "load_dataset", lambda p: pytest.fail("Must not read annual prices"))
    with pytest.raises(FileExistsError, match="before any test"):
        cap.select(main, dataset, output)


def test_replay_parameter_or_member_log_tampering_blocks_selection(prepared, monkeypatch):
    main, dataset, output, _, _ = validation_ready(prepared, monkeypatch)
    (output / "validation/balanced/base/ensemble/member_targets.csv").write_bytes(b"changed")
    with pytest.raises(ValueError, match="replay/source/parameter"):
        cap.select(main, dataset, output)


def test_changed_sealed_group_blocks_test_without_loading_prices(prepared, monkeypatch):
    main, dataset, output, _, _ = validation_ready(prepared, monkeypatch)
    cap.select(main, dataset, output)
    path = output / "validation/balanced/summary.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    monkeypatch.setattr(cap.formal, "load_dataset", lambda p: pytest.fail("Must not read annual prices"))
    with pytest.raises(ValueError, match="input changed"):
        cap.validate_selection(main, dataset, output)


@pytest.mark.parametrize("field,value", [("initial_capital", 5000.), ("seed", 12.), ("end_exclusive", "2026-10-23T00:00:00Z")])
def test_wrong_capital_seed_or_dates_rejected(prepared, monkeypatch, field, value):
    main, dataset, output, _, _ = validation_ready(prepared, monkeypatch)
    path = output / "validation/balanced/summary.csv"
    frame = pd.read_csv(path, float_precision="round_trip"); frame.loc[0, field] = value
    frame.to_csv(path, index=False, float_format="%.17g")
    with pytest.raises(ValueError):
        cap.select(main, dataset, output)


def test_running_fit_blocks_validation_before_gpu_or_data(prepared, monkeypatch):
    main, dataset, output, _, _ = prepared
    write(main / "fit_process.json", {"status": "running", "phase": "fit"})
    monkeypatch.setattr(cap, "run_gpu_groups", lambda *a: pytest.fail("No GPU work while fit running"))
    with pytest.raises(ValueError, match="exit0"):
        cap.run_validation(main, dataset, output)


def test_performance_gates_cannot_use_sac_only_or_best_test_seed(prepared):
    _, _, _, spec, _ = prepared
    rows = []
    for scenario in cap.formal.COSTS:
        for seed in (*cap.formal.SEEDS, None):
            value = .50 if seed == 0 else .01
            item = row(spec, "tqc", "balanced", scenario, seed, ret=value, family="cap075")
            item["split"] = "test"; item["run_id"] = item["run_id"].replace("validation__", "test__")
            rows.append(item)
    rows.append(dict(run_id="test__base__equal_hold", split="test", algorithm="baseline", profile="baseline", scenario="base", primary_policy=False, seed=None, net_return=.02, max_drawdown=-.3))
    metrics, gates = goal.performance_gates(pd.DataFrame(rows), {"algorithm": "tqc", "profile": "balanced"}, spec["research_targets"])
    assert metrics["base_net_return"] == .01 and not all(gates.values())
    with pytest.raises(ValueError, match="SAC"):
        goal.performance_gates(pd.DataFrame(rows), {"algorithm": "sac", "profile": "balanced"}, spec["research_targets"])


def start_main_test(main):
    write(main / "test_started.json", dict(at="2026-10-29T17:04:00Z",
        validation_selection_sha256=cap.sha(main / "validation_selection.json"),
        frozen_spec_sha256=cap.sha(main / "frozen_spec.json")))


def test_late_receipt_even_with_rehashed_selection_refused(prepared, monkeypatch):
    main, dataset, output, _, _ = validation_ready(prepared, monkeypatch)
    cap.select(main, dataset, output)
    start_main_test(main)
    receipt = cap.read_json(output / "selection_receipt.json")
    receipt["sealed_at"] = "2026-10-29T17:05:00Z"
    write(output / "selection_receipt.json", receipt)
    with pytest.raises(ValueError, match="before original test"):
        cap.validate_selection(main, dataset, output, require_test=True)


@pytest.mark.parametrize("audit", [
    {"passed": True, "status": "partial_test_pending", "test_complete": False, "failures": []},
    {"passed": True, "status": "passed", "test_complete": True, "failures": ["accounting failed"]},
    {"passed": True, "status": "passed", "test_complete": True, "failures": []},
])
def test_partial_or_unpinned_final_audit_refused(prepared, monkeypatch, audit):
    main, dataset, output, _, _ = validation_ready(prepared, monkeypatch)
    cap.select(main, dataset, output)
    start_main_test(main)
    write(output / "independent_audit.json", audit)
    with pytest.raises(ValueError, match="audit"):
        cap.verify(main, dataset, output)
    assert not (output / "goal_candidate_evidence.json").exists()


def test_raw_winner_reuses_original_fixed_test_cases_without_model_inference(prepared, monkeypatch):
    main, dataset, output, spec, _ = validation_ready(prepared, monkeypatch, capped_ret=.10)
    selected = cap.select(main, dataset, output)
    assert selected["candidate"]["family"] == "raw"
    start_main_test(main)
    write(main / "test_completion.json", dict(model_groups=4, baselines=7, frozen_spec_sha256=cap.sha(main / "frozen_spec.json")))
    algorithm, profile = selected["candidate"]["algorithm"], selected["candidate"]["profile"]
    folder = main / f"evaluation/test/{algorithm}/{profile}"
    folder.mkdir(parents=True)
    frame = pd.read_csv(main / f"evaluation/validation/{algorithm}/{profile}/summary.csv", float_precision="round_trip")
    frame["run_id"] = frame.run_id.str.replace("validation__", "test__", regex=False)
    frame["split"], frame["start"], frame["end_exclusive"] = "test", spec["test"][0], spec["test"][1]
    frame.to_csv(folder / "summary.csv", index=False, float_format="%.17g")
    for name in cap.GROUP_FILES[1:]:
        (folder / name).write_text("synthetic_test_only\n" if name.endswith(".csv") else "{}")
    for scenario in cap.formal.COSTS:
        for key in (*map(str, cap.formal.SEEDS), "ensemble"):
            case = folder / scenario / key
            case.mkdir(parents=True)
            for name in cap.CASE_FILES[:6]:
                (case / name).write_bytes(b"TEST_ONLY")
    monkeypatch.setattr(cap, "terminal_process", lambda *a: {"status": "completed", "exit_code": 0})
    monkeypatch.setattr(cap.formal, "load_dataset", lambda p: pytest.fail("Raw reuse must not infer prices"))
    monkeypatch.setattr(cap, "run_gpu_groups", lambda *a: pytest.fail("Raw winner must not run new GPU inference"))
    cap.run_test(main, dataset, output)
    done = cap.read_json(output / "test_completion.json")
    assert done["mode"] == "raw_reference" and len(done["raw_source_sha256"]) == 76
    final = pd.read_csv(output / "summary.csv")
    assert len(final) == 36 and len(final.loc[final.family.eq("raw")]) == 12
    assert not (output / "test").exists()


def test_original_financial_sources_and_frozen_goal_helper_were_not_modified():
    spec = json.loads((cap.ROOT / "results/backtests/btceth_rl_20261029/frozen_spec.json").read_text(encoding="utf-8"))
    for name, expected in spec["code_sha256"].items():
        assert cap.sha(cap.ROOT / name) == expected
    assert cap.sha(cap.ROOT / "scripts/btceth_rl_goal_candidate.py") == "911034a4861d79dbc9ca9933302e177d645d783f99d89e33ac01438dfa107109"
