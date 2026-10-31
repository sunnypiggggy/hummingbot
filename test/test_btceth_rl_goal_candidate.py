"""Synthetic-only tests of the pre-test new-model goal contract; no RL training."""
from copy import deepcopy
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from scripts import btceth_rl_goal_candidate as goal


ALGOS = ("recurrent_ppo", "tqc", "crossq", "sac")
PROFILES = ("return", "balanced")
BASELINES = ("cash", "btc_hold", "eth_hold", "equal_hold", "equal_rebalance", "sma200_trend", "momentum180")
DATES = {"train": ["2020-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
         "validation": ["2025-01-01T00:00:00Z", "2025-10-23T00:00:00Z"],
         "test": ["2025-10-23T00:00:00Z", "2026-10-23T00:00:00Z"]}
TARGETS = {"primary_base_net_return_at_least": .05, "primary_base_max_drawdown_magnitude_at_most": .20,
           "primary_stress_net_return_at_least": 0., "primary_beats_equal_hold_return_and_drawdown": True,
           "positive_base_training_seeds_at_least": 3, "stronger_claim_requires_beating_simple_trend_controls": True,
           "targets_are_not_guarantees": True}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def score(base, stress):
    return sum(math.log1p(r) - .5 * .1 for r in (base, stress)) / 2


def financial_row(split, scenario, algorithm, profile, seed, result, dd=-.1):
    baseline = algorithm == "baseline"
    key = profile if baseline else "ensemble" if seed is None else str(seed)
    run_id = f"{split}__{scenario}__{key}" if baseline else f"{split}__{scenario}__{algorithm}__{profile}__{key}"
    return {"run_id": run_id, "algorithm": algorithm, "profile": "baseline" if baseline else profile,
            "strategy": profile if baseline else algorithm, "scenario": scenario, "split": split,
            "seed": seed, "primary_policy": not baseline and seed is None,
            "initial_capital": 10000., "start": DATES[split][0], "end_exclusive": DATES[split][1],
            "net_return": result, "max_drawdown": dd, "final_equity": 10000. * (1 + result), "fees": 10.}


def make_pretest(root, *, control_wins=True, new_tie=False):
    root.mkdir(parents=True, exist_ok=True)
    spec = {**DATES, "frozen_at": "2026-10-29T12:00:00Z", "algorithms": list(ALGOS),
            "profiles": {"return": {}, "balanced": {}}, "seeds": list(range(5)), "baseline_names": list(BASELINES),
            "capital": 10000., "research_targets": TARGETS,
            "validation_selection": {"score": "mean(base,stress)[log(final/initial)-0.5*abs(maxDD)-2*max(abs(maxDD)-.20,0)]"},
            "code_sha256": {f"scripts/frozen_{i}.py": str(i) * 64 for i in range(4)}}
    write(root / "frozen_spec.json", spec)
    spec_hash = goal.sha(root / "frozen_spec.json")
    records, training = [], {}
    for algorithm in ALGOS:
        for profile in PROFILES:
            for seed in range(5):
                folder = root / f"training/{algorithm}__{profile}__seed{seed}"
                folder.mkdir(parents=True)
                (folder / "model.zip").write_bytes(b"SYNTHETIC_TEST_FIXTURE_NOT_A_CHECKPOINT")
                record = {"algorithm": algorithm, "profile": profile, "seed": seed,
                          "frozen_spec_sha256": spec_hash, "model_sha256": goal.sha(folder / "model.zip")}
                write(folder / "record.json", record)
                records.append(record)
                for name in ("record.json", "model.zip"):
                    training[(folder / name).relative_to(root).as_posix()] = goal.sha(folder / name)
    write(root / "training_completion.json", {"model_count": 40, "records": records, "frozen_spec_sha256": spec_hash})
    training["training_completion.json"] = goal.sha(root / "training_completion.json")
    candidates, validation_hashes = [], {}
    for algorithm in ALGOS:
        for profile in PROFILES:
            base = {"recurrent_ppo": .05, "tqc": .20, "crossq": .20 if new_tie else .10,
                    "sac": .30 if control_wins else .12}[algorithm]
            base -= .03 if profile == "return" else 0.
            stress = base - .05
            rows = []
            for scenario, result in (("base", base), ("stress", stress)):
                for seed in (*range(5), None):
                    # An exceptional individual validation seed must never be
                    # substituted for the actual five-model ensemble.
                    value = .99 if algorithm == "crossq" and seed == 0 else result
                    rows.append(financial_row("validation", scenario, algorithm, profile, seed, value))
            frame = pd.DataFrame(rows)
            path = root / f"evaluation/validation/{algorithm}/{profile}/summary.csv"
            path.parent.mkdir(parents=True)
            frame.to_csv(path, index=False, float_format="%.17g")
            validation_hashes[path.relative_to(root).as_posix()] = goal.sha(path)
            candidates.append({"algorithm": algorithm, "profile": profile, "score": score(base, stress),
                               "base": {"net_return": base, "max_drawdown": -.1, "final_equity": 10000. * (1 + base), "fees": 10.},
                               "stress": {"net_return": stress, "max_drawdown": -.1, "final_equity": 10000. * (1 + stress), "fees": 10.}})
    ordered = sorted(candidates, key=lambda row: (-row["score"], row["algorithm"], row["profile"]))
    global_leader = {name: ordered[0][name] for name in ("algorithm", "profile")}
    original = {"frozen_at": "2026-10-30T00:30:00Z", "frozen_spec_sha256": spec_hash,
                "selected_profiles": {algorithm: "balanced" for algorithm in ALGOS},
                "primary_validation_leader": global_leader, "candidates": candidates,
                "training_source_sha256": training, "validation_source_sha256": validation_hashes,
                "selection_uses_test": False, "test_results_available_to_selector": False}
    write(root / "validation_selection.json", original)
    write(root / "validation_completion.json", {"model_groups": 8, "baselines": 7, "frozen_spec_sha256": spec_hash})
    return root


@pytest.fixture
def pretest(tmp_path, monkeypatch):
    monkeypatch.setattr(goal, "now", lambda: "2026-10-30T01:00:00Z")
    root = make_pretest(tmp_path / "study")
    goal.declare_protocol(root)
    return root


def add_final_test(root, *, case="new_selected_passes"):
    original = goal.read_json(root / "validation_selection.json")
    frames = [pd.read_csv(root / name, float_precision="round_trip") for name in original["validation_source_sha256"]]
    rows = []
    for scenario in ("base", "stress"):
        for name in BASELINES:
            rows.append(financial_row("validation", scenario, "baseline", name, None, 0. if name == "cash" else .02, 0. if name == "cash" else -.30))
        for algorithm in ALGOS:
            if algorithm == "sac":
                value, dd = (.30 if scenario == "base" else .20), -.05
            elif case == "only_sac_passes":
                value, dd = (.01 if scenario == "base" else -.01), -.25
            elif algorithm == "tqc":
                value = (.01 if scenario == "base" else -.01) if case == "other_new_only_passes" else (.06 if scenario == "base" else .02)
                dd = -.10
            elif algorithm == "crossq":
                value, dd = (.35 if scenario == "base" else .25), -.05
            else:
                value, dd = (.01 if scenario == "base" else -.01), -.25
            for seed in (*range(5), None):
                rows.append(financial_row("test", scenario, algorithm, "balanced", seed, value, dd))
        for name in BASELINES:
            rows.append(financial_row("test", scenario, "baseline", name, None, 0. if name == "cash" else .02, 0. if name == "cash" else -.30))
    frame = pd.concat([*frames, pd.DataFrame(rows)], ignore_index=True)
    assert len(frame) == 172
    frame.to_csv(root / "summary.csv", index=False, float_format="%.17g")
    spec_hash = goal.sha(root / "frozen_spec.json")
    write(root / "test_started.json", {"at": "2026-10-30T02:00:00Z", "frozen_spec_sha256": spec_hash,
                                      "validation_selection_sha256": goal.sha(root / "validation_selection.json")})
    write(root / "test_completion.json", {"model_groups": 4, "baselines": 7, "frozen_spec_sha256": spec_hash})
    required = ("summary.csv", "frozen_spec.json", "validation_selection.json", "training_completion.json", "validation_completion.json", "test_started.json", "test_completion.json")
    audit = {"status": "passed", "passed": True, "test_complete": True, "failure_count": 0, "failures": [],
             "run_count": 172, "runs_checked": 172, "frozen_spec_sha256": spec_hash,
             "input_sha256": {name: goal.sha(root / name) for name in required},
             "runs": {name: {"reconstructed": True, "hours": 8760} for name in frame.run_id},
             "research_targets": {"primary_validation_leader": {"algorithm": "sac", "profile": "balanced"},
                                  "all_predeclared_primary_targets_met": True}}
    write(root / "independent_audit.local.json", audit)
    return frame


def test_declare_pins_new_algorithms_control_source_and_preserves_original_spec(tmp_path, monkeypatch):
    root = make_pretest(tmp_path / "study")
    monkeypatch.setattr(goal, "now", lambda: "2026-10-30T01:00:00Z")
    before = (root / "frozen_spec.json").read_bytes()
    declared = goal.declare_protocol(root)
    assert declared["new_algorithms"] == ["recurrent_ppo", "tqc", "crossq"]
    assert declared["control_algorithm"] == "sac"
    assert declared["sac_only_success_is_not_goal_success"] is True
    assert declared["research_targets"] == TARGETS
    assert declared["source_sha256"] == {"scripts/btceth_rl_goal_candidate.py": goal.sha(Path(goal.__file__))}
    assert (root / "frozen_spec.json").read_bytes() == before
    original = (root / goal.PROTOCOL_FILE).read_bytes()
    with pytest.raises(FileExistsError, match="Preserve"):
        goal.declare_protocol(root)
    assert (root / goal.PROTOCOL_FILE).read_bytes() == original


def test_sac_validation_winner_is_kept_as_control_and_new_candidate_uses_ensemble(pretest):
    original = (pretest / "validation_selection.json").read_bytes()
    selected = goal.select_candidate(pretest)
    assert selected["original_global_validation_leader"] == {"algorithm": "sac", "profile": "balanced"}
    assert selected["new_model_validation_leader"] == {"algorithm": "tqc", "profile": "balanced"}
    assert len(selected["new_validation_candidates"]) == 6
    assert selected["seed_selection"] is False
    assert (pretest / "validation_selection.json").read_bytes() == original
    receipt = goal.read_json(pretest / goal.RECEIPT_FILE)
    assert receipt["selection_sha256"] == goal.sha(pretest / goal.SELECTION_FILE)
    assert receipt["declared_protocol_sha256"] == goal.sha(pretest / goal.PROTOCOL_FILE)


def test_new_global_validation_winner_equals_new_only_candidate(tmp_path, monkeypatch):
    root = make_pretest(tmp_path / "study", control_wins=False)
    monkeypatch.setattr(goal, "now", lambda: "2026-10-30T01:00:00Z")
    goal.declare_protocol(root)
    selected = goal.select_candidate(root)
    assert selected["new_model_validation_leader"] == selected["original_global_validation_leader"] == {"algorithm": "tqc", "profile": "balanced"}


def test_new_candidate_tie_uses_lexicographic_algorithm_not_test_or_seed(tmp_path, monkeypatch):
    root = make_pretest(tmp_path / "study", new_tie=True)
    monkeypatch.setattr(goal, "now", lambda: "2026-10-30T01:00:00Z")
    goal.declare_protocol(root)
    assert goal.select_candidate(root)["new_model_validation_leader"] == {"algorithm": "crossq", "profile": "balanced"}


@pytest.mark.parametrize("artifact", ["test_started.json", "test_completion.json", "evaluation/test"])
def test_select_refuses_any_existing_test_attempt(pretest, artifact):
    path = pretest / artifact
    if artifact == "evaluation/test":
        path.mkdir(parents=True)
    else:
        write(path, {})
    with pytest.raises(FileExistsError, match="before any test"):
        goal.select_candidate(pretest)
    assert not (pretest / goal.SELECTION_FILE).exists()


@pytest.mark.parametrize("name", ["training/tqc__balanced__seed0/model.zip", "evaluation/validation/tqc/balanced/summary.csv"])
def test_select_refuses_bad_training_or_validation_source_sha(pretest, name):
    with (pretest / name).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="SHA changed"):
        goal.select_candidate(pretest)


def test_read_only_contract_gate_never_reads_test_pnl(pretest, monkeypatch):
    goal.select_candidate(pretest)
    write(pretest / "test_started.json", {"at": "2026-10-30T02:00:00Z", "frozen_spec_sha256": goal.sha(pretest / "frozen_spec.json"),
                                         "validation_selection_sha256": goal.sha(pretest / "validation_selection.json")})
    (pretest / "summary.csv").write_bytes(b"TEST_PNL_MUST_NOT_BE_READ")
    original_read, paths = goal.read_summary, []
    def read(path):
        paths.append(Path(path).relative_to(pretest).as_posix())
        return original_read(path)
    monkeypatch.setattr(goal, "read_summary", read)
    result = goal.validate_frozen_candidate(pretest, require_test_started=True)
    assert result["test_pnl_read"] is False
    assert len(paths) == 8 and all(path.startswith("evaluation/validation/") for path in paths)
    assert not (pretest / goal.EVIDENCE_FILE).exists()


def test_changed_candidate_bytes_are_rejected_by_pretest_seal(pretest):
    goal.select_candidate(pretest)
    with (pretest / goal.SELECTION_FILE).open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="receipt SHA"):
        goal.validate_frozen_candidate(pretest)


def test_selection_created_after_test_cannot_pass_by_resealing(pretest):
    goal.select_candidate(pretest)
    selected = goal.read_json(pretest / goal.SELECTION_FILE)
    selected["selected_at"] = "2026-10-30T03:00:00Z"
    write(pretest / goal.SELECTION_FILE, selected)
    receipt = goal.read_json(pretest / goal.RECEIPT_FILE)
    receipt.update(selection_sha256=goal.sha(pretest / goal.SELECTION_FILE), sealed_at="2026-10-30T03:01:00Z")
    write(pretest / goal.RECEIPT_FILE, receipt)
    write(pretest / "test_started.json", {"at": "2026-10-30T02:00:00Z", "frozen_spec_sha256": goal.sha(pretest / "frozen_spec.json"),
                                         "validation_selection_sha256": goal.sha(pretest / "validation_selection.json")})
    with pytest.raises(ValueError, match="before test dispatch"):
        goal.validate_frozen_candidate(pretest, require_test_started=True)


@pytest.mark.parametrize("case,achieved", [("new_selected_passes", True), ("only_sac_passes", False), ("other_new_only_passes", False)])
def test_final_goal_uses_only_preselected_new_model_and_never_sac_only_or_best_test(pretest, case, achieved):
    selected = goal.select_candidate(pretest)
    frame = add_final_test(pretest, case=case)
    result = goal.verify_candidate(pretest)
    assert result["candidate"] == selected["new_model_validation_leader"] == {"algorithm": "tqc", "profile": "balanced"}
    assert result["original_global_validation_leader"]["algorithm"] == "sac"
    assert result["goal_performance_achieved"] is achieved
    assert result["accounting_audit_passed"] is True
    assert result["new_algorithms_executed"] == ["recurrent_ppo", "tqc", "crossq"]
    assert result["execution_coverage"] == {a: {"validation_runs": 24, "test_runs": 12} for a in ("recurrent_ppo", "tqc", "crossq")}
    sac = frame.loc[frame.run_id.eq("test__base__sac__balanced__ensemble")].iloc[0]
    assert sac.net_return >= .05 and abs(sac.max_drawdown) <= .20
    if case != "only_sac_passes":
        other = frame.loc[frame.run_id.eq("test__base__crossq__balanced__ensemble")].iloc[0]
        assert other.net_return > result["metrics"]["base_net_return"]
    assert (pretest / goal.EVIDENCE_FILE).is_file()


def test_final_verification_rejects_changed_audited_summary(pretest):
    goal.select_candidate(pretest)
    add_final_test(pretest)
    with (pretest / "summary.csv").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="audited input SHA"):
        goal.verify_candidate(pretest)
    assert not (pretest / goal.EVIDENCE_FILE).exists()


def test_partial_accounting_audit_cannot_produce_final_goal_success(pretest):
    goal.select_candidate(pretest)
    add_final_test(pretest)
    path = pretest / "independent_audit.local.json"
    audit = goal.read_json(path)
    audit.update(test_complete=False, status="partial_test_pending", run_count=110)
    write(path, audit)
    with pytest.raises(ValueError, match="complete 172-run accounting audit"):
        goal.verify_candidate(pretest)


def test_sac_cannot_be_passed_directly_as_goal_candidate(pretest):
    goal.select_candidate(pretest)
    frame = add_final_test(pretest)
    with pytest.raises(ValueError, match="SAC cannot"):
        goal.performance_gates(frame, {"algorithm": "sac", "profile": "balanced"}, TARGETS)
