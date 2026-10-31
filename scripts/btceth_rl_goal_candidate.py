"""Independent pre-test goal contract for newly researched BTC/ETH RL models.

This supplement never changes financial rules, the original global validation
leader, a training checkpoint, or an accounting audit. SAC remains a control.
Selection uses only frozen validation ensembles; final verification evaluates
that preselected new model rather than the best observed test performer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import uuid

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OBJECTIVE = "使用新调研的模型，达到比较好的效果，使用eth、btc对usdt交易对"
NEW_ALGORITHMS = ("recurrent_ppo", "tqc", "crossq")
CONTROL = "sac"
ALGORITHMS = (*NEW_ALGORITHMS, CONTROL)
PROFILES, SCENARIOS, SEEDS = ("return", "balanced"), ("base", "stress"), tuple(range(5))
BASELINES = ("cash", "btc_hold", "eth_hold", "equal_hold", "equal_rebalance", "sma200_trend", "momentum180")
SPLITS = {"train": ("2020-01-01T00:00:00Z", "2025-01-01T00:00:00Z"),
          "validation": ("2025-01-01T00:00:00Z", "2025-10-23T00:00:00Z"),
          "test": ("2025-10-23T00:00:00Z", "2026-10-23T00:00:00Z")}
TARGETS = {"primary_base_net_return_at_least": .05,
           "primary_base_max_drawdown_magnitude_at_most": .20,
           "primary_stress_net_return_at_least": 0.,
           "primary_beats_equal_hold_return_and_drawdown": True,
           "positive_base_training_seeds_at_least": 3,
           "stronger_claim_requires_beating_simple_trend_controls": True,
           "targets_are_not_guarantees": True}
SCORE_DESCRIPTION = "mean(base,stress)[log(final/initial)-0.5*abs(maxDD)-2*max(abs(maxDD)-.20,0)]"
PROTOCOL_FILE = "goal_candidate_protocol.json"
SELECTION_FILE = "goal_candidate_selection.json"
RECEIPT_FILE = "goal_candidate_selection_receipt.json"
EVIDENCE_FILE = "goal_candidate_evidence.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def stamp(value):
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        raise ValueError("Evidence timestamps must include an explicit UTC offset")
    return result.tz_convert("UTC")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def exclusive_json(path, value):
    """Publish complete new bytes without replacing any previous artifact."""
    path = Path(path)
    if path.exists():
        raise FileExistsError("Preserve the existing goal-contract artifact: " + str(path))
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic, exclusive creation on the same volume.
    finally:
        temporary.unlink(missing_ok=True)


def relative_file(root, name):
    text = str(name).replace("\\", "/")
    if text.startswith("/") or ":" in text or ".." in text.split("/"):
        raise ValueError("Pinned input paths must remain study-root-relative")
    result = (Path(root).resolve() / text).resolve()
    result.relative_to(Path(root).resolve())
    return result


def source_hashes():
    return {"scripts/btceth_rl_goal_candidate.py": sha(Path(__file__).resolve())}


def ensure_no_test(root):
    for name in ("test_started.json", "test_completion.json", "evaluation/test", EVIDENCE_FILE):
        if (Path(root) / name).exists():
            raise FileExistsError("Goal candidate must be declared/selected before any test attempt: " + name)


def load_spec(root):
    root = Path(root).resolve()
    spec = read_json(root / "frozen_spec.json")
    for split, dates in SPLITS.items():
        if [stamp(value) for value in spec[split]] != [stamp(value) for value in dates]:
            raise ValueError("Original chronological BTC/ETH split changed")
    if (spec["algorithms"] != list(ALGORITHMS) or set(spec["profiles"]) != set(PROFILES)
            or spec["seeds"] != list(SEEDS) or spec["baseline_names"] != list(BASELINES)
            or spec["capital"] != 10000. or spec["research_targets"] != TARGETS
            or spec["validation_selection"]["score"] != SCORE_DESCRIPTION):
        raise ValueError("Original frozen algorithms, capital, validation score or research targets changed")
    return spec, sha(root / "frozen_spec.json")


def protocol_contract(spec, spec_hash):
    return {"schema_version": 1, "objective": OBJECTIVE,
            "scope": "Fixed BTCUSDT/ETHUSDT/USDT funded-spot research; no trading authorization",
            "new_algorithms": list(NEW_ALGORITHMS), "control_algorithm": CONTROL,
            "new_model_success_required": True, "sac_only_success_is_not_goal_success": True,
            "original_global_leader_preserved": True, "financial_rules_unchanged": True,
            "original_frozen_spec_sha256": spec_hash, "original_financial_code_sha256": spec["code_sha256"],
            "source_sha256": source_hashes(), "dates_utc": {name: list(dates) for name, dates in SPLITS.items()},
            "research_targets": spec["research_targets"], "seeds": list(SEEDS),
            "selection_score": SCORE_DESCRIPTION, "selection_tie_break": "lexicographic algorithm/profile identifier",
            "selection_population": "All six new-algorithm/profile validation ensembles; five seeds retained, never seed-selected",
            "selection_file": SELECTION_FILE, "selection_receipt_file": RECEIPT_FILE,
            "selection_must_precede_test": True,
            "accounting_audit_is_not_performance_success": True,
            "known_test_year_already_seen": True}


def declare_protocol(root):
    root = Path(root).resolve()
    if (root / PROTOCOL_FILE).exists():
        raise FileExistsError("Preserve the existing declared protocol; do not redeclare it")
    ensure_no_test(root)
    spec, spec_hash = load_spec(root)
    result = {**protocol_contract(spec, spec_hash), "declared_at": now()}
    if stamp(result["declared_at"]) < stamp(spec["frozen_at"]):
        raise ValueError("Additional goal declaration predates the original frozen experiment")
    ensure_no_test(root)
    exclusive_json(root / PROTOCOL_FILE, result)
    return result


def load_protocol(root):
    spec, spec_hash = load_spec(root)
    protocol = read_json(Path(root) / PROTOCOL_FILE)
    expected = protocol_contract(spec, spec_hash)
    if {key: value for key, value in protocol.items() if key != "declared_at"} != expected:
        raise ValueError("Declared goal contract, frozen spec or goal-helper source changed")
    if stamp(protocol["declared_at"]) < stamp(spec["frozen_at"]):
        raise ValueError("Goal declaration must follow original experiment freeze")
    return spec, protocol


def verify_hashes(root, recorded, expected_names, label):
    if not isinstance(recorded, dict) or set(recorded) != set(expected_names):
        raise ValueError(label + " requires the exact declared file set")
    for name, expected in recorded.items():
        if sha(relative_file(root, name)) != expected:
            raise ValueError(label + " SHA changed: " + name)


def training_names():
    return {"training_completion.json", *[f"training/{a}__{p}__seed{s}/{name}"
            for a in ALGORITHMS for p in PROFILES for s in SEEDS for name in ("record.json", "model.zip")]}


def validation_names():
    return {f"evaluation/validation/{a}/{p}/summary.csv" for a in ALGORITHMS for p in PROFILES}


def read_summary(path):
    return pd.read_csv(path, float_precision="round_trip")


def validate_rows(frame, *, split, algorithm, profile, spec):
    required = {"run_id", "algorithm", "profile", "scenario", "split", "seed", "primary_policy",
                "initial_capital", "start", "end_exclusive", "net_return", "max_drawdown", "final_equity", "fees"}
    if not required.issubset(frame.columns):
        raise ValueError("Missing actual financial summary fields")
    expected_ids = {f"{split}__{scenario}__{algorithm}__{profile}__{key}"
                    for scenario in SCENARIOS for key in (*map(str, SEEDS), "ensemble")}
    if len(frame) != 12 or frame.run_id.duplicated().any() or set(frame.run_id) != expected_ids:
        raise ValueError("Need both costs, all five seeds and one actual ensemble")
    for row in frame.itertuples(index=False):
        key = row.run_id.rsplit("__", 1)[1]
        primary = key == "ensemble"
        if ((row.algorithm, row.profile, row.split) != (algorithm, profile, split)
                or row.scenario not in SCENARIOS or row.initial_capital != spec["capital"]
                or not isinstance(row.primary_policy, (bool, np.bool_)) or bool(row.primary_policy) != primary
                or (not pd.isna(row.seed) if primary else pd.isna(row.seed) or row.seed != int(key))):
            raise ValueError("Summary identity, capital, primary marker or seed mismatch")
        if stamp(row.start) != stamp(spec[split][0]) or stamp(row.end_exclusive) != stamp(spec[split][1]):
            raise ValueError("Summary split dates differ from original frozen protocol")
        values = [row.net_return, row.max_drawdown, row.final_equity, row.fees]
        if (not np.isfinite(values).all() or row.net_return <= -1 or not -1 < row.max_drawdown <= 0
                or row.final_equity <= 0 or row.fees < 0
                or not math.isclose(row.final_equity / row.initial_capital - 1, row.net_return, abs_tol=1e-10, rel_tol=1e-10)):
            raise ValueError("Non-finite or inconsistent financial summary")


def candidate_score(base, stress):
    def one(row):
        dd = abs(float(row["max_drawdown"]))
        return math.log1p(float(row["net_return"])) - .5 * dd - 2 * max(dd - .20, 0.)
    return (one(base) + one(stress)) / 2


def candidate_row(frame, algorithm, profile):
    values = {}
    for scenario in SCENARIOS:
        row = frame.loc[frame.scenario.eq(scenario) & frame.primary_policy.eq(True)].iloc[0]
        values[scenario] = {"run_id": str(row.run_id), "net_return": float(row.net_return),
                            "max_drawdown": float(row.max_drawdown), "final_equity": float(row.final_equity), "fees": float(row.fees)}
    return {"algorithm": algorithm, "profile": profile, "score": candidate_score(values["base"], values["stress"]), **values}


def rank(candidates):
    return sorted(candidates, key=lambda row: (-row["score"], row["algorithm"], row["profile"]))


def validation_candidates(root, spec, original):
    verify_hashes(root, original.get("training_source_sha256"), training_names(), "Training freeze")
    verify_hashes(root, original.get("validation_source_sha256"), validation_names(), "Validation freeze")
    spec_hash = sha(Path(root) / "frozen_spec.json")
    if (original.get("frozen_spec_sha256") != spec_hash or original.get("selection_uses_test") is not False
            or original.get("test_results_available_to_selector") is not False):
        raise ValueError("Original selection must be tied to the pre-test frozen experiment")
    completion = read_json(Path(root) / "training_completion.json")
    identities = {(a, p, s) for a in ALGORITHMS for p in PROFILES for s in SEEDS}
    records = completion.get("records", [])
    if (completion.get("model_count") != 40 or len(records) != 40
            or {(r["algorithm"], r["profile"], r["seed"]) for r in records} != identities
            or completion.get("frozen_spec_sha256") != spec_hash):
        raise ValueError("All 40 frozen training jobs must have completed")
    for record in records:
        relative = f"training/{record['algorithm']}__{record['profile']}__seed{record['seed']}"
        if (read_json(Path(root) / relative / "record.json") != record
                or record.get("frozen_spec_sha256") != spec_hash
                or record.get("model_sha256") != sha(Path(root) / relative / "model.zip")):
            raise ValueError("Training completion differs from the original record/checkpoint")
    done = read_json(Path(root) / "validation_completion.json")
    if done.get("model_groups") != 8 or done.get("baselines") != 7 or done.get("frozen_spec_sha256") != spec_hash:
        raise ValueError("All 110 original validation runs must have formally completed")
    candidates = []
    for algorithm in ALGORITHMS:
        for profile in PROFILES:
            frame = read_summary(Path(root) / f"evaluation/validation/{algorithm}/{profile}/summary.csv")
            validate_rows(frame, split="validation", algorithm=algorithm, profile=profile, spec=spec)
            candidates.append(candidate_row(frame, algorithm, profile))
    choices = {algorithm: rank([row for row in candidates if row["algorithm"] == algorithm])[0]["profile"] for algorithm in ALGORITHMS}
    global_leader = {key: rank(candidates)[0][key] for key in ("algorithm", "profile")}
    if original.get("selected_profiles") != choices or original.get("primary_validation_leader") != global_leader:
        raise ValueError("Original validation choice does not match independently recomputed scores")
    previous = {(row["algorithm"], row["profile"]): row for row in original.get("candidates", [])}
    if len(previous) != 8 or len(original.get("candidates", [])) != 8:
        raise ValueError("Original selection must retain all eight candidates")
    for row in candidates:
        old = previous.get((row["algorithm"], row["profile"]), {})
        if not math.isclose(float(old.get("score", math.nan)), row["score"], abs_tol=1e-12, rel_tol=1e-12):
            raise ValueError("Original candidate score differs from independent recomputation")
        for scenario in SCENARIOS:
            if any(not math.isclose(float(old.get(scenario, {}).get(name, math.nan)), row[scenario][name], abs_tol=1e-10, rel_tol=1e-10)
                   for name in ("net_return", "max_drawdown", "final_equity", "fees")):
                raise ValueError("Original candidate metrics differ from frozen validation")
    return candidates


def select_candidate(root):
    root = Path(root).resolve()
    if (root / SELECTION_FILE).exists() or (root / RECEIPT_FILE).exists():
        raise FileExistsError("Preserve an existing goal candidate selection/receipt")
    ensure_no_test(root)
    spec, protocol = load_protocol(root)
    original_path = root / "validation_selection.json"
    original = read_json(original_path)
    candidates = validation_candidates(root, spec, original)
    new_candidates = [row for row in candidates if row["algorithm"] in NEW_ALGORITHMS]
    chosen = rank(new_candidates)[0]
    leader = {key: chosen[key] for key in ("algorithm", "profile")}
    if leader["profile"] != original["selected_profiles"][leader["algorithm"]]:
        raise AssertionError("New-only candidate must use its original validation-selected profile")
    if (original["primary_validation_leader"]["algorithm"] in NEW_ALGORITHMS
            and leader != original["primary_validation_leader"]):
        raise AssertionError("When the global leader is new, both validation selections must coincide")
    result = {"schema_version": 1, "selected_at": now(), "objective": OBJECTIVE,
              "declared_protocol_sha256": sha(root / PROTOCOL_FILE), "source_sha256": source_hashes(),
              "original_frozen_spec_sha256": sha(root / "frozen_spec.json"),
              "original_validation_selection_sha256": sha(original_path),
              "original_global_validation_leader": original["primary_validation_leader"],
              "new_model_validation_leader": leader, "candidate": chosen,
              "new_validation_candidates": rank(new_candidates),
              "training_source_sha256": original["training_source_sha256"],
              "validation_source_sha256": original["validation_source_sha256"],
              "seed_selection": False, "selection_uses_test": False,
              "sac_role": "control only; never eligible for new-model goal success",
              "original_global_leader_preserved": True}
    if stamp(result["selected_at"]) < stamp(protocol["declared_at"]):
        raise ValueError("Candidate selection predates its declaration")
    ensure_no_test(root)
    exclusive_json(root / SELECTION_FILE, result)
    receipt = {"schema_version": 1, "sealed_at": now(), "selection_file": SELECTION_FILE,
               "selection_sha256": sha(root / SELECTION_FILE),
               "declared_protocol_sha256": sha(root / PROTOCOL_FILE),
               "original_validation_selection_sha256": sha(original_path)}
    exclusive_json(root / RECEIPT_FILE, receipt)
    return result


def expected_final_ids(original):
    expected = set()
    for split in ("validation", "test"):
        for scenario in SCENARIOS:
            for algorithm in ALGORITHMS:
                profiles = PROFILES if split == "validation" else (original["selected_profiles"][algorithm],)
                for profile in profiles:
                    expected.update(f"{split}__{scenario}__{algorithm}__{profile}__{key}" for key in (*map(str, SEEDS), "ensemble"))
            expected.update(f"{split}__{scenario}__{name}" for name in BASELINES)
    return expected


def performance_gates(summary, leader, targets):
    if leader["algorithm"] not in NEW_ALGORITHMS:
        raise ValueError("SAC cannot be the new-model goal candidate")
    rows = summary.loc[summary.split.eq("test") & summary.algorithm.eq(leader["algorithm"]) & summary.profile.eq(leader["profile"])]
    base = rows.loc[rows.scenario.eq("base") & rows.primary_policy.eq(True)].iloc[0]
    stress = rows.loc[rows.scenario.eq("stress") & rows.primary_policy.eq(True)].iloc[0]
    equal = summary.loc[summary.run_id.eq("test__base__equal_hold")].iloc[0]
    seeds = rows.loc[rows.scenario.eq("base") & rows.seed.notna()].sort_values("seed")
    if len(seeds) != 5 or set(seeds.seed.astype(int)) != set(SEEDS):
        raise ValueError("Goal candidate requires all five actual training seeds")
    positive = int((seeds.net_return > 0).sum())
    gates = {"base_return_at_least_5pct": bool(base.net_return >= targets["primary_base_net_return_at_least"]),
             "base_max_drawdown_at_most_20pct": bool(abs(base.max_drawdown) <= targets["primary_base_max_drawdown_magnitude_at_most"]),
             "stress_return_nonnegative": bool(stress.net_return >= targets["primary_stress_net_return_at_least"]),
             "beats_equal_hold_return_and_drawdown": bool(base.net_return > equal.net_return and base.max_drawdown > equal.max_drawdown),
             "at_least_three_positive_training_seeds": positive >= targets["positive_base_training_seeds_at_least"]}
    metrics = {"base_net_return": float(base.net_return), "base_max_drawdown": float(base.max_drawdown),
               "stress_net_return": float(stress.net_return), "stress_max_drawdown": float(stress.max_drawdown),
               "base_fees_usdt": float(base.fees), "stress_fees_usdt": float(stress.fees),
               "base_run_id": str(base.run_id), "stress_run_id": str(stress.run_id),
               "positive_base_training_seeds": positive,
               "base_seed_returns": [{"seed": int(row.seed), "net_return": float(row.net_return)} for row in seeds.itertuples(index=False)],
               "equal_hold_base_net_return": float(equal.net_return), "equal_hold_base_max_drawdown": float(equal.max_drawdown)}
    return metrics, gates


def validate_frozen_candidate(root, *, require_test_started=False):
    """Read-only goal-contract gate; reads validation evidence, never test PNL.

    Other precommitted comparisons can reuse this gate before their own test
    calculation. All source/hash names remain relative to the main study root.
    """
    root = Path(root).resolve()
    spec, protocol = load_protocol(root)
    selected = read_json(root / SELECTION_FILE)
    receipt = read_json(root / RECEIPT_FILE)
    original_path = root / "validation_selection.json"
    original = read_json(original_path)
    if (receipt.get("selection_file") != SELECTION_FILE or receipt.get("selection_sha256") != sha(root / SELECTION_FILE)
            or receipt.get("declared_protocol_sha256") != sha(root / PROTOCOL_FILE)
            or receipt.get("original_validation_selection_sha256") != sha(original_path)):
        raise ValueError("Goal candidate selection receipt SHA no longer matches frozen files")
    if (selected.get("declared_protocol_sha256") != sha(root / PROTOCOL_FILE)
            or selected.get("source_sha256") != source_hashes()
            or selected.get("original_frozen_spec_sha256") != sha(root / "frozen_spec.json")
            or selected.get("original_validation_selection_sha256") != sha(original_path)
            or selected.get("selection_uses_test") is not False or selected.get("seed_selection") is not False):
        raise ValueError("Goal selection no longer matches the declared pre-test contract")
    if not stamp(protocol["declared_at"]) <= stamp(selected["selected_at"]) <= stamp(receipt["sealed_at"]):
        raise ValueError("Goal declaration, selection and seal timestamps are inconsistent")
    marker_path = root / "test_started.json"
    if require_test_started and not marker_path.is_file():
        raise FileNotFoundError("Original formal test_started.json is required")
    if marker_path.is_file():
        marker = read_json(marker_path)
        if (marker.get("validation_selection_sha256") != sha(original_path)
                or marker.get("frozen_spec_sha256") != sha(root / "frozen_spec.json")
                or not stamp(receipt["sealed_at"]) < stamp(marker["at"])):
            raise ValueError("Goal candidate must have been declared, selected and sealed before test dispatch")
    elif (root / "evaluation/test").exists() or (root / "test_completion.json").exists():
        raise ValueError("Test evidence without its original dispatch marker cannot prove pre-test selection")
    candidates = validation_candidates(root, spec, original)
    new_candidates = rank([row for row in candidates if row["algorithm"] in NEW_ALGORITHMS])
    leader = {key: new_candidates[0][key] for key in ("algorithm", "profile")}
    if leader["profile"] != original["selected_profiles"][leader["algorithm"]]:
        raise ValueError("Goal candidate must not introduce an undeclared test profile")
    if (selected.get("new_model_validation_leader") != leader or selected.get("candidate") != new_candidates[0]
            or selected.get("new_validation_candidates") != new_candidates
            or selected.get("original_global_validation_leader") != original["primary_validation_leader"]
            or selected.get("training_source_sha256") != original["training_source_sha256"]
            or selected.get("validation_source_sha256") != original["validation_source_sha256"]):
        raise ValueError("Goal candidate differs from independently reproduced validation-only choice")
    if original["primary_validation_leader"]["algorithm"] in NEW_ALGORITHMS and leader != original["primary_validation_leader"]:
        raise ValueError("Original new-algorithm global winner must equal the goal candidate")
    return {"spec": spec, "protocol": protocol, "selection": selected, "receipt": receipt,
            "original_selection": original, "new_model_validation_leader": leader,
            "test_started": marker_path.is_file(), "test_pnl_read": False}


def verify_candidate(root, audit_path=None):
    root = Path(root).resolve()
    if (root / EVIDENCE_FILE).exists():
        raise FileExistsError("Preserve the existing final goal evidence")
    contract = validate_frozen_candidate(root, require_test_started=True)
    spec, protocol = contract["spec"], contract["protocol"]
    selected, original = contract["selection"], contract["original_selection"]
    leader = contract["new_model_validation_leader"]
    anchor_names = (PROTOCOL_FILE, SELECTION_FILE, RECEIPT_FILE, "frozen_spec.json", "validation_selection.json", "summary.csv", "test_started.json")
    anchors = {name: sha(root / name) for name in anchor_names}
    if audit_path is None:
        audit_path = root / ("independent_audit.local.json" if (root / "independent_audit.local.json").is_file() else "independent_audit.json")
    audit_path = Path(audit_path).resolve()
    audit_hash = sha(audit_path)
    audit = read_json(audit_path)
    if (audit.get("status") != "passed" or audit.get("passed") is not True or audit.get("test_complete") is not True
            or audit.get("failure_count") != 0 or audit.get("failures") != []
            or audit.get("run_count", audit.get("runs_checked")) != 172):
        raise ValueError("Final passed, complete 172-run accounting audit required; audit passage is not a performance claim")
    audit_inputs = audit.get("input_sha256", {})
    required = {"summary.csv", "frozen_spec.json", "validation_selection.json", "training_completion.json", "validation_completion.json", "test_started.json", "test_completion.json"}
    if not isinstance(audit_inputs, dict) or not required.issubset(audit_inputs):
        raise ValueError("Final audit must pin all required financial and test-dispatch inputs")
    for name, expected in audit_inputs.items():
        if sha(relative_file(root, name)) != expected:
            raise ValueError("Final audited input SHA changed: " + name)
    if audit.get("frozen_spec_sha256") != sha(root / "frozen_spec.json"):
        raise ValueError("Final audit belongs to another frozen experiment")
    frame = read_summary(root / "summary.csv")
    expected_ids = expected_final_ids(original)
    if len(frame) != 172 or frame.run_id.duplicated().any() or set(frame.run_id) != expected_ids or set(audit.get("runs", {})) != expected_ids:
        raise ValueError("Need exactly all172 audited validation/test rows")
    for algorithm in ALGORITHMS:
        for profile in PROFILES:
            validate_rows(frame.loc[frame.split.eq("validation") & frame.algorithm.eq(algorithm) & frame.profile.eq(profile)],
                          split="validation", algorithm=algorithm, profile=profile, spec=spec)
        validate_rows(frame.loc[frame.split.eq("test") & frame.algorithm.eq(algorithm)], split="test", algorithm=algorithm,
                      profile=original["selected_profiles"][algorithm], spec=spec)
    completed = read_json(root / "test_completion.json")
    if completed.get("model_groups") != 4 or completed.get("baselines") != 7 or completed.get("frozen_spec_sha256") != sha(root / "frozen_spec.json"):
        raise ValueError("Original formal test phase is incomplete")
    metrics, gates = performance_gates(frame, leader, protocol["research_targets"])
    for name in (metrics["base_run_id"], metrics["stress_run_id"]):
        audited = audit["runs"][name]
        if audited.get("reconstructed") is not True or audited.get("hours") != 365 * 24:
            raise ValueError("Candidate ensemble must be an independently reconstructed complete actual ledger")
    controls = {}
    for name in ("sma200_trend", "momentum180"):
        row = frame.loc[frame.run_id.eq("test__base__" + name)].iloc[0]
        controls[name] = bool(metrics["base_net_return"] > row.net_return and metrics["base_max_drawdown"] > row.max_drawdown)
    result = {"schema_version": 1, "verified_at": now(), "objective": OBJECTIVE,
              "new_algorithms_executed": list(NEW_ALGORITHMS),
              "execution_coverage": {a: {"validation_runs": int((frame.split.eq("validation") & frame.algorithm.eq(a)).sum()),
                                         "test_runs": int((frame.split.eq("test") & frame.algorithm.eq(a)).sum())} for a in NEW_ALGORITHMS},
              "candidate": leader, "candidate_validation_score": selected["candidate"]["score"],
              "original_global_validation_leader": original["primary_validation_leader"],
              "original_global_leader_preserved": True, "sac_role": "control only; SAC-only performance cannot satisfy this goal",
              "metrics": metrics, "gates": gates, "goal_performance_achieved": bool(all(gates.values())),
              "simple_trend_controls_dominated": controls,
              "stronger_claim_supported": bool(all(gates.values()) and all(controls.values())),
              "accounting_audit_passed": True, "accounting_audit_is_not_performance_success": True,
              "selection_uses_test": False, "seed_selection": False, "known_test_year_already_seen": True,
              "audit_path": str(audit_path), "audit_sha256": audit_hash,
              "source_sha256": source_hashes(),
              "frozen_inputs_sha256": anchors}
    for name, expected in {**audit_inputs, **anchors}.items():
        if sha(relative_file(root, name)) != expected:
            raise ValueError("Frozen goal/financial input changed during verification: " + name)
    if sha(audit_path) != audit_hash or source_hashes() != protocol["source_sha256"]:
        raise ValueError("Audit or goal-helper source changed during verification")
    exclusive_json(root / EVIDENCE_FILE, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("declare", "select", "verify"))
    parser.add_argument("--output-dir", "--study-root", dest="output_dir", type=Path,
                        default=ROOT / "results/backtests/btceth_rl_20261029")
    parser.add_argument("--audit-path", type=Path)
    args = parser.parse_args()
    functions = {"declare": declare_protocol, "select": select_candidate, "verify": verify_candidate}
    result = functions[args.phase](args.output_dir, args.audit_path) if args.phase == "verify" else functions[args.phase](args.output_dir)
    name = {"declare": PROTOCOL_FILE, "select": SELECTION_FILE, "verify": EVIDENCE_FILE}[args.phase]
    response = {"phase": args.phase, "path": str(args.output_dir.resolve() / name), "sha256": sha(args.output_dir.resolve() / name)}
    response.update({key: result[key] for key in ("new_model_validation_leader", "candidate", "gates", "goal_performance_achieved", "new_algorithms_executed") if key in result})
    print(json.dumps(response, ensure_ascii=False))


if __name__ == "__main__":
    main()
