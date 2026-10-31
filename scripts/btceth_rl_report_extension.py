"""Data-only report integration of a complete, sealed exposure-cap extension.

No selector, environment, model inference or market simulator is executed.
The original172 rows and global/new-only choices are retained as evidence.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from scripts.crypto_rl_reporting import clean_json
from scripts.audit_btceth_rl_exposure_cap import (ALGORITHMS, CASE_FILES, GROUP_FILES, MULTIPLIER,
    PROFILES, SPLITS, extension_validation_files, objective_evidence, validate_group_rows)

DISPLAY_CAP = "tqc_cap075"
ROOT = Path(__file__).resolve().parents[1]
MAIN_ANCHORS = ("goal_candidate_protocol.json", "goal_candidate_selection.json", "goal_candidate_selection_receipt.json",
                "frozen_spec.json", "validation_selection.json", "summary.csv", "test_started.json")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe(root, name):
    text = str(name).replace("\\", "/")
    p = PurePosixPath(text)
    if not text or p.is_absolute() or ":" in text or ".." in p.parts:
        raise ValueError("External evidence pins must remain relative to their explicitly supplied root")
    root = Path(root).resolve()
    result = root
    for part in p.parts:
        result = result / part
        if result.is_symlink() or (hasattr(result, "is_junction") and result.is_junction()):
            raise ValueError("External report evidence cannot cross links or junctions")
    result.resolve().relative_to(root)
    return result


def stamp(value):
    p = pd.Timestamp(value)
    if p.tzinfo is None:
        raise ValueError("Pre-test report contracts require timezone-aware UTC timestamps")
    return p.tz_convert("UTC")


class Sources:
    def __init__(self):
        self.hashes = {}

    def file(self, path):
        path = Path(path).resolve()
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError("Complete external evidence required: " + str(path))
        key = path.as_posix()
        actual = sha(path)
        if key in self.hashes and self.hashes[key] != actual:
            raise ValueError("External evidence changed between reads: " + key)
        self.hashes[key] = actual
        return path

    def json(self, path):
        return json.loads(self.file(path).read_text(encoding="utf-8-sig"))

    def csv(self, path):
        return pd.read_csv(self.file(path), float_precision="round_trip")

    def pins(self, root, values, *, required=(), label="Evidence"):
        if not isinstance(values, dict) or not values or not set(required).issubset(values):
            raise ValueError(label + " has partial SHA evidence")
        for name, expected in values.items():
            if sha(self.file(safe(root, name))) != expected:
                raise ValueError(label + " SHA changed: " + name)

    def unchanged(self):
        verify_external_inputs(self.hashes)


def verify_external_inputs(values):
    for name, expected in values.items():
        path = Path(name)
        if not path.is_absolute() or sha(path) != expected:
            raise ValueError("Absolute external report source changed: " + name)


def final_audit(sources, root, path, count, *, required):
    value = sources.json(path)
    if (value.get("passed") is not True or value.get("status") != "passed"
            or value.get("test_complete") is not True or value.get("failure_count") != 0
            or value.get("failures") != [] or value.get("run_count", value.get("runs_checked")) != count):
        raise ValueError(f"A full passed {count}-run final audit is required, never partial validation")
    sources.pins(root, value.get("input_sha256"), required=required, label="Final audit")
    return value


def same_metrics(actual, expected, label):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(label + " metric field mismatch")
        for key in expected:
            same_metrics(actual[key], expected[key], label + ":" + key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(label + " metric list mismatch")
        for a, b in zip(actual, expected):
            same_metrics(a, b, label)
    elif isinstance(expected, (int, float, np.number)) and not isinstance(expected, (bool, np.bool_)):
        if not isinstance(actual, (int, float, np.number)) or not np.isfinite([actual, expected]).all() or not np.isclose(actual, expected, atol=1e-10, rtol=1e-10):
            raise ValueError(label + " metric value mismatch")
    elif actual != expected:
        raise ValueError(label + " identity/value mismatch")


def audit_identity(root, primary, retained, sources):
    root, primary, retained = map(lambda p: Path(p).resolve(), (root, primary, retained))
    return {"primary_audit_file": primary.as_posix(), "primary_audit_sha256": sha(sources.file(primary)),
            "primary_audit_external_to_extension": not primary.is_relative_to(root),
            "retained_remote_audit_file": retained.as_posix(), "retained_remote_audit_sha256": sha(sources.file(retained)),
            "retained_remote_is_primary": primary == retained,
            "role": "Explicit local re-audit and retained remote goal-verification audit are separate identities"}


def goal_contracts(sources, main, extension, selection, protocol):
    original = sources.json(main / "validation_selection.json")
    raw_protocol = sources.json(main / "goal_candidate_protocol.json")
    raw_selection = sources.json(main / "goal_candidate_selection.json")
    raw_receipt = sources.json(main / "goal_candidate_selection_receipt.json")
    raw_evidence = sources.json(main / "goal_candidate_evidence.json")
    marker = sources.json(main / "test_started.json")
    ext_receipt = sources.json(extension / "selection_receipt.json")
    proto_receipt = sources.json(extension / "protocol_receipt.json")
    spec_sha = sha(sources.file(main / "frozen_spec.json"))
    if (protocol.get("original_frozen_spec_sha256") != spec_sha
            or protocol.get("original_goal_protocol_sha256") != sha(main / "goal_candidate_protocol.json")
            or proto_receipt.get("protocol_sha256") != sha(extension / "protocol.json")
            or proto_receipt.get("protocol_file") != "protocol.json"
            or ext_receipt.get("selection_file") != "selection.json"
            or ext_receipt.get("selection_sha256") != sha(extension / "selection.json")
            or not (ext_receipt.get("protocol_sha256") == selection.get("protocol_sha256") == sha(extension / "protocol.json"))):
        raise ValueError("Extension protocol/selection does not match original byte-bound receipts")
    if selection.get("protocol_sha256") != sha(extension / "protocol.json"):
        raise ValueError("Extension selection belongs to another fixed protocol")
    if (raw_receipt.get("selection_file") != "goal_candidate_selection.json"
            or raw_receipt.get("selection_sha256") != sha(main / "goal_candidate_selection.json")
            or not (raw_receipt.get("declared_protocol_sha256") == raw_selection.get("declared_protocol_sha256") == sha(main / "goal_candidate_protocol.json"))
            or raw_receipt.get("original_validation_selection_sha256") != sha(main / "validation_selection.json")):
        raise ValueError("Original new-only selection receipt changed")
    if (raw_selection.get("declared_protocol_sha256") != sha(main / "goal_candidate_protocol.json")
            or raw_selection.get("original_validation_selection_sha256") != sha(main / "validation_selection.json")
            or raw_selection.get("original_frozen_spec_sha256") != spec_sha
            or raw_selection.get("selection_uses_test") is not False or raw_selection.get("seed_selection") is not False
            or selection.get("selection_uses_test") is not False or selection.get("seed_selection") is not False):
        raise ValueError("Both new-model choices must be frozen without test/seed selection")
    if not (stamp(protocol["prepared_at"]) <= stamp(proto_receipt["sealed_at"]) <= stamp(selection["selected_at"])
            <= stamp(ext_receipt["sealed_at"]) < stamp(marker["at"])
            and stamp(raw_protocol["declared_at"]) <= stamp(raw_selection["selected_at"])
            <= stamp(raw_receipt["sealed_at"]) < stamp(marker["at"])):
        raise ValueError("Both original and extension choices must have been sealed before main test")
    if (marker.get("validation_selection_sha256") != sha(main / "validation_selection.json")
            or marker.get("frozen_spec_sha256") != spec_sha
            or selection.get("original_validation_selection_sha256") != sha(main / "validation_selection.json")
            or selection.get("original_goal_selection_sha256") != sha(main / "goal_candidate_selection.json")
            or selection.get("original_goal_receipt_sha256") != sha(main / "goal_candidate_selection_receipt.json")
            or selection.get("original_global_validation_leader") != original["primary_validation_leader"]
            or selection.get("original_new_model_validation_leader") != raw_selection["new_model_validation_leader"]):
        raise ValueError("Original global/new-only choices cannot be replaced by report/test ranking")
    if (raw_evidence.get("candidate") != raw_selection["new_model_validation_leader"]
            or raw_evidence.get("original_global_validation_leader") != original["primary_validation_leader"]
            or raw_evidence.get("accounting_audit_passed") is not True
            or raw_evidence.get("selection_uses_test") is not False or raw_evidence.get("seed_selection") is not False):
        raise ValueError("Original frozen new-model goal evidence is incomplete")
    sources.pins(main, raw_evidence.get("frozen_inputs_sha256"), required=MAIN_ANCHORS, label="Original new-model goal anchors")
    return original, raw_selection, raw_evidence


def gate_rows(summary, chosen, targets):
    rows = []
    metrics = targets["metrics"]
    descriptions = {
        "base_return_at_least_5pct": ("主成本净收益至少5%", f"{metrics['base_net_return']:.6%}", "≥5%"),
        "base_max_drawdown_at_most_20pct": ("小时最大回撤不超过20%", f"{metrics['base_max_drawdown']:.6%}", "回撤幅度≤20%"),
        "stress_return_nonnegative": ("压力成本收益非负", f"{metrics['stress_net_return']:.6%}", "≥0%"),
        "at_least_three_positive_training_seeds": ("至少三个训练种子测试收益为正", f"{metrics['positive_base_training_seeds']}/5", "≥3/5；全部保留"),
        "beats_equal_hold_return_and_drawdown": ("收益与回撤均优于等权被动持有", f"{metrics['equal_hold_base_net_return']:.6%} / {metrics['equal_hold_base_max_drawdown']:.6%}", "同窗口、同资金逐项比较"),
    }
    for key, value in targets["gates"].items():
        label, observed, criterion = descriptions[key]
        rows.append(dict(gate=key, label=label, observed=observed, criterion=criterion, met=bool(value), category="冻结新模型研究条件"))
    controls = {}
    for name, label in (("sma200_trend", "SMA200趋势对照"), ("momentum180", "180日动量对照")):
        other = summary.loc[summary.run_id.eq("test__base__" + name)].iloc[0]
        value = bool(metrics["base_net_return"] > other.net_return and metrics["base_max_drawdown"] > other.max_drawdown)
        controls[name] = value
        rows.append(dict(gate=name, label="收益及回撤均优于" + label, observed=f"{other.net_return:.6%} / {other.max_drawdown:.6%}",
                         criterion="净收益更高且回撤幅度更小", met=value, category="简单规则对照"))
    return rows, controls


def original_new_model_targets(summary, raw_selection, raw_evidence):
    """An accounting pass or the SAC control cannot stand in for the raw new model."""
    leader = raw_selection["new_model_validation_leader"]
    rows = summary.loc[summary.split.eq("test") & summary.algorithm.eq(leader["algorithm"])
                       & summary.profile.eq(leader["profile"])].copy()
    equal = summary.loc[summary.run_id.eq("test__base__equal_hold")]
    if len(equal) != 1:
        raise ValueError("Original new-model goal needs its actual same-window equal-hold control")
    calculated = objective_evidence(rows, {"family":"raw", **leader}, equal.iloc[0].to_dict())
    if (raw_evidence.get("gates") != calculated["gates"]
            or raw_evidence.get("goal_performance_achieved") is not calculated["goal_performance_achieved"]):
        raise ValueError("Original raw new-model goal differs from its actual twelve test accounts; SAC cannot substitute")
    same_metrics(raw_evidence.get("metrics"), calculated["metrics"], "Original raw new-model goal")
    return calculated


def load_extension(main_root, extension_root, *, extension_audit_path=None, validate_history=None):
    main, extension = Path(main_root).resolve(), Path(extension_root).resolve()
    if main == extension or extension.is_relative_to(main) or main.is_relative_to(extension):
        raise ValueError("Extension report sources must be a separate explicit study root")
    sources = Sources()
    contracts = ("protocol.json", "protocol_receipt.json", "selection.json", "selection_receipt.json", "validation_completion.json",
                 "test_started.json", "test_completion.json", "summary.csv", "daily_equity.csv", "monthly_returns.csv")
    primary_path = Path(extension_audit_path).resolve() if extension_audit_path else extension / "independent_audit.json"
    primary = final_audit(sources, extension, primary_path, 36, required=contracts)
    remote_path = extension / "independent_audit.json"
    remote = primary if primary_path == remote_path else final_audit(sources, extension, remote_path, 36, required=contracts)
    # The report executes these pure financial checks, so bind their actual
    # imported implementations as well as archived experiment source bytes.
    sources.file(Path(__file__))
    for name in ("scripts/audit_btceth_rl_exposure_cap.py", "scripts/audit_btceth_rl.py"):
        actual = sha(sources.file(ROOT / name))
        if any(value.get("source_sha256", {}).get(name) != actual for value in (primary, remote)):
            raise ValueError("Actual independent report calculation source differs from the complete audit: " + name)
    sources.pins(main, primary.get("main_input_sha256"), required=("frozen_spec.json", "validation_selection.json", "test_started.json", "test_completion.json"), label="Main inputs to extension audit")
    if remote is not primary:
        sources.pins(main, remote.get("main_input_sha256"), required=("frozen_spec.json", "test_completion.json"), label="Retained remote main inputs")
    protocol, selection = sources.json(extension / "protocol.json"), sources.json(extension / "selection.json")
    if (protocol.get("multiplier") != MULTIPLIER or protocol.get("capital") != 10000.
            or protocol.get("profiles") != list(PROFILES) or protocol.get("seeds") != list(range(5))
            or protocol.get("effective_max_coin_target_total") != .7125):
        raise ValueError("Only the single fixed TQC75% budget extension may be displayed")
    for name, expected in protocol.get("source_sha256", {}).items():
        if sha(sources.file(safe(extension / "sources", name))) != expected:
            raise ValueError("Frozen extension source snapshot changed")
    sources.pins(main, protocol.get("training_source_sha256"), label="Original ten TQC checkpoints")
    sources.pins(extension, selection.get("extension_source_sha256"), required=extension_validation_files(), label="Immutable24 capped validation cases")
    if any(not extension_validation_files().issubset(value["input_sha256"]) for value in (primary, remote)):
        raise ValueError("Both complete audits must bind every immutable capped validation account")
    sources.pins(main, selection.get("training_source_sha256"), label="Original40 model freeze")
    sources.pins(main, selection.get("validation_source_sha256"), label="Original eight immutable validation groups")
    original, raw_selection, raw_evidence = goal_contracts(sources, main, extension, selection, protocol)
    main_remote = final_audit(sources, main, main / "independent_audit.json", 172,
        required=("summary.csv", "frozen_spec.json", "validation_selection.json", "training_completion.json",
                  "validation_completion.json", "test_started.json", "test_completion.json"))
    if (main_remote.get("passed") is not True or main_remote.get("test_complete") is not True
            or main_remote.get("run_count", main_remote.get("runs_checked")) != 172
            or raw_evidence.get("audit_sha256") != sha(main / "independent_audit.json")):
        raise ValueError("Original new-model evidence must pin the retained full remote172 audit")
    evidence = sources.json(extension / "goal_candidate_evidence.json")
    chosen = selection.get("candidate", {})
    pool = selection.get("candidates", [])
    expected_pool = {("raw", a, p) for a in ALGORITHMS for p in PROFILES} | {("cap075", "tqc", p) for p in PROFILES}
    if (len(pool) != 8 or {(r.get("family"), r.get("algorithm"), r.get("profile")) for r in pool} != expected_pool
            or pool != sorted(pool, key=lambda r: (-r["score"], 0 if r["family"] == "raw" else 1, r["algorithm"], r["profile"]))
            or chosen != pool[0]):
        raise ValueError("Exactly eight frozen new-model candidates select the primary; two SAC profiles are controls only")
    if chosen.get("family") not in ("raw", "cap075") or chosen.get("algorithm") not in ALGORITHMS:
        raise ValueError("SAC/cash cannot replace the sealed new-model primary")
    if (evidence.get("candidate") != {key: chosen[key] for key in ("family", "algorithm", "profile")}
            or evidence.get("audit_sha256") != sha(remote_path)
            or evidence.get("protocol_sha256") != sha(extension / "protocol.json")
            or evidence.get("selection_sha256") != sha(extension / "selection.json")
            or evidence.get("accounting_audit_passed") is not True
            or evidence.get("selection_uses_test") is not False or evidence.get("seed_selection") is not False):
        raise ValueError("Extension goal evidence must match the sealed primary and retained remote audit bytes")
    summary = sources.csv(extension / "summary.csv")
    history, months = sources.csv(extension / "daily_equity.csv"), sources.csv(extension / "monthly_returns.csv")
    expected = {f"validation__{c}__cap075__tqc__{p}__{k}" for p in PROFILES for c in ("base", "stress") for k in (*map(str, range(5)), "ensemble")}
    test_prefix = "cap075__" if chosen["family"] == "cap075" else ""
    expected.update(f"test__{c}__{test_prefix}{chosen['algorithm']}__{chosen['profile']}__{k}" for c in ("base", "stress") for k in (*map(str, range(5)), "ensemble"))
    if len(summary) != 36 or summary.run_id.duplicated().any() or set(summary.run_id) != expected or set(primary.get("runs", {})) != expected or set(remote.get("runs", {})) != expected:
        raise ValueError("Full extension report requires exactly24 cap-validation and12 sealed test/reference rows")
    for audit in (primary, remote):
        for run_id in expected:
            row = audit["runs"][run_id]
            if row.get("reconstructed") is not True or row.get("hours") != (7080 if run_id.startswith("validation__") else 8760):
                raise ValueError("Every extension/raw reference needs its complete real ledger audit")
    for p in PROFILES:
        validate_group_rows(summary.loc[summary.split.eq("validation") & summary.profile.eq(p)], split="validation", algorithm="tqc", profile=p, multiplier=MULTIPLIER)
    selected_frame = summary.loc[summary.split.eq("test")].copy()
    validate_group_rows(selected_frame, split="test", algorithm=chosen["algorithm"], profile=chosen["profile"], multiplier=MULTIPLIER if chosen["family"] == "cap075" else 1.)
    if validate_history is not None:
        validate_history(history, months, summary)
    main_summary = sources.csv(main / "summary.csv")
    raw_targets = original_new_model_targets(main_summary, raw_selection, raw_evidence)
    done = sources.json(extension / "test_completion.json")
    dispatch = sources.json(extension / "test_started.json")
    if (done.get("case_count") != 12 or done.get("repeat_count") != 12
            or done.get("protocol_sha256") != sha(extension / "protocol.json")
            or done.get("selection_sha256") != sha(extension / "selection.json")
            or done.get("main_test_completion_sha256") != sha(main / "test_completion.json")
            or dispatch.get("selection_sha256") != sha(extension / "selection.json")
            or dispatch.get("protocol_sha256") != sha(extension / "protocol.json")
            or dispatch.get("main_test_started_sha256") != sha(main / "test_started.json")):
        raise ValueError("Selected test/reference must be bound to both immutable dispatches")
    for item in pool:
        frame = summary if item["family"] == "cap075" else main_summary
        for scenario in ("base", "stress"):
            actual = frame.loc[frame.run_id.eq(item[scenario]["run_id"])]
            if (len(actual) != 1 or actual.iloc[0].split != "validation" or actual.iloc[0].scenario != scenario
                    or actual.iloc[0].algorithm != item["algorithm"] or actual.iloc[0].profile != item["profile"]
                    or not bool(actual.iloc[0].primary_policy)):
                raise ValueError("Candidate table must refer to its actual validation ensemble")
            for metric in ("net_return", "max_drawdown", "final_equity", "fees"):
                same_metrics(item[scenario][metric], actual.iloc[0][metric], "Validation candidate " + metric)
    if chosen["family"] == "raw":
        folder = f"evaluation/test/{chosen['algorithm']}/{chosen['profile']}"
        names = {*(f"{folder}/{f}" for f in GROUP_FILES),
                 *(f"{folder}/{c}/{k}/{f}" for c in ("base", "stress") for k in (*map(str, range(5)), "ensemble") for f in CASE_FILES[:6])}
        if (done.get("mode") != "raw_reference" or (extension / "test").exists()
                or set(done.get("raw_source_sha256", {})) != names):
            raise ValueError("Raw winner must reuse exactly76 original files without extra test inference")
        sources.pins(main, done.get("raw_source_sha256"), label="Reused original raw test files")
        if any(not names.issubset(value.get("main_input_sha256", {})) for value in (primary, remote)):
            raise ValueError("Both complete audits must bind all76 original raw test reference files")
        for row in selected_frame.itertuples(index=False):
            other = main_summary.loc[main_summary.run_id.eq(row.run_id)]
            if len(other) != 1:
                raise ValueError("Raw test reference is absent from original172 rows")
            for key in ("net_return", "max_drawdown", "fees", "final_equity", "average_exposure"):
                same_metrics(getattr(row, key), other.iloc[0][key], "Raw reuse " + key)
    else:
        if done.get("mode") != "cap075_evaluation" or done.get("raw_source_sha256") != {}:
            raise ValueError("Cap winner requires its own actual test account, never raw post-test replacement")
        names = {*(f"test/{chosen['profile']}/{f}" for f in GROUP_FILES),
                 *(f"test/{chosen['profile']}/{c}/{k}/{f}" for c in ("base", "stress") for k in (*map(str, range(5)), "ensemble") for f in CASE_FILES)}
        if not names.issubset(primary["input_sha256"]) or not names.issubset(remote["input_sha256"]):
            raise ValueError("Both audits must bind all twelve actual capped test accounts")
    equal = main_summary.loc[main_summary.run_id.eq("test__base__equal_hold")].iloc[0]
    calculated = objective_evidence(selected_frame, chosen, equal.to_dict())
    for value in (primary.get("research_targets", {}), remote.get("research_targets", {}), evidence):
        if value.get("evaluated", True) is not True or value.get("gates") != calculated["gates"] or value.get("goal_performance_achieved") is not calculated["goal_performance_achieved"]:
            raise ValueError("Displayed new-model success differs from final audit/goal evidence")
        same_metrics(value.get("metrics"), calculated["metrics"], "New-model goal")
    gate_table, controls = gate_rows(main_summary, chosen, calculated)
    identity = audit_identity(extension, primary_path, remote_path, sources)
    source = {"label": "测试前封存的TQC75%币仓预算扩展与原模型对照", "classification": "modeled", "timezone": "UTC",
        "period": {"start": SPLITS["validation"][0], "endExclusive": SPLITS["test"][1]},
        "files": list(sources.hashes), "fileHashes": dict(sources.hashes), "extensionAuditEvidence": identity,
        "caveats": ["75%预算方案是在观察验证表现后设计，不把该验证称为全新样本。", "原40模型未重训；新预算使用自己的真实资金状态重新推理，未缩放旧收益曲线。", "SAC仅为原算法对照；测试排名不能替换封存的新模型主方案。"]}
    sources.unchanged()
    return dict(summary=summary, history=history, months=months, selection=selection, original_selection=original,
        raw_selection=raw_selection, raw_goal_evidence=raw_evidence, raw_targets=raw_targets, evidence=evidence, targets=calculated,
        gate_rows=gate_table, trend_controls=controls, audit=primary, audit_identity=identity, source=source,
        external_input_sha256=dict(sources.hashes))


def source_with_extension(original, extra):
    source = copy.deepcopy(original)
    source["files"] = list(dict.fromkeys([*source.get("files", []), *extra["files"]]))
    source["fileHashes"] = {**source.get("fileHashes", {}), **extra["fileHashes"]}
    source["extensionAuditEvidence"] = extra["extensionAuditEvidence"]
    source["caveats"] = list(dict.fromkeys([*source.get("caveats", []), *extra.get("caveats", [])]))
    return source


def augment_snapshot(snapshot, evidence, *, main_root, extension_root, extension_audit_path=None, validate_history=None):
    payload = load_extension(main_root, extension_root, extension_audit_path=extension_audit_path, validate_history=validate_history)
    snapshot, evidence = copy.deepcopy(snapshot), copy.deepcopy(evidence)
    queries, report = snapshot["queries"], snapshot["report"]
    selected = payload["selection"]["candidate"]
    cap_profile = next(r["profile"] for r in payload["selection"]["candidates"] if r["family"] == "cap075")
    additional = payload["summary"].loc[payload["summary"].family.eq("cap075")].copy()
    labels, lookup = {}, {}
    for row in additional.itertuples(index=False):
        labels[row.run_id] = "TQC75%预算" + ("五模型主组合" if row.primary_policy else f" seed {int(row.seed)}")
        lookup[row.run_id] = dict(algorithm=DISPLAY_CAP, source_algorithm="tqc", family="cap075", profile=row.profile,
            primary_policy=bool(row.primary_policy), policy_role="primary" if row.primary_policy else "training_seed",
            selected_profile=row.profile == cap_profile)
    frames = {"strategy_summary": additional,
              "equity_history": payload["history"].loc[payload["history"].run_id.isin(lookup)].copy(),
              "monthly_returns": payload["months"].loc[payload["months"].run_id.isin(lookup)].copy()}
    for key, frame in frames.items():
        frame["label"] = frame.run_id.map(labels)
        for name in ("algorithm", "source_algorithm", "family", "profile", "primary_policy", "policy_role", "selected_profile"):
            frame[name] = frame.run_id.map(lambda run: lookup[run][name])
        existing = queries[key]["rows"]
        if set(row["run_id"] for row in existing) & set(frame.run_id):
            raise ValueError("Raw test references must not be counted as new report runs")
        queries[key]["rows"] = [*existing, *clean_json(frame.astype(object).where(pd.notna(frame), None).to_dict("records"))]
    statistics = []
    for (split, cost, profile), values in additional.loc[additional.seed.notna()].groupby(["split", "scenario", "profile"]):
        if len(values) != 5 or set(values.seed.astype(int)) != set(range(5)):
            raise ValueError("Cap display ranges require all five actual seed accounts")
        statistics.append(dict(split=split, scenario=cost, algorithm=DISPLAY_CAP, source_algorithm="tqc", family="cap075",
            profile=profile, seeds=5, best_return=float(values.net_return.max()), worst_return=float(values.net_return.min()),
            median_return=float(values.net_return.median()), worst_drawdown=float(values.max_drawdown.min()),
            best_drawdown=float(values.max_drawdown.max()), positive_seeds=int(values.net_return.gt(0).sum()), selected_profile=profile == cap_profile))
    queries["seed_statistics"]["rows"].extend(statistics)
    by_identity = {(r["family"], r["algorithm"], r["profile"]): r for r in payload["selection"]["candidates"]}
    table = []
    for row in queries["validation_selection"]["rows"]:
        row = dict(row)
        identity = ("raw", row["algorithm"], row["profile"])
        row.update(family="raw", source_algorithm=row["algorithm"], eligible_for_primary=row["algorithm"] != "sac",
            original_global_leader=payload["original_selection"]["primary_validation_leader"] == {"algorithm":row["algorithm"],"profile":row["profile"]},
            original_raw_new_leader=payload["raw_selection"]["new_model_validation_leader"] == {"algorithm":row["algorithm"],"profile":row["profile"]},
            primary_leader=identity == (selected["family"], selected["algorithm"], selected["profile"]))
        table.append(row)
    for profile in PROFILES:
        row = by_identity[("cap075", "tqc", profile)]
        table.append(dict(family="cap075", algorithm=DISPLAY_CAP, source_algorithm="tqc", profile=profile, score=row["score"],
            base_return=row["base"]["net_return"], base_drawdown=row["base"]["max_drawdown"],
            stress_return=row["stress"]["net_return"], stress_drawdown=row["stress"]["max_drawdown"],
            selected=profile == cap_profile, eligible_for_primary=True, original_global_leader=False, original_raw_new_leader=False,
            primary_leader=selected["family"] == "cap075" and selected["profile"] == profile))
    if len(table) != 10 or sum(r["eligible_for_primary"] for r in table) != 8 or sum(r["primary_leader"] for r in table) != 1:
        raise ValueError("Eight new candidates plus two SAC controls must identify exactly one sealed new primary")
    queries["validation_selection"]["rows"] = table
    queries["research_targets"]["rows"] = payload["gate_rows"]
    queries["research_targets"]["methods"] = [{"language":"text", "code":"Use the pre-test sealed extension candidate's actual twelve test/reference accounts. Recompute the five original objective gates and require the retained remote goal evidence plus selected local/full36 audit to agree. SAC is a control, not a substitute primary."}]
    for key in ("strategy_summary", "equity_history", "monthly_returns", "seed_statistics", "validation_selection", "research_targets", "audit_evidence"):
        queries[key]["source"] = source_with_extension(queries[key]["source"], payload["source"])
    identity = payload["audit_identity"]
    queries["audit_evidence"]["rows"].append(dict(status="passed", passed=True, audit_role="独立预算扩展（含引用时不重复计新账本）",
        checks_total=payload["audit"]["checks"], runs_checked=36, failures=0,
        audit_completed_at=payload["audit"].get("audited_at", payload["audit"].get("audit_completed_at")),
        dataset_sha256=payload["audit"].get("dataset_sha256"), **identity))
    primary = {"algorithm": DISPLAY_CAP if selected["family"] == "cap075" else selected["algorithm"],
               "source_algorithm":selected["algorithm"],"profile":selected["profile"],"family":selected["family"]}
    met = bool(payload["targets"]["goal_performance_achieved"])
    report["originalGlobalLeader"] = copy.deepcopy(report["primaryLeader"])
    report["rawNewLeader"] = copy.deepcopy(payload["raw_selection"]["new_model_validation_leader"])
    report["rawNewModelTargets"] = copy.deepcopy(payload["raw_targets"])
    report["primaryLeader"] = primary
    report["primaryBaseReturn"] = payload["targets"]["metrics"]["base_net_return"]
    report["researchTargets"] = {**payload["targets"], "primary_validation_leader":primary,
        "all_predeclared_primary_targets_met":met,"dominates_simple_trend_controls":payload["trend_controls"],
        "stronger_claim_supported":met and all(payload["trend_controls"].values())}
    report["extensionSelection"] = payload["selection"]
    report["extension"] = {"candidate":{k:selected[k] for k in ("family","algorithm","profile")},
        "eligibleCandidates":8,"sacControlProfiles":2,"scopeRuns":36,"additionalUniqueRuns":len(additional),
        "rawTestReferencesReused":selected["family"]=="raw","auditEvidence":identity,
        "designedAfterValidation":True,"additionalTrainingModels":0}
    snapshot["title"] = "新模型主方案达到本轮研究条件" if met else "新模型主方案尚未满足全部研究条件"
    evidence.update(main_runs=172, extension_scope_runs=36, additional_unique_extension_runs=len(additional),
        runs=len(queries["strategy_summary"]["rows"]), extension_audit_path=identity["primary_audit_file"],
        extension_retained_remote_audit_path=identity["retained_remote_audit_file"],
        external_input_sha256={**evidence.get("external_input_sha256",{}),**payload["external_input_sha256"]})
    verify_external_inputs(evidence["external_input_sha256"])
    return snapshot, evidence


def augment_risk_controls(snapshot, evidence, *, reference_root, dataset_path, main_root, extension_root=None):
    from scripts.btceth_rl_report_risk_controls import load_risk_controls
    payload = load_risk_controls(reference_root, dataset_path=dataset_path, main_output=main_root, cap_output=extension_root)
    if payload.get("audit",{}).get("passed") is not True or payload["audit"].get("cases") != 6 or payload["audit"].get("full_repeats") != 6:
        raise ValueError("Optional risk reference requires all six complete, repeated independent test accounts")
    snapshot, evidence = copy.deepcopy(snapshot), copy.deepcopy(evidence)
    for key in ("risk_budget_summary","risk_budget_equity","risk_budget_monthly"):
        snapshot["queries"][key] = copy.deepcopy(payload["queries"][key])
    snapshot["report"]["riskBudgetControls"] = {"cases":6,"audit":payload["audit"],
        "includedInModelRanking":False,"exactRiskMatchClaimed":False}
    evidence["external_input_sha256"] = {**evidence.get("external_input_sha256",{}),**payload["external_input_sha256"]}
    evidence["risk_reference_runs"] = 6
    verify_external_inputs(evidence["external_input_sha256"])
    return snapshot,evidence

