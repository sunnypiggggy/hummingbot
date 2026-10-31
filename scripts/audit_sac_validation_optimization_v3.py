"""Final fallback audit preserving prior passes and correcting a digest binding.

Does not import a runner, adapter, environment or learner. Native zip metadata
is read as JSON; model tensors and Python-serialized payloads are never loaded.
The unchanged public financial oracle performs the hourly accounting audit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import importlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
import zipfile

import numpy as np
import pandas as pd

PLAN_SHA256 = "8ab80ceb21720691af055c7ed3505b69997087437f87315484ba2c64af625c36"
ORIGINAL_AUDITOR_SHA256 = "f84e18ea2d587cbce962f3c9ec9f7385fb56ce5f91284f7dd8c102f6d92fe3ad"
PARENT_AUDITOR_SHA256 = "a53f412c748899bec6cc961316eedad52c354f522248a5f8b9d6d99b279ab468"
REVISION_ID = "native_sac_policy_kwargs_default_false_v2"
REVISION_REASON = "sb3_sac_saved_policy_kwargs_use_sde_false"
REVISION_AUDIT_PATH = "audit_revision_source/scripts/audit_sac_validation_optimization_v2.py"
REVISION_CONTROLLER_PATH = "audit_revision_source/scripts/continue_sac_validation_optimization.py"
REVISION_NORMALIZATION = "net_arch unchanged; use_sde absent or boolean False only; reject unknown policy kwargs"
IMPORT_REVISION_ID = "retained_import_population_shadowing_v3"
IMPORT_REVISION_REASON = "import_manifest_expected_digest_shadowing"
IMPORT_AUDIT_PATH = "audit_import_revision_source/scripts/audit_sac_validation_optimization_v3.py"
IMPORT_FINISHER_PATH = "audit_import_revision_source/scripts/finish_sac_fallback_audit_v3.py"
IMPORT_PERMITTED_CHANGE = "rename inner imported-file expected digest; preserve exact population and byte hashes"
FINAL_VALIDATION_AUDIT = "final/validation_optimization_audit_v3.json"
FINAL_TEST_AUDIT = "final/test_optimization_audit_v3.json"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
TUNING_SEEDS = tuple(range(100, 105))
FINAL_SEEDS = tuple(range(30))
CANDIDATE_ORDER = ("c00_baseline", "c01_lr1e4", "c02_gamma099", "c03_lr1e4_gamma099")
BASELINES = ("cash", "buy_hold", "sma200_trend", "momentum180")
COSTS = {"base": {"fee": .001, "slippage": .0002}, "stress": {"fee": .002, "slippage": .0005}}
BOUNDS = {"train": ["2020-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
          "validation": ["2025-01-01T00:00:00Z", "2025-10-23T00:00:00Z"],
          "test": ["2025-10-23T00:00:00Z", "2026-10-23T00:00:00Z"]}
MARKET_FEATURES = ("close23utc_return_1d/.1", "close23utc_return_7d/.3", "close23utc_return_30d/.5",
    "close23utc_return_90d/1", "close23utc_return_180d/2", "close23utc_sma20_deviation/.2",
    "close23utc_sma60_deviation/.35", "close23utc_sma200_deviation/.5", "close23utc_volatility30/.1", "volume24h_relative30_minus1/2")
EXTENSION_FILES = ("scripts/backtest_sac_validation_optimization.py", "scripts/sac_validation_optimization_models.py", "scripts/audit_sac_validation_optimization.py")
RESULT_FILES = ("equity_1h.csv.gz", "fills.csv", "weights.csv", "boundary_equity.csv", "metadata.json", "fingerprints.json", "actions.json", "summary.json", "daily_equity.csv", "monthly_returns.csv")


def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()
def unlinked(path):
    path = Path(path).absolute()
    if any(p.is_symlink() or hasattr(p, "is_junction") and p.is_junction() for p in (path, *path.parents)):
        raise ValueError("Evidence cannot cross a link")
    return path.resolve()
def safe(root, name):
    name = str(name); parsed = PurePosixPath(name)
    if not name or parsed.as_posix() != name or parsed.is_absolute() or ".." in parsed.parts or ":" in name or "\\" in name:
        raise ValueError("Canonical root-relative evidence path required")
    root = unlinked(root); path = unlinked(root.joinpath(*parsed.parts)); path.relative_to(root)
    if not path.is_file(): raise FileNotFoundError(name)
    return path
def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")


class Checks:
    def __init__(self): self.count, self.failures = 0, []
    def check(self, okay, label):
        self.count += 1
        if not bool(okay): self.failures.append(label)
    def close(self, a, b, label):
        try: okay = math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(float(a), float(b), abs_tol=1e-12, rel_tol=1e-12)
        except (TypeError, ValueError): okay = False
        self.check(okay, label)
    def result(self): return {"passed": not self.failures, "checks_total": self.count, "failure_count": len(self.failures), "failures": self.failures}


def load_plan(top):
    path = safe(top, "search_plan.json"); plan = read(path)
    if sha(path) != PLAN_SHA256 or read(safe(top, "search_plan_receipt.json")).get("sha256") != PLAN_SHA256:
        raise ValueError("Use the original plan frozen before any search training")
    return plan


def quantile(values, probability):
    values = sorted(float(v) for v in values)
    if not values or any(not math.isfinite(v) for v in values): raise ValueError("Finite nonempty score population required")
    position = (len(values) - 1) * probability; low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def utility(net_return, drawdown):
    net_return, drawdown = float(net_return), abs(float(drawdown))
    if not math.isfinite(net_return) or not math.isfinite(drawdown) or net_return <= -1: raise ValueError("Invalid validation economics")
    return math.log1p(net_return) - .5 * drawdown - 2 * max(drawdown - .20, 0.)


def continuous_window_metrics(frame, start, end, *, initial_start=BOUNDS["validation"][0]):
    """Include pre-window NAV and every cost-bearing hourly close; never reset positions."""
    frame = frame.copy()
    if "timestamp" in frame:
        frame.index = pd.to_datetime(frame.timestamp, utc=True)
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None or frame.index.has_duplicates:
        raise ValueError("Unique timezone-aware hourly marks required")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    expected = pd.date_range(start, end, freq="h", inclusive="left")
    inside = frame.loc[(frame.index >= start) & (frame.index < end), "equity"]
    if not inside.index.equals(expected): raise ValueError("Every subwindow hour must exist")
    before = 10000. if start == pd.Timestamp(initial_start) else float(frame.loc[start - pd.Timedelta(hours=1), "equity"])
    values = np.r_[before, inside.to_numpy(dtype=float)]
    if not np.isfinite(values).all() or (values <= 0).any(): raise ValueError("Positive complete continuous NAV required")
    return float(values[-1] / before - 1), float((values / np.maximum.accumulate(values) - 1).min())


def compute_candidate_metrics(candidate_root, plan=None):
    """Independent score calculation from validation only; no annual path is opened."""
    candidate_root = Path(candidate_root)
    if plan is None: plan = load_plan(candidate_root.parents[1])
    frame = pd.read_csv(safe(candidate_root, "validation_summary.csv"), float_precision="round_trip")
    expected = {(s, a, n, c) for s in SYMBOLS for a in ("sac",) for n in TUNING_SEEDS for c in COSTS}
    expected.update((s, a, None, c) for s in SYMBOLS for a in BASELINES for c in COSTS)
    observed = {(r.symbol, r.algorithm, None if pd.isna(r.seed) else int(r.seed), r.scenario) for r in frame.itertuples()}
    if len(frame) != 36 or observed != expected or set(frame.split) != {"validation"} or not frame.initial_capital.eq(10000.).all():
        raise ValueError("Exactly36 complete tuning validation accounts required")
    result = {}
    for symbol in SYMBOLS:
        scores, base_returns, stress_returns, stress_dd, window_scores = [], [], [], [], [[] for _ in plan["subwindows"]]
        for seed in TUNING_SEEDS:
            group = frame.loc[frame.symbol.eq(symbol) & frame.algorithm.eq("sac") & frame.seed.eq(seed)]
            if len(group) != 2 or set(group.scenario) != set(COSTS): raise ValueError("Two actual cost accounts per tuning seed required")
            cost_scores, per_window = [], [[] for _ in plan["subwindows"]]
            for row in group.itertuples():
                cost_scores.append(utility(row.net_return, row.max_drawdown))
                if row.scenario == "base": base_returns.append(float(row.net_return))
                else: stress_returns.append(float(row.net_return)); stress_dd.append(abs(float(row.max_drawdown)))
                p = safe(candidate_root, f"evaluation/validation/{symbol}/sac/seed{seed}/{row.scenario}/equity_1h.csv.gz")
                hourly = pd.read_csv(p, float_precision="round_trip")
                for i, bounds in enumerate(plan["subwindows"]):
                    r, d = continuous_window_metrics(hourly, *bounds); per_window[i].append(utility(r, d))
            scores.append(math.fsum(cost_scores) / 2)
            for i, values in enumerate(per_window): window_scores[i].append(math.fsum(values) / 2)
        median = quantile(scores, .5); iqr = quantile(scores, .75) - quantile(scores, .25)
        result[symbol] = {"seed_scores": scores, "score_median": median, "score_iqr": iqr,
            "robust_score": median - .25 * iqr, "base_positive_seeds": sum(v > 0 for v in base_returns),
            "stress_median_return": quantile(stress_returns, .5), "stress_p90_drawdown": quantile(stress_dd, .9),
            "subwindow_median_scores": [quantile(values, .5) for values in window_scores]}
    return result


def compute_expected_selection(metrics_by_candidate, plan):
    if set(metrics_by_candidate) != set(CANDIDATE_ORDER): raise ValueError("Full predeclared candidate grid required")
    recipes = {r["candidate_id"]: r["native_recipe"] for r in plan["candidates"]}; bounds = plan["acceptance"]
    result = {}
    for symbol in SYMBOLS:
        by_candidate = {cid: metrics_by_candidate[cid][symbol] for cid in CANDIDATE_ORDER}; baseline = by_candidate[CANDIDATE_ORDER[0]]
        gates, eligible = {}, [CANDIDATE_ORDER[0]]
        for cid in CANDIDATE_ORDER:
            stat = by_candidate[cid]
            gates[cid] = {
                "score_improvement_at_least": stat["robust_score"] >= baseline["robust_score"] + bounds["score_improvement_at_least"],
                "paired_better_seed_count_at_least": sum(a > b for a, b in zip(stat["seed_scores"], baseline["seed_scores"])) >= bounds["paired_better_seed_count_at_least"],
                "positive_base_seed_count_at_least": stat["base_positive_seeds"] >= bounds["positive_base_seed_count_at_least"],
                "stress_return_median_at_least": stat["stress_median_return"] >= bounds["stress_return_median_at_least"],
                "stress_p90_drawdown_above_baseline_at_most": stat["stress_p90_drawdown"] <= baseline["stress_p90_drawdown"] + bounds["stress_p90_drawdown_above_baseline_at_most"],
                "nonworse_subwindows_at_least": sum(a >= b for a, b in zip(stat["subwindow_median_scores"], baseline["subwindow_median_scores"])) >= bounds["nonworse_subwindows_at_least"]}
            if cid != CANDIDATE_ORDER[0] and all(gates[cid].values()): eligible.append(cid)
        ordered = sorted(eligible, key=lambda cid: (-by_candidate[cid]["robust_score"], cid != CANDIDATE_ORDER[0], cid))
        selected = ordered[0]
        result[symbol] = {"candidate_id": selected, "selected_recipe": recipes[selected], "fallback_to_baseline": selected == CANDIDATE_ORDER[0],
            "metrics_by_candidate": by_candidate, "gates_by_candidate": gates, "eligible_candidates": ordered}
    return result


def recursive_equal(checks, actual, expected, label):
    if isinstance(expected, dict):
        checks.check(isinstance(actual, dict) and set(actual) == set(expected), label + ":keys")
        if isinstance(actual, dict):
            for key, value in expected.items(): recursive_equal(checks, actual.get(key), value, label + ":" + key)
    elif isinstance(expected, list):
        checks.check(isinstance(actual, list) and len(actual) == len(expected), label + ":length")
        if isinstance(actual, list):
            for i, (a, b) in enumerate(zip(actual, expected)): recursive_equal(checks, a, b, label + ":" + str(i))
    elif isinstance(expected, bool) or isinstance(expected, str) or expected is None: checks.check(actual == expected and (not isinstance(expected, bool) or type(actual) is bool), label)
    elif isinstance(expected, (int, float)): checks.close(actual, expected, label)
    else: checks.check(actual == expected, label)


def _take(top, name, pins):
    path = safe(top, name); pins[name] = sha(path); return read(path)


def _bind_files(top, files, checks, pins, label):
    for name, expected in files.items():
        actual = sha(safe(top, name)); checks.check(actual == expected, label + ":" + name)
        checks.check(name not in pins or pins[name] == actual, "consistent input binding:" + name); pins[name] = actual


def _audit_revision(top, plan, checks, pins):
    revision = _take(top, "audit_revision.json", pins)
    receipt = _take(top, "audit_revision_receipt.json", pins)
    checks.check(receipt.get("sha256") == pins["audit_revision.json"], "versioned audit revision receipt")
    checks.check(pd.Timestamp(receipt["at_utc"]) >= pd.Timestamp(revision["created_at_utc"]), "versioned audit revision receipt chronology")
    expected = {"schema_version": 1, "revision_id": REVISION_ID, "reason_code": REVISION_REASON,
        "original_protocol_sha256": pins["protocol.json"], "search_plan_sha256": PLAN_SHA256,
        "original_extension_manifest_sha256": pins["extension_manifest.json"], "original_audit_source_sha256": ORIGINAL_AUDITOR_SHA256,
        "revised_audit_source_path": REVISION_AUDIT_PATH, "revised_audit_source_sha256": PARENT_AUDITOR_SHA256,
        "controller_source_path": REVISION_CONTROLLER_PATH,
        "financial_oracle_source_sha256": plan["frozen_core_sources"]["scripts/audit_single_pair_rl.py"],
        "permitted_metadata_normalization": REVISION_NORMALIZATION, "financial_math_changed": False,
        "model_recipe_changed": False, "training_restarted": False}
    for key, value in expected.items():
        checks.check(revision.get(key) == value and (not isinstance(value, bool) or type(revision.get(key)) is bool), "explicit audit-only revision:" + key)
    _bind_files(top, {REVISION_AUDIT_PATH: revision["revised_audit_source_sha256"], REVISION_CONTROLLER_PATH: revision["controller_source_sha256"]}, checks, pins, "versioned audit/controller source")
    _take(top, "pipeline_launch.json", pins)
    checks.check(revision.get("original_pipeline_launch_sha256") == pins["pipeline_launch.json"], "revision retains original training launch")
    checks.check(pd.Timestamp(revision["created_at_utc"]) >= pd.Timestamp(read(safe(top, "protocol.json"))["frozen_at"]), "audit metadata correction follows original protocol freeze")
    return revision


def _audit_import_revision(top, checks, pins):
    revision = _take(top, "audit_import_revision.json", pins)
    receipt = _take(top, "audit_import_revision_receipt.json", pins)
    failed = _take(top, "final/validation_optimization_audit.json", pins)
    selected = _take(top, "selection.json", pins)
    selected_audit = _take(top, "selection_audit.json", pins)
    expected = {"schema_version": 1, "revision_id": IMPORT_REVISION_ID, "reason_code": IMPORT_REVISION_REASON,
        "original_protocol_sha256": pins["protocol.json"], "search_plan_sha256": PLAN_SHA256,
        "parent_audit_revision_sha256": pins["audit_revision.json"], "parent_auditor_source_sha256": PARENT_AUDITOR_SHA256,
        "revised_audit_source_path": IMPORT_AUDIT_PATH, "revised_audit_source_sha256": sha(__file__),
        "finisher_source_path": IMPORT_FINISHER_PATH,
        "failed_final_validation_audit_sha256": pins["final/validation_optimization_audit.json"],
        "selection_sha256": pins["selection.json"], "selection_audit_sha256": pins["selection_audit.json"],
        "permitted_change": IMPORT_PERMITTED_CHANGE, "financial_math_changed": False, "model_recipe_changed": False,
        "selection_changed": False, "training_restarted": False}
    for key, value in expected.items():
        checks.check(revision.get(key) == value and (not isinstance(value, bool) or type(revision.get(key)) is bool), "explicit import-audit-only revision:" + key)
    checks.check(receipt.get("sha256") == pins["audit_import_revision.json"] and pd.Timestamp(receipt["at_utc"]) >= pd.Timestamp(revision["created_at_utc"]), "import audit revision receipt and chronology")
    checks.check(failed.get("passed") is False and failed.get("failure_count", 0) > 0 and failed.get("phase") == "final-validation"
        and failed.get("audit_source_sha256") == PARENT_AUDITOR_SHA256 and failed.get("core_financial_audit_passed") is False,
        "preserved failed parent import gate, not a financial failure")
    checks.check(pd.Timestamp(revision["created_at_utc"]) >= pd.Timestamp(failed["audited_at"]) >= pd.Timestamp(selected["at"]), "import correction follows preserved failure and frozen selection")
    checks.check(selected_audit.get("passed") is True and selected_audit.get("failure_count") == 0
        and selected_audit.get("audit_source_sha256") == PARENT_AUDITOR_SHA256 and selected_audit.get("selection_sha256") == pins["selection.json"], "unchanged passed parent selection")
    checks.check(set(selected.get("selection", {})) == set(SYMBOLS) and all(selected["selection"][s].get("candidate_id") == "c00_baseline"
        and selected["selection"][s].get("fallback_to_baseline") is True for s in SYMBOLS), "final branch only reuses both frozen c00 fallbacks")
    _bind_files(top, {IMPORT_AUDIT_PATH: revision["revised_audit_source_sha256"], IMPORT_FINISHER_PATH: revision["finisher_source_sha256"]}, checks, pins, "versioned import auditor/finisher source")
    return revision


def native_policy_kwargs_match(actual, declared):
    """SB3 SAC serializes its unchanged default use_sde=False into policy kwargs."""
    if not isinstance(actual, dict) or declared != {"net_arch": [128, 128]}: return False
    if set(actual) not in ({"net_arch"}, {"net_arch", "use_sde"}): return False
    return actual.get("net_arch") == declared["net_arch"] and ("use_sde" not in actual or actual["use_sde"] is False)


def _base_contract(top, checks, pins):
    plan = load_plan(top); _take(top, "search_plan.json", pins); _take(top, "search_plan_receipt.json", pins)
    spec = _take(top, "protocol.json", pins); receipt = _take(top, "protocol_receipt.json", pins)
    checks.check(receipt.get("protocol_sha256") == pins["protocol.json"], "top protocol receipt")
    for key in ("symbols", "capital_per_account", "ensemble", "train", "validation", "test", "costs", "environment", "dataset_sha256", "runtime_packages"):
        checks.check(spec.get(key) == plan[key], "unchanged top financial/runtime contract:" + key)
    checks.check(spec.get("sources") == plan["frozen_core_sources"], "exact original11 financial sources")
    checks.check(spec.get("tuning_seeds") == list(TUNING_SEEDS) and spec.get("final_seeds") == list(FINAL_SEEDS), "disjoint tuning and final seed populations")
    checks.check(spec.get("candidate_order") == list(CANDIDATE_ORDER) and spec.get("candidate_recipes") == {x["candidate_id"]: x["native_recipe"] for x in plan["candidates"]}, "complete fixed2x2 candidate grid")
    checks.check(spec.get("search_plan_sha256") == PLAN_SHA256, "protocol binds predeclared plan")
    checks.check(spec.get("steps_per_model") == 32768 and spec.get("algorithm") == "sac", "fixed native SAC budget and family")
    checks.check(spec.get("capital_shared_between_pairs") is False and spec.get("primary_seed") == 0 and spec.get("primary_seed_fixed_before_training") is True,
        "independent accounts and predeclared primary seed")
    checks.check(spec.get("base_recipe") == plan["candidates"][0]["native_recipe"], "unchanged original default SAC recipe")
    checks.check(spec.get("selection_contract") == {k: plan[k] for k in ("score", "acceptance", "subwindows", "subwindow_metric", "tie_break", "fallback")}, "exact original predeclared selection contract")
    extension = _take(top, "extension_manifest.json", pins)
    checks.check(extension.get("protocol_sha256") == pins["protocol.json"] and extension.get("search_plan_sha256") == PLAN_SHA256, "extension sealed to exact protocol and original plan")
    checks.check(set(extension.get("extension_sources", {})) == set(EXTENSION_FILES), "full independent optimizer extension sources")
    for name, expected in plan["frozen_core_sources"].items():
        relative = "frozen_source/" + name; pins[relative] = sha(safe(top, relative)); checks.check(pins[relative] == expected, "original frozen source:" + name)
    for name, expected in extension.get("extension_sources", {}).items():
        relative = "extension_source/" + name; pins[relative] = sha(safe(top, relative)); checks.check(pins[relative] == expected, "fixed optimization extension:" + name)
    checks.check(extension.get("extension_sources", {}).get(EXTENSION_FILES[2]) == ORIGINAL_AUDITOR_SHA256, "preserved original independent optimization auditor")
    _audit_revision(top, plan, checks, pins)
    _audit_import_revision(top, checks, pins)
    references = _take(top, "reference_manifest.json", pins); reference_receipt = _take(top, "reference_receipt.json", pins)
    checks.check(reference_receipt.get("reference_manifest_sha256") == pins["reference_manifest.json"], "original reference import receipt")
    checks.check(references.get("sources") == spec.get("reference_sources") and set(spec.get("reference_sources", {})) == set(SYMBOLS), "two separate original reference studies")
    checks.check(not any("test" in PurePosixPath(n).parts or PurePosixPath(n).name == "test_audit.json" for n in references.get("files_sha256", {})), "prepare reference manifest excludes all annual values")
    _bind_files(top, references.get("files_sha256", {}), checks, pins, "original preselection reference")
    for symbol in SYMBOLS:
        old_name = f"references/{symbol}/protocol.json"; old = _take(top, old_name, pins)
        checks.check(pins[old_name] == spec["reference_sources"][symbol]["protocol_sha256"], "original reference protocol:" + symbol)
        for key in ("train", "validation", "test", "capital_per_account", "ensemble", "costs", "environment", "dataset_sha256", "sources", "runtime_packages"):
            checks.check(old.get(key) == spec.get(key), "same reference financial semantics:" + symbol + ":" + key)
        checks.check(old.get("symbols") == [symbol] and old.get("seeds") == list(FINAL_SEEDS) and old.get("recipes", {}).get("sac") == plan["candidates"][0]["native_recipe"], "real original thirty-seed baseline:" + symbol)
    return plan, spec, extension


def _selection_contract(top, plan, spec, checks, pins, *, recompute=True):
    selection = _take(top, "selection.json", pins); receipt = _take(top, "selection_receipt.json", pins)
    checks.check(receipt.get("sha256") == pins["selection.json"], "immutable per-coin selection receipt")
    checks.check(selection.get("protocol_sha256") == pins["protocol.json"] and selection.get("uses_test") is False
        and selection.get("uses_final_validation") is False and selection.get("seed_selection") is False and selection.get("algorithm_selection") is False,
        "selection uses only tuning validation, no test/final/seed/algorithm selection")
    checks.check(selection.get("tuning_seeds") == list(TUNING_SEEDS) and selection.get("final_seeds") == list(FINAL_SEEDS), "selection seed partition")
    checks.check(selection.get("search_plan_sha256") == PLAN_SHA256, "selection bound to original frozen search plan")
    checks.check(set(selection.get("selection", {})) == set(SYMBOLS), "each coin independently selected")
    metrics = {}
    for cid in CANDIDATE_ORDER:
        prefix = "tuning/" + cid + "/"
        financial = _take(top, prefix + "validation_audit.json", pins); contract = _take(top, prefix + "validation_optimization_audit.json", pins)
        checks.check(financial.get("passed") is True and financial.get("failure_count") == 0 and financial.get("accounts_checked") == financial.get("repeats_checked") == 36, "complete candidate financial audit:" + cid)
        checks.check(contract.get("passed") is True and contract.get("failure_count") == 0, "complete candidate contract audit:" + cid)
        checks.check(contract.get("protocol_sha256") == pins["protocol.json"] and contract.get("audit_source_sha256") == PARENT_AUDITOR_SHA256, "candidate audit exact source/protocol:" + cid)
        _bind_files(top, contract.get("inputs_sha256", {}), checks, pins, "selection immutable validation inputs")
        _bind_files(top, {prefix + "validation_summary.csv": sha(safe(top, prefix + "validation_summary.csv"))}, checks, pins, "selection summary")
        checks.check(datetime.fromisoformat(selection["at"]) >= datetime.fromisoformat(contract["audited_at"]) >= datetime.fromisoformat(financial["audited_at"]), "candidate audits before selection:" + cid)
        evidence = selection.get("validation_evidence_sha256", {}).get(cid, {})
        checks.check(evidence == {"summary": sha(safe(top, prefix + "validation_summary.csv")), "financial_audit": pins[prefix + "validation_audit.json"], "contract_audit": pins[prefix + "validation_optimization_audit.json"]}, "selection binds complete validation evidence:" + cid)
        if recompute: metrics[cid] = compute_candidate_metrics(Path(top) / "tuning" / cid, plan)
    if recompute:
        expected = compute_expected_selection(metrics, plan)
        for symbol in SYMBOLS:
            actual = selection["selection"].get(symbol, {})
            recursive_equal(checks, {k: actual.get(k) for k in expected[symbol]}, expected[symbol], "independent per-coin score/gates/fallback:" + symbol)
    return selection


def _directory_hashes(top, relative):
    directory = unlinked(Path(top) / relative)
    directory.relative_to(unlinked(top))
    return {p.relative_to(directory).as_posix(): sha(safe(top, p.relative_to(top).as_posix())) for p in directory.rglob("*") if p.is_file()}


def _reference_external_inputs(top, checks, *, annual=False):
    """Bind copied evidence to its immutable original, without opening annual values early."""
    spec = read(safe(top, "protocol.json")); mappings = read(safe(top, "reference_manifest.json"))["files_sha256"]
    if annual: mappings = {**mappings, **read(safe(top, "reference_annual_import.json"))["files_sha256"]}
    external = {}
    for name, expected in mappings.items():
        parts = PurePosixPath(name).parts
        if len(parts) < 3 or parts[0] != "references" or parts[1] not in SYMBOLS: raise ValueError("Canonical original reference file required")
        source = safe(spec["reference_sources"][parts[1]]["source_root"], PurePosixPath(*parts[2:]).as_posix())
        checks.check(sha(source) == expected, "actual unchanged original reference:" + name)
        external[str(source)] = expected
    return external


def _import_checks(top, name, selection_sha, checks, pins):
    manifest = _take(top, name, pins)
    checks.check(manifest.get("selection_sha256") == selection_sha, "import follows immutable selection:" + name)
    selection = read(safe(top, "selection.json"))
    checks.check(datetime.fromisoformat(manifest["at"]) >= datetime.fromisoformat(selection["at"]), "import time after selection:" + name)
    fallback = {s for s in SYMBOLS if selection["selection"][s]["candidate_id"] == "c00_baseline"}
    expected = {}
    if name == "final/import_manifest.json":
        for symbol in fallback:
            for seed in FINAL_SEEDS:
                tail = f"training/{symbol}/sac/seed{seed}"
                expected["final/" + tail] = ("references/" + symbol + "/" + tail, "retained_baseline_fallback")
    else:
        split = "test" if name == "final/test_import_manifest.json" else "validation"
        for symbol in SYMBOLS:
            identities = [(a, None) for a in BASELINES] + ([("sac", n) for n in FINAL_SEEDS] if symbol in fallback else [])
            for algorithm, seed in identities:
                for cost in COSTS:
                    for kind in ("evaluation", "repeat"):
                        tail = f"{kind}/{split}/{symbol}/{algorithm}/{'baseline' if seed is None else 'seed'+str(seed)}/{cost}"
                        expected["final/" + tail] = ("references/" + symbol + "/" + tail, "retained_fixed_baseline" if seed is None else "retained_baseline_fallback")
    entries = manifest.get("entries", [])
    checks.check(len(entries) == len(expected) and {r["destination_directory"] for r in entries} == set(expected), "exact required fallback/fixed import population:" + name)
    for entry in manifest.get("entries", []):
        source, destination = entry["source_directory"], entry["destination_directory"]
        checks.check(expected.get(destination) == (source, entry.get("origin")), "same coin/seed/cost exact import identity:" + destination)
        source_files, copied_files = _directory_hashes(top, source), _directory_hashes(top, destination)
        checks.check(source_files == copied_files == entry.get("file_sha256"), "byte-identical complete imported directory:" + destination)
        for filename, expected_digest in entry.get("file_sha256", {}).items():
            _bind_files(top, {source + "/" + filename: expected_digest, destination + "/" + filename: expected_digest}, checks, pins, "immutable baseline import")
        checks.check(entry.get("origin") in ("retained_baseline_fallback", "retained_fixed_baseline"), "honest import origin:" + destination)
    return manifest


def _training_checks(top, child, plan, selected_recipes, checks, pins, *, candidate_id=None):
    prefix = child.relative_to(top).as_posix() + "/"; seeds = TUNING_SEEDS if candidate_id else FINAL_SEEDS
    count = len(SYMBOLS) * len(seeds); expected = {(s, n) for s in SYMBOLS for n in seeds}
    paths = {p.relative_to(top).as_posix() for p in (child / "training").rglob("record.json")}
    checks.check(paths == {prefix + f"training/{s}/sac/seed{n}/record.json" for s, n in expected}, "exact declared trained population:" + prefix)
    fit = _take(top, prefix + "fit_complete.json", pins); counter = _take(top, prefix + "training_counter_audit.json", pins)
    checks.check(fit.get("model_count") == count and len(fit.get("records", [])) == count, "complete fit:" + prefix)
    checks.check(counter.get("passed") is True and counter.get("failure_count") == 0 and counter.get("models_checked") == count, "complete native counter audit:" + prefix)
    checks.check(counter.get("core_protocol_sha256") == pins[prefix + "protocol.json"] and counter.get("parent_protocol_sha256") == pins["protocol.json"]
        and counter.get("extension_source_sha256") == read(safe(top, "extension_manifest.json"))["extension_sources"][EXTENSION_FILES[0]], "native counter source/protocol binding:" + prefix)
    rows = {(r["symbol"], r["seed"]): r for r in counter.get("models", [])}
    checks.check(len(counter.get("models", [])) == count and set(rows) == expected, "native counter identities:" + prefix)
    imported = {}
    if not candidate_id:
        manifest = _import_checks(top, "final/import_manifest.json", pins["selection.json"], checks, pins)
        imported = {(r["symbol"], r["seed"]): r for r in manifest.get("entries", [])}
    records = []
    for symbol, seed in sorted(expected):
        name = prefix + f"training/{symbol}/sac/seed{seed}/record.json"; record = _take(top, name, pins); records.append(record)
        recipe = selected_recipes[symbol]; checkpoint = record["checkpoint"]
        checks.check(record.get("symbol") == symbol and record.get("algorithm") == "sac" and type(record.get("seed")) is int and record["seed"] == seed
            and record.get("capital") == 10000. and record.get("ensemble") is False and record.get("observation_shape") == [13] and record.get("action_shape") == [1], "independent native training identity:" + name)
        checks.check(record["training"]["actual_steps"] == 32768 and pd.Timestamp(record["maximum_input_hour"]) < pd.Timestamp(BOUNDS["train"][1]), "training budget/cutoff:" + name)
        c = record["training"]["counters"]
        checks.check(c.get("environment_steps") == c.get("replay_add_calls") == c.get("replay_transitions") == 32768
            and c.get("gradient_update_rounds") == 32256 and c.get("optimizer_steps") == {"actor": 32256, "critic": 32256, "entropy": 32256}, "same native optimizer and replay budget:" + name)
        checks.check(rows[(symbol, seed)].get("record_sha256") == pins[name] and rows[(symbol, seed)].get("counters") == c, "training audit bound actual record:" + name)
        for filename, expected_sha in record.get("files", {}).items():
            relative = name.rsplit("/", 1)[0] + "/" + filename; _bind_files(top, {relative: expected_sha}, checks, pins, "actual checkpoint")
        checks.check(checkpoint.get("recipe") == recipe and checkpoint.get("training") == record["training"], "actual recipe and training checkpoint manifest:" + name)
        directory = name.rsplit("/", 1)[0]
        expected_files = {"checkpoint/model.zip", "checkpoint/single_pair_model.json"}
        checks.check(set(record.get("files", {})) == expected_files and read(safe(top, directory + "/checkpoint/single_pair_model.json")) == checkpoint,
            "complete actual checkpoint manifest matches record:" + name)
        checks.check(checkpoint.get("symbol") == symbol and checkpoint.get("algorithm") == "sac" and checkpoint.get("seed") == seed
            and checkpoint.get("observation_shape") == [13] and checkpoint.get("action_shape") == [1]
            and checkpoint.get("independent_account") is True and checkpoint.get("multi_model_target_average") is False, "actual native checkpoint identity:" + name)
        checks.check(checkpoint.get("checkpoint_files") == {"model.zip": record["files"].get("checkpoint/model.zip")}, "native manifest binds model zip:" + name)
        if (symbol, seed) in imported:
            checks.check(recipe == plan["candidates"][0]["native_recipe"] and imported[(symbol, seed)].get("origin") == "retained_baseline_fallback", "c00 mandatory retained model reuse:" + name)
            checks.check(checkpoint.get("schema_version") == 1 and checkpoint.get("adapter_sha256") == plan["frozen_core_sources"]["scripts/single_pair_rl_models.py"], "unchanged original fallback adapter:" + name)
        else:
            checks.check(checkpoint.get("nonfinancial_preflight") is False, "formal training is not smoke evidence:" + name)
            checks.check(checkpoint.get("schema_version") == 2 and checkpoint.get("base_adapter_sha256") == plan["frozen_core_sources"]["scripts/single_pair_rl_models.py"], "new explicit recipe adapter provenance:" + name)
            extension = read(safe(top, "extension_manifest.json"))
            checks.check(checkpoint.get("adapter_sha256") == extension["extension_sources"][EXTENSION_FILES[1]], "actual optimized adapter source:" + name)
            expected_cid = candidate_id or read(safe(top, "selection.json"))["selection"][symbol]["candidate_id"]
            checks.check(expected_cid != "c00_baseline" or candidate_id is not None, "never wastefully retrain final fallback:" + name)
            context = checkpoint.get("model_context", {})
            checks.check(context.get("stage") == ("tuning" if candidate_id else "final") and context.get("symbol") == symbol
                and context.get("candidate_id") == expected_cid and context.get("seed") == seed and context.get("parent_protocol_sha256") == pins["protocol.json"], "predeclared trained candidate context:" + name)
            if not candidate_id:
                checks.check(datetime.fromisoformat(record["started_at"]) >= datetime.fromisoformat(read(safe(top, "selection.json"))["at"]), "new final fit after selection:" + name)
        data_path = safe(top, name.rsplit("/", 1)[0] + "/checkpoint/model.zip")
        with zipfile.ZipFile(data_path) as archive: actual = json.loads(archive.read("data"))
        for key in ("learning_rate", "gamma", "buffer_size", "learning_starts", "batch_size", "tau", "gradient_steps", "ent_coef"):
            checks.check(actual.get(key) == recipe[key], "actual native zip hyperparameter:" + name + ":" + key)
        checks.check(native_policy_kwargs_match(actual.get("policy_kwargs"), recipe["policy_kwargs"]) and actual.get("use_sde") is False,
            "same native policy architecture and boolean-false SDE default:" + name)
        effective = record["training"].get("effective_parameters", {})
        checks.check(effective.get("native_class") == "SAC" and effective.get("train_freq") == [1, "step"]
            and effective.get("observation_shape") == [13] and effective.get("action_shape") == [1], "actual native SAC spaces and update frequency:" + name)
        for key in ("learning_rate", "gamma", "buffer_size", "learning_starts", "batch_size", "tau", "gradient_steps", "ent_coef"):
            checks.check(effective.get(key) == recipe[key], "observed native effective hyperparameter:" + name + ":" + key)
        completed = [r for r in record["training"]["training_episode_history"] if r["completed_steps"] > 0]
        checks.check(len(completed) == 1024 and sum(r["completed_steps"] for r in completed) == 32768, "same complete32-day fragment budget:" + name)
        for episode in completed:
            start, end = datetime.fromisoformat(episode["start"]), datetime.fromisoformat(episode["end_exclusive"])
            checks.check(episode["completed_steps"] == 32 and episode["terminated"] is False and episode["truncated"] is True
                and datetime.fromisoformat(episode["completed_until"]) == end == start + timedelta(days=32)
                and pd.Timestamp(BOUNDS["train"][0]) <= pd.Timestamp(start) < pd.Timestamp(end) <= pd.Timestamp(BOUNDS["train"][1]), "train-only timeout fragment:" + name)
    checks.check(sorted(fit["records"], key=lambda r: (r["symbol"], r["seed"])) == sorted(records, key=lambda r: (r["symbol"], r["seed"])), "fit receipt exact records:" + prefix)


def _case_population(top, child, split, seeds, checks, pins):
    prefix = child.relative_to(top).as_posix() + "/"
    identities = [(s, a, n, c) for s in SYMBOLS for a in ("sac",) for n in seeds for c in COSTS]
    identities += [(s, a, None, c) for s in SYMBOLS for a in BASELINES for c in COSTS]
    wanted = set()
    for kind in ("evaluation", "repeat"):
        expected_meta = set()
        for symbol, algorithm, seed, cost in identities:
            directory = f"{kind}/{split}/{symbol}/{algorithm}/{'baseline' if seed is None else 'seed'+str(seed)}/{cost}"
            name = prefix + directory + "/metadata.json"; meta = _take(top, name, pins); expected_meta.add(name)
            features = [symbol + ":" + field for field in MARKET_FEATURES] + ["coin_weight", "cash_weight", "current_drawdown"]
            checks.check(meta.get("observation_features") == features and meta.get("pair") == symbol and meta.get("tradable_symbols") == [symbol]
                and set(meta.get("final_positions", {})) <= {symbol}, "exact own-coin feature/inventory dependencies:" + name)
            checks.check(meta.get("observation_dim") == 13 and meta.get("action_dim") == 1 and meta.get("no_ensemble") is True
                and meta.get("independent_capital") == 10000. and meta.get("bnb_fee_discount") is False, "same cash-funded spot dimensions:" + name)
            if kind == "evaluation": wanted.add(directory)
        actual = {p.relative_to(top).as_posix() for p in (child / kind / split).rglob("metadata.json")}
        checks.check(actual == expected_meta, "complete unique case metadata population:" + prefix + kind + "/" + split)
    return wanted


def _financial_bindings(top, child, split, wanted, checks, pins):
    prefix = child.relative_to(top).as_posix() + "/"; name = prefix + split + "_audit.json"; audit = _take(top, name, pins)
    checks.check(audit.get("passed") is True and audit.get("failure_count") == 0 and not audit.get("failures")
        and audit.get("accounts_checked") == audit.get("repeats_checked") == len(wanted), "complete independent financial oracle:" + name)
    checks.check(audit.get("protocol_sha256") == sha(safe(top, prefix + "protocol.json"))
        and audit.get("summary_sha256") == sha(safe(top, prefix + split + "_summary.csv")), "financial oracle current protocol/summary binding:" + name)
    checks.check(len(audit.get("cases", [])) == len(wanted) and {c["relative_path"] for c in audit.get("cases", [])} == wanted, "financial oracle exact case population:" + name)
    _bind_files(top, {prefix + n: h for n, h in audit.get("inputs_sha256", {}).items()}, checks, pins, "current financial phase input")
    for case in audit.get("cases", []):
        checks.check(case.get("passed") is True and case.get("failure_count") == 0 and not case.get("failures"), "each actual case passed:" + case["relative_path"])
        directory = case["relative_path"]
        expected = {directory + "/" + n for n in RESULT_FILES} | {directory.replace("evaluation/", "repeat/", 1) + "/" + n for n in RESULT_FILES}
        checks.check(set(case.get("inputs_sha256", {})) == expected, "all20 original/repeat case files bound:" + directory)
        _bind_files(top, {prefix + n: h for n, h in case.get("inputs_sha256", {}).items()}, checks, pins, "current financial case input")
    return audit


def _child_contract(top, child, plan, spec, checks, pins, candidate):
    prefix = child.relative_to(top).as_posix() + "/"; child_spec = _take(top, prefix + "protocol.json", pins)
    child_receipt = _take(top, prefix + "protocol_receipt.json", pins)
    checks.check(child_receipt.get("protocol_sha256") == pins[prefix + "protocol.json"], "child immutable protocol:" + prefix)
    checks.check(child_spec.get("parent_protocol_sha256") == pins["protocol.json"], "child binds parent protocol:" + prefix)
    seeds = TUNING_SEEDS if candidate else FINAL_SEEDS
    checks.check(child_spec.get("symbols") == list(SYMBOLS) and child_spec.get("algorithms") == ["sac"] and child_spec.get("seeds") == list(seeds)
        and child_spec.get("training_model_count") == len(SYMBOLS) * len(seeds), "exact native stage population:" + prefix)
    for key in ("train", "validation", "test", "capital_per_account", "ensemble", "costs", "environment", "dataset_sha256", "sources", "runtime_packages"):
        checks.check(child_spec.get(key) == spec.get(key), "unchanged child financial context:" + prefix + key)
    for key, expected in {"capital_shared_between_pairs": False, "primary_seed": 0, "primary_seed_fixed_before_training": True,
                          "fresh_holdout": False, "test_year_already_observed": True, "baselines": list(BASELINES)}.items():
        checks.check(child_spec.get(key) == expected, "same original financial declaration:" + prefix + key)
    if candidate:
        checks.check(candidate in CANDIDATE_ORDER and child_spec.get("stage") == "tuning" and child_spec.get("candidate_id") == candidate, "fixed tuning context:" + prefix)
        recipes = {s: next(r["native_recipe"] for r in plan["candidates"] if r["candidate_id"] == candidate) for s in SYMBOLS}
        checks.check(not (child / "evaluation/test").exists() and not (child / "repeat/test").exists() and not (child / "test_started.json").exists(), "tuning never opens annual cases:" + prefix)
    else:
        selection = _selection_contract(top, plan, spec, checks, pins)
        recipes = {s: selection["selection"][s]["selected_recipe"] for s in SYMBOLS}
        checks.check(child_spec.get("stage") == "final" and child_spec.get("selection_sha256") == pins["selection.json"], "final context follows fixed selection")
        checks.check(child_spec.get("recipe_ids_by_symbol") == {s: selection["selection"][s]["candidate_id"] for s in SYMBOLS}, "each final coin uses exactly its selected candidate")
    checks.check(child_spec.get("recipes_by_symbol") == recipes, "honest actual recipes per independent coin:" + prefix)
    for name, expected in plan["frozen_core_sources"].items():
        _bind_files(top, {prefix + "frozen_source/" + name: expected}, checks, pins, "same child financial source")
    return child_spec, recipes, seeds


def _annual_gate(top, checks, pins):
    validation = _take(top, FINAL_VALIDATION_AUDIT, pins)
    selection_audit = _take(top, "selection_audit.json", pins)
    for label, value in (("final validation", validation), ("selection", selection_audit)):
        checks.check(value.get("passed") is True and value.get("failure_count") == 0 and not value.get("failures"), label + " independent contract passed before annual")
        _bind_files(top, value.get("inputs_sha256", {}), checks, pins, "annual gate still-current " + label)
    checks.check(validation.get("audit_source_sha256") == sha(__file__) and selection_audit.get("audit_source_sha256") == PARENT_AUDITOR_SHA256, "explicit parent-selection/final-v3 audit provenance")
    started = _take(top, "final/test_started.json", pins)
    checks.check(datetime.fromisoformat(started["at"]) >= datetime.fromisoformat(validation["audited_at"]), "annual starts after final validation audits")
    expected = {"selection_sha256": pins["selection.json"], "selection_audit_sha256": pins["selection_audit.json"],
        "validation_audit_sha256": sha(safe(top, "final/validation_audit.json")), "validation_optimization_audit_sha256": pins[FINAL_VALIDATION_AUDIT],
        "protocol_sha256": pins["protocol.json"], "final_protocol_sha256": pins["final/protocol.json"]}
    checks.check(started.get("pins") == expected, "annual gate exact frozen selection/validation bindings")
    checks.check(started.get("validation_optimization_audit_path") == FINAL_VALIDATION_AUDIT and started.get("active_auditor_source_sha256") == sha(__file__)
        and started.get("audit_import_revision_sha256") == pins["audit_import_revision.json"], "annual gate explicit retained v3 path and revision")
    compatible = _take(top, "final/validation_selection.json", pins)
    checks.check(started.get("oracle_compatibility_selection_sha256") == pins["final/validation_selection.json"], "annual gate binds original-oracle compatibility receipt")
    checks.check(compatible.get("uses_test") is False and compatible.get("seed_selection") is False and compatible.get("algorithm_selection") is False
        and compatible.get("mode") == "fixed_sac_recipe_comparison" and compatible.get("selection") == {s: {"algorithm": "sac", "seed": 0} for s in SYMBOLS}, "compatibility receipt never reselects parameters or seeds")
    checks.check({key: compatible.get(key) for key in ("top_selection_sha256", "protocol_sha256", "validation_audit_sha256", "validation_optimization_audit_sha256")} == {
        "top_selection_sha256": expected["selection_sha256"], "protocol_sha256": expected["final_protocol_sha256"], "validation_audit_sha256": expected["validation_audit_sha256"],
        "validation_optimization_audit_sha256": expected["validation_optimization_audit_sha256"]}, "compatibility receipt bound to frozen final inputs")
    checks.check(compatible.get("validation_optimization_audit_path") == FINAL_VALIDATION_AUDIT and compatible.get("active_auditor_source_sha256") == sha(__file__)
        and compatible.get("audit_import_revision_sha256") == pins["audit_import_revision.json"], "compatibility receipt explicit v3 path and revision")
    checks.check(datetime.fromisoformat(validation["audited_at"]) <= datetime.fromisoformat(compatible["at"]) <= datetime.fromisoformat(started["at"]), "compatibility receipt only follows passed final validation")


def audit_phase(study_root, dataset, phase, *, candidate=None, workers=None):
    top = unlinked(study_root); checks, pins, external = Checks(), {}, {}; financial = None; started = now()
    if phase not in ("final-validation", "final-test"): raise ValueError("The v3 branch audits final retained accounts only")
    if phase == "tuning-validation" and candidate not in CANDIDATE_ORDER: raise ValueError("One predeclared tuning candidate required")
    child = top / "tuning" / candidate if phase == "tuning-validation" else top / "final"
    split = "test" if phase == "final-test" else "validation"
    relative = FINAL_TEST_AUDIT if split == "test" else FINAL_VALIDATION_AUDIT
    if (top / relative).exists(): raise FileExistsError("Preserve the existing independent optimization audit")
    if phase != "selection" and (child / (split + "_audit.json")).exists(): raise FileExistsError("Preserve the original financial audit")
    try:
        plan, spec, extension = _base_contract(top, checks, pins)
        external.update(_reference_external_inputs(top, checks, annual=phase == "final-test"))
        dataset = unlinked(dataset); external[str(dataset)] = sha(dataset); external[str(dataset.parent / "manifest.json")] = sha(dataset.parent / "manifest.json")
        checks.check(external[str(dataset)] == plan["dataset_sha256"] and external[str(dataset.parent / "manifest.json")] == spec["dataset_manifest_sha256"], "same immutable market source")
        if phase == "selection":
            checks.check(not (top / "final/test_started.json").exists() and not (top / "reference_annual_import.json").exists(), "freeze selection before any annual import/start")
            _selection_contract(top, plan, spec, checks, pins)
        else:
            child_spec, recipes, seeds = _child_contract(top, child, plan, spec, checks, pins, candidate if phase == "tuning-validation" else None)
            _training_checks(top, child, plan, recipes, checks, pins, candidate_id=candidate if phase == "tuning-validation" else None)
            if phase == "final-test":
                _annual_gate(top, checks, pins)
                _import_checks(top, "final/test_import_manifest.json", pins["selection.json"], checks, pins)
            elif phase == "final-validation":
                _import_checks(top, "final/validation_import_manifest.json", pins["selection.json"], checks, pins)
            wanted = _case_population(top, child, split, seeds, checks, pins)
            if checks.failures: raise ValueError("Independent optimization provenance/recipe/leakage gate failed")
            source = top / "frozen_source"
            if str(source) not in sys.path: sys.path.insert(0, str(source))
            oracle = importlib.import_module("scripts.audit_single_pair_rl")
            if Path(oracle.__file__).resolve() != safe(source, "scripts/audit_single_pair_rl.py").resolve(): raise ValueError("Use a clean process with exact frozen independent financial oracle")
            oracle.SYMBOLS, oracle.SEEDS = SYMBOLS, tuple(seeds)
            financial = oracle.audit_phase(child, dataset, split, workers=workers)
            _financial_bindings(top, child, split, wanted, checks, pins)
        for name, expected in pins.items(): checks.check(sha(safe(top, name)) == expected, "audit inputs remained unchanged:" + name)
        for name, expected in external.items(): checks.check(sha(name) == expected, "external source remained unchanged:" + name)
    except Exception as exc:
        checks.check(False, type(exc).__name__ + ": " + str(exc))
    result = {**checks.result(), "phase": phase, "candidate_id": candidate, "started_at": started, "audited_at": now(),
        "protocol_sha256": pins.get("protocol.json"), "selection_sha256": pins.get("selection.json"), "inputs_sha256": pins,
        "external_inputs_sha256": external, "core_financial_audit_passed": financial is not None and financial.get("passed") is True,
        "core_financial_audit_sha256": pins.get(child.relative_to(top).as_posix() + "/" + split + "_audit.json") if phase != "selection" else None,
        "audit_source_sha256": sha(__file__), "no_environment_or_model_imports": True, "independent_math_unchanged": True}
    write_new(top / relative, result)
    return result


def verify_completed_study(study_root):
    """Read-only final contract for the collector; does not execute economic arithmetic."""
    top = unlinked(study_root); checks, pins, external = Checks(), {}, {}
    plan, spec, extension = _base_contract(top, checks, pins)
    external.update(_reference_external_inputs(top, checks, annual=True))
    selection = _selection_contract(top, plan, spec, checks, pins)
    names = [f"tuning/{cid}/validation_optimization_audit.json" for cid in CANDIDATE_ORDER]
    names += ["selection_audit.json", FINAL_VALIDATION_AUDIT, FINAL_TEST_AUDIT]
    for name in names:
        audit = _take(top, name, pins)
        checks.check(audit.get("passed") is True and audit.get("failure_count") == 0 and not audit.get("failures") and audit.get("checks_total", 0) > 0, "every completed independent stage passed:" + name)
        expected_source = sha(__file__) if name in (FINAL_VALIDATION_AUDIT, FINAL_TEST_AUDIT) else PARENT_AUDITOR_SHA256
        checks.check(audit.get("protocol_sha256") == pins["protocol.json"] and audit.get("audit_source_sha256") == expected_source, "stage binds exact protocol/auditor:" + name)
        _bind_files(top, audit.get("inputs_sha256", {}), checks, pins, "completed stage current inputs")
        for path, expected in audit.get("external_inputs_sha256", {}).items():
            checks.check(path not in external or external[path] == expected, "same external source across phases:" + path)
            checks.check(sha(path) == expected, "current external source:" + path); external[path] = expected
    for cid in CANDIDATE_ORDER:
        child = top / "tuning" / cid; wanted = _case_population(top, child, "validation", TUNING_SEEDS, checks, pins)
        _financial_bindings(top, child, "validation", wanted, checks, pins)
    final = top / "final"
    for split in ("validation", "test"):
        wanted = _case_population(top, final, split, FINAL_SEEDS, checks, pins); _financial_bindings(top, final, split, wanted, checks, pins)
    for name in ("final/fit_complete.json", "final/training_counter_audit.json", "reference_annual_import.json", "final/import_manifest.json", "final/validation_import_manifest.json", "final/test_import_manifest.json"):
        _take(top, name, pins)
    annual = read(safe(top, "reference_annual_import.json"))
    checks.check(annual.get("selection_sha256") == pins["selection.json"] and datetime.fromisoformat(annual["at"]) >= datetime.fromisoformat(selection["at"]), "annual reference import only after selection")
    _bind_files(top, annual.get("files_sha256", {}), checks, pins, "post-selection original annual comparison")
    checks.check(not any(name.endswith("pipeline_status.json") or name == "completion_contract_audit.json" for name in pins), "completion excludes mutable status and itself")
    return {**checks.result(), "protocol_sha256": pins["protocol.json"], "selection_sha256": pins["selection.json"],
        "inputs_sha256": pins, "external_inputs_sha256": external, "expected_tuning_models": 40, "final_models": 60,
        "audit_source_sha256": sha(__file__), "financial_arithmetic_rerun": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path); parser.add_argument("--dataset", type=Path)
    parser.add_argument("--phase", required=True, choices=("final-validation", "final-test", "completion"))
    parser.add_argument("--candidate", choices=CANDIDATE_ORDER); parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    if args.phase == "completion":
        target = args.study_root / "completion_contract_audit.json"
        if target.exists(): raise FileExistsError("Preserve original completion audit")
        result = verify_completed_study(args.study_root); write_new(target, result)
    else:
        if args.dataset is None: parser.error("Dataset is required for independent phase accounting")
        result = audit_phase(args.study_root, args.dataset, args.phase, candidate=args.candidate, workers=args.workers)
    print(json.dumps({k: result[k] for k in ("passed", "checks_total", "failure_count")}))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__": main()
