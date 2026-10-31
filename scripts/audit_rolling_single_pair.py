"""Read-only independent audits for the single-pair retraining-frequency study.

No runner, environment, policy, or learner is imported.  The unchanged original
single-pair arithmetic oracle checks execution economics; this module checks the
calendar, native training evidence, model identities, and continuous accounts.
Failed attempts remain separate evidence. A completion gate is written only on
an entirely successful, current-input audit.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import re
import zipfile

for _thread_option in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_option] = "1"

import numpy as np
import pandas as pd

UTC = timezone.utc
TEST = ("2025-10-23T00:00:00Z", "2026-10-23T00:00:00Z")
SYMBOLS = ("BTCUSDT", "ETHUSDT")
CADENCES = {"fixed": None, "every_4w": 28, "every_week": 7}
COSTS = {"base": {"fee": .001, "slippage": .0002}, "stress": {"fee": .002, "slippage": .0005}}
ALGORITHMS = {"sac": "SAC", "tqc": "TQC", "td3": "TD3"}
SOURCE_ORACLE = "scripts/audit_single_pair_rl.py"
ORACLE_SHA256 = "a980b357cfec5c0b7b734506cbcb5f95af411efdbc7c728da507a91da09291bb"
SNAPSHOT_FIELDS = {
    "cash", "positions", "cost_basis", "realized_pnl", "nav", "high_water",
    "fill_count", "fills_digest", "equity_rows", "weight_rows", "boundary_rows",
    "fees", "notional", "slippage_cost", "steps", "last_prices", "date",
}
HEX = re.compile(r"^[0-9a-f]{64}$")


class AuditError(ValueError):
    pass


def utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise AuditError("Explicit timezone-aware timestamps are required")
    return value.astimezone(UTC)


def stamp(value):
    return utc(value).isoformat().replace("+00:00", "Z")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def safe_path(root, relative):
    root = Path(root).resolve()
    relative = str(relative).replace("\\", "/")
    parts = relative.split("/")
    if not relative or ":" in relative or relative.startswith("/") or any(p in ("", ".", "..") for p in parts):
        raise AuditError("Audit paths must be safe, root-relative paths")
    path = root.joinpath(*parts)
    for entry in (path, *list(path.parents)[:len(parts) - 1]):
        if entry.is_symlink():
            raise AuditError("Symlink evidence is not accepted")
    path.resolve().relative_to(root)
    return path


def _require(condition, message):
    if not bool(condition):
        raise AuditError(message)


def _finite(value):
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite(v) for v in value)
    return not isinstance(value, (int, float)) or isinstance(value, bool) or math.isfinite(value)


def schedule_versions(start=TEST[0], end=TEST[1], cadence_days=7, *, train_days=1827,
                      lag_days=1, preheat_days=200):
    """Independent UTC calendar, with a common weekly version-number system."""
    start, end = utc(start), utc(end)
    _require(start < end and all(v.hour == v.minute == v.second == v.microsecond == 0 for v in (start, end)), "Require nonempty UTC whole-day boundaries")
    _require(cadence_days is None or type(cadence_days) is int and cadence_days in (7, 28), "Cadence must be fixed(None),7,28")
    _require((train_days, lag_days, preheat_days) == (1827, 1, 200), "Frozen 1827-day training,1-day information lag,200-day preheat required")
    days = [start] if cadence_days is None else [start + timedelta(days=i) for i in range(0, (end - start).days, cadence_days)]
    result = []
    for day in days:
        train_end = day - timedelta(days=lag_days)
        train_start = train_end - timedelta(days=train_days)
        result.append({"effective_day": stamp(day), "version_index": (day - start).days // 7,
                       "train_start": stamp(train_start), "train_end": stamp(train_end),
                       "preheat_start": stamp(train_start - timedelta(days=preheat_days))})
    return result


def model_context(version, *, symbol, algorithm, recipe_id, seed, protocol_sha256):
    _require(symbol in SYMBOLS and algorithm in ALGORITHMS, "Unknown pair or native learner")
    _require(type(seed) is int and seed in range(5), "Only the predeclared five integer seeds are allowed")
    _require(isinstance(recipe_id, str) and re.fullmatch(r"[a-z][a-z0-9_]*", recipe_id), "Unsafe recipe identifier")
    _require(isinstance(protocol_sha256, str) and HEX.fullmatch(protocol_sha256), "Protocol SHA256 required")
    return {key: utc(version[key]).isoformat() for key in ("effective_day", "train_start", "train_end")} | {
        "version_index": version["version_index"],
        "symbol": symbol, "algorithm": algorithm, "recipe_id": recipe_id, "seed": seed,
        "protocol_sha256": protocol_sha256}


def input_market_digest(data, symbol):
    """Hash only physical cropped own-coin OHLCV, never derived/cross-coin data."""
    _require(symbol in SYMBOLS, "Only owned coin market input is permitted")
    digest = hashlib.sha256()
    reference = None
    for field in ("open", "high", "low", "close", "volume"):
        frame = data["hourly_" + field].loc[:, [symbol]]
        _require(isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is not None, "Market index must carry timezone")
        index = frame.index.tz_convert("UTC")
        _require(not index.has_duplicates and index.is_monotonic_increasing, "Raw market index must be ordered and unique")
        _require(reference is None or index.equals(reference), "Raw own-coin OHLCV must align")
        reference = index
        digest.update(field.encode("utf-8") + b"\0")
        digest.update(np.asarray(frame.shape, dtype="<i8").tobytes())
        digest.update(index.asi8.astype("<i8").tobytes())
        digest.update(frame.to_numpy(dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def cropped_market(data, symbol, context):
    lower = utc(context["train_start"]) - timedelta(days=200)
    upper = utc(context["train_end"])
    result = {}
    for field in ("open", "high", "low", "close", "volume"):
        source = data["hourly_" + field]
        frame = source.loc[(source.index >= lower) & (source.index < upper), [symbol]].copy()
        wanted = pd.date_range(lower, upper, freq="h", inclusive="left")
        _require(frame.index.tz_convert("UTC").equals(wanted), "Physical training crop must retain every requested timestamp")
        result["hourly_" + field] = frame
    return result


def audit_swap(before, after):
    """A model replacement must be financially inert, including accumulated DD."""
    _require(isinstance(before, dict) and isinstance(after, dict), "Swap must expose both financial snapshots")
    _require(set(before) == SNAPSHOT_FIELDS and set(after) == SNAPSHOT_FIELDS, "Complete fixed financial snapshot required")
    _require(_finite(before) and _finite(after), "Swap snapshot must be finite")
    _require(before == after, "Policy replacement reset or changed financial account state")
    _require(before["cash"] >= -1e-7 and before["nav"] > 0 and before["high_water"] >= before["nav"] - 1e-7, "Invalid cash/NAV/high-water snapshot")
    _require(all(v >= 0 for v in before["positions"].values()), "Swap contains negative inventory")
    _require(HEX.fullmatch(before["fills_digest"]) is not None, "Cumulative fill digest required")
    return True


def _check_native_archive(path, algorithm, recipe):
    """Read native JSON without unpickling third-party Python objects."""
    with zipfile.ZipFile(path) as archive:
        _require("data" in archive.namelist(), "Native checkpoint has no SB3 data document")
        native = json.loads(archive.read("data"))
    for key in ("learning_rate", "gamma", "buffer_size", "learning_starts", "batch_size", "tau", "gradient_steps",
                "n_steps", "policy_delay", "target_policy_noise", "target_noise_clip", "top_quantiles_to_drop_per_net"):
        if key in recipe:
            _require(native.get(key) == recipe[key], "Native serialized parameter differs: " + key)
    if "ent_coef" in recipe:
        _require(native.get("ent_coef") == recipe["ent_coef"], "Native entropy recipe differs")
    if algorithm in ("sac", "tqc"):
        kwargs = native.get("policy_kwargs")
        _require(isinstance(kwargs, dict) and ":serialized:" not in kwargs, "Inspectable native actor architecture required")
        kwargs = dict(kwargs)
        if "use_sde" in kwargs and "use_sde" not in recipe.get("policy_kwargs", {}):
            _require(kwargs.pop("use_sde") is False, "Only native use_sde=False serialization default is permitted")
        _require(kwargs == recipe["policy_kwargs"], "Native policy architecture differs")


def audit_model_evidence(record, manifest, expected_context, expected_recipe, expected_steps,
                         model_dir, *, manifest_name="rolling_model.json", expected_market_sha256=None):
    """Validate actual saved native model, counters, data boundaries and episodes."""
    model_dir = Path(model_dir)
    algorithm = expected_context["algorithm"]
    _require(algorithm in ALGORITHMS, "Unsupported selected algorithm")
    _require(expected_steps == (65536 if algorithm == "td3" else 32768), "Original per-model budget changed")
    for key in ("symbol", "algorithm", "seed", "recipe_id"):
        _require(record.get(key) == expected_context[key] and manifest.get(key) == expected_context[key], "Model identity differs: " + key)
    _require(record.get("context") == expected_context and manifest.get("rolling_context") == expected_context, "Exact rolling model context required")
    _require(record.get("checkpoint") == manifest and record.get("training") == manifest.get("training"), "Training record and saved checkpoint manifest disagree")
    _require(manifest.get("recipe") == expected_recipe, "Saved complete native recipe differs")
    _require(manifest.get("observation_shape") == [13] and manifest.get("action_shape") == [1], "Only native single-pair spaces are permitted")
    _require(manifest.get("independent_account") is True and manifest.get("multi_model_target_average") is False, "Model is not an independent own-coin policy")
    _require(manifest.get("no_optimizer_or_replay_resume") is True, "Every scheduled model must be freshly trained")
    for field in ("parameter_sha256", "adapter_sha256"):
        _require(isinstance(manifest.get(field), str) and HEX.fullmatch(manifest[field]), "Missing model evidence digest:" + field)
    checkpoint_files = manifest.get("checkpoint_files")
    _require(isinstance(checkpoint_files, dict) and set(checkpoint_files) == {"model.zip"}, "Expected exactly one native SB3 checkpoint")
    checkpoint_dir = model_dir / "checkpoint"
    for name, digest in checkpoint_files.items():
        _require(sha(safe_path(checkpoint_dir, name)) == digest, "Saved checkpoint hash changed")
    files = record.get("files")
    _require(isinstance(files, dict) and {"checkpoint/model.zip", "checkpoint/" + manifest_name} <= set(files), "All native model/manifest file hashes required")
    for name, digest in files.items():
        _require(sha(safe_path(model_dir, name)) == digest, "Recorded model input changed: " + name)
    training = manifest["training"]
    for key in ("symbol", "algorithm", "seed"):
        _require(training.get(key) == expected_context[key], "Actual training history identity differs: " + key)
    _require(training.get("actual_steps") == expected_steps, "Actual learner interactions differ")
    effective, counters = training.get("effective_parameters", {}), training.get("counters", {})
    if algorithm in ("sac", "tqc"):
        _require(effective.get("native_class") == ALGORITHMS[algorithm], "Recorded native class differs")
        _require(effective.get("observation_shape") == [13] and effective.get("action_shape") == [1], "Recorded native spaces differ")
    for key in ("learning_rate", "gamma", "buffer_size", "learning_starts", "batch_size", "tau", "gradient_steps",
                "n_steps", "policy_delay", "target_policy_noise", "target_noise_clip", "top_quantiles_to_drop_per_net", "ent_coef"):
        if key in expected_recipe:
            _require(effective.get(key) == expected_recipe[key], "Actual native training parameter differs: " + key)
    updates = expected_steps - expected_recipe["learning_starts"]
    wanted_optimizers = {"actor": updates // 2, "critic": updates} if algorithm == "td3" else {"actor": updates, "critic": updates, "entropy": updates}
    _require(counters.get("environment_steps") == expected_steps and counters.get("gradient_update_rounds") == updates,
             "Native environment/gradient counters differ")
    _require(counters.get("optimizer_steps") == wanted_optimizers, "Native optimizer rounds differ")
    _require(counters.get("replay_add_calls") == expected_steps and counters.get("replay_transitions") == expected_steps, "Every learner interaction must enter replay once")
    cutoff, start = utc(expected_context["train_end"]), utc(expected_context["train_start"])
    _require(cutoff - start == timedelta(days=1827), "Fixed physical training lookback changed")
    _require(utc(record["maximum_input_hour"]) == cutoff - timedelta(hours=1), "Training includes unavailable or omitted final hour")
    _require(expected_market_sha256 is None or record.get("input_market_sha256") == expected_market_sha256, "Private own-coin market crop changed")
    quality = training.get("training_window_quality", {})
    _require(quality.get("requested_train_days") == 1827 and quality.get("episode_days") == 32,
             "Requested training window/episode length differs")
    _require(0 < quality.get("eligible_windows", 0) <= 1796 and 0 < quality.get("covered_training_days", 0) <= 1827,
             "Training lacks valid complete causal episodes")
    episodes = training.get("training_episode_history")
    _require(isinstance(episodes, list) and len(episodes) in (expected_steps // 32, expected_steps // 32 + 1), "Complete native episode history required")
    completed = 0
    for index, episode in enumerate(episodes):
        first, last, until = map(utc, (episode["start"], episode["end_exclusive"], episode["completed_until"]))
        _require(start <= first < last <= cutoff and last - first == timedelta(days=32), "Sampled episode crosses physical training bounds")
        steps = episode["completed_steps"]
        _require(type(steps) is int and steps in (0, 32) and until == first + timedelta(days=steps), "Native sampled episode incomplete or time-shifted")
        _require(episode.get("terminated") is False and episode.get("truncated") is (steps == 32), "Time limits must be truncations, not economic terminations")
        _require(steps != 0 or index == len(episodes) - 1, "Only the final unused native auto-reset episode may have zero steps")
        completed += steps
    _require(completed == expected_steps, "Sampled episode steps do not equal actual learner interactions")
    _check_native_archive(checkpoint_dir / "model.zip", algorithm, expected_recipe)
    return {"passed": True, "parameter_sha256": manifest["parameter_sha256"],
            "manifest_sha256": sha(checkpoint_dir / manifest_name), "context": expected_context}


def validate_protocol(spec):
    _require(spec.get("test") == list(TEST), "The original full 365-day period is required")
    _require(spec.get("seeds") == list(range(5)) and all(type(s) is int for s in spec["seeds"]), "Five complete predeclared seeds required")
    _require(spec.get("cadences") == CADENCES and spec.get("costs") == COSTS, "Cadence/cost contract changed")
    _require(spec.get("capital_per_account") == 10000 and spec.get("selection_uses_test") is False, "Own 10000USDT accounts and validation-only parameter choice required")
    _require(spec.get("capital_shared_between_pairs") is False and spec.get("ensemble") is False, "No capital or policy aggregation is permitted")
    _require(spec.get("fresh_holdout") is False and spec.get("test_year_already_observed") is True, "Prior year observation must be disclosed")
    _require((spec.get("train_days"), spec.get("lag_days"), spec.get("preheat_days")) == (1827, 1, 200), "Frozen physical training boundary rules changed")
    selected, recipes = spec.get("selected", {}), spec.get("recipes", {})
    _require(set(selected) == set(SYMBOLS) and all(len(v) == 3 and len(set(v)) == 3 for v in selected.values()), "Exactly three distinct recipes per coin required")
    for symbol, recipe_ids in selected.items():
        for recipe_id in recipe_ids:
            _require(recipe_id in recipes, "Selected recipe is missing")
            recipe = recipes[recipe_id]
            _require(recipe.get("algorithm") in ALGORITHMS and isinstance(recipe.get("parameters"), dict), "Full native recipe required")
            _require(recipe.get("timesteps") == (65536 if recipe["algorithm"] == "td3" else 32768), "Native interaction budget changed")
    _require(spec.get("expected_models") == 1590 and spec.get("expected_accounts") == 180, "Declared complete comparison grid changed")
    for key in ("dataset_sha256", "dataset_manifest_sha256", "inventory_sha256", "selection_sha256"):
        _require(isinstance(spec.get(key), str) and HEX.fullmatch(spec[key]), "Required frozen input SHA missing:" + key)
    _require(isinstance(spec.get("sources"), dict) and spec["sources"].get(SOURCE_ORACLE) == ORACLE_SHA256, "Unchanged independent arithmetic oracle is required")
    return True


def _source_root(study, spec):
    matches = []
    for name in ("source", "frozen_source"):
        root = Path(study) / name
        if root.is_dir() and all(safe_path(root, relative).is_file() for relative in spec["sources"]):
            matches.append(root)
    _require(len(matches) == 1, "One unambiguous complete frozen source root is required")
    return matches[0]


def _load_oracle(source_root):
    path = safe_path(source_root, SOURCE_ORACLE)
    _require(sha(path) == ORACLE_SHA256, "Financial oracle bytes changed")
    module_spec = importlib.util.spec_from_file_location("_unchanged_rolling_financial_oracle", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def audit_financial_case(account_root, market, symbol, algorithm, seed, scenario, *, oracle):
    result = oracle.audit_case(Path(account_root), market, "test", symbol, algorithm, seed, scenario,
                               pd.Timestamp(TEST[0]), pd.Timestamp(TEST[1]))
    return result


def _all_expected_models(spec, protocol_sha):
    result = []
    for symbol, recipe_ids in spec["selected"].items():
        for recipe_id in recipe_ids:
            recipe = spec["recipes"][recipe_id]
            for seed in spec["seeds"]:
                for version in schedule_versions(*spec["test"], 7):
                    result.append(model_context(version, symbol=symbol, algorithm=recipe["algorithm"], recipe_id=recipe_id,
                                                seed=seed, protocol_sha256=protocol_sha))
    _require(len(result) == spec["expected_models"], "Complete unique model grid does not reconcile")
    return result


def _audit_timeout_trace(path, algorithm, steps, checks):
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    checks.check(len(rows) == steps // 32, "every genuine native time-limit transition recorded")
    for index, row in enumerate(rows):
        checks.check(row.get("transition_number") == (index + 1) * 32, "timeout actual environment step")
        checks.check(row.get("done") == 1. and row.get("timeout") == 1. and row.get("effective_done") == 0., "time limit bootstraps instead of economic termination")
        terminal = np.asarray(row.get("terminal_observation_values"), dtype="<f4")
        stored = np.asarray(row.get("stored_next_observation_values"), dtype="<f4")
        checks.check(terminal.shape == (13,) and stored.shape == (13,) and np.isfinite(terminal).all() and np.isfinite(stored).all(), "actual finite own-coin terminal transition")
        checks.vector(stored, terminal, "native replay stores actual terminal state", atol=0., rtol=0.)
        checks.check(row.get("terminal_observation_sha256") == hashlib.sha256(terminal.tobytes()).hexdigest()
                     and row.get("stored_next_observation_sha256") == hashlib.sha256(stored.tobytes()).hexdigest(), "terminal observation trace digests")


def _model_worker(task):
    study, context, recipe, runtime, source_pins, expected_market = task
    study = Path(study)
    relative = _model_relative(context)
    directory = study / relative
    checks, inputs = Checks(relative.as_posix()), {}
    pin = None
    try:
        record = read(directory / "record.json")
        manifest = read(directory / "checkpoint/rolling_model.json")
        pin = audit_model_evidence(record, manifest, context, recipe["parameters"], recipe["timesteps"], directory,
                                   expected_market_sha256=expected_market)
        checks.check(True, "complete native model identity, recipe, fresh-training budget and physical data window")
        checks.check(manifest.get("adapter_sha256") == source_pins["scripts/rolling_single_pair_models.py"]
                     and manifest.get("base_adapter_sha256") == source_pins["scripts/single_pair_rl_models.py"], "actual model source adapters are the frozen source bytes")
        checks.check(manifest.get("nonfinancial_preflight") is False and "nonfinancial_preflight" not in context,
                     "nonfinancial smoke checkpoint cannot enter real financial experiment")
        checks.check(manifest.get("native_optimizer_state_retained") is True, "actual native optimizer state retained as evidence")
        checks.check(manifest.get("runtime_packages") == {key: value for key, value in runtime.items() if key != "pandas"},
                     "actual learner dependency versions frozen")
        checks.check(isinstance(manifest.get("rng_before_training"), dict) and isinstance(manifest.get("rng_after_training"), dict),
                     "fresh RNG provenance recorded without claiming complete resume")
        _audit_timeout_trace(directory / "timeout_trace.jsonl", context["algorithm"], recipe["timesteps"], checks)
        files = {"record.json", "started.json", "timeout_trace.jsonl", "progress.jsonl", *record["files"]}
        for name in files:
            part = safe_path(directory, name)
            inputs[(relative / name).as_posix()] = sha(part)
        checks.check(read(directory / "started.json").get("context") == context, "started process bound to same frozen model context")
        progress = [json.loads(line) for line in (directory / "progress.jsonl").read_text().splitlines() if line.strip()]
        checks.check(bool(progress) and progress[-1].get("actual_steps") == recipe["timesteps"], "actual process progress reached exact frozen budget")
    except Exception as error:
        checks.exception(error)
    return {"context": context, "passed": not checks.failures, "checks_total": checks.count,
            "failure_count": len(checks.failures), "failures": checks.failures, "inputs_sha256": inputs, "pin": pin}


def _model_relative(context):
    return Path("models") / context["symbol"] / context["recipe_id"] / f"seed{context['seed']}" / f"v{context['version_index']:02d}"


def _write_new(path, document):
    path = Path(path)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


class Checks:
    def __init__(self, label=""):
        self.label, self.count, self.failures = label, 0, []

    def check(self, condition, name, details=None):
        self.count += 1
        if not bool(condition):
            self.failures.append({"case": self.label, "check": name, **({"details": details} if details is not None else {})})
        return bool(condition)

    def close(self, actual, expected, name, atol=1e-6, rtol=1e-10):
        try:
            okay = math.isfinite(float(actual)) and math.isfinite(float(expected)) and abs(float(actual) - float(expected)) <= atol + rtol * abs(float(expected))
        except (TypeError, ValueError, OverflowError):
            okay = False
        self.check(okay, name)

    def vector(self, actual, expected, name, atol=1e-6, rtol=1e-10):
        try:
            left, right = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
            if not self.check(left.shape == right.shape, name + ":shape"):
                return
            self.count += left.size
            if not np.isfinite(left).all() or not np.isfinite(right).all() or not np.allclose(left, right, atol=atol, rtol=rtol):
                self.failures.append({"case": self.label, "check": name})
        except (ValueError, TypeError):
            self.check(False, name + ":numeric")

    def exception(self, error):
        self.check(False, "exception", {"type": type(error).__name__, "error": str(error)})


def own_market_features(data, symbol, days):
    """Independently reconstruct causal ten-feature observations from raw hours."""
    field = data["hourly_close"].loc[:, [symbol]]
    hourly_index = field.index.tz_convert("UTC")
    close = field.loc[hourly_index.hour == 23].copy()
    close.index = close.index.normalize()
    complete = np.ones((len(field), 1), dtype=bool)
    for name in ("open", "high", "low", "close", "volume"):
        values = data["hourly_" + name].loc[:, [symbol]].to_numpy(dtype=float)
        complete &= np.isfinite(values) & (values >= 0 if name == "volume" else values > 0)
    day_index = pd.DataFrame(complete, index=field.index).resample("D").sum().index
    close = close.reindex(day_index).where(lambda f: np.isfinite(f) & f.gt(0))
    hourly_volume = data["hourly_volume"].loc[:, [symbol]]
    valid_volume = np.isfinite(hourly_volume) & hourly_volume.ge(0)
    volume = hourly_volume.where(valid_volume).resample("D").sum(min_count=24).where(valid_volume.resample("D").sum().eq(24))
    returns = close.pct_change(fill_method=None)
    parts = [returns / .1, (close / close.shift(7) - 1) / .3,
             (close / close.shift(30) - 1) / .5, close / close.shift(90) - 1,
             (close / close.shift(180) - 1) / 2,
             (close / close.rolling(20, min_periods=20).mean() - 1) / .2,
             (close / close.rolling(60, min_periods=60).mean() - 1) / .35,
             (close / close.rolling(200, min_periods=200).mean() - 1) / .5,
             returns.rolling(30, min_periods=30).std(ddof=1) / .1,
             (volume / volume.rolling(30, min_periods=30).mean() - 1) / 2]
    matrix = pd.DataFrame(np.column_stack([v.iloc[:, 0].to_numpy() for v in parts]), index=day_index + pd.Timedelta(days=1))
    result = matrix.loc[pd.DatetimeIndex(days)].to_numpy(dtype=float)
    _require(result.shape == (len(days), 10) and np.isfinite(result).all(), "Complete causal own-coin features are required on every account day")
    return np.clip(result, -10., 10.)


def _audit_selection(study, spec, checks, inputs):
    selection = read(study / "selection.json")
    checks.check(sha(study / "selection.json") == spec["selection_sha256"], "selection bytes frozen")
    checks.check(sha(study / "inventory.json") == spec["inventory_sha256"] == selection.get("inventory_sha256"), "inventory bytes frozen separately from validation-only selection")
    checks.check(selection.get("uses_test") is False and selection.get("seed_selection") is False, "no annual/seed selection")
    checks.check(selection.get("selected") == spec["selected"], "selected recipes match protocol")
    declared_files = selection.get("source_sha256", {})
    required_files = {"selection_inputs/native_protocol.json", "selection_inputs/native_validation_summary.csv",
                      *{"selection_inputs/" + c + "_validation_summary.csv" for c in ("c01_lr1e4", "c02_gamma099", "c03_lr1e4_gamma099")}}
    _require(set(declared_files) == required_files, "Selection must bind precisely all native and SAC-variant validation sources")
    for name, digest in declared_files.items():
        checks.check(sha(safe_path(study, name)) == digest, "validation-only source hash:" + name)
        inputs[name] = sha(safe_path(study, name))
    native_spec = read(study / "selection_inputs/native_protocol.json")
    native_frame = pd.read_csv(study / "selection_inputs/native_validation_summary.csv", float_precision="round_trip")
    populations = [("native_" + a, a, native_frame[native_frame.algorithm.eq(a)], tuple(range(5))) for a in native_spec["algorithms"]]
    for candidate in ("c01_lr1e4", "c02_gamma099", "c03_lr1e4_gamma099"):
        frame = pd.read_csv(study / f"selection_inputs/{candidate}_validation_summary.csv", float_precision="round_trip")
        populations.append(("sac_" + candidate, "sac", frame[frame.algorithm.eq("sac")], tuple(range(100, 105))))
    observed_candidates = selection.get("candidates", [])
    mapping = {"native_sac": "sac_default", "native_td3": "td3_default", "native_tqc": "tqc_default", "sac_c02_gamma099": "sac_gamma099"}
    for symbol in SYMBOLS:
        ranked = []
        for identifier, algorithm, frame, seeds in populations:
            group = frame[frame.symbol.eq(symbol)]
            expected = {(seed, cost) for seed in seeds for cost in COSTS}
            observed = {(int(r.seed), r.scenario) for r in group.itertuples()}
            checks.check(len(group) == 10 and not group.duplicated(["seed", "scenario"]).any() and observed == expected, "five validation seeds and two costs:" + symbol + identifier)
            checks.check(group.split.eq("validation").all() and group.initial_capital.eq(10000.).all(), "independent validation identity:" + identifier)
            checks.check(pd.to_datetime(group.start, utc=True).eq(pd.Timestamp("2025-01-01T00:00:00Z")).all()
                         and pd.to_datetime(group.end_exclusive, utc=True).eq(pd.Timestamp(TEST[0])).all(), "only prior validation dates:" + identifier)
            utilities = []
            for seed in seeds:
                costs = group[group.seed.eq(seed)]
                utilities.append(float(np.mean([math.log1p(float(r.net_return)) - .5 * abs(float(r.max_drawdown))
                    - 2 * max(abs(float(r.max_drawdown)) - .2, 0.) for r in costs.itertuples()])))
            score = float(np.median(utilities) - .25 * (np.quantile(utilities, .75, method="linear") - np.quantile(utilities, .25, method="linear")))
            degenerate = bool(group.fills.eq(0).all())
            candidates = [v for v in observed_candidates if v.get("symbol") == symbol and v.get("candidate_id") == identifier]
            checks.check(len(candidates) == 1, "one preserved candidate:" + symbol + identifier)
            if candidates:
                checks.close(candidates[0].get("robust_score"), score, "independent validation robust score:" + identifier, atol=1e-12)
                checks.vector(candidates[0].get("seed_utilities"), utilities, "validation seed utility:" + identifier, atol=1e-12)
                checks.check(candidates[0].get("cashlike_degenerate") is degenerate, "zero-fill candidates separately identified:" + identifier)
                expected_recipe = json.loads(json.dumps(native_spec["recipes"][algorithm]))
                if identifier == "sac_c01_lr1e4":
                    expected_recipe["learning_rate"] = .0001
                elif identifier == "sac_c02_gamma099":
                    expected_recipe["gamma"] = .99
                elif identifier == "sac_c03_lr1e4_gamma099":
                    expected_recipe.update(learning_rate=.0001, gamma=.99)
                checks.check(candidates[0].get("recipe") == expected_recipe and candidates[0].get("algorithm") == algorithm
                             and candidates[0].get("tuning_seeds") == list(seeds), "full source-bound validation recipe and seed population:" + identifier)
                selected_id = mapping.get(identifier)
                if selected_id in spec["selected"][symbol]:
                    checks.check(spec["recipes"][selected_id]["parameters"] == expected_recipe
                                 and spec["recipes"][selected_id]["algorithm"] == algorithm,
                                 "new model recipe equals selected prior validation recipe:" + symbol + selected_id)
            if not degenerate:
                ranked.append((score, identifier))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        ranking = [v[1] for v in ranked]
        checks.check(selection.get("rankings", {}).get(symbol) == ranking, "full validation-only ranking:" + symbol)
        checks.check([mapping.get(name) for name in ranking[:3]] == spec["selected"][symbol], "top three selected solely by frozen robust score:" + symbol)
    return selection


def _snapshot_from_fills(snapshot, *, symbol, day_number, day, market, fills, path_values, checks):
    """Cross-check policy-swap state against the actual continuous fill history."""
    prefix = fills[pd.to_datetime(fills.timestamp, utc=True) < pd.Timestamp(day)]
    records = prefix.to_dict("records")
    actual_digest = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    checks.check(snapshot["fill_count"] == len(records) and snapshot["fills_digest"] == actual_digest, "swap cumulative fills and digest")
    checks.check(snapshot["steps"] == day_number and snapshot["equity_rows"] == 24 * day_number
                 and snapshot["weight_rows"] == day_number and snapshot["boundary_rows"] == day_number + 1,
                 "swap continuous full account history lengths")
    checks.check(utc(snapshot["date"]) == utc(day), "swap financial date matches effective model day")
    cash, qty, basis, realized, fees, notional, slippage = 10000., 0., 0., 0., 0., 0., 0.
    for row in records:
        amount, price, fee = row["qty"], row["price"], row["fee"]
        if row["side"] == "BUY":
            qty += amount
            basis += amount * price + fee
            cash -= amount * price + fee
        else:
            removed = basis * amount / qty
            qty -= amount
            cash += amount * price - fee
            realized += amount * price - fee - removed
            basis -= removed
            if qty <= 1e-14:
                qty, basis = 0., 0.
        fees += fee
        notional += row["notional"]
        slippage += row["slippage_cost"]
    checks.check(set(snapshot["positions"]).issubset({symbol}) and set(snapshot["cost_basis"]).issubset({symbol}), "swap inventory belongs only to own coin")
    for actual, wanted, name in ((snapshot["cash"], cash, "cash"), (snapshot["positions"].get(symbol, 0.), qty, "coin quantity"),
                                 (snapshot["cost_basis"].get(symbol, 0.), basis, "cost basis"),
                                 (snapshot["realized_pnl"], realized, "realized PNL"), (snapshot["fees"], fees, "fees"),
                                 (snapshot["notional"], notional, "notional"), (snapshot["slippage_cost"], slippage, "slippage")):
        checks.close(actual, wanted, "swap independently continuous " + name)
    previous_price = market["previous_close"] if day_number == 0 else market["close"][day_number * 24 - 1]
    nav = 10000. if day_number == 0 else path_values[day_number * 24 - 1, 0]
    peak = max(10000., float(path_values[:day_number * 24, 0].max())) if day_number else 10000.
    checks.close(snapshot["nav"], nav, "swap continuous NAV")
    checks.close(snapshot["high_water"], peak, "swap continuous historical high water")
    checks.check(set(snapshot["last_prices"]) == {symbol}, "swap own-coin mark only")
    checks.close(snapshot["last_prices"].get(symbol), previous_price, "swap causal last mark")


def audit_account_schedule(account_root, symbol, recipe_id, seed, cadence, scenario, *, spec,
                           protocol_sha256, market, features, model_pins, oracle):
    """Read one already-computed actual account and its exact deterministic repeat."""
    account_root = Path(account_root)
    algorithm = spec["recipes"][recipe_id]["algorithm"]
    relative = Path("evaluation/test") / symbol / algorithm / f"seed{seed}" / scenario
    directory = account_root / relative
    repeated = account_root / Path("repeat/test") / symbol / algorithm / f"seed{seed}" / scenario
    checks, inputs = Checks(f"{symbol}/{recipe_id}/{cadence}/seed{seed}/{scenario}"), {}
    try:
        actions, swaps = read(directory / "actions.json"), read(directory / "swaps.json")
        for name in ("actions.json", "swaps.json"):
            inputs[(relative / name).as_posix()] = sha(directory / name)
            repeat_relative = Path("repeat/test") / symbol / algorithm / f"seed{seed}" / scenario / name
            inputs[repeat_relative.as_posix()] = sha(repeated / name)
            checks.check(sha(directory / name) == sha(repeated / name), "repeated actual policy/swap records identical:" + name)
        planned = schedule_versions(*spec["test"], CADENCES[cadence])
        checks.check(len(swaps) == len(planned), "exact scheduled swap count including common initial model")
        target_check = oracle.Checks(checks.label)
        targets = oracle._targets(actions, market, algorithm, seed, target_check)
        checks.count += target_check.count
        checks.failures.extend(target_check.failures)
        arithmetic = oracle.arithmetic_path(market, targets, COSTS[scenario])
        path_values = arithmetic["values"]
        fills = pd.read_csv(directory / "fills.csv", float_precision="round_trip")
        for actual, version in zip(swaps, planned):
            expected = model_context(version, symbol=symbol, algorithm=algorithm, recipe_id=recipe_id, seed=seed,
                                     protocol_sha256=protocol_sha256)
            pin = model_pins[(symbol, recipe_id, seed, version["version_index"])]
            checks.check(utc(actual["date"]) == utc(version["effective_day"]) and actual.get("version_index") == version["version_index"], "model effective exactly at declared daily boundary")
            checks.check(actual.get("context") == expected and actual.get("model_manifest_sha256") == pin["manifest_sha256"], "swap binds exact own-coin scheduled trained checkpoint")
            try:
                audit_swap(actual["before"], actual["after"])
                checks.check(True, "complete financial state invariant across policy load")
            except AuditError as error:
                checks.check(False, "complete financial state invariant across policy load", str(error))
            day_number = (utc(actual["date"]) - utc(TEST[0])).days
            _snapshot_from_fills(actual["before"], symbol=symbol, day_number=day_number, day=actual["date"],
                                 market=market, fills=fills, path_values=path_values, checks=checks)
        peaks = np.maximum.accumulate(np.r_[10000., path_values[:, 0]])
        for index, (day, action) in enumerate(zip(market["days"], actions)):
            offset = index if CADENCES[cadence] is None else index // CADENCES[cadence] * CADENCES[cadence]
            version_index = 0 if CADENCES[cadence] is None else offset // 7
            pin = model_pins[(symbol, recipe_id, seed, version_index)]
            checks.check(action.get("version_index") == version_index and action.get("model_manifest_sha256") == pin["manifest_sha256"], "actual prediction binds shared scheduled model")
            if index == 0:
                own = [0., 1., 0.]
            else:
                previous = path_values[index * 24 - 1]
                own = [previous[3], previous[1] / previous[0], max(0., 1 - previous[0] / peaks[index * 24])]
            expected_obs = np.clip(np.r_[features[index], own], -10, 10).astype("<f4")
            actual_obs = np.asarray(action.get("observation"), dtype="<f4")
            checks.vector(actual_obs, expected_obs, "actual own-coin causal observation and continuous account state", atol=1e-6, rtol=1e-6)
            checks.check(actual_obs.shape == (13,) and np.isfinite(actual_obs).all(), "one genuine 13-value observation per daily prediction")
            checks.check(action.get("observation_sha256") == hashlib.sha256(actual_obs.tobytes()).hexdigest(), "actual FP32 observation digest")
        summary = read(directory / "summary.json")
        expected_id = f"{symbol}__{recipe_id}__{cadence}__seed{seed}__{scenario}"
        checks.check(summary.get("account_id") == expected_id and summary.get("recipe_id") == recipe_id
                     and summary.get("cadence") == cadence and summary.get("algorithm") == algorithm, "account namespaces preserve native algorithm identity")
    except Exception as error:
        checks.exception(error)
    return {"checks_total": checks.count, "failure_count": len(checks.failures), "failures": checks.failures,
            "inputs_sha256": inputs}


_WORKER_STATE = None


def _initialize_audit_worker(study, spec, protocol_sha256, source_root, markets, features, model_pins):
    global _WORKER_STATE
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    _WORKER_STATE = (Path(study), spec, protocol_sha256, _load_oracle(source_root), markets, features, model_pins)


def _account_worker(case):
    study, spec, protocol_sha, oracle, markets, features, model_pins = _WORKER_STATE
    symbol, recipe_id, seed, cadence, scenario = case
    account_root = study / "accounts" / symbol / recipe_id / cadence
    algorithm = spec["recipes"][recipe_id]["algorithm"]
    finance = audit_financial_case(account_root, markets[symbol], symbol, algorithm, seed, scenario, oracle=oracle)
    method = audit_account_schedule(account_root, symbol, recipe_id, seed, cadence, scenario,
        spec=spec, protocol_sha256=protocol_sha, market=markets[symbol], features=features[symbol], model_pins=model_pins, oracle=oracle)
    relative = account_root.relative_to(study)
    inputs = {str(relative / key).replace("\\", "/"): value for key, value in finance.get("inputs_sha256", {}).items()}
    inputs.update({str(relative / key).replace("\\", "/"): value for key, value in method["inputs_sha256"].items()})
    # The original oracle counts every scalar arithmetic comparison separately.
    return {"case": list(case), "passed": finance.get("passed") is True and method["failure_count"] == 0,
            "checks_total": finance.get("checks_total", finance.get("checks", 0)) + method["checks_total"],
            "failure_count": finance.get("failure_count", 0) + method["failure_count"],
            "failures": finance.get("failures", []) + method["failures"], "inputs_sha256": inputs,
            "summary": finance.get("summary")}


def audit_study(study, dataset, max_workers=None):
    """Full completion gate. The CLI and parent runner may call this directly."""
    return _audit_study_impl(Path(study), Path(dataset), max_workers)


def verify_completed_study(study):
    """Read-only, current-byte verification of an already completed gate."""
    study = Path(study)
    gate = read(study / "completion_gate.json")
    _require(gate.get("passed") is True and gate.get("failure_count") == 0, "Existing completion gate did not pass")
    _require((gate.get("expected_models"), gate.get("models_checked")) == (1590, 1590)
             and (gate.get("expected_accounts"), gate.get("accounts_checked"), gate.get("repeats_checked")) == (180, 180, 180),
             "Existing gate did not audit the complete real grid")
    _require(gate.get("protocol_sha256") == sha(study / "protocol.json") and gate.get("summary_sha256") == sha(study / "summary.csv"),
             "Completed research contract or summary changed")
    _require(gate.get("source_sha256") == sha(__file__) and gate.get("oracle_source_sha256") == ORACLE_SHA256,
             "Completed gate was produced by another auditor")
    _require(bool(gate.get("inputs_sha256")) and bool(gate.get("external_inputs_sha256")), "Complete current input binding required")
    for name, expected in gate["inputs_sha256"].items():
        _require(sha(safe_path(study, name)) == expected, "Completed study input changed:" + name)
    for name, expected in gate["external_inputs_sha256"].items():
        _require(sha(name) == expected, "Completed study external market changed")
    return {"passed": True, "completion_gate_sha256": sha(study / "completion_gate.json"),
            "inputs_verified": len(gate["inputs_sha256"]) + len(gate["external_inputs_sha256"]),
            "no_training_inference_or_repair": True}


def _audit_study_impl(study, dataset, max_workers):
    started = datetime.now(UTC).isoformat()
    checks = Checks("study-completion")
    inputs, model_results, account_results = {}, [], []
    protocol_sha = summary_sha = None
    external = {}
    try:
        spec = read(study / "protocol.json")
        validate_protocol(spec)
        checks.check(True, "complete immutable rolling research specification")
        protocol_sha = sha(study / "protocol.json")
        _require(read(study / "protocol_receipt.json").get("protocol_sha256") == protocol_sha, "Frozen protocol receipt changed")
        source_root = _source_root(study, spec)
        for name, expected in spec["sources"].items():
            actual = sha(safe_path(source_root, name))
            checks.check(actual == expected, "frozen source:" + name)
            inputs[(source_root.relative_to(study) / name).as_posix()] = actual
        checks.check(sha(__file__) == spec["sources"].get("scripts/audit_rolling_single_pair.py"), "actually executing the frozen independent completion auditor")
        dataset_manifest = dataset.parent / "manifest.json"
        _require(sha(dataset) == spec["dataset_sha256"] and sha(dataset_manifest) == spec["dataset_manifest_sha256"], "Physical market archive or manifest differs from frozen inputs")
        external = {str(dataset.resolve()): sha(dataset), str(dataset_manifest.resolve()): sha(dataset_manifest)}
        for name in ("protocol.json", "protocol_receipt.json", "inventory.json", "selection.json", "fit_complete.json",
                     "summary.csv", "accounts_manifest.json", "evaluation_complete.json"):
            inputs[name] = sha(safe_path(study, name))
        _audit_selection(study, spec, checks, inputs)
        _require(not checks.failures, "Source or validation-selection checks failed; do not trust derived final accounts")
        with gzip.open(dataset, "rb") as stream:
            data = pickle.load(stream)
        oracle = _load_oracle(source_root)
        markets = {symbol: oracle._market(data, symbol, pd.Timestamp(TEST[0]), pd.Timestamp(TEST[1])) for symbol in SYMBOLS}
        features = {symbol: own_market_features(data, symbol, markets[symbol]["days"]) for symbol in SYMBOLS}
        expected_contexts = _all_expected_models(spec, protocol_sha)
        expected_record_paths = {(_model_relative(context) / "record.json").as_posix() for context in expected_contexts}
        observed_record_paths = {path.relative_to(study).as_posix() for path in (study / "models").rglob("record.json")}
        checks.check(observed_record_paths == expected_record_paths, "every planned unique model and no undeclared model record")
        market_digests = {}
        for symbol in SYMBOLS:
            for version in schedule_versions():
                context = model_context(version, symbol=symbol, algorithm="sac", recipe_id="sac_default", seed=0, protocol_sha256=protocol_sha)
                crop = cropped_market(data, symbol, context)
                market_digests[(symbol, version["version_index"])] = input_market_digest(crop, symbol)
        model_tasks = [(str(study), context, spec["recipes"][context["recipe_id"]], spec["runtime_packages"],
                        spec["sources"], market_digests[(context["symbol"], context["version_index"])]) for context in expected_contexts]
        cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
        workers = max(1, min(cpus, max_workers or max(1, cpus - 2), len(model_tasks)))
        if workers == 1:
            model_results = [_model_worker(task) for task in model_tasks]
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
                model_results = list(pool.map(_model_worker, model_tasks, chunksize=1))
        model_pins = {}
        for result in model_results:
            checks.count += result["checks_total"]
            checks.failures.extend(result["failures"])
            inputs.update(result["inputs_sha256"])
            context = result["context"]
            if result["pin"] is not None:
                model_pins[(context["symbol"], context["recipe_id"], context["seed"], context["version_index"])] = result["pin"]
        _require(len(model_pins) == spec["expected_models"] and not checks.failures, "One or more actual trained models failed schedule/recipe/cutoff verification")
        fit = read(study / "fit_complete.json")
        checks.check(fit.get("models") == spec["expected_models"], "training completion reflects full actual model grid")
        cases = [(symbol, recipe_id, seed, cadence, scenario) for symbol in SYMBOLS
                 for recipe_id in spec["selected"][symbol] for seed in spec["seeds"] for cadence in CADENCES for scenario in COSTS]
        _require(len(cases) == spec["expected_accounts"], "Declared annual account grid does not reconcile")
        expected_case_summaries = {
            (Path("accounts") / s / r / c / prefix / "test" / s / spec["recipes"][r]["algorithm"] /
             f"seed{seed}" / scenario / "summary.json").as_posix()
            for s, r, seed, c, scenario in cases for prefix in ("evaluation", "repeat")}
        actual_case_summaries = {path.relative_to(study).as_posix() for path in (study / "accounts").rglob("summary.json")}
        checks.check(actual_case_summaries == expected_case_summaries, "all actual independent accounts and repeats exactly match declared grid")
        if workers == 1:
            _initialize_audit_worker(study, spec, protocol_sha, source_root, markets, features, model_pins)
            account_results = [_account_worker(case) for case in cases]
        else:
            with ProcessPoolExecutor(max_workers=min(workers, len(cases)), mp_context=mp.get_context("spawn"),
                                     initializer=_initialize_audit_worker,
                                     initargs=(str(study), spec, protocol_sha, str(source_root), markets, features, model_pins)) as pool:
                account_results = list(pool.map(_account_worker, cases, chunksize=1))
        for result in account_results:
            checks.count += result["checks_total"]
            checks.failures.extend(result["failures"])
            checks.check(result["passed"], "financial and causal-schedule case completed:" + "/".join(map(str, result["case"])))
            inputs.update(result["inputs_sha256"])
        summary = pd.read_csv(study / "summary.csv", float_precision="round_trip")
        summary_sha = sha(study / "summary.csv")
        manifest = read(study / "accounts_manifest.json").get("accounts", [])
        expected_ids = {f"{s}__{r}__{c}__seed{seed}__{cost}" for s, r, seed, c, cost in cases}
        checks.check(len(summary) == len(cases) and set(summary.account_id) == expected_ids and not summary.account_id.duplicated().any(), "one complete summary row per independent account")
        checks.check(len(manifest) == len(cases) and {r.get("account_id") for r in manifest} == expected_ids, "actual account file manifest covers exact independent grid")
        manifest_by_id = {row["account_id"]: row for row in manifest}
        for result in account_results:
            row = result.get("summary")
            if not row:
                continue
            match = summary[summary.account_id.eq(row.get("account_id"))]
            checks.check(len(match) == 1, "one final summary row for audited actual account")
            if len(match) == 1:
                actual = match.iloc[0].to_dict()
                for key, expected in row.items():
                    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
                        value = actual.get(key)
                        checks.check(value == expected or expected is None and pd.isna(value), "summary reconciles actual account:" + key)
                    else:
                        checks.close(actual.get(key), expected, "summary reconciles actual account:" + key, atol=1e-10)
            declared = manifest_by_id.get(row.get("account_id"), {})
            s, r, seed, c, cost = result["case"]
            alg = spec["recipes"][r]["algorithm"]
            directory = Path("accounts") / s / r / c / "evaluation/test" / s / alg / f"seed{seed}" / cost
            checks.check(declared.get("directory") == directory.as_posix(), "manifest actual account directory identity")
            hashes = declared.get("files_sha256", {})
            checks.check(set(hashes) == set(oracle.RESULT_FILES) | {"swaps.json"}, "all financial and swap evidence hashes declared")
            for name, expected in hashes.items():
                checks.check(sha(safe_path(study, (directory / name).as_posix())) == expected, "account result manifest input:" + name)
        evaluation = read(study / "evaluation_complete.json")
        checks.check(evaluation.get("accounts") == len(cases) and evaluation.get("summary_sha256") == summary_sha, "annual completion bound to actual account summary")
        # Recheck every source, checkpoint and account after the full audit to
        # reject mixed generations or records still being mutated in parallel.
        for name, expected in inputs.items():
            checks.check(sha(safe_path(study, name)) == expected, "input unchanged during completion audit:" + name)
        for name, expected in external.items():
            checks.check(sha(name) == expected, "external market unchanged during completion audit")
    except Exception as error:
        checks.exception(error)
    result = {"passed": not checks.failures, "checks_total": checks.count, "failure_count": len(checks.failures),
              "failures": checks.failures, "started_at_utc": started, "audited_at_utc": datetime.now(UTC).isoformat(),
              "expected_accounts": 180, "accounts_checked": len(account_results), "repeats_checked": len(account_results),
              "expected_models": 1590, "models_checked": len(model_results), "protocol_sha256": protocol_sha,
              "summary_sha256": summary_sha, "inputs_sha256": inputs, "external_inputs_sha256": external,
              "source_sha256": sha(__file__), "oracle_source_sha256": ORACLE_SHA256,
              "no_training_inference_or_repair": True}
    if result["passed"]:
        if (study / "completion_gate.json").exists():
            existing = read(study / "completion_gate.json")
            if (existing.get("passed") is True and existing.get("inputs_sha256") == inputs
                    and existing.get("external_inputs_sha256") == external
                    and existing.get("source_sha256") == result["source_sha256"]):
                result["reused_completion_gate_sha256"] = sha(study / "completion_gate.json")
            else:
                result["passed"], result["failure_count"] = False, result["failure_count"] + 1
                result["failures"].append({"case": "study-completion", "check": "preserve incompatible existing completion gate"})
        else:
            _write_new(study / "completion_gate.json", result)
    if not result["passed"]:
        filename = "audit_failed_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + ".json"
        _write_new(study / filename, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--max-workers", type=int)
    args = parser.parse_args()
    result = audit_study(args.study, args.dataset, args.max_workers)
    print(json.dumps(result, ensure_ascii=False))
    if not result.get("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
