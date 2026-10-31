"""Single pre-test TQC exposure multiplier .75, CPU validation diagnostics.

Every policy sees this diagnostic ledger's actual weights/drawdown. Predictions
are generated anew; old actions, trades or equity curves are never scaled.
"""
from __future__ import annotations

import os
for _name, _value in {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                      "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}.items():
    os.environ[_name] = _value

import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts import backtest_btceth_rl as formal
from scripts.diagnose_btceth_rl_validation import availability, cpu_setup, now, sha
from scripts.btceth_rl_env import BTCETHPortfolioEnv
from scripts.crypto_rl_reporting import result_fingerprints, save_result, summarize, write_json

MULTIPLIER = .75
PROFILES = ("return", "balanced")


def evaluate_overlay(data, models, profile, costs):
    """New real funded account, with mean of five individually mapped targets."""
    import torch
    before = [formal.parameter_digest(model.policy) for model in models]
    env = BTCETHPortfolioEnv(data, *formal.VALIDATION, costs=costs, record_history=True, **formal.PROFILES[profile])
    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        targets = []
        for model in models:
            action, _ = model.predict(obs, deterministic=True)
            target = env.action_to_weights(action)
            target[:2] *= MULTIPLIER
            target[2] = 1 - target[:2].sum()
            targets.append(target)
        target = np.mean(targets, axis=0)
        if target[:2].sum() > .95 * MULTIPLIER + 1e-12 or target[2] < .05:
            raise AssertionError("Overlay target exceeded the declared funded budget")
        obs, _, terminated, truncated, _ = env.step_target(target)
        done = bool(terminated or truncated)
    result = env.get_result()
    env.close()
    after = [formal.parameter_digest(model.policy) for model in models]
    if before != after or torch.cuda.is_initialized():
        raise AssertionError("Overlay prediction changed weights or initialized CUDA")
    result["metadata"].update(algorithm="tqc", profile=profile, backend="cpu", model_count=5,
        exposure_multiplier=MULTIPLIER, parameter_sha256_before=before, parameter_sha256_after=after,
        ensemble="mean five individually mapped and capped targets; one actual10000 USDT ledger",
        observations="prior completed market features plus this overlay ledger's actual weights/current drawdown",
        old_actions_or_equity_scaled=False, formal_evaluation=False)
    return result


def diagnostic(formal_output, dataset_path, raw_output, output):
    import torch
    formal_output, dataset_path, raw_output, output = map(lambda p: Path(p).resolve(), (formal_output, dataset_path, raw_output, output))
    if output.exists():
        raise FileExistsError("Preserve the existing single-multiplier diagnostic")
    if output == formal_output or output.is_relative_to(formal_output) or output == raw_output or output.is_relative_to(raw_output):
        raise ValueError("Overlay diagnostics must be separate from formal and raw results")
    spec = formal.check_frozen(formal_output, dataset_path)
    runtime = cpu_setup()
    if any(runtime["packages"].get(name) != value for name, value in spec.get("runtime_packages", {}).items()):
        raise ValueError("Use the unchanged formal research runtime")
    records, status = availability(formal_output, spec)
    if "tqc" not in records:
        raise ValueError("Both TQC profiles require all five original completed model records")
    raw_manifest = json.loads((raw_output / "file_manifest.json").read_text(encoding="utf-8"))
    protected = {raw_output / "file_manifest.json": sha(raw_output / "file_manifest.json"),
                 formal_output / "frozen_spec.json": sha(formal_output / "frozen_spec.json"),
                 dataset_path: sha(dataset_path), dataset_path.parent / "manifest.json": sha(dataset_path.parent / "manifest.json")}
    for name, expected in raw_manifest.items():
        path = (raw_output / name).resolve()
        path.relative_to(raw_output)
        if sha(path) != expected:
            raise ValueError("Original raw diagnostic source changed: " + name)
        protected[path] = expected
    completed = json.loads((raw_output / "completion.json").read_text(encoding="utf-8"))
    if completed.get("passed") is not True or completed.get("cases") != 8 or completed.get("test_evaluations") != 0:
        raise ValueError("The prior audited8-case CPU diagnostic is required")
    raw_summary = pd.read_csv(raw_output / "summary.csv", float_precision="round_trip")
    raw_months = pd.read_csv(raw_output / "monthly_returns.csv", float_precision="round_trip")
    raw_proofs = json.loads((raw_output / "replay_and_audit_proofs.json").read_text(encoding="utf-8"))
    protected.update({ROOT / name: value for name, value in spec["code_sha256"].items()})
    for _, _, record_path, checkpoint, _ in records["tqc"]:
        protected[record_path], protected[checkpoint] = sha(record_path), sha(checkpoint)
    output.mkdir(parents=True)
    sources = output / "sources"
    sources.mkdir()
    for path in (Path(__file__).resolve(), ROOT / "scripts/diagnose_btceth_rl_validation.py", *[ROOT / name for name in spec["code_sha256"]]):
        shutil.copyfile(path, sources / path.name)
    shutil.copyfile(formal_output / "frozen_spec.json", sources / "original_frozen_spec.json")
    audit_path = sources / "independent_audit_source_snapshot.py"
    shutil.copyfile(raw_output / "sources/reference_independent_audit_source_snapshot.py", audit_path)
    protocol = {"declared_at": now(), "scope": "validation-only limited exposure-budget diagnostic; not a formal extension or test",
                "motivation": "Prior CPU TQC balanced validation returned23.66% but22.10% base /24.06% stress DD, exceeding the20% target; test one limited.75 exposure budget before any annual test.",
                "multipliers_evaluated": [MULTIPLIER], "parameter_search": False, "new_training": 0,
                "test_evaluations": 0, "formal_output_modified": False, "initial_capital": 10000.,
                "algorithm": "tqc", "profiles": list(PROFILES), "seeds": list(formal.SEEDS),
                "mapping": "Each model: original action_to_weights; multiply only BTC/ETH targets by.75, USDT=1-sum(coins); average five targets and execute one real ledger. Feed the resulting true weights/DD into the next policy observation.",
                "cash_floor_environment": .05, "environment_asset_cap": .95, "effective_max_coin_target_total": .95 * MULTIPLIER,
                "rebalance_band": .025, "execution": "original daily00UTC sells,01UTC buys; unchanged per-leg costs",
                "costs": formal.COSTS, "validation": list(formal.VALIDATION),
                "price_data_cutoff_exclusive": formal.VALIDATION[1], "runtime": runtime,
                "source_sha256": sha(Path(__file__).resolve()), "evaluation_helper_sha256": sha(ROOT / "scripts/diagnose_btceth_rl_validation.py"),
                "raw_summary_sha256": sha(raw_output / "summary.csv"), "raw_completion_sha256": sha(raw_output / "completion.json"),
                "raw_manifest_sha256": sha(raw_output / "file_manifest.json"), "original_frozen_spec_sha256": sha(formal_output / "frozen_spec.json"),
                "interpretation": "Designed after seeing validation, not new unseen validation evidence. CPU/GPU may differ slightly; formal GPU extension, selection and five-seed tests would require a separate pre-test contract."}
    write_json(output / "protocol.json", protocol)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--formal-output", str(formal_output), "--dataset", str(dataset_path),
               "--raw-output", str(raw_output), "--output-dir", str(output)]
    (output / "exact_command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    write_json(output / "input_sha_before.json", {str(path): value for path, value in protected.items()})
    original = formal.load_dataset(dataset_path)
    data = {name: value for name, value in original.items() if not name.startswith("hourly_")}
    for field in ("open", "high", "low", "close", "volume"):
        frame = original["hourly_" + field]
        data["hourly_" + field] = frame.loc[frame.index < pd.Timestamp(formal.VALIDATION[1])].copy()
    del original
    if any(frame.index.max() >= pd.Timestamp(formal.VALIDATION[1]) for name, frame in data.items() if name.startswith("hourly_")):
        raise AssertionError("No test-year prices permitted in feature or ledger calculations")
    module_spec = importlib.util.spec_from_file_location("tqc_cap_independent_audit", audit_path)
    audit = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = audit
    module_spec.loader.exec_module(audit)
    rows, histories, months, proofs = [], [], [], []
    started = time.monotonic()
    try:
        for profile in PROFILES:
            models, model_records = [], []
            for candidate_profile, seed, record_path, checkpoint, record in records["tqc"]:
                if candidate_profile != profile:
                    continue
                model = formal.model_class("tqc").load(checkpoint, device="cpu")
                if any(p.device.type != "cpu" for p in model.policy.parameters()) or formal.parameter_digest(model.policy) != record["parameter_sha256"]:
                    raise ValueError("CPU overlay model does not match the original frozen checkpoint")
                models.append(model)
                model_records.append({"seed": seed, "model_sha256": record["model_sha256"], "parameter_sha256": record["parameter_sha256"],
                                      "record_sha256": sha(record_path), "record_path": str(record_path)})
            raw_proof = next(row for row in raw_proofs if row["run_id"] == f"early_validation__base__tqc__{profile}__ensemble")
            if len(models) != 5 or model_records != raw_proof["model_records"]:
                raise AssertionError("Overlay must reuse exactly the same five models as the raw diagnostic")
            for scenario, costs in formal.COSTS.items():
                run_id = f"tqc_cap075_validation__{scenario}__{profile}__ensemble"
                before = [formal.parameter_digest(model.policy) for model in models]
                result = evaluate_overlay(data, models, profile, costs)
                fingerprints = save_result(output / "cases" / profile / scenario, result)
                repeat = evaluate_overlay(data, models, profile, costs)
                if result_fingerprints(repeat) != fingerprints:
                    raise AssertionError("Overlay replay must be exact: " + run_id)
                after = [formal.parameter_digest(model.policy) for model in models]
                if before != after or before != raw_proof["parameter_sha256_before"]:
                    raise AssertionError("Original trained parameter state changed")
                checked = audit.reconstruct_fills(data, result["fills"], result["equity"], result["metadata"],
                          start=formal.VALIDATION[0], end=formal.VALIDATION[1], **costs, label=run_id)
                if checked["checks"].failed:
                    raise AssertionError("Independent overlay ledger reconstruction failed: " + str(checked["checks"].failures))
                target_totals = result["weights"].groupby("date").target_weight.sum()
                if target_totals.max() > .95 * MULTIPLIER + 1e-10 or target_totals.min() < -1e-12:
                    raise AssertionError("Saved targets violate the single declared multiplier")
                row, daily, monthly = summarize(result, run_id=run_id, strategy="tqc", seed=None, scenario=scenario,
                    split="validation", start=formal.VALIDATION[0], end=formal.VALIDATION[1], capital=10000., **costs)
                for name in ("net_return", "max_drawdown", "fees", "turnover", "average_exposure", "final_equity"):
                    if not np.isclose(row[name], checked["metrics"][name], atol=1e-9, rtol=1e-10):
                        raise AssertionError("Reconstructed overlay summary differs: " + name)
                raw_id = f"early_validation__{scenario}__tqc__{profile}__ensemble"
                raw_row = raw_summary.loc[raw_summary.run_id.eq(raw_id)].iloc[0]
                raw_monthly = raw_months.loc[raw_months.run_id.eq(raw_id), "return"]
                row.update(algorithm="tqc", profile=profile, exposure_multiplier=MULTIPLIER, backend="cpu", model_count=5,
                    old_actions_or_equity_scaled=False, formal_evaluation=False, test_evaluations=0,
                    positive_months=int((monthly["return"] > 0).sum()), observed_months=len(monthly),
                    worst_month=float(monthly["return"].min()), best_month=float(monthly["return"].max()),
                    monthly_return_std_ddof1=float(monthly["return"].std(ddof=1)), raw_run_id=raw_id,
                    raw_positive_months=int((raw_monthly > 0).sum()), raw_worst_month=float(raw_monthly.min()))
                for name in ("net_return", "max_drawdown", "fees", "turnover", "average_exposure"):
                    row["raw_" + name] = float(raw_row[name])
                    row[name + "_minus_raw"] = float(row[name] - raw_row[name])
                daily["profile"], monthly["profile"] = profile, profile
                rows.append(row); histories.append(daily); months.append(monthly)
                proof = {"run_id": run_id, "same_five_models_as_raw": True, "model_records": model_records,
                         "parameter_sha256_before": before, "parameter_sha256_after": after,
                         "fingerprints": fingerprints, "repeat_exact": True,
                         "independent_checks": checked["checks"].count, "independent_failures": 0,
                         "maximum_reconstruction_errors": checked["checks"].maximum_errors,
                         "metrics": checked["metrics"], "max_requested_coin_weight_sum": float(target_totals.max()),
                         "cuda_initialized": torch.cuda.is_initialized()}
                proofs.append(proof)
                write_json(output / "cases" / profile / scenario / "independent_reconstruction.json", proof)
                pd.DataFrame(rows).to_csv(output / "summary.csv", index=False, float_format="%.17g")
                print(f"{run_id} return={row['net_return']:.6%} DD={row['max_drawdown']:.6%} fees={row['fees']:.4f} turnover={row['turnover']:.4f} exposure={row['average_exposure']:.6%} positive_months={row['positive_months']}/{row['observed_months']}", flush=True)
            del models
        pd.concat(histories, ignore_index=True).to_csv(output / "daily_equity.csv", index=False, float_format="%.17g")
        pd.concat(months, ignore_index=True).to_csv(output / "monthly_returns.csv", index=False, float_format="%.17g")
        for path, expected in protected.items():
            if sha(path) != expected:
                raise AssertionError("Protected raw/formal/checkpoint input changed: " + str(path))
        if torch.cuda.is_initialized():
            raise AssertionError("CPU-only overlay initialized CUDA")
        completion = {"passed": True, "completed_at": now(), "elapsed_seconds": time.monotonic() - started,
            "cases": 4, "exact_repeats": 4, "independent_checks": sum(row["independent_checks"] for row in proofs),
            "independent_failures": 0, "runtime": runtime, "all_inputs_unchanged": True,
            "multipliers_evaluated": [MULTIPLIER], "same_five_models_as_raw": True,
            "test_evaluations": 0, "new_training": 0, "formal_output_modified": False,
            "old_actions_or_equity_scaled": False, "validation": list(formal.VALIDATION),
            "latest_price_used": str(data["hourly_close"].index.max()),
            "partial_month": "October2025 includes01--22 only; monthly counts coverJan--Oct10 buckets",
            "scope": "CPU validation diagnostic after seeing prior validation; not a formal GPU extension or holdout test"}
        write_json(output / "completion.json", completion)
        write_json(output / "replay_and_audit_proofs.json", proofs)
        write_json(output / "input_sha_after.json", {str(path): sha(path) for path in protected})
        write_json(output / "file_manifest.json", {path.relative_to(output).as_posix(): sha(path) for path in sorted(output.rglob("*")) if path.is_file() and path.name != "file_manifest.json"})
        print(json.dumps(completion, ensure_ascii=False), flush=True)
        return completion
    except BaseException as error:
        write_json(output / "failure.json", {"at": now(), "type": type(error).__name__, "error": str(error), "completed_cases": len(rows)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-output", type=Path, default=ROOT / "results/backtests/btceth_rl_20261029")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/crypto_rl_btceth_20261029/dataset.pkl.gz")
    parser.add_argument("--raw-output", type=Path, default=ROOT / "results/backtests/btceth_rl_pretest_reference_20261029/early_validation")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/backtests/btceth_rl_pretest_reference_20261029/tqc_exposure_cap_075_validation")
    args = parser.parse_args()
    diagnostic(args.formal_output, args.dataset, args.raw_output, args.output_dir)


if __name__ == "__main__":
    main()
