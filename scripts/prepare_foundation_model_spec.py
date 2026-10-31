"""Freeze source, weights and a single policy before real-market inference."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.backtest_crypto_rl_top10 import sha, COSTS, VAL_START, TEST_START, TEST_END
from scripts.crypto_foundation_portfolio import OUTPUT, DATASETS, CONTEXT_DAYS
from scripts.crypto_rl_reporting import write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def evaluation_provenance():
    """Keep source facts intact and label this experiment's added settings."""
    original_path = OUTPUT / "model_provenance.json"
    original = read(original_path)
    result = {"source": {**original["source"],
                         "files": original["source"]["files"] + [str(original_path.resolve()),
                                                                  str((OUTPUT / "policy_precommit.json").resolve())]},
              "original_provenance_path": str(original_path.resolve()),
              "original_provenance_sha256": sha(original_path), "rows": []}
    for row in original["rows"]:
        result["rows"].append({**row, "label": row["model"],
                               "repository": row["source_repository"], "source_url": row["source_repository"],
                               "weight_revision": row["weights_revision"], "weight_sha256": row["weights_sha256"],
                               "first_verified_available_at": row["first_verified_public_notice_utc"],
                               "first_decision_date": row["first_usable_decision_utc"],
                               "context_days": CONTEXT_DAYS, "horizon_days": 1,
                               "evaluation_settings_are_source_facts": False,
                               "sampling_description": "5 fixed stochastic paths, average mapped weights" if row["model_key"].startswith("kronos") else
                                                       "1 deterministic median forecast, no sampling ensemble",
                               "evaluation_note": "128-day context and 1-day horizon are this experiment's fixed settings, not model pretraining facts"})
    path = OUTPUT / "model_provenance_evaluation.json"
    if path.exists() and read(path) != result:
        raise AssertionError("Preserve a different existing provenance contract")
    write_json(path, result)
    return path, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--model-workers", type=int, default=3)
    parser.add_argument("--threads-per-model", type=int, default=2)
    parser.add_argument("--backend-evidence", type=Path, required=True)
    args = parser.parse_args()
    path = OUTPUT / "frozen_spec.json"
    if path.exists():
        raise FileExistsError("Never overwrite a frozen native-model experiment")
    if not args.backend_evidence.is_file():
        raise ValueError("Verify the actual inference backend before freezing it")
    backend = read(args.backend_evidence)
    if backend.get("status") != "passed" or backend.get("device") != args.device:
        raise ValueError("Actual model backend checks must pass before freezing")
    precommit = read(OUTPUT / "policy_precommit.json")
    if (precommit["policy_module_sha256"] != sha(ROOT / "scripts/crypto_foundation_portfolio.py") or
            precommit["request_plan_sha256"] != sha(OUTPUT / "forecast_request_plan.csv") or
            precommit["context_days"] != CONTEXT_DAYS or precommit["horizon_days"] != 1 or
            precommit["hurdle"] != .0024 or precommit["volatility_floor"] != .01 or
            precommit["sampling_seeds"] != list(range(5)) or precommit["parameter_search"] is not False):
        raise AssertionError("The rule precommitted before real forecasts must remain unchanged")
    source = read(OUTPUT / "sources/source_manifest.json")
    weights = read(OUTPUT / "sources/weights_manifest.json")
    provenance_path, provenance = evaluation_provenance()
    sources = {row["id"]: row for row in source["sources"]}
    models = {row["name"]: row for row in weights["models"]}
    details = {row["model_key"]: row for row in provenance["rows"]}
    if set(details) != {"kronos_small", "kronos_base", "chronos2"}:
        raise ValueError("Three independently sourced pretrained model records are required")
    configs = {}
    for key in ("kronos_small", "kronos_base", "chronos2"):
        native = sources["kronos" if key.startswith("kronos") else "chronos2"]
        vendor = Path(native["vendor_path"])
        if key == "chronos2" and vendor.name != "src":
            vendor /= "src"
        if details[key].get("local_fit") is not False or details[key]["context_days"] != CONTEXT_DAYS or details[key]["horizon_days"] != 1:
            raise ValueError("Provenance and frozen inference contract disagree")
        configs[key] = {"label": details[key]["label"], "vendor_path": str(vendor),
                        "weights_path": models[key]["local_directory"], "device": args.device,
                        "source_commit": native["commit"], "weight_revision": models[key]["hf_revision"],
                        "first_decision_date": details[key]["first_decision_date"],
                        "sampling_seeds": list(range(5)) if key.startswith("kronos") else [],
                        "sampling": {"T": 1., "top_p": .9, "top_k": 0, "sample_count": 1} if key.startswith("kronos") else
                                    {"quantile": .5, "predict_batches_jointly": False, "future_covariates": False}}
        if key.startswith("kronos"):
            configs[key]["tokenizer_path"] = models["kronos_tokenizer"]["local_directory"]
        else:
            if backend.get("chronos_synchronous_loader_full_quantile_parity") is not True:
                raise ValueError("Synchronous loader requires actual full-quantile bitwise parity evidence")
            configs[key]["synchronous_chronos_loader"] = True
    pinned = {}
    for row in (*source["sources"], *weights["models"]):
        for entry in row["files"]:
            file = entry["local_path"]
            if sha(file) != entry["sha256"]:
                raise AssertionError("Official source/weights differ from their manifest")
            pinned[file] = entry["sha256"]
    spec = {"frozen_at": datetime.now(timezone.utc).isoformat(), "local_fit": False,
            "models": configs, "validation": [VAL_START, TEST_START], "test": [TEST_START, TEST_END],
            "costs": COSTS, "capital": 10000, "cash_target_floor": .05, "single_coin_target_cap": .25,
            "context_days": CONTEXT_DAYS, "horizon_days": 1,
            "forecast_inputs": "128 complete past UTC daily OHLCVA candles; true quote turnover from the same original Binance archives",
            "forecast_to_weights": {"formula": "score_i=max(predicted_close/last_close-1-0.0024,0)/max(std(last30 daily returns,ddof=1),0.01); weight_i=min(0.95*score_i/sum(scores),0.25); excess USDT; allzero scores mean100% cash",
                                    "hurdle": .0024, "volatility_floor": .01,
                                    "not_exact_net_roi": True, "no_renormalization_after_cap": True,
                                    "first_decision_availability_gate": True},
            "ensemble": "Kronos: mean of5 independently mapped target weights in one real ledger; Chronos: one deterministic median policy",
            "sampling_seed_policy": "SHA256(model_key,base_seed,UTC decision date,ordered same-day union stable asset IDs), first4bytes modulo2**31; reset each date; reuse identical forecasts across pool variants",
            "native_repeat_dates": ["2025-07-25T00:00:00+00:00", "2025-10-23T00:00:00+00:00", "2026-10-22T00:00:00+00:00"],
            "parameter_search": False, "validation_or_test_fit": False, "prior_test_year_already_observed": True,
            "ex_ada_is_post_hoc_sensitivity": True, "amount_is_estimated": False,
            "forecast_execution": "At daily00UTC use only prior complete candles, sell00open and buy01open; hourly close mark",
            "model_workers": args.model_workers, "threads_per_model": args.threads_per_model,
            "date_shards": {"kronos_small": 2, "kronos_base": 4, "chronos2": 8},
            "chronos_loader_adaptation": "Windows synchronous DataLoader scheduling only; num_workers=0, pin_memory=False; official source unchanged; all native quantiles verified bitwise equal",
            "backend_evidence_path": str(args.backend_evidence.resolve()), "backend_evidence_sha256": sha(args.backend_evidence),
            "model_provenance_path": str(provenance_path.resolve()),
            "model_provenance_sha256": sha(provenance_path),
            "policy_precommit_sha256": sha(OUTPUT / "policy_precommit.json"),
            "market_manifest_sha256": sha(OUTPUT / "market/manifest.json"),
            "dataset_sha256_by_variant": {name: sha(file) for name, file in DATASETS.items()},
            "pinned_files": [{"path": file, "sha256": digest} for file, digest in sorted(pinned.items())],
            "reproduction_argv": [sys.executable, "scripts/backtest_crypto_foundation_models.py", "--phase", "all"],
            "production_changes_authorized": False}
    write_json(path, spec)
    write_json(OUTPUT / "forecast_diagnostics_source.json", {"label": "Native forecast coverage and fixed signal rule",
               "classification": "modeled", "timezone": "UTC", "period": {"start": VAL_START, "end": TEST_END},
               "files": ["forecast_diagnostics.csv", "forecast_request_plan.csv", "frozen_spec.json", "market/manifest.json"],
               "caveats": ["Positive-score count is the mean over5 stochastic Kronos paths; Chronos has1 deterministic path.",
                           "Forecast coverage is counted among the10 current historical slots, not all20 cached assets.",
                           "Model-unavailable validation days are USDT cash, not filled with retrospective forecasts."]})
    print(str(path))


if __name__ == "__main__":
    main()
