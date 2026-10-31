"""Verify saved foundation-model delivery without inference or simulation."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/backtests/crypto_foundation_models_20261024"
sys.path.insert(0, str(ROOT))
from scripts.foundation_model_proof_paths import ProofPaths


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(output=OUTPUT):
    output = Path(output).resolve()
    report = output / "report"
    protection = read(output / "original_research_preservation.json")
    for name, expected in protection["files"].items():
        path = ROOT / name
        assert path.stat().st_size == expected["size_bytes"] and sha(path) == expected["sha256"], name
    frozen = read(output / "inference_frozen_inputs.json")
    assert sha(output / "frozen_spec.json") == frozen["protocol_sha256"]
    for name, expected in frozen["source_sha256"].items():
        assert sha(ROOT / name) == expected, name
    spec = read(output / "frozen_spec.json")
    proofs = ProofPaths(output, spec)
    for item in spec["pinned_files"]:
        assert sha(proofs.resolve(item["path"])) == item["sha256"], item["path"]
    assert sha(proofs.resolve(spec["model_provenance_path"])) == spec["model_provenance_sha256"]
    assert sha(output / "policy_precommit.json") == spec["policy_precommit_sha256"]
    assert sha(proofs.resolve(spec["backend_evidence_path"])) == spec["backend_evidence_sha256"]
    complete, audit = read(output / "numeric_completion.json"), read(output / "independent_audit.json")
    assert complete["status"] == "complete" and complete["evaluations"] == complete["repeat_checks"] == 200
    assert complete["new_models_trained"] == 0 and complete["pretrained_models"] == 3
    assert audit["status"] == "passed" and audit["checks_failed"] == 0 and audit["runs_checked"] == 200
    assert audit["phase"] == "full_pretrained_foundation_independent_audit"
    manifest, build = read(report / "manifest.json"), read(report / "report_build.json")
    for name, expected in manifest["input_sha256"].items():
        assert sha(name) == expected, name
    for name, expected in manifest["generated_sha256"].items():
        assert sha(report / name) == expected, name
    for name, expected in build["authoring_source_sha256"].items():
        assert sha(ROOT / "scripts" / name) == expected, name
    assert sha(report / "report.html") == build["delivered_sha256"]
    reviewed = read(report / "reviewed.json")
    expected_counts = {"strategy_summary": 376, "training_metrics": 50, "seed_statistics": 56,
                       "universe_changes": 48, "model_provenance": 3,
                       "forecast_diagnostics": 2730, "implementation_review": 8}
    assert reviewed["status"] == "reviewed" and reviewed["report"]["asOf"] == "2026-10-22"
    for key, count in expected_counts.items():
        assert len(reviewed["queries"][key]["rows"]) == count, key
    summary = reviewed["queries"]["strategy_summary"]["rows"]
    assert len({row["run_id"] for row in summary}) == len(summary)
    assert sum(row.get("primary_policy") is True or row.get("policy_role") == "primary" or
               str(row["strategy"]).endswith("_ensemble") for row in summary) == 64
    ui = read(report / "verification/ui_verification.json")
    assert ui["checks"] and all(value is True for value in ui["checks"].values())
    assert not ui["new_console_errors"]
    assert ui["geometry"]["width"] == ui["geometry"]["scrollWidth"]
    screenshots = {name: sha(report / "verification" / name) for name in ui["screenshots"]}
    assert screenshots
    result = {"status": "passed", "verified_at": datetime.now(timezone.utc).isoformat(),
              "new_evaluations": 200, "repeat_checks": 200, "all_saved_comparison_evaluations": 376,
              "pretrained_models": 3, "new_models_trained": 0, "old_rl_training_records": 50,
              "independent_checks": audit["checks_total"], "independent_checks_failed": 0,
              "old_files_preserved": len(protection["files"]),
              "read_only_report_inputs_verified": len(manifest["input_sha256"]),
              "html_sha256": sha(report / "report.html"), "html_bytes": (report / "report.html").stat().st_size,
              "ui_verified": True, "url": ui["url"], "tab_id": ui["tab_id"],
              "screenshot_sha256": screenshots, "frozen_financial_sources_unchanged": True,
              "production_changes": False, "git_commit_or_push_performed": False}
    if proofs.imported is not None:
        assert complete["gpu_protocol_sha256"] == sha(output / "frozen_spec.json")
        assert complete["local_import_manifest_sha256"] == sha(output / "local_import_manifest.json")
        result.update(execution="dual_v100_forecasts_local_original_ledger", cpu_forecasts_reused=False,
                      gpu_protocol_sha256=complete["gpu_protocol_sha256"],
                      local_import_manifest_sha256=complete["local_import_manifest_sha256"])
    path = output / "delivery_verification.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=OUTPUT)
    main(parser.parse_args().study_root)
