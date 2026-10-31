"""Verify final saved delivery evidence; never train or simulate economics."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.backtest_crypto_rl_top10 import sha
from scripts.crypto_rl_reporting import write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    output = ROOT / "results/backtests/crypto_rl_elegantrl_20261024"
    report = output / "report"
    snapshot = read(output / "original_research_preservation.json")
    for name, expected in snapshot["files"].items():
        path = ROOT / name
        assert path.stat().st_size == expected["size_bytes"] and sha(path) == expected["sha256"], name
    manifest = read(report / "manifest.json")
    for name, expected in manifest["input_sha256"].items():
        assert sha(Path(name)) == expected, name
    for name, expected in manifest["generated_sha256"].items():
        assert sha(report / name) == expected, name
    build = read(report / "report_build.json")
    assert sha(report / "report.html") == build["delivered_sha256"]
    for name, expected in build["authoring_source_sha256"].items():
        assert sha(ROOT / "scripts" / name) == expected, name
    frozen = read(output / "training_frozen_inputs.json")
    for name, expected in frozen["code_sha256_before_training"].items():
        assert sha(ROOT / name) == expected, name
    audit, complete = read(output / "independent_audit.json"), read(output / "numeric_completion.json")
    assert audit["status"] == "passed" and audit["checks_failed"] == 0 and audit["runs_checked"] == 240
    assert complete["status"] == "complete" and complete["evaluations"] == complete["repeat_checks"] == 240 and complete["models"] == 30
    ui = read(report / "verification/ui_verification.json")
    assert all(ui["checks"].values()) and not ui["new_console_errors"]
    assert ui["geometry"]["width"] == ui["geometry"]["scrollWidth"]
    reviewed = read(report / "reviewed.json")
    assert reviewed["status"] == "reviewed" and reviewed["report"]["asOf"] == "2026-10-22"
    assert {k: len(v["rows"]) for k, v in reviewed["queries"].items() if k in ("strategy_summary", "training_metrics", "seed_statistics", "universe_changes", "implementation_review")} == {
        "strategy_summary": 272, "training_metrics": 50, "seed_statistics": 40,
        "universe_changes": 48, "implementation_review": 5}
    result = {"verified_at": datetime.now(timezone.utc).isoformat(), "status": "passed",
              "original_files_preserved": len(snapshot["files"]), "read_only_inputs_verified": len(manifest["input_sha256"]),
              "evaluations": 240, "repeat_checks": 240, "new_models": 30, "all_comparison_models": 50,
              "independent_checks": audit["checks_total"], "independent_checks_failed": 0,
              "html_sha256": sha(report / "report.html"), "html_bytes": (report / "report.html").stat().st_size,
              "ui_verified": True, "ui_tab_id": ui["tab_id"], "local_preview": ui["url"],
              "screenshot_sha256": {name: sha(report / "verification" / name) for name in ui["screenshots"]},
              "frozen_financial_sources_unchanged": True, "production_changes": False,
              "git_commit_or_push_performed": False}
    write_json(output / "delivery_verification.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
