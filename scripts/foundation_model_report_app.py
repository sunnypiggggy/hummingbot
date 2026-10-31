"""Author the reviewed pretrained-forecast versus RL portfolio comparison."""
from __future__ import annotations

import copy
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.spot_strategy_suite_report_app import NODE_DEFAULT, PLUGIN_DEFAULT, _json, _run, _sha, _write


QUERY_IDS = ("strategy_summary", "equity_history", "monthly_returns", "seed_statistics",
             "training_metrics", "universe_changes", "model_provenance", "forecast_diagnostics")


def _primary(row):
    return row.get("primary_policy") is True or row.get("policy_role") == "primary" or str(row.get("strategy", "")).endswith("_ensemble")


def build_report(output_dir: Path, node_path=None, plugin_root=None) -> dict:
    output_dir = Path(output_dir).resolve()
    reviewed_path = output_dir / "reviewed.json"
    reviewed = _json(reviewed_path)
    if reviewed.get("status") != "reviewed":
        raise ValueError("Complete reviewed forecasting and RL evidence is required")
    for query_id in QUERY_IDS:
        query = reviewed.get("queries", {}).get(query_id)
        if not query or not isinstance(query.get("rows"), list) or not query["rows"] or not query.get("source"):
            raise ValueError(f"Missing reviewed evidence: {query_id}")
    rows = reviewed["queries"]["strategy_summary"]["rows"]
    if len({r.get("run_id") for r in rows}) != len(rows):
        raise ValueError("Every saved run must have one globally unique identity")
    primary = [r for r in rows if _primary(r)]
    combinations = {(r["implementation"], r["algorithm"]) for r in primary}
    foundation = {(r["implementation"], r["algorithm"]) for r in primary if r["implementation"] == "foundation"}
    if len(foundation) != 3:
        raise ValueError("The three actual pretrained model policies are required")
    for implementation, algorithm in combinations:
        for variant in ("original", "ex_ada"):
            for split in ("validation", "test"):
                for scenario in ("base", "stress"):
                    matches = [r for r in primary if (r["implementation"], r["algorithm"], r["universe_variant"], r["split"], r["scenario"]) == (implementation, algorithm, variant, split, scenario)]
                    if len(matches) != 1:
                        raise ValueError(f"Missing or duplicated primary policy: {implementation}/{algorithm}/{variant}/{split}/{scenario}")
    # Deterministic policies are not relabelled as independent stochastic seeds.
    chronos = [r for r in rows if r.get("implementation") == "foundation" and (r.get("model_key") == "chronos2" or "chronos" in str(r.get("algorithm", "")).lower())]
    if any(r.get("seed") is not None for r in chronos):
        raise ValueError("Chronos deterministic primary must not carry invented seed replication")
    node, plugin = Path(node_path) if node_path else NODE_DEFAULT, Path(plugin_root) if plugin_root else PLUGIN_DEFAULT
    if not node.is_file() or not (plugin / "scripts/data-app.mjs").is_file():
        raise FileNotFoundError("Pinned Node and installed Data plugin are required")
    project = output_dir / "report_app"
    created = not project.exists()
    if created:
        preparation = _run(node, plugin / "scripts/prepare-data-app.mjs", "--surface", "report", "--output", str(project), "--snapshot", str(reviewed_path), cwd=output_dir)
    else:
        preparation = _run(node, plugin / "scripts/data-app.mjs", "prepare", "--project-dir", str(project), cwd=output_dir)
    guide = (project / "AGENTS.md").read_text(encoding="utf-8")
    reference = Path(preparation["documentation"]["entryPoint"])
    reference.read_text(encoding="utf-8")
    for name in ("reports.md", "charts.md", "tables.md", "controls.md"):
        (reference.parent / name).read_text(encoding="utf-8")
    if "src/content/" not in guide:
        raise RuntimeError("Unexpected protected runtime boundary")
    ownership_path = project / ".foundation-model-authoring.json"
    previous = _json(ownership_path) if ownership_path.exists() else {}
    if not created and not previous:
        raise RuntimeError("Existing app has no foundation-model ownership record; preserving it")
    for relative, expected in previous.get("authored_sha256", {}).items():
        path = project / relative
        if not path.is_file() or _sha(path) != expected:
            raise RuntimeError(f"Preserving independent presentation edits: {path}")
    content = project / "src/content/report"
    content.mkdir(parents=True, exist_ok=True)
    templates = {Path(__file__).with_suffix(".jsx"): content / "ReportContent.jsx",
                 Path(__file__).with_suffix(".css"): content / "foundation-model-report.css"}
    for source, destination in templates.items():
        shutil.copyfile(source, destination)
    old = _json(project / "src/data.json")
    snapshot = copy.deepcopy(reviewed)
    snapshot["id"] = old["id"]
    if "legacyPresentationTitle" in old:
        snapshot["legacyPresentationTitle"] = old["legacyPresentationTitle"]
    snapshot["surface"], snapshot["buildStatus"] = "report", "complete"
    for query_id in QUERY_IDS:
        snapshot["queries"][query_id].setdefault("methods", []).append({"language": "text", "code": "Presentation only: percentage aliases multiply reviewed fractions by 100. Pool deltas match the same implementation/model, split and cost; ex_ada minus original. Primary roles are explicit: stochastic Kronos primary averages declared mapped target weights in one ledger, deterministic Chronos has one policy. Sampling seeds of one pretrained checkpoint are not independently trained models. No missing metrics become zero and no best test seed/checkpoint is selected."})
    _write(project / "src/data.json", snapshot)
    _write(ownership_path, {"authored_sha256": {str(path.relative_to(project)).replace("\\", "/"): _sha(path) for path in templates.values()}, "reviewed_sha256": _sha(reviewed_path)})
    build = _run(node, plugin / "scripts/data-app.mjs", "build", "--project-dir", str(project), "--separate-data", cwd=output_dir)
    export_path = project / ".data-app-offline/exports/foundation-model-comparison.html"
    exported = _run(node, plugin / "scripts/data-app.mjs", "export-offline", "--project-dir", str(project), "--output", str(export_path), cwd=output_dir)
    delivered = output_dir / "report.html"
    shutil.copyfile(export_path, delivered)
    return {"project_dir": str(project), "html_path": str(delivered), "dist_dir": str(project / "dist"),
            "preview_path": str(project / "dist/index.html"), "offline_path": str(export_path),
            "reviewed_sha256": _sha(reviewed_path), "delivered_sha256": _sha(delivered),
            "delivered_bytes": delivered.stat().st_size,
            "authoring_source_sha256": {path.name: _sha(path) for path in [Path(__file__), *templates]},
            "runtime": build, "export": exported, "published": False, "numeric_results_changed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=ROOT / "results/backtests/crypto_foundation_models_20261024")
    args = parser.parse_args()
    result = build_report(args.study_root.resolve() / "report")
    print(json.dumps(result, ensure_ascii=False))
