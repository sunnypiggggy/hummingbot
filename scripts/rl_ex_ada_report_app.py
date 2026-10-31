"""Author the separate, post-hoc ADA exclusion sensitivity report."""
from __future__ import annotations

import copy
from pathlib import Path
import shutil

from scripts.spot_strategy_suite_report_app import NODE_DEFAULT, PLUGIN_DEFAULT, _json, _run, _sha, _write


QUERY_IDS = ("strategy_summary", "equity_history", "monthly_returns", "seed_statistics", "universe_changes", "training_metrics")


def build_report(output_dir: Path, node_path=None, plugin_root=None) -> dict:
    """Consume reviewed evidence beneath the experiment's report directory."""
    output_dir = Path(output_dir).resolve()
    reviewed_path = output_dir / "reviewed.json"
    reviewed = _json(reviewed_path)
    if reviewed.get("status") != "reviewed":
        raise ValueError("The sensitivity report requires reviewed evidence")
    for query_id in QUERY_IDS:
        query = reviewed.get("queries", {}).get(query_id)
        if not query or not isinstance(query.get("rows"), list) or not query["rows"] or not query.get("source"):
            raise ValueError(f"Missing reviewed rows or source: {query_id}")
    summary = reviewed["queries"]["strategy_summary"]["rows"]
    for variant in ("original", "ex_ada"):
        for strategy in ("ppo_ensemble", "a2c_ensemble"):
            for split in ("test", "validation"):
                for scenario in ("base", "stress"):
                    matches = [r for r in summary if r.get("universe_variant") == variant and r.get("strategy") == strategy and r.get("split") == split and r.get("scenario") == scenario]
                    if len(matches) != 1:
                        raise ValueError(f"Expected one actual ensemble result: {variant}/{strategy}/{split}/{scenario}")
    if len({r.get("run_id") for r in summary}) != len(summary):
        raise ValueError("Comparison run_id must include universe identity and be globally unique")
    node = Path(node_path) if node_path else NODE_DEFAULT
    plugin = Path(plugin_root) if plugin_root else PLUGIN_DEFAULT
    if not node.is_file() or not (plugin / "scripts/data-app.mjs").is_file():
        raise FileNotFoundError("Pinned Codex Node and installed Data plugin are required")
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
    ownership_path = project / ".rl-ex-ada-authoring.json"
    previous = _json(ownership_path) if ownership_path.exists() else {}
    if not created and not previous:
        raise RuntimeError("Existing app has no ADA report ownership record; preserving it")
    for relative, expected in previous.get("authored_sha256", {}).items():
        path = project / relative
        if not path.is_file() or _sha(path) != expected:
            raise RuntimeError(f"Independent presentation edits are preserved: {path}")
    content = project / "src/content/report"
    content.mkdir(parents=True, exist_ok=True)
    templates = {Path(__file__).with_suffix(".jsx"): content / "ReportContent.jsx", Path(__file__).with_suffix(".css"): content / "rl-ex-ada-report.css"}
    for source, destination in templates.items():
        shutil.copyfile(source, destination)
    old = _json(project / "src/data.json")
    snapshot = copy.deepcopy(reviewed)
    snapshot["id"] = old["id"]
    if "legacyPresentationTitle" in old:
        snapshot["legacyPresentationTitle"] = old["legacyPresentationTitle"]
    snapshot["surface"], snapshot["buildStatus"] = "report", "complete"
    for query_id in QUERY_IDS:
        query = snapshot["queries"][query_id]
        query.setdefault("methods", []).append({"language": "text", "code": "Presentation only: fractional returns are multiplied by 100 for percent charts. Paired deltas equal ADA-excluded minus original for the same algorithm, seed, split and cost; missing matches stay null. Each ensemble is an actual separately executed average-target-weight portfolio, not a mean return series. No source rows are changed."})
    _write(project / "src/data.json", snapshot)
    authored = {str(path.relative_to(project)).replace("\\", "/"): _sha(path) for path in templates.values()}
    _write(ownership_path, {"authored_sha256": authored, "reviewed_sha256": _sha(reviewed_path)})
    build = _run(node, plugin / "scripts/data-app.mjs", "build", "--project-dir", str(project), "--separate-data", cwd=output_dir)
    export_path = project / ".data-app-offline/exports/rl-ada-exclusion.html"
    exported = _run(node, plugin / "scripts/data-app.mjs", "export-offline", "--project-dir", str(project), "--output", str(export_path), cwd=output_dir)
    delivered = output_dir / "report.html"
    shutil.copyfile(export_path, delivered)
    return {"project_dir": str(project), "html_path": str(delivered), "dist_dir": str(project / "dist"),
            "preview_path": str(project / "dist/index.html"), "offline_path": str(export_path),
            "reviewed_sha256": _sha(reviewed_path), "delivered_sha256": _sha(delivered), "delivered_bytes": delivered.stat().st_size,
            "authoring_source_sha256": {path.name: _sha(path) for path in [Path(__file__), *templates]},
            "runtime": build, "export": exported, "published": False, "numeric_results_changed": False}
