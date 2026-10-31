"""Build the separate, source-backed framework and RL algorithm comparison."""
from __future__ import annotations

import copy
from pathlib import Path
import shutil

from scripts.spot_strategy_suite_report_app import NODE_DEFAULT, PLUGIN_DEFAULT, _json, _run, _sha, _write


QUERY_IDS = ("strategy_summary", "equity_history", "monthly_returns", "framework_comparison", "seed_statistics", "training_metrics")


def build_report(output_dir: Path, node_path=None, plugin_root=None) -> dict:
    """Read assembled evidence and author only this comparison's new Data app."""
    output_dir = Path(output_dir).resolve()
    reviewed_path = output_dir / "reviewed.json"
    reviewed = _json(reviewed_path)
    if reviewed.get("status") != "reviewed":
        raise ValueError("Only reviewed comparison evidence can be delivered")
    for query_id in QUERY_IDS:
        query = reviewed.get("queries", {}).get(query_id)
        if not query or not isinstance(query.get("rows"), list) or not query["rows"] or not query.get("source"):
            raise ValueError(f"Missing reviewed rows or provenance: {query_id}")
    strategies = {row.get("strategy") for row in reviewed["queries"]["strategy_summary"]["rows"]}
    if not {"ppo_ensemble", "a2c_ensemble", "cash", "top10_equal", "top10_cap", "btc_eth_hold"}.issubset(strategies):
        raise ValueError("The two actual ensembles and all four shared baselines are required")
    if len(reviewed["queries"]["framework_comparison"]["rows"]) != 5:
        raise ValueError("The five requested framework reviews must all be present")
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
    ownership_path = project / ".rl-framework-authoring.json"
    previous = _json(ownership_path) if ownership_path.exists() else {}
    if not created and not previous:
        raise RuntimeError("Existing project has no comparison ownership record; preserving it")
    for relative, expected in previous.get("authored_sha256", {}).items():
        path = project / relative
        if not path.is_file() or _sha(path) != expected:
            raise RuntimeError(f"Report has independent edits; preserving them: {path}")
    content = project / "src/content/report"
    content.mkdir(parents=True, exist_ok=True)
    templates = {Path(__file__).with_suffix(".jsx"): content / "ReportContent.jsx", Path(__file__).with_suffix(".css"): content / "rl-framework-report.css"}
    for source, destination in templates.items():
        shutil.copyfile(source, destination)
    old = _json(project / "src/data.json")
    snapshot = copy.deepcopy(reviewed)
    snapshot["id"] = old["id"]
    if "legacyPresentationTitle" in old:
        snapshot["legacyPresentationTitle"] = old["legacyPresentationTitle"]
    snapshot["surface"], snapshot["buildStatus"] = "report", "complete"
    snapshot.setdefault("report", {})["officialReferences"] = [
        {"label": "Stable-Baselines3 PPO", "url": "https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html"},
        {"label": "Stable-Baselines3 A2C", "url": "https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html"},
        {"label": "Gymnasium自定义环境", "url": "https://gymnasium.farama.org/introduction/create_custom_env/"},
        {"label": "CMC历史市值快照", "url": "https://coinmarketcap.com/historical/"},
        {"label": "Binance公开行情说明", "url": "https://github.com/binance/binance-public-data"},
    ]
    for query_id in ("strategy_summary", "equity_history", "monthly_returns", "seed_statistics", "training_metrics"):
        query = snapshot["queries"][query_id]
        query.setdefault("methods", []).append({"language": "text", "code": "Display only: percentage aliases multiply reviewed fractional values by 100. Model curves use actual independently executed ensemble ledgers. The four baseline series are included once, after cross-study byte-equality verification. Source rows remain unmodified; no seed is selected using test results."})
    _write(project / "src/data.json", snapshot)
    authored = {str(path.relative_to(project)).replace("\\", "/"): _sha(path) for path in templates.values()}
    _write(ownership_path, {"authored_sha256": authored, "reviewed_sha256": _sha(reviewed_path)})
    build = _run(node, plugin / "scripts/data-app.mjs", "build", "--project-dir", str(project), "--separate-data", cwd=output_dir)
    export_path = project / ".data-app-offline/exports/rl-framework-comparison.html"
    exported = _run(node, plugin / "scripts/data-app.mjs", "export-offline", "--project-dir", str(project), "--output", str(export_path), cwd=output_dir)
    delivered = output_dir / "report.html"
    shutil.copyfile(export_path, delivered)
    return {"project_dir": str(project), "html_path": str(delivered), "dist_dir": str(project / "dist"),
            "preview_path": str(project / "dist/index.html"), "offline_path": str(export_path),
            "reviewed_sha256": _sha(reviewed_path), "delivered_sha256": _sha(delivered), "delivered_bytes": delivered.stat().st_size,
            "authoring_source_sha256": {path.name: _sha(path) for path in [Path(__file__), *templates]},
            "runtime": build, "export": exported, "published": False, "numeric_results_changed": False}
