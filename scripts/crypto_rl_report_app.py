"""Build the isolated crypto PPO report with the protected Data runtime."""
from __future__ import annotations

import copy
from pathlib import Path
import shutil

from scripts.spot_strategy_suite_report_app import NODE_DEFAULT, PLUGIN_DEFAULT, _json, _run, _sha, _write


QUERY_IDS = ("strategy_summary", "equity_history", "monthly_returns", "universe_history", "training_metrics")


def build_report(output_dir: Path, node_path=None, plugin_root=None) -> dict:
    output_dir = Path(output_dir).resolve()
    reviewed_path = output_dir / "reviewed.json"
    reviewed = _json(reviewed_path)
    if reviewed.get("status") != "reviewed":
        raise ValueError("Only reviewed RL evidence can be delivered")
    for query_id in QUERY_IDS:
        query = reviewed.get("queries", {}).get(query_id)
        if not query or not isinstance(query.get("rows"), list) or not query["rows"] or not query.get("source"):
            raise ValueError(f"Missing reviewed rows or provenance: {query_id}")
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
    ownership_path = project / ".crypto-rl-authoring.json"
    previous = _json(ownership_path) if ownership_path.exists() else {}
    if not created and not previous:
        raise RuntimeError("Existing project has no RL report ownership record; preserving it")
    for relative, expected in previous.get("authored_sha256", {}).items():
        path = project / relative
        if not path.is_file() or _sha(path) != expected:
            raise RuntimeError(f"Report has independent edits; preserving them: {path}")
    content = project / "src/content/report"
    content.mkdir(parents=True, exist_ok=True)
    templates = {Path(__file__).with_suffix(".jsx"): content / "ReportContent.jsx", Path(__file__).with_suffix(".css"): content / "crypto-rl-report.css"}
    for source, destination in templates.items():
        shutil.copyfile(source, destination)
    old = _json(project / "src/data.json")
    snapshot = copy.deepcopy(reviewed)
    snapshot["id"] = old["id"]
    if "legacyPresentationTitle" in old:
        snapshot["legacyPresentationTitle"] = old["legacyPresentationTitle"]
    snapshot["surface"], snapshot["buildStatus"] = "report", "complete"
    config_path = output_dir / "config.json"
    snapshot.setdefault("report", {})["studyConfig"] = _json(config_path) if config_path.exists() else {}
    official_links = [
        {"label": "Stable-Baselines3 PPO", "url": "https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html"},
        {"label": "Gymnasium自定义环境", "url": "https://gymnasium.farama.org/introduction/create_custom_env/"},
        {"label": "CMC历史市值快照", "url": "https://coinmarketcap.com/historical/"},
        {"label": "Binance官方公开行情说明", "url": "https://github.com/binance/binance-public-data"},
        {"label": "Binance 2023-03-24恢复现货公告", "url": "https://www.binance.com/en/support/announcement/detail/813a31506e9f478ea8c1058b425df87a"},
    ]
    snapshot["report"]["officialReferences"] = official_links
    for query in snapshot["queries"].values():
        source = query["source"]
        source["files"] = list(dict.fromkeys([*source.get("files", []), "METHODOLOGY.md", "sources/data_manifest.json", "frozen_spec.json", "training_frozen_inputs.json"]))
        source["links"] = list(source.get("links", []))
        known_urls = {item.get("url") for item in source["links"]}
        source["links"].extend(item for item in official_links if item["url"] not in known_urls)
        source["caveats"] = list(dict.fromkeys([*source.get("caveats", []),
            "名义训练936天；剔除跨真实缺失小时的128日片段后，可用片段并集覆盖853天，不代表每个种子均遍历全部853天。验证和测试未删除日期。",
            "需要31个完整历史UTC日及有效成交量特征才允许投资；池内10个资产槽位不等于任何时刻实际持有10币。",
            "使用小数币数量，未重建历史数量步长、最小下单量和最小名义金额。",
        ]))
        query.setdefault("methods", []).append({
            "language": "text", "code": "Only plotted percentage aliases multiply reviewed fractional returns by 100. Daily curves contain actual UTC day-end valuations; summary maximum drawdown uses every hourly close and initial capital. Seed median is descriptive across runs, never a synthetic tradable equity curve."
        })
    _write(project / "src/data.json", snapshot)
    authored = {str(path.relative_to(project)).replace("\\", "/"): _sha(path) for path in templates.values()}
    _write(ownership_path, {"authored_sha256": authored, "reviewed_sha256": _sha(reviewed_path)})
    build = _run(node, plugin / "scripts/data-app.mjs", "build", "--project-dir", str(project), "--separate-data", cwd=output_dir)
    export_path = project / ".data-app-offline/exports/crypto-rl-portfolio.html"
    exported = _run(node, plugin / "scripts/data-app.mjs", "export-offline", "--project-dir", str(project), "--output", str(export_path), cwd=output_dir)
    delivered = output_dir / "report.html"
    shutil.copyfile(export_path, delivered)
    return {"project_dir": str(project), "html_path": str(delivered), "dist_dir": str(project / "dist"),
            "preview_path": str(project / "dist/index.html"), "offline_path": str(export_path),
            "reviewed_sha256": _sha(reviewed_path), "delivered_sha256": _sha(delivered),
            "authoring_source_sha256": {path.name: _sha(path) for path in [Path(__file__), *templates]},
            "runtime": build, "export": exported, "published": False}
