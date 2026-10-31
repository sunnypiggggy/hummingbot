"""Build the second, independent spot-strategy study as an offline Data report.

Only reviewed research artifacts are read. Existing report identities and
authored edits are preserved; the first study and all trading state are untouched.
"""
from __future__ import annotations

import copy
from pathlib import Path
import shutil

from scripts.spot_strategy_suite_report_app import (
    NODE_DEFAULT, PLUGIN_DEFAULT, _json, _run, _sha, _write,
)


QUERY_IDS = ("strategy_summary", "equity_history", "monthly_returns", "previous_summary")


def build_report(output_dir: Path, node_path=None, plugin_root=None) -> dict:
    """Prepare reviewed data, build protected runtime, and deliver report.html."""
    output_dir = Path(output_dir).resolve()
    reviewed_path = output_dir / "reviewed.json"
    reviewed = _json(reviewed_path)
    if reviewed.get("status") != "reviewed":
        raise ValueError("Second-round report requires a reviewed snapshot")
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
        preparation = _run(node, plugin / "scripts/prepare-data-app.mjs", "--surface", "report",
                           "--output", str(project), "--snapshot", str(reviewed_path), cwd=output_dir)
    else:
        preparation = _run(node, plugin / "scripts/data-app.mjs", "prepare", "--project-dir", str(project), cwd=output_dir)
    guide = (project / "AGENTS.md").read_text(encoding="utf-8")
    reference = Path(preparation["documentation"]["entryPoint"])
    reference.read_text(encoding="utf-8")
    for name in ("reports.md", "charts.md", "tables.md", "controls.md"):
        (reference.parent / name).read_text(encoding="utf-8")
    if "src/content/" not in guide:
        raise RuntimeError("Unexpected Data app authoring boundary")
    ownership_path = project / ".spot-round2-authoring.json"
    previous = _json(ownership_path) if ownership_path.exists() else {}
    if not created and not previous:
        raise RuntimeError("Existing report has no round-two ownership record; preserving it")
    for relative, expected in previous.get("authored_sha256", {}).items():
        path = project / relative
        if not path.is_file() or _sha(path) != expected:
            raise RuntimeError(f"Report content was independently edited; preserving it: {path}")
    content = project / "src/content/report"
    content.mkdir(parents=True, exist_ok=True)
    templates = {
        Path(__file__).with_suffix(".jsx"): content / "ReportContent.jsx",
        Path(__file__).with_suffix(".css"): content / "spot-round2-report.css",
    }
    for source, destination in templates.items():
        shutil.copyfile(source, destination)
    old = _json(project / "src/data.json")
    snapshot = copy.deepcopy(reviewed)
    snapshot["id"] = old["id"]
    if "legacyPresentationTitle" in old:
        snapshot["legacyPresentationTitle"] = old["legacyPresentationTitle"]
    snapshot["surface"] = "report"
    snapshot["buildStatus"] = "complete"
    previous_source = snapshot["queries"]["previous_summary"]["source"]
    comparison_detail = (
        "第一轮前四组各10,000 USDT，BTC/ETH各5,000独立资金；第五组轮动共用10,000 USDT。"
        "第一轮与本轮采用相同固定365天窗口、入场现金上限和两档研究成本；旧结果只读引用。"
    )
    flow = previous_source.setdefault("evidenceFlow", [])
    comparison_step = next((step for step in flow if step.get("title") == "比较"), None)
    if comparison_step is None:
        flow.append({"title": "比较", "detail": comparison_detail})
    else:
        comparison_step["detail"] = comparison_detail
    old_caveats = previous_source.setdefault("caveats", [])
    selection_caveat = "两轮均为同一年历史样本上的候选筛选，不构成新增样本外验证。"
    if selection_caveat not in old_caveats:
        old_caveats.append(selection_caveat)
    config_path = output_dir / "config.json"
    snapshot.setdefault("report", {})["suiteConfig"] = _json(config_path) if config_path.exists() else {}
    source_path = output_dir / "sources/sources.json"
    if source_path.exists():
        source_document = _json(source_path)
        references = source_document.get("sources", []) if isinstance(source_document, dict) else source_document
        references = [r for r in references if r.get("source_url")]
        for item in references:
            primary = next((reference for reference in item.get("references", [])
                            if reference.get("url") == item["source_url"]), None)
            if primary and primary.get("summary"):
                item["preview"] = primary["summary"]
        snapshot["report"]["strategySources"] = references
        for query_id in QUERY_IDS[:3]:
            source = snapshot["queries"][query_id]["source"]
            source["links"] = list(source.get("links", []))
            existing = {link.get("url") for link in source["links"]}
            for reference in references:
                if reference["source_url"] not in existing:
                    source["links"].append({"label": reference.get("label", reference.get("title", reference.get("strategy_id", "策略原始参考"))), "url": reference["source_url"]})
                    existing.add(reference["source_url"])
            source["files"] = list(dict.fromkeys([*source.get("files", []), "sources/sources.json"]))
    for query_id in QUERY_IDS:
        snapshot["queries"][query_id].setdefault("methods", []).append({
            "language": "text",
            "code": "Fractional returns and drawdowns are multiplied by 100 only in plotted percentage aliases. Curves show UTC daily close boundaries; maximum drawdown remains the independently recorded complete 1-minute metric. Round 1 is a comparison on the same historical window, not an out-of-sample validation."
        })
    _write(project / "src/data.json", snapshot)
    authored = {str(path.relative_to(project)).replace("\\", "/"): _sha(path) for path in templates.values()}
    _write(ownership_path, {"authored_sha256": authored, "reviewed_sha256": _sha(reviewed_path)})
    build = _run(node, plugin / "scripts/data-app.mjs", "build", "--project-dir", str(project), "--separate-data", cwd=output_dir)
    export_path = project / ".data-app-offline/exports/spot-strategy-round2.html"
    exported = _run(node, plugin / "scripts/data-app.mjs", "export-offline", "--project-dir", str(project), "--output", str(export_path), cwd=output_dir)
    delivered = output_dir / "report.html"
    shutil.copyfile(export_path, delivered)
    return {
        "project_dir": str(project), "html_path": str(delivered), "dist_dir": str(project / "dist"),
        "preview_path": str(project / "dist/index.html"), "offline_path": str(export_path),
        "reviewed_sha256": _sha(reviewed_path), "delivered_sha256": _sha(delivered),
        "authoring_source_sha256": {path.name: _sha(path) for path in [Path(__file__), *templates]},
        "runtime": build, "export": exported, "published": False,
    }
