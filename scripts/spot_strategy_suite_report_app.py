"""Build the source-backed local Data report without publishing anything.

Only a reviewed snapshot is accepted. The installed Data plugin supplies the
protected runtime; this module authors report content and a portable export.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path


NODE_DEFAULT = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe"
PLUGIN_DEFAULT = Path.home() / ".codex/plugins/cache/openai-curated-remote/data-analytics/1.0.11"
TEMPLATE_STEM = "spot_strategy_suite_report_app"
QUERY_IDS = ("strategy_summary", "equity_history", "monthly_returns")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _run(node: Path, script: Path, *arguments: str, cwd: Path) -> dict:
    process = subprocess.run([str(node), str(script), *map(str, arguments)], cwd=cwd,
                             capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    if process.returncode:
        raise RuntimeError(f"Data report command failed: {script.name}\n{process.stderr[-12000:]}\n{process.stdout[-3000:]}")
    for line in reversed(process.stdout.splitlines()):
        if line.strip().startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"Data report command returned no JSON result: {script.name}")


def build_report(output_dir: Path, node_path=None, plugin_root=None) -> dict:
    """Create/update report_app, build with split data, and export report.html."""
    output_dir = Path(output_dir).resolve()
    reviewed_path = output_dir / "reviewed.json"
    if not reviewed_path.is_file():
        raise FileNotFoundError(f"Reviewed evidence is required before building: {reviewed_path}")
    reviewed = _json(reviewed_path)
    if reviewed.get("status") != "reviewed":
        raise ValueError("Only status=reviewed snapshots may be delivered")
    for query_id in QUERY_IDS:
        query = reviewed.get("queries", {}).get(query_id)
        if not query or not isinstance(query.get("rows"), list) or not query["rows"] or not query.get("source"):
            raise ValueError(f"Missing reviewed rows or provenance: {query_id}")
    node = Path(node_path) if node_path else NODE_DEFAULT
    plugin = Path(plugin_root) if plugin_root else PLUGIN_DEFAULT
    if not node.is_file() or not (plugin / "scripts/data-app.mjs").is_file():
        raise FileNotFoundError("The pinned Codex Node runtime and installed Data plugin are required")
    project = output_dir / "report_app"
    created = not project.exists()
    if created:
        preparation = _run(node, plugin / "scripts/prepare-data-app.mjs", "--surface", "report",
                           "--output", str(project), "--snapshot", str(reviewed_path), cwd=output_dir)
    else:
        preparation = _run(node, plugin / "scripts/data-app.mjs", "prepare", "--project-dir", str(project), cwd=output_dir)
    # Consume the exact copied guide/current reference; neither is modified.
    guide = (project / "AGENTS.md").read_text(encoding="utf-8")
    reference = Path(preparation["documentation"]["entryPoint"])
    reference.read_text(encoding="utf-8")
    for name in ("reports.md", "charts.md", "tables.md", "controls.md"):
        (reference.parent / name).read_text(encoding="utf-8")
    if "src/content/" not in guide:
        raise RuntimeError("Unexpected Data app authoring boundary")
    ownership_path = project / ".spot-suite-authoring.json"
    previous = _json(ownership_path) if ownership_path.exists() else {}
    if not created and not previous:
        raise RuntimeError("Existing report_app has no suite ownership record; preserve it for manual review")
    for relative, expected in previous.get("authored_sha256", {}).items():
        path = project / relative
        if not path.is_file() or _sha(path) != expected:
            raise RuntimeError(f"Authored report was edited outside the builder; preserving edits: {path}")
    content = project / "src/content/report"
    content.mkdir(parents=True, exist_ok=True)
    templates = {
        Path(__file__).with_suffix(".jsx"): content / "ReportContent.jsx",
        Path(__file__).with_suffix(".css"): content / "spot-suite-report.css",
    }
    for source, destination in templates.items():
        shutil.copyfile(source, destination)
    old_snapshot = _json(project / "src/data.json")
    snapshot = reviewed
    snapshot["id"] = old_snapshot["id"]
    if "legacyPresentationTitle" in old_snapshot:
        snapshot["legacyPresentationTitle"] = old_snapshot["legacyPresentationTitle"]
    snapshot["surface"] = "report"
    snapshot["buildStatus"] = "complete"
    config_path = output_dir / "config.json"
    config = _json(config_path) if config_path.exists() else {}
    snapshot.setdefault("report", {})["suiteConfig"] = config
    source_path = output_dir / "sources/sources.json"
    if source_path.exists():
        references = [row for row in _json(source_path).get("sources", []) if row.get("kind") == "strategy_reference"]
        snapshot["report"]["strategySources"] = references
        for query_id in QUERY_IDS:
            source = snapshot["queries"][query_id]["source"]
            links = source.setdefault("links", [])
            existing = {link.get("url") for link in links}
            for item in references:
                if item["source_url"] not in existing:
                    links.append({"label": f"{item['repository']} · {item['repo_path']}", "url": item["source_url"]})
            source.setdefault("files", []).append("sources/sources.json")
    for query_id in QUERY_IDS:
        snapshot["queries"][query_id].setdefault("methods", []).append({
            "language": "text", "code": "Display-only percentage aliases equal the reviewed fractional return/drawdown multiplied by 100. UTC daily curve points remain unchanged; headline maximum drawdown retains the reviewed 1-minute calculation."
        })
    _write(project / "src/data.json", snapshot)
    authored = {str(destination.relative_to(project)).replace("\\", "/"): _sha(destination) for destination in templates.values()}
    _write(ownership_path, {"authored_sha256": authored, "reviewed_sha256": _sha(reviewed_path)})
    build = _run(node, plugin / "scripts/data-app.mjs", "build", "--project-dir", str(project), "--separate-data", cwd=output_dir)
    export_path = project / ".data-app-offline/exports/spot-strategy-suite.html"
    exported = _run(node, plugin / "scripts/data-app.mjs", "export-offline", "--project-dir", str(project), "--output", str(export_path), cwd=output_dir)
    delivered = output_dir / "report.html"
    shutil.copyfile(export_path, delivered)
    return {
        "project_dir": str(project), "html_path": str(delivered), "dist_dir": str(project / "dist"),
        "preview_path": str(project / "dist/index.html"), "offline_path": str(export_path),
        "reviewed_sha256": _sha(reviewed_path), "delivered_sha256": _sha(delivered),
        "authoring_source_sha256": {source.name: _sha(source) for source in [Path(__file__), *templates]},
        "runtime": build, "export": exported, "published": False,
    }
