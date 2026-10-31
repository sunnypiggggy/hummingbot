"""Deliver historical experiment evidence while the new rolling study runs.

This is a separate historical artifact, not an incomplete rolling-results report.
It never reads new annual accounts, fabricates outcomes, or relaxes the final gate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import all_algorithms_rolling_report_app as rolling
from scripts import rolling_single_pair_remote_ops as ops
from scripts.btceth_rl_report_app import DataRuntime, host_path

SCOPE = "all_experiments_historical_summary_and_frozen_rolling_method"


def history_inputs(top):
    top = rolling.root_path(top)
    required = ("inventory.json", "protocol.json", "protocol_receipt.json", "selection.json")
    pins = {n: rolling.sha(rolling.relative(top, n)) for n in required}
    spec, inventory, selection = (rolling.read(top / n) for n in ("protocol.json", "inventory.json", "selection.json"))
    rolling.verify_protocol_contract(spec)
    rolling.require(rolling.read(top / "protocol_receipt.json").get("protocol_sha256") == pins["protocol.json"], "Frozen protocol receipt mismatch")
    rolling.require(spec["inventory_sha256"] == rolling.INVENTORY_SHA256 == pins["inventory.json"] and spec["selection_sha256"] == pins["selection.json"], "Historical inventory or frozen shortlist changed")
    rolling.require(selection.get("uses_test") is False and selection.get("seed_selection") is False and selection.get("selected") == spec["selected"], "Validation-only frozen shortlist required")
    for name, digest in selection["source_sha256"].items():
        rolling.require(rolling.sha(rolling.relative(top, name)) == digest, "Frozen selection input changed")
        pins[name] = digest
    for name, digest in spec["sources"].items():
        source_name = "source/" + name
        rolling.require(rolling.sha(rolling.relative(top, source_name)) == digest, "Frozen training source changed")
        pins[source_name] = digest
    external = dict(inventory["source_sha256"])
    script = "results/analysis/all_experiments_rolling_20261031/summarize_existing_experiments_v2.py"
    external[script] = inventory["analysis_script_sha256"]
    for name, digest in external.items():
        rolling.require(rolling.sha(rolling.relative(ROOT, name)) == digest, "Historical source changed: " + name)
    candidates = rolling.validation_rows(inventory)
    frozen = {(r["symbol"], r["candidate_id"]): r for r in selection["candidates"]}
    rolling.require(set(frozen) == {(r["symbol"], r["candidate_id"]) for r in candidates}, "Frozen candidate population differs from original validation evidence")
    shortlist = []
    for symbol, recipes in rolling.EXPECTED.items():
        rolling.require(selection["rankings"][symbol][:3] == inventory["rankings"][symbol]["by_robust_score"][:3] == [rolling.INVENTORY_IDS[r] for r in recipes], "Frozen validation shortlist mismatch")
        for rank, recipe in enumerate(recipes, 1):
            item = next(r for r in candidates if r["symbol"] == symbol and r["candidate_id"] == rolling.INVENTORY_IDS[recipe])
            rolling.require(abs(item["robust_score"] - frozen[(symbol, item["candidate_id"])]["robust_score"]) < 1e-14, "Frozen candidate score mismatch")
            rolling.require(json.loads(item["recipe_json"]) == spec["recipes"][recipe]["parameters"], "Candidate training parameters changed")
            shortlist.append({**item, "recipe_id": recipe, "rank": rank, "recipe_json": json.dumps(spec["recipes"][recipe]["parameters"], ensure_ascii=False, sort_keys=True)})
    return spec, inventory, selection, pins, external, candidates, shortlist


def flatten_parameters(shortlist):
    rows = []
    def leaves(value, prefix=""):
        if isinstance(value, dict):
            for key, child in value.items():
                yield from leaves(child, prefix + "." + key if prefix else key)
        else:
            yield prefix, json.dumps(value, ensure_ascii=False, allow_nan=False)
    for item in shortlist:
        for key, value in leaves(json.loads(item["recipe_json"])):
            rows.append({"key": item["key"] + ":" + key, "symbol": item["symbol"], "recipe_id": item["recipe_id"], "parameter": key, "value": value})
    return rows


def capture_status():
    captured = datetime.now(timezone.utc).isoformat()
    try:
        data = ops.remote_request("status", {}, timeout=45)
        rolling.require(data.get("remote_root") == ops.REMOTE_ROOT, "Status belongs to another study")
        return {"captured_at_utc": captured, "status_available": True, "read_only_status": data}
    except Exception as error:
        return {"captured_at_utc": captured, "status_available": False, "error_type": type(error).__name__, "error": str(error)}


def make_history_snapshot(top, status, status_source):
    spec, inventory, selection, pins, external, candidates, shortlist = history_inputs(top)
    pins[status_source] = rolling.sha(rolling.relative(top, status_source))
    framework_name = "results/backtests/rl_framework_comparison_20261024/framework_review_rows.json"
    provenance_name = "results/backtests/crypto_foundation_models_v100_20261024/model_provenance_evaluation.json"
    framework, provenance = rolling.read(rolling.relative(ROOT, framework_name)), rolling.read(rolling.relative(ROOT, provenance_name))
    external[framework_name], external[provenance_name] = rolling.sha(ROOT / framework_name), rolling.sha(ROOT / provenance_name)
    framework_rows = [{k: r[k] for k in ("name", "task", "decision", "native_backtest_status", "commit", "repository") if k in r} for r in framework["rows"]]
    provenance_rows = [{k: r[k] for k in ("model_key", "model", "source_repository", "source_commit", "weights_revision", "weights_sha256", "first_usable_decision_utc", "local_fit", "pretraining_cutoff_independently_verified", "checkpoint_dates_note") if k in r} for r in provenance["rows"]]
    historical = rolling.historical_rows(inventory)
    families = [{"study": f["study"], "scope": f["scope"], "local_present": f["local_present"], "summary_sets": len(f.get("summary_sets", [])), "recorded_audits": len(f.get("audits", []))} for f in inventory["families"]]
    raw = status.get("read_only_status", {})
    phase = raw.get("markers", {}).get("pipeline_status.json", {})
    research = [{"captured_at_utc": status["captured_at_utc"], "status_available": status["status_available"], "status": phase.get("status", "unavailable"), "phase": phase.get("phase"),
        "models_started": raw.get("models", {}).get("started"), "models_completed": raw.get("models", {}).get("completed"), "expected_models": 1590, "expected_accounts": 180,
        "new_financial_results_included": False, "new_financial_results_status": "not_included_in_historical_artifact", "active_recorded_phases": raw.get("active_recorded_phases", [])}]
    base = {"classification": "modeled", "label": "原实验的只读历史总结与本轮冻结方案", "timezone": "UTC", "period": {"start": rolling.TEST[0], "end": rolling.TEST[1]},
        "metricDefinitions": [{"label": "历史扣费收益", "definition": "按各原账户资金分母计算的期末净值变化；保留手续费、不利滑点与原年度或验证窗口。"}, {"label": "最大回撤", "definition": "按各原实验记录的采样粒度及历史峰值计算，显示绝对跌幅；独立六算法与扩30seed为小时收盘，其他以原来源为准。"}, {"label": "seed分布", "definition": "原六算法用5个seed、扩展SAC/TD3用30个seed；固定规则单账户不伪造seed分布，跨seed不是独立行情样本。"}],
        "caveats": ["本页只总结已经完成的旧实验；不展示新固定/4周/每周重训的收益或排名。", "旧年度行情已经被研究观察，本轮不是新的市场盲测。", "共享Top10或BTC/ETH组合不等于每币独立10,000账户；不同资金和标的不能直接按收益排算法。", "SAC原配方seed100..104是重复实验，不能作为不同超参数占一个候选名额。", "候选仅用原验证集的净收益、回撤和seed分散评分选择；入选研究不等于通过原实盘替换门槛。", "历史审计状态来自原记录，本页核对其文件哈希与统计，不声称重跑过全部旧账本。"],
        "evidenceFlow": [{"title": "历史结果保持原身份", "detail": "保留18实验族资金、标的、窗口、seed数、实际规则或算法及每一份来源SHA。"}, {"title": "参数只从验证期选择", "detail": "新研究冻结selection通过原五份验证输入重新计算，前3候选与9不同配方排名对账。"}, {"title": "新回测仍在单独运行", "detail": "进度只读快照附采集时间；本历史页面不读取新年度账户，不生成任何新滚动收益。"}]}
    queries = {}
    anchors = ["inventory.json", "protocol.json", "protocol_receipt.json", "selection.json"]
    def query(key, rows, files, method, *, period=rolling.TEST, external_files=()):
        source = deepcopy(base)
        if key in ("framework_reviews", "foundation_provenance", "rolling_method", "candidate_parameters"):
            source["classification"] = "source_review"
        elif key == "research_status":
            source["classification"] = "observed"
        names = list(dict.fromkeys([*files, *anchors]))
        source.update(files=[*names, *["workspace/" + n for n in external_files]], fileHashes={**{n: pins[n] for n in names}, **{"workspace/" + n: external[n] for n in external_files}}, period={"start": period[0], "end": period[1]})
        queries[key] = {"rows": rows, "source": source, "methods": [{"language": "text", "code": method}]}
    query("historical_results", historical, ["inventory.json"], "Each original family/cost/split/configuration retains account funding, seed count and source SHA; no mixed-family ranking.", period=(rolling.VALIDATION[0], rolling.TEST[1]))
    query("experiment_families", families, ["inventory.json"], "18 experiment families with original scope and recorded audits; derived comparison studies may cite the same accounts.")
    query("validation_candidates", candidates, ["inventory.json"], "Nine distinct recipes per coin, original5-seed verification statistics; all-zero-fill R-PPO separately labeled.", period=rolling.VALIDATION)
    query("candidate_selection", shortlist, ["selection.json"], "Same frozen validation-only robust-score top3 per coin, exact parameters and steps; annual results do not select candidates.", period=rolling.VALIDATION)
    query("candidate_parameters", flatten_parameters(shortlist), ["protocol.json"], "Every scalar/list of every shortlisted coin/recipe's exact native parameters, using dot paths and JSON literals.", period=rolling.VALIDATION)
    query("sac_replication", inventory["sac_baseline_replication"], ["inventory.json"], "SAC default seeds100..104 remain a second replication, not a different hyperparameter recipe.", period=rolling.VALIDATION)
    query("expanded_seed_results", rolling.expanded_seed_rows(inventory), ["inventory.json"], "OriginalSAC/TD3 all30-seed validation and365-day results kept separately from native5-seed cohorts.", period=(rolling.VALIDATION[0], rolling.TEST[1]))
    query("framework_reviews", framework_rows, [], "Five original-code adaptation reviews; no native comparable backtest results are claimed.", external_files=(framework_name,))
    query("foundation_provenance", provenance_rows, [], "Three pinned pretrained model provenance records; exact complete pretraining data cutoff remains unverified.", external_files=(provenance_name,))
    query("research_status", research, [status_source], "A timestamped read-only status snapshot; progress counts are not performance or completion claims.", period=(status["captured_at_utc"], status["captured_at_utc"]))
    query("rolling_method", [{"protocol_sha256": pins["protocol.json"], "train_days": spec["train_days"], "lag_days": spec["lag_days"], "preheat_days": spec["preheat_days"], "capital_per_account": spec["capital_per_account"], "expected_models": spec["expected_models"], "expected_accounts": spec["expected_accounts"], "seeds": spec["seeds"], "cadences": spec["cadences"], "protocol_json": json.dumps(spec, ensure_ascii=False, sort_keys=True)}], ["protocol.json"], "Frozen new experiment design only; no new rolling account outcome included.")
    queries["rolling_method"]["payloadColumns"] = ["protocol_json"]
    snapshot = {"surface": "report", "status": "reviewed", "buildStatus": "complete", "title": "全部实验回顾：BTC与ETH候选，以及三种重训频率方案", "generatedAt": status["captured_at_utc"], "filters": [], "queries": queries,
        "report": {"scope": SCOPE, "historyPreparedAtUtc": status["captured_at_utc"], "historicalMarketCutoff": "2026-10-22", "rollingFinancialResultsIncluded": False, "studyConfig": spec}}
    evidence = {"input_sha256": pins, "external_input_sha256": external, "report_source_sha256": {p.relative_to(ROOT).as_posix(): rolling.sha(p) for p in (Path(__file__), Path(__file__).with_suffix(".jsx"), Path(__file__).with_suffix(".css"), ROOT / "scripts/all_algorithms_rolling_report_app.py", ROOT / "scripts/btceth_rl_report_app.py", ROOT / "scripts/crypto_rl_reporting.py", ROOT / "scripts/rolling_single_pair_remote_ops.py")}, "query_counts": {k: len(v["rows"]) for k, v in queries.items()}, "new_financial_results_included": False, "completion_gate_relaxed": False, "no_training_inference_or_production_change": True}
    rolling.verify_inputs(top, evidence)
    return snapshot, evidence


def build_history(study_root, *, output_dir=None, status_snapshot=None, build_mode="wsl", wsl_node=None, wsl_distro="Ubuntu-26.04", node_path=None, plugin_root=None):
    top = rolling.root_path(study_root)
    # Verify historical sources before creating even the separate history output.
    history_inputs(top)
    destination = rolling.root_path(output_dir) if output_dir else top / "history_report"
    rolling.require(destination != top / "report" and not destination.is_relative_to(top / "report"), "Historical output cannot replace the final audit-gated report")
    project = destination / "report_app"
    owner = project / ".all-experiments-history-authoring.json"
    old = rolling.read(owner) if owner.exists() else None
    if project.exists():
        rolling.require(old and old.get("scope") == SCOPE, "Preserve unowned history app")
        for name, digest in old.get("authored_sha256", {}).items():
            rolling.require(rolling.sha(rolling.relative(project, name)) == digest, "Preserve edited historical presentation")
    else:
        rolling.require(not destination.exists() or not any(destination.iterdir()), "Preserve nonempty unowned history destination")
    destination.mkdir(parents=True, exist_ok=True)
    status = rolling.read(status_snapshot) if status_snapshot else capture_status()
    captured_name = "history_report/preparation_status.json" if destination == top / "history_report" else None
    rolling.require(captured_name is not None, "This history artifact uses its own named study child")
    rolling.write(top / captured_name, status)
    snapshot, evidence = make_history_snapshot(top, status, captured_name)
    reviewed = destination / "reviewed.json"
    rolling.write(reviewed, snapshot)
    rolling.write(destination / "report_input_manifest.json", evidence)
    runtime = DataRuntime(node_path=node_path, plugin_root=plugin_root, build_mode=build_mode, wsl_node=wsl_node, wsl_distro=wsl_distro)
    prepared = runtime.run("data-app.mjs", "prepare", "--project-dir", project, cwd=destination) if project.exists() else runtime.run("prepare-data-app.mjs", "--surface", "report", "--output", project, "--snapshot", reviewed, cwd=destination)
    rolling.require("src/content/" in (project / "AGENTS.md").read_text(encoding="utf-8"), "Unexpected official authoring boundary")
    docs = prepared["documentation"]
    host_path(docs["entryPoint"]).read_text(encoding="utf-8")
    refs = host_path(docs["referenceRoot"])
    for name in ("narrative/reports.md", "charts/charts.md", "tables/tables.md", "controls/controls.md", "sources/sources.md"):
        (refs / name).read_text(encoding="utf-8")
    targets = {Path(__file__).with_suffix(".jsx"): project / "src/content/report/ReportContent.jsx", Path(__file__).with_suffix(".css"): project / "src/content/report/all-algorithms-history.css"}
    for source, target in targets.items():
        shutil.copyfile(source, target)
    app = rolling.read(project / "src/data.json")
    snapshot["id"] = app["id"]
    if "legacyPresentationTitle" in app:
        snapshot["legacyPresentationTitle"] = app["legacyPresentationTitle"]
    rolling.write(project / "src/data.json", snapshot)
    authored = {p.relative_to(project).as_posix(): rolling.sha(p) for p in targets.values()}
    rolling.write(owner, {"scope": SCOPE, "authored_sha256": authored})
    built = runtime.run("data-app.mjs", "build", "--project-dir", project, "--separate-data", cwd=destination)
    exported = project / ".data-app-offline/exports/all-experiments-history.html"
    export = runtime.run("data-app.mjs", "export-html", "--project-dir", project, "--output", exported, cwd=destination)
    rolling.verify_inputs(top, evidence)
    delivered = destination / "report.html"
    shutil.copyfile(exported, delivered)
    result = {"scope": SCOPE, "html_path": str(delivered), "preview_path": str(project / "dist/index.html"), "project_dir": str(project), "reviewed_sha256": rolling.sha(reviewed), "html_sha256": rolling.sha(delivered), "runtime": built, "export": export, "query_counts": evidence["query_counts"], "report_source_sha256": evidence["report_source_sha256"], "new_financial_results_included": False, "completion_gate_relaxed": False, "published": False}
    rolling.write(destination / "report_build.json", result)
    rolling.write(owner, {"scope": SCOPE, "authored_sha256": authored})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--status-snapshot", type=Path)
    p.add_argument("--build-mode", choices=("native", "wsl"), default="wsl")
    p.add_argument("--wsl-node", required=True)
    p.add_argument("--wsl-distro", default="Ubuntu-26.04")
    a = p.parse_args()
    print(json.dumps(build_history(a.study_root, output_dir=a.output_dir, status_snapshot=a.status_snapshot, build_mode=a.build_mode, wsl_node=a.wsl_node, wsl_distro=a.wsl_distro), ensure_ascii=False))


if __name__ == "__main__":
    main()
