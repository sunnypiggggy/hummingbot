"""Read-only assembly of audited pretrained forecasts and saved RL controls.

No trainer, forecaster, ledger simulator or model deserializer is imported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.build_rl_ex_ada_comparison import (
    Inputs, DATASETS, BASELINES, records, require_audit, read, sha, write, _run_artifacts,
)
from scripts.foundation_model_proof_paths import ProofPaths


NEW_ROOT = ROOT / "results/backtests/crypto_foundation_models_20261024"
V100_ROOT = ROOT / "results/backtests/crypto_foundation_models_v100_20261024"
OUTPUT = NEW_ROOT / "report"
LEGACY_ROOT = ROOT / "results/backtests/crypto_rl_elegantrl_20261024"
LEGACY_REPORT = LEGACY_ROOT / "report"
MODEL_LABELS = {"kronos_small": "Kronos-small", "kronos_base": "Kronos-base", "chronos2": "Chronos-2"}
FRAME_MAP = {"strategy_summary": "summary", "equity_history": "daily_equity",
             "monthly_returns": "monthly_returns", "seed_statistics": "seed_statistics",
             "training_metrics": "training_metrics", "universe_changes": "universe_changes",
             "model_provenance": "model_provenance", "forecast_diagnostics": "forecast_diagnostics",
             "implementation_review": "implementation_review"}
OWNED_NAMES = tuple(name + ".csv" for name in FRAME_MAP.values()) + (
    "reviewed.json", "README.md", "input_integrity_before.json", "input_integrity_after.json")
COMMON_CONFIG = ("validation_start", "validation_end", "test_start", "test_end", "costs", "capital",
                 "cash_target_floor", "single_coin_target_cap")


def primary(frame):
    answer = frame.strategy.astype(str).str.endswith("_ensemble") if "strategy" in frame else pd.Series(False, index=frame.index)
    if "primary_policy" in frame:
        answer |= frame.primary_policy.eq(True)
    if "policy_role" in frame:
        answer |= frame.policy_role.eq("primary")
    return answer


def _check_output(output, study_root):
    if study_root not in {NEW_ROOT.resolve(), V100_ROOT.resolve()} or output != study_root / "report":
        raise ValueError("Only the separate foundation-model report is writable")
    path = output / "manifest.json"
    if not path.exists():
        if any((output / name).exists() for name in OWNED_NAMES):
            raise ValueError("Preserving existing report files without an ownership manifest")
        return
    previous = read(path)
    if previous.get("assembler") != Path(__file__).name:
        raise ValueError("Preserving a report owned by another assembler")
    for name, expected in previous.get("generated_sha256", {}).items():
        if sha(output / name) != expected:
            raise ValueError("Preserving independent report edits: " + name)


def _preservation(inputs, study_root, proof):
    path = proof.local_evidence("original_research_preservation.json")
    contract = inputs.json(path)
    if not contract.get("files"):
        raise ValueError("The original-research protection contract is empty")
    for relative, expected in contract["files"].items():
        path = inputs.track(ROOT / relative.replace("\\", "/"))
        if path.stat().st_size != expected["size_bytes"] or inputs.files[path.as_posix()] != expected["sha256"]:
            raise ValueError("An existing research artifact changed: " + relative)
    for relative in contract.get("originally_absent", []):
        if (ROOT / relative.replace("\\", "/")).exists():
            raise ValueError("A historical artifact appeared after protection freeze: " + relative)
    return {"files_checked": len(contract["files"]), "passed": True}


def _legacy(inputs):
    """Verify and reuse the previously delivered complete 272-run comparison."""
    manifest = inputs.json(LEGACY_REPORT / "manifest.json")
    delivery = inputs.json(LEGACY_ROOT / "delivery_verification.json")
    if manifest.get("status") != "complete" or delivery.get("status") != "passed" or manifest.get("global_independent_audit_status") != "passed":
        raise ValueError("The saved RL comparison is incomplete or unaudited")
    for path, expected in manifest["input_sha256"].items():
        item = inputs.track(Path(path))
        if inputs.files[item.as_posix()] != expected:
            raise ValueError("A saved RL source input changed: " + path)
    for filename, expected in manifest["generated_sha256"].items():
        path = inputs.track(LEGACY_REPORT / filename)
        if inputs.files[path.as_posix()] != expected:
            raise ValueError("A saved RL output changed: " + filename)
    audited = manifest.get("independent_audit_summaries", {})
    if len(audited) != 10 or any(r.get("status") != "passed" or r.get("checks_failed") != 0 or r.get("runs_checked") != 40 for r in audited.values()):
        raise ValueError("All ten prior RL study audits are required")
    reviewed = inputs.json(LEGACY_REPORT / "reviewed.json")
    if reviewed.get("status") != "reviewed":
        raise ValueError("The old source-scoped report is not reviewed")
    frames = {name: inputs.csv(LEGACY_REPORT / (name + ".csv")) for name in
              ("summary", "daily_equity", "monthly_returns", "seed_statistics", "training_metrics", "universe_changes", "implementation_review")}
    if len(frames["summary"]) != 272 or not frames["summary"].run_id.is_unique or len(frames["training_metrics"]) != 50 or len(frames["seed_statistics"]) != 40:
        raise ValueError("The complete 272-run/50-model RL comparison is required")
    # Preserve structured historical training evidence rather than CSV repr text.
    frames["training_metrics"] = pd.DataFrame(reviewed["queries"]["training_metrics"]["rows"])
    frames["implementation_review"] = pd.DataFrame(reviewed["queries"]["implementation_review"]["rows"])
    frames["universe_changes"] = pd.DataFrame(reviewed["queries"]["universe_changes"]["rows"])
    for name in ("summary", "daily_equity", "monthly_returns"):
        frame = frames[name]
        frame["primary_policy"] = primary(frame)
        frame["policy_role"] = frame.apply(lambda r: "baseline" if r.algorithm == "SHARED" else "primary" if r.primary_policy else "training_seed", axis=1)
        frame["model_key"] = frame.algorithm.astype(str).str.lower()
    old_manifests = {}
    for variant in DATASETS:
        folder = Path(manifest["source_studies"][variant + "__sb3__ppo"])
        old_manifests[variant] = inputs.json(folder / "manifest.json")
    return frames, manifest, reviewed, old_manifests


def _model_definitions(spec):
    models = spec.get("models")
    if not isinstance(models, dict) or set(models) != set(MODEL_LABELS):
        raise ValueError("The frozen protocol must declare the three requested pretrained models")
    result = {}
    for model_key, definition in models.items():
        seeds = next((definition[k] for k in ("sampling_seeds", "inference_seeds", "seeds") if k in definition), None)
        if seeds is None:
            raise ValueError("Sampling identity must be explicit in the frozen protocol: " + model_key)
        expected = [] if model_key == "chronos2" else list(range(5))
        if seeds != expected:
            raise ValueError("Chronos is one deterministic policy; each Kronos retains five declared sampling seeds")
        result[model_key] = {"definition": definition, "seeds": seeds, "runs": 4 * (5 + len(seeds))}
    return result


def _completion(inputs, definitions, study_root):
    completion = inputs.json(study_root / "numeric_completion.json")
    expected_total = 2 * sum(d["runs"] for d in definitions.values())
    if completion.get("status") != "complete" or completion.get("evaluations") != expected_total or completion.get("repeat_checks") != expected_total:
        raise ValueError("The complete actual forecast-policy evaluations and repeats are required")
    cases = completion.get("cases", [])
    mapping = {(row["variant"], row.get("model_key", row.get("model"))): row for row in cases}
    if len(mapping) != len(cases) or set(mapping) != {(v, m) for v in DATASETS for m in definitions}:
        raise ValueError("Numeric completion must cover each model and pool exactly once")
    if any(row["runs"] != definitions[model]["runs"] for (_, model), row in mapping.items()):
        raise ValueError("A completed family has the wrong real sampling count")
    audit = inputs.json(study_root / "independent_audit.json")
    require_audit(audit, expected_total, "pretrained forecast global")
    if audit.get("phase") != "full_pretrained_foundation_independent_audit":
        raise ValueError("A source preflight cannot replace the full independent financial audit")
    return completion, mapping, audit


def _annotate(frame, variant, model_key, frame_name):
    frame = frame.copy()
    if "strategy" in frame:
        frame = frame.loc[~frame.strategy.isin(BASELINES)].copy()
        if "primary_policy" not in frame and "policy_role" not in frame:
            raise ValueError("Forecast primary/sample role must be explicit, including deterministic Chronos")
        frame["primary_policy"] = primary(frame)
        frame["policy_role"] = frame.primary_policy.map({True: "primary", False: "sample_seed"})
    frame["universe_variant"], frame["implementation"], frame["model_key"], frame["algorithm"] = variant, "foundation", model_key, MODEL_LABELS[model_key]
    if "run_id" in frame:
        frame["raw_run_id"] = frame.run_id
        frame["run_id"] = "foundation__" + variant + "__" + frame.run_id
    if frame_name == "daily_equity":
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    return frame


def _provenance(inputs, spec, study_root, resolve_proof):
    path = resolve_proof(spec.get("model_provenance_path", study_root / "model_provenance.json"))
    document = inputs.json(path)
    if spec.get("model_provenance_sha256") != inputs.files[path.resolve().as_posix()]:
        raise ValueError("The checkpoint provenance differs from the frozen inference protocol")
    rows = document.get("rows")
    if not document.get("source") or not isinstance(rows, list) or len(rows) != 3 or {r["model_key"] for r in rows} != set(MODEL_LABELS):
        raise ValueError("Three actual pretrained checkpoint provenance rows with source metadata are required")
    policy_path = study_root / "policy_precommit.json"
    policy = inputs.json(policy_path)
    if policy.get("context_days") != 128 or policy.get("horizon_days") != 1 or policy.get("local_fit") is not False:
        raise ValueError("The precommitted no-local-fit 128-day/one-day inference policy differs")
    normalized = []
    for source_row in rows:
        row = dict(source_row)
        row["weight_revision"] = row.get("weight_revision", row.get("weights_revision"))
        row["weight_sha256"] = row.get("weight_sha256", row.get("weights_sha256"))
        row["first_verified_available_at"] = row.get("first_verified_available_at", row.get("first_verified_public_notice_utc"))
        row["first_decision_date"] = row.get("first_decision_date", row.get("first_usable_decision_utc"))
        row["source_url"] = row.get("source_url", row.get("weights_repository", row.get("source_repository")))
        if not row.get("training_cutoff_status") and row.get("pretraining_cutoff_independently_verified") is False:
            row["training_cutoff_status"] = "确切预训练语料截止未独立核实；公开权重日期不是完整语料审计"
        row.setdefault("context_days", policy["context_days"])
        row.setdefault("horizon_days", policy["horizon_days"])
        if not row.get("sampling_description"):
            seeds = spec["models"][row["model_key"]]["sampling_seeds"]
            row["sampling_description"] = f"同一权重的{len(seeds)}个固定采样路径；每路径sample_count=1" if seeds else "单一确定性0.5分位数；不制造五种子分布"
        if not row.get("weight_revision") or not row.get("weight_sha256") or not row.get("training_cutoff_status"):
            raise ValueError("Weight revision/hash and honest cutoff evidence must be explicit")
        if row.get("context_days") != 128 or row.get("horizon_days") != 1 or row.get("local_fit") is not False:
            raise ValueError("The declared fixed 128-day, one-day, no-local-training model protocol differs")
        normalized.append(row)
    document = json.loads(json.dumps(document, ensure_ascii=False))
    document["source"]["files"] = [*document["source"].get("files", []), str(policy_path), str(study_root / "frozen_spec.json")]
    return document, pd.DataFrame(normalized)


def _mapping_description(spec):
    """Describe the executed mapper, never infer a strategy from its return."""
    mapping = spec.get("forecast_to_weights", {})
    if (mapping.get("hurdle") != .0024 or mapping.get("volatility_floor") != .01 or
            mapping.get("no_renormalization_after_cap") is not True):
        raise ValueError("The described fixed forecast mapper differs from the executed protocol")
    return ("输入为128个已完成UTC日的OHLC、币成交量及原始成交额；成交额来自同一Binance归档的报价币成交额，不以收盘价乘成交量估算。"
            "以预测次日收盘价相对最近收盘价的收益，减去0.24%固定门槛，再将正值除以过去30个日收益的样本标准差（下限1%）。"
            "正分数按比例分配最多95%本金，单币目标截断为25%，超出部分留现金且不再重新分配；全部分数为零时持USDT。"
            "0.24%是固定研究门槛，不代表精确净ROI；没有根据本年收益调节门槛或仓位公式。")


def _execution_description(spec, backend):
    devices = sorted({row.get("device", "unspecified") for row in spec["models"].values()})
    probes = []
    if any(str(device).startswith("cuda") for device in devices):
        probes = backend.get("gpu_probe", [])
        if (len(probes) != 2 or len({row['uuid'] for row in probes}) != 2 or
                any("V100" not in row["name"] for row in probes) or spec["model_workers"] != 2):
            raise ValueError("The dual-V100 report requires the actually verified two-device backend")
        names = "、".join(row["name"] for row in probes)
        description = (f"三组模型在双V100（{names}）完整重新推理，每张物理GPU独立一个进程，以UUID隔离，进程内均使用cuda:0；"
                       f"每进程{spec['threads_per_model']}个CPU辅助线程。未复用先前CPU预测，也未声明CPU与GPU预测逐位相同。"
                       "日期分段保持每日固定采样种子、同日资产顺序和完整批次，GPU同卡重复核验另存证据；返还预测后在本机原现货账本中扣费评估。")
    else:
        description = (f"实际推理设备为{'、'.join(devices)}，最多{spec['model_workers']}个进程、每进程{spec['threads_per_model']}个CPU线程；"
                       "日期分段仅改变任务调度，每日固定采样种子、同日资产顺序及完整预测批次。")
    if spec["models"]["chronos2"].get("synchronous_chronos_loader") is True:
        description += "Chronos保留同步加载调度适配，官方源码和权重保持固定；Windows原生／同步加载全分位数合成输入一致性及实际GPU重复核验分别保存在复现证据中。" if probes else "Chronos采用Windows同步加载调度适配，官方源码和权重保持固定；全分位数合成输入一致性校验保存在复现证据中。"
    return description


def _implementation_rows(document, spec, backend):
    """Keep source facts, derive execution notes from the actual backend proof."""
    rows = json.loads(json.dumps(document["rows"], ensure_ascii=False))
    if not any(str(row.get("device", "")).startswith("cuda") for row in spec["models"].values()):
        return rows
    runtime = _execution_description(spec, backend)
    for row in rows:
        if row.get("implementation") == "foundation":
            notes = row.get("adapter_changes", [])
            if not isinstance(notes, list):
                raise ValueError("Native implementation adapter facts must retain list structure")
            row["adapter_changes"] = [note for note in notes if "CPU inference with" not in note and "Windows synchronous" not in note]
            row["adapter_changes"].append(runtime)
    return rows


def _availability_events(history, provenance):
    """Join verified publication facts, leaving every financial column untouched."""
    history = history.copy()
    history["timestamp"] = pd.to_datetime(history.timestamp, utc=True)
    history["availability_event"] = None
    history["model_first_decision_date"] = None
    for row in provenance.to_dict("records"):
        date = row.get("first_decision_date")
        if not date:
            continue
        at = pd.Timestamp(date)
        at = at.tz_localize("UTC") if at.tzinfo is None else at.tz_convert("UTC")
        mask = history.implementation.eq("foundation") & history.model_key.eq(row["model_key"])
        history.loc[mask, "model_first_decision_date"] = at.isoformat()
        history.loc[mask & history.timestamp.eq(at), "availability_event"] = MODEL_LABELS[row["model_key"]] + "首个可核实可用决策日；此前持现金"
    return history


def _source(inputs, spec, preservation, study_root):
    return {"label": "固定预训练预测策略与既有强化学习在同一现货账本中的实际保存结果",
        "classification": "modeled", "timezone": "UTC",
        "period": {"start": "2025-07-25T00:00:00Z", "end": "2026-10-23T00:00:00Z"},
        "evidenceFlow": [
            {"title": "冻结预测与交易规则", "detail": "固定权重版本、128个过去完整日K线、一日预测和事前声明的预测转仓位规则；不向模型提供未来价格或未来外生变量。所有推理与资金执行来源均保留。"},
            {"title": "可用时间与样本边界", "detail": "模型首次可核实可用时间按证据保留；尚不可用的历史决策日持现金。首次发布证据不自动证明全部预训练截止时间，不将未知截止补成已确认。"},
            {"title": "真实重复与独立对账", "detail": "Kronos采样策略与平均目标权重主组合、Chronos单确定性策略均在现货账本模拟执行；每份评估重复核验，独立审计从逐笔成交重建现金、库存、小时权益和指标。"},
            {"title": "旧结果只读引用", "detail": f"验证既有RL完整报告、底层输入及{preservation['files_checked']}份旧研究保护证据；不重训旧RL、不重新模拟旧收益、不挑本年最优种子。"}],
        "caveats": [
            "本年度已用于前序研究；剔除ADA来自此前亏损归因，属于已见市场的事后比较。",
            "预训练权重首次可核实可用时间与训练数据截止是不同证据，无法证明截止时不称为严格未见样本验证。",
            "可用日期按权重公开证据设门；本轮使用固定推理代码重演历史，不表示对应日期已经部署了同一交易系统。",
            "Kronos五采样来自同一预训练权重，不等同于五个独立训练模型；Chronos单确定性策略不制造种子分布。",
            "模型可用前现金等待及128日完整上下文会改变实际交易机会与风险暴露，需结合预测覆盖和平均仓位比较。",
            "币池替补、策略输入、预测映射及模型路径同时变化，不能解释为单一框架或单币移除的因果效应。",
            "回撤按完整小时收盘及初始本金，图表只采样日末；小时内更深跌幅未捕捉。",
            "费用和滑点为统一研究假设；小数币数量未重建历史数量步长、最小名义金额或盘口深度。",
            "历史市值快照是回溯版本，原始发布版本未完全核实；BTC/ETH被动持有属于25%上限例外。"],
        "metricDefinitions": [
            {"label": "扣费净收益", "definition": "期末现金加真实持币市值相对10000 USDT的变化，已扣每腿模拟费用与滑点；假设退出成本后的净清算收益另列。"},
            {"label": "小时最大回撤", "definition": "完整小时收盘权益相对包含初始资金的历史峰值最大跌幅；日图不会重算为日粒度回撤。"},
            {"label": "平均现金占比", "definition": "1减同一小时收盘估值序列中持币市值占总权益比例的算术均值；来自原保存的average_exposure，不根据目标现金下限假设实际现金。", "formula": "1-average_exposure", "componentIds": ["foundation-main"]},
            {"label": "模型可用日标记", "definition": "从固定权重来源的first_decision_date合并到同模型同日的已存在权益行，availability_event是发布事实，不引入未来行情或额外金融样本。", "componentIds": ["foundation-return", "foundation-drawdown"]}],
        "files": [p for p in inputs.files if "/runs/" not in p and
                  ("/forecasts/" not in p or p.endswith("/manifest.json")) and
                  (p.startswith(study_root.as_posix() + "/") or p in {
                      (LEGACY_REPORT / "manifest.json").resolve().as_posix(),
                      (LEGACY_REPORT / "reviewed.json").resolve().as_posix(),
                      (LEGACY_ROOT / "delivery_verification.json").resolve().as_posix()})],
        "frozen_forecast_protocol": spec}


def _readme(summary, preservation, spec, backend):
    lines = ["# 预训练行情模型与既有强化学习的同口径组合比较", "",
             "测试365天[2025-10-23,2026-10-23)UTC；主组合均为实际执行的独立10000 USDT账本。Kronos的多采样来自同一权重，Chronos只有单一确定性策略，旧RL模型分别训练。模型尚未可用时现金等待，完整窗口不删天。", "",
             "| 资产池 | 模型／实现 | 主成本净收益 | 主成本小时回撤 | 压力净收益 | 压力小时回撤 | 主成本平均仓位 |", "|---|---|---:|---:|---:|---:|---:|"]
    selected = summary.loc[primary(summary) & summary.split.eq("test")]
    for (variant, implementation, algorithm), rows in selected.groupby(["universe_variant", "implementation", "algorithm"], sort=False):
        by_cost = rows.set_index("scenario")
        a, b = by_cost.loc["base"], by_cost.loc["stress"]
        lines.append(f"| {variant} | {implementation} {algorithm} | {a.net_return:.2%} | {a.max_drawdown:.2%} | {b.net_return:.2%} | {b.max_drawdown:.2%} | {a.average_exposure:.2%} |")
    lines += ["", "主成本单边手续费0.10%＋滑点0.02%；压力0.20%＋0.05%。目标现金至少5%、单币至多25%，每日00卖／01买。BTC/ETH被动参考初始各47.5%，是限仓例外。", "",
              "数据与权重发布证据见model_provenance，真实预测覆盖与等待见forecast_diagnostics；预训练截止无法完全证明时保留未知，不把结果标为严格独立样本外。该年度已用于前序研究，ADA选择是事后敏感性分析。", "",
              _execution_description(spec, backend), "",
              f"本汇总只读验证{preservation['files_checked']}份旧证据，保留全部真实采样、旧模型及共同基准，未重训、撮合、修改生产或提交。原始逐笔成交、小时权益、重复指纹与独立审计在上级模型case目录。", ""]
    return "\n".join(lines)


def main(output_dir=None, study_root=NEW_ROOT):
    study_root = Path(study_root).resolve()
    output = Path(output_dir).resolve() if output_dir else study_root / "report"
    _check_output(output, study_root)
    inputs = Inputs()
    spec = inputs.json(study_root / "frozen_spec.json")
    imported_path = study_root / "local_import_manifest.json"
    imported = inputs.json(imported_path) if imported_path.exists() else None
    proof = ProofPaths(study_root, spec, imported)
    inputs.track(ROOT / "scripts/foundation_model_proof_paths.py")
    if imported:
        for item in imported["files"]:
            inputs.track(proof.local_evidence(item["path"]))
    backend = inputs.json(proof.resolve(spec["backend_evidence_path"]))
    if backend.get("status") != "passed":
        raise ValueError("The executed inference backend was not verified")
    definitions = _model_definitions(spec)
    completion, cases, global_audit = _completion(inputs, definitions, study_root)
    frozen_path = next((study_root / name for name in ("inference_frozen_inputs.json", "frozen_inputs.json", "training_frozen_inputs.json") if (study_root / name).exists()), None)
    if frozen_path is None:
        raise FileNotFoundError("The inference freeze manifest is required")
    frozen = inputs.json(frozen_path)
    if frozen.get("protocol_sha256") != inputs.files[(study_root / "frozen_spec.json").resolve().as_posix()]:
        raise ValueError("The forecast protocol changed after inference freeze")
    for row in spec["pinned_files"]:
        path = inputs.track(proof.resolve(row["path"]))
        if inputs.files[path.as_posix()] != row["sha256"]:
            raise ValueError("A pinned official source or pretrained weight changed")
    for key, name in (("market_manifest_sha256", "market/manifest.json"), ("source_manifest_sha256", "sources/source_manifest.json")):
        path = inputs.track(study_root / name)
        if inputs.files[path.as_posix()] != frozen[key]:
            raise ValueError("The frozen market or source manifest changed")
    for name, expected in frozen["source_sha256"].items():
        path = inputs.track(ROOT / name)
        if inputs.files[path.as_posix()] != expected:
            raise ValueError("A frozen financial/inference source changed: " + name)
    preservation = _preservation(inputs, study_root, proof)
    frames, old_manifest, old_reviewed, baselines = _legacy(inputs)
    provenance_document, provenance = _provenance(inputs, spec, study_root, proof.resolve)
    parts = {name: [frames[name]] for name in ("summary", "daily_equity", "monthly_returns", "seed_statistics")}
    baseline_checks, case_audits, case_manifests = [], {}, {}
    reference_config = baselines["original"]["config"]
    for variant, model_key in ((v, m) for v in DATASETS for m in MODEL_LABELS):
        completed = cases[(variant, model_key)]
        folder = study_root / variant / model_key
        manifest = inputs.json(folder / "manifest.json")
        if inputs.files[(folder / "manifest.json").resolve().as_posix()] != completed["manifest_sha256"]:
            raise ValueError("Numeric completion pins another case manifest")
        audit = inputs.json(folder / "independent_audit.json")
        require_audit(audit, definitions[model_key]["runs"], variant + "/" + model_key)
        case_audits[(variant, model_key)], case_manifests[(variant, model_key)] = audit, manifest
        config = manifest["config"]
        if any(config.get(k) != reference_config.get(k) for k in COMMON_CONFIG):
            raise ValueError("A common economic window/cost/cash contract differs")
        if manifest.get("model_key", config.get("model_key")) != model_key or manifest.get("universe_variant", config.get("universe_variant")) != variant:
            raise ValueError("Case identity differs from its completed path")
        if config.get("local_fit") is not False or config.get("model_config") != spec["models"][model_key] or config.get("policy") != spec["forecast_to_weights"]:
            raise ValueError("A case differs from the frozen inference-only model or forecast mapper")
        forecast_path = inputs.track(study_root / "forecasts" / model_key / "manifest.json")
        if inputs.files[forecast_path.as_posix()] != manifest["forecast_manifest_sha256"]:
            raise ValueError("The executed forecast manifest changed")
        forecasts = inputs.json(forecast_path)
        if forecasts != manifest["forecast_manifest"] or forecasts.get("local_fit") is not False or forecasts.get("inference_only") is not True or forecasts.get("config") != spec["models"][model_key]:
            raise ValueError("The actual forecast cache is not the fixed inference-only model")
        if not forecasts.get("parameter_sha256_before") or forecasts["parameter_sha256_before"] != forecasts["parameter_sha256_after"]:
            raise ValueError("Pretrained model tensors were changed during prediction")
        expected_seeds = definitions[model_key]["seeds"] or [None]
        entries = forecasts.get("forecast_files", [])
        if len(entries) != len(expected_seeds) or {r["seed"] for r in entries} != set(expected_seeds):
            raise ValueError("The native forecast paths do not match the real sampling declarations")
        for cache in entries:
            path = inputs.track(Path(cache["path"]))
            if inputs.files[path.as_posix()] != cache["sha256"] or cache["model_key"] != model_key or cache["rows"] <= 0:
                raise ValueError("An executed native forecast cache changed or is empty")
        native_repeats = forecasts.get("native_repeat_verifications", [])
        if not native_repeats or not all(r.get("passed") is True for r in native_repeats):
            raise ValueError("The declared native inference-repeat checks did not complete")
        for path, expected in manifest.get("source_sha256", {}).items():
            actual = inputs.track(ROOT / path)
            if inputs.files[actual.as_posix()] != expected:
                raise ValueError("A frozen financial/inference implementation changed: " + path)
        dataset = inputs.track(DATASETS[variant])
        if manifest["frozen_inputs"]["dataset_sha256"] != inputs.files[dataset.as_posix()]:
            raise ValueError("A forecast case used different frozen market data")
        table = inputs.csv(folder / "summary.csv")
        expected_runs = definitions[model_key]["runs"]
        if len(table) != expected_runs or not table.run_id.is_unique or set(table.run_id) != set(manifest["run_results"]) or set(table.run_id) != set(audit["runs"]):
            raise ValueError("A real evaluation family is incomplete")
        repeats = manifest.get("repeat_verifications", [])
        if len(repeats) != expected_runs or {r["run_id"] for r in repeats} != set(table.run_id) or not all(r.get("passed") is True and r.get("fingerprints") == manifest["run_results"][r["run_id"]]["fingerprints"] for r in repeats):
            raise ValueError("Exact repeated forecast-policy evaluation evidence is incomplete")
        for (split, scenario), group in table.groupby(["split", "scenario"]):
            if split not in ("validation", "test") or scenario not in ("base", "stress") or len(group) != expected_runs // 4:
                raise ValueError("Every split and cost must retain all planned real policies")
            active = group.loc[~group.strategy.isin(BASELINES)]
            p = primary(active)
            if p.sum() != 1 or active.loc[p, "seed"].notna().any() or sorted(active.loc[~p, "seed"].astype(int)) != definitions[model_key]["seeds"]:
                raise ValueError("Primary and actual sampling identities differ from the frozen model protocol")
            if set(group.loc[group.strategy.isin(BASELINES), "strategy"]) != set(BASELINES):
                raise ValueError("All four true shared baseline evaluations are required")
        for row in table.to_dict("records"):
            audited = audit["runs"][row["run_id"]]
            if any(not math.isfinite(float(row[f])) or abs(float(row[f]) - audited[f]) > 1e-8 for f in ("net_return", "max_drawdown", "fees")):
                raise ValueError("Saved metrics do not reconcile with independent ledger audit")
            fingerprints = _run_artifacts(inputs, folder, row["run_id"], manifest["run_results"][row["run_id"]]["fingerprints"])
            if row["strategy"] in BASELINES:
                reference = baselines[variant]["run_results"][row["run_id"]]["fingerprints"]
                if fingerprints != reference:
                    raise ValueError("A same-pool baseline differs from its actual frozen RL ledger")
                baseline_checks.append({"universe_variant": variant, "model_key": model_key, "run_id": row["run_id"], "all_four_financial_fingerprints_identical": True})
        parts["summary"].append(_annotate(table, variant, model_key, "summary"))
        for frame_name in ("daily_equity", "monthly_returns"):
            parts[frame_name].append(_annotate(inputs.csv(folder / (frame_name + ".csv")), variant, model_key, frame_name))
        if definitions[model_key]["seeds"]:
            stats = inputs.csv(folder / "seed_statistics.csv")
            if len(stats) != 4 or not stats.seeds.eq(len(definitions[model_key]["seeds"])).all():
                raise ValueError("Actual stochastic sample statistics are incomplete")
            parts["seed_statistics"].append(_annotate(stats, variant, model_key, "seed_statistics"))
    for name, values in parts.items():
        frames[name] = pd.concat(values, ignore_index=True)
    frames["model_provenance"] = provenance
    diagnostics_path = Path(spec.get("forecast_diagnostics_path", study_root / "forecast_diagnostics.csv"))
    frames["forecast_diagnostics"] = inputs.csv(diagnostics_path)
    if not len(frames["forecast_diagnostics"]) or not {"model_key", "universe_variant"}.issubset(frames["forecast_diagnostics"].columns) or set(frames["forecast_diagnostics"].model_key) != set(definitions):
        raise ValueError("Real forecast coverage diagnostics for all three models are required")
    frames["daily_equity"] = _availability_events(frames["daily_equity"], provenance)
    expected_summary = 272 + 2 * sum(4 * (1 + len(d["seeds"])) for d in definitions.values())
    if len(frames["summary"]) != expected_summary or not frames["summary"].run_id.is_unique or len(frames["training_metrics"]) != 50:
        raise ValueError("The deduplicated comparison or preserved RL training count is incomplete")
    expected_stats = 40 + 2 * sum(4 for d in definitions.values() if d["seeds"])
    if len(frames["seed_statistics"]) != expected_stats or len(baseline_checks) != len(cases) * 16:
        raise ValueError("Sampling statistics or exact shared benchmark coverage is incomplete")
    for _, group in frames["summary"].groupby(["universe_variant", "split", "scenario"]):
        if primary(group).sum() != 8 or group.policy_role.eq("baseline").sum() != 4:
            raise ValueError("Each view must contain eight real primary policies and four common baselines")
    document = inputs.json(proof.local_evidence("sources/implementation_review.json"))
    if not document.get("source") or not isinstance(document.get("rows"), list):
        raise ValueError("New implementation review requires actual source-scoped rows")
    frames["implementation_review"] = pd.concat([frames["implementation_review"], pd.DataFrame(_implementation_rows(document, spec, backend))], ignore_index=True).drop_duplicates(["implementation", "algorithm"], keep="last")
    source = _source(inputs, spec, preservation, study_root)
    diagnostics_source_path = Path(spec.get("forecast_diagnostics_source_path", study_root / "forecast_diagnostics_source.json"))
    diagnostics_source = inputs.json(proof.local_evidence(diagnostics_source_path.relative_to(study_root).as_posix())) if diagnostics_source_path.exists() else None
    queries = {}
    for query_id, frame_name in FRAME_MAP.items():
        meta = json.loads(json.dumps(source, ensure_ascii=False))
        meta["files"] = [frame_name + ".csv", "manifest.json", "input_integrity_before.json", "input_integrity_after.json", *source["files"]]
        if query_id == "model_provenance":
            meta = json.loads(json.dumps(provenance_document["source"], ensure_ascii=False))
            meta["files"] = ["model_provenance.csv", *meta.get("files", [])]
        elif query_id == "forecast_diagnostics" and diagnostics_source:
            meta = json.loads(json.dumps(diagnostics_source.get("source", diagnostics_source), ensure_ascii=False))
            meta["files"] = ["forecast_diagnostics.csv", *meta.get("files", [])]
        elif query_id == "training_metrics":
            meta["period"] = {"start": "2023-01-01T00:00:00Z", "end": "2025-07-25T00:00:00Z"}
        queries[query_id] = {"source": meta, "rows": records(frames[frame_name])}
    after = inputs.verify()
    output.mkdir(parents=True, exist_ok=True)
    write(output / "input_integrity_before.json", {"sha256": inputs.files})
    write(output / "input_integrity_after.json", {"sha256": after, "passed": True})
    for name, frame in frames.items():
        frame.to_csv(output / (name + ".csv"), index=False, float_format="%.17g", encoding="utf-8-sig")
    selected = frames["summary"].loc[frames["summary"].implementation.eq("foundation") & frames["summary"].split.eq("test") & primary(frames["summary"])]
    title = "这套固定映射下，三组模型组合仍未转正" if len(selected) and selected.net_return.lt(0).all() else "固定映射下，预训练模型与强化学习的收益和回撤"
    as_of = (pd.Timestamp(reference_config["test_end"]).tz_convert("UTC") - pd.Timedelta(hours=1)).date().isoformat()
    write(output / "reviewed.json", {"surface": "report", "status": "reviewed", "buildStatus": "complete", "title": title,
        "generatedAt": datetime.now(timezone.utc).isoformat(), "report": {"asOf": as_of, "title": title,
            "forecastMappingDescription": _mapping_description(spec),
            "inferenceExecutionDescription": _execution_description(spec, backend),
            "studyConfig": {"test_start": reference_config["test_start"], "test_end": reference_config["test_end"], "validation_start": reference_config["validation_start"], "validation_end": reference_config["validation_end"], "costs": reference_config["costs"], "hourly_drawdown": True}},
        "filters": [], "queries": queries})
    (output / "README.md").write_text(_readme(frames["summary"], preservation, spec, backend), encoding="utf-8")
    inputs.verify()
    write(output / "manifest.json", {"assembler": Path(__file__).name, "assembler_source_sha256": sha(Path(__file__)),
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "complete", "study_root": str(study_root), "source_legacy_report": str(LEGACY_REPORT),
        "input_sha256": inputs.files, "input_unchanged_after_assembly": True, "original_preservation": preservation,
        "row_counts": {name: len(frame) for name, frame in frames.items()}, "shared_baseline_verifications": baseline_checks,
        "numeric_completion_status": completion["status"], "global_independent_audit_status": global_audit["status"],
        "case_independent_audits": {"__".join(k): {f: doc[f] for f in ("status", "checks_total", "checks_failed", "runs_checked")} for k, doc in case_audits.items()},
        "generated_sha256": {name: sha(output / name) for name in OWNED_NAMES}, "numeric_results_changed": False,
        "pretraining_cutoff_not_assumed": True, "post_hoc_research_extension": True, "production_changes": False})
    print(f"Assembled {len(frames['summary'])} saved evaluations, eight primary policies per view, {len(provenance)} actual pretrained checkpoints: {output}")


def self_check():
    """Identity/role/annotation checks only, not published financial fixtures."""
    sample = pd.DataFrame([{"run_id": "test__base__chronos2", "strategy": "chronos2", "seed": None,
        "primary_policy": True, "policy_role": "primary", "timestamp": "2025-10-31 00:00:00+00:00"}])
    value = _annotate(sample, "original", "chronos2", "daily_equity")
    assert value.run_id.iloc[0] == "foundation__original__test__base__chronos2" and primary(value).all()
    value = _availability_events(value, pd.DataFrame([{"model_key": "chronos2", "first_decision_date": "2025-10-31T00:00:00Z"}]))
    assert value.availability_event.notna().all() and records(value)[0]["timestamp"] == "2025-10-31T00:00:00.000Z"
    assert 272 + 2 * (4 * 6 + 4 * 6 + 4) == 376
    cpu = {"models": {"chronos2": {"device": "cpu", "synchronous_chronos_loader": True}}, "model_workers": 4, "threads_per_model": 2}
    assert "4个进程" in _execution_description(cpu, {}) and "GPU预测" not in _execution_description(cpu, {})
    gpu = {**cpu, "models": {"chronos2": {"device": "cuda:0", "synchronous_chronos_loader": True}}, "model_workers": 2}
    hardware_fixture = {"gpu_probe": [{"uuid": "synthetic-device-a", "name": "V100 test fixture"}, {"uuid": "synthetic-device-b", "name": "V100 test fixture"}]}
    description = _execution_description(gpu, hardware_fixture)
    assert "双V100" in description and "UUID隔离" in description and "未复用先前CPU预测" in description and "4个进程" not in description
    assert (V100_ROOT.resolve() / "report").parent == V100_ROOT.resolve()
    print("SELF_CHECK_PASS: deterministic primary, global identity, same-query availability and CPU/GPU scheduling descriptions; no economics executed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=NEW_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    self_check() if args.self_check else main(args.output_dir, args.study_root)
