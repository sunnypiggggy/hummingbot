"""Assemble audited native ElegantRL runs and read-only SB3 controls.

This entry point hashes saved evidence and copies financial tables only. It does
not import a trainer, deserialize checkpoints, simulate orders or select models.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.build_rl_ex_ada_comparison import (
    Inputs, ROOT, DATASETS, BASELINES, FRAME_NAMES, _expected_runs,
    _run_artifacts, _verify_source_versions, _compare_weight_economics,
    _universe_changes, records, require_audit, read, sha, write,
)


NEW_ROOT = ROOT / "results/backtests/crypto_rl_elegantrl_20261024"
OUTPUT = NEW_ROOT / "report"
OLD_ROOT = ROOT / "results/backtests/crypto_rl_ex_ada_20261024"
LEGACY = {
    ("original", "sb3", "ppo"): ROOT / "results/backtests/crypto_rl_top10_20261024",
    ("original", "sb3", "a2c"): ROOT / "results/backtests/crypto_rl_a2c_20261024",
    ("ex_ada", "sb3", "ppo"): OLD_ROOT / "ppo",
    ("ex_ada", "sb3", "a2c"): OLD_ROOT / "a2c",
}
COMMON_CORE = ("scripts/backtest_crypto_rl_top10.py", "scripts/crypto_rl_data.py",
               "scripts/crypto_rl_portfolio_env.py", "scripts/crypto_rl_reporting.py")
COMMON_CONFIG = ("train_start", "train_end", "validation_start", "validation_end", "test_start", "test_end",
                 "costs", "capital", "seeds", "timesteps_per_seed", "net_arch", "cash_target_floor", "single_coin_target_cap")
QUERY_MAP = {"strategy_summary": "summary", "equity_history": "daily_equity", "monthly_returns": "monthly_returns",
             "seed_statistics": "seed_statistics", "training_metrics": "training_metrics", "universe_changes": "universe_changes"}
OWNED_NAMES = tuple(name + ".csv" for name in (*FRAME_NAMES, "universe_changes", "implementation_review")) + (
    "reviewed.json", "README.md", "input_integrity_before.json", "input_integrity_after.json")


def _check_output(output: Path) -> None:
    if output != OUTPUT.resolve():
        raise ValueError("Only the separate ElegantRL report directory is writable")
    path = output / "manifest.json"
    if not path.exists():
        if any((output / name).exists() for name in OWNED_NAMES):
            raise ValueError("Preserving comparison files without an ownership manifest")
        return
    previous = read(path)
    if previous.get("assembler") != Path(__file__).name:
        raise ValueError("Preserving a comparison owned by another assembler")
    for name, expected in previous.get("generated_sha256", {}).items():
        if sha(output / name) != expected:
            raise ValueError("Preserving independently edited comparison file: " + name)


def _preservation(inputs: Inputs) -> dict:
    path = NEW_ROOT / "original_research_preservation.json"
    contract = inputs.json(path)
    files = contract.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("The dynamic preservation contract is empty")
    for relative, expected in files.items():
        tracked = inputs.track(ROOT / relative.replace("\\", "/"))
        if tracked.stat().st_size != expected["size_bytes"] or inputs.files[tracked.as_posix()] != expected["sha256"]:
            raise ValueError("An immutable prior artifact changed: " + relative)
    absent = contract.get("originally_absent", [])
    for relative in absent:
        if (ROOT / relative.replace("\\", "/")).exists():
            raise ValueError("A previously absent historical artifact appeared: " + relative)
    return {"files_checked": len(files), "originally_absent_checked": len(absent),
            "contract_sha256": inputs.files[path.resolve().as_posix()], "passed": True}


def _vendor(inputs: Inputs, spec: dict, frozen: dict) -> dict:
    vendor_path = Path(spec["vendor_manifest"])
    manifest = inputs.json(vendor_path)
    if manifest.get("commit") != spec["upstream_commit"]:
        raise ValueError("The source review and frozen upstream commit differ")
    if frozen.get("vendor_manifest_sha256") != inputs.files[vendor_path.resolve().as_posix()]:
        raise ValueError("The vendor manifest changed after training freeze")
    official = {row["path"]: row for row in manifest["files"]}
    expected_members = {"helloworld/erl_agent.py", "helloworld/erl_config.py", "LICENSE"}
    if set(official) != expected_members or len(manifest["files"]) != 3 or set(spec["vendor_file_sha256"]) != expected_members:
        raise ValueError("The selected training vendor must contain the two official helloworld sources and license")
    for relative, expected in spec["vendor_file_sha256"].items():
        path = inputs.track(Path(spec["vendor_path"]) / relative)
        if inputs.files[path.as_posix()] != expected or official.get(relative, {}).get("sha256") != expected:
            raise ValueError("A pinned official source differs from the training version: " + relative)
    sidecar_path = NEW_ROOT / "sources/training_source_archive.json"
    sidecar = inputs.json(sidecar_path)
    if sidecar.get("commit") != manifest["commit"] or sidecar.get("training_source_manifest_sha256") != frozen["vendor_manifest_sha256"] or sidecar.get("algorithm_source_modified") is not False:
        raise ValueError("The training-source archive sidecar does not match the frozen selected vendor")
    archived_files = {row["path"]: row for row in sidecar["files"]}
    if set(archived_files) != expected_members or len(sidecar["files"]) != 3:
        raise ValueError("The offline training archive must contain exactly three declared files")
    path = inputs.track(Path(sidecar["archive_path"]))
    if inputs.files[path.as_posix()] != sidecar["archive_sha256"] or path.stat().st_size != sidecar["archive_bytes"]:
        raise ValueError("The actual offline training-source ZIP no longer matches its sidecar")
    with zipfile.ZipFile(path) as zipped:
        if len(zipped.infolist()) != 3 or set(zipped.namelist()) != expected_members:
            raise ValueError("The offline ZIP is not the exact selected helloworld training source set")
        for relative in sorted(expected_members):
            entry = official[relative]
            if archived_files[relative].get("sha256") != entry["sha256"] or archived_files[relative].get("git_blob_sha1") != entry["git_blob_sha1"] or zipped.getinfo(relative).file_size != entry["bytes"]:
                raise ValueError("An archived source declaration differs from the frozen official file")
            raw = zipped.read(relative)
            blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            if hashlib.sha256(raw).hexdigest() != entry["sha256"] or blob != entry["git_blob_sha1"] or raw != Path(entry["vendor_path"]).read_bytes():
                raise ValueError("Archived bytes do not match the actual official training source: " + relative)
    archive = {"path": path.as_posix(), "sha256": sidecar["archive_sha256"], "bytes": sidecar["archive_bytes"],
               "exact_members_verified": sorted(expected_members), "member_sha256_and_git_blob_sha1_verified": True,
               "sidecar_path": sidecar_path.as_posix(), "sidecar_sha256": inputs.files[sidecar_path.resolve().as_posix()]}
    return {"commit": manifest["commit"], "repository": manifest["repository"],
            "manifest_sha256": inputs.files[vendor_path.resolve().as_posix()],
            "official_files_verified": len(spec["vendor_file_sha256"]), "algorithm_source_modified": False,
            "license": manifest.get("license"), "source_archive": archive}


def _annotate(frame: pd.DataFrame, variant: str, implementation: str, algorithm: str, frame_name: str) -> pd.DataFrame:
    frame = frame.copy()
    if "strategy" in frame:
        keep_baselines = implementation == "sb3" and algorithm == "ppo"
        frame = frame.loc[frame.strategy.isin((algorithm, algorithm + "_ensemble", *BASELINES) if keep_baselines else (algorithm, algorithm + "_ensemble"))].copy()
    frame["universe_variant"], frame["implementation"], frame["algorithm"] = variant, implementation, algorithm.upper()
    if "strategy" in frame:
        mask = frame.strategy.isin(BASELINES)
        frame.loc[mask, "algorithm"] = "SHARED"
        frame.loc[mask, "implementation"] = "shared"
    if "run_id" in frame:
        frame["raw_run_id"] = frame.run_id
        frame["run_id"] = frame.implementation + "__" + variant + "__" + frame.raw_run_id
    if frame_name == "daily_equity":
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="raise")
    return frame


def _completion_cases(document: dict, algorithms: tuple[str, ...]) -> dict:
    count = len(algorithms) * 2 * 40
    if document.get("status") != "complete" or document.get("evaluations") != count or document.get("repeat_checks") != count or document.get("models") != len(algorithms) * 2 * 5:
        raise ValueError("Native numeric completion is missing or partial")
    cases = document.get("cases", [])
    mapping = {(row["variant"], row["algorithm"]): row for row in cases}
    if len(mapping) != len(cases) or set(mapping) != {(v, a) for v in DATASETS for a in algorithms} or any(row["runs"] != 40 for row in cases):
        raise ValueError("Numeric completion does not cover each native case exactly once")
    return mapping


def _source(inputs: Inputs, manifests: dict, preservation: dict, vendor: dict) -> dict:
    return {
        "label": "固定ElegantRL原生算法与SB3控制在同一现货账本中的保存结果",
        "classification": "modeled", "timezone": "UTC",
        "period": {"start": "2025-07-25T00:00:00Z", "end": "2026-10-23T00:00:00Z"},
        "evidenceFlow": [
            {"title": "共同输入与经济账本", "detail": "原池/剔ADA池各使用同一已冻结小时数据；现货资金、库存、权重投影、成交时序及费用实现均保持一致。每组合10000 USDT。"},
            {"title": "原生算法与显式适配", "detail": "ElegantRL使用固定官方源码的Agent更新，显式采集/重置/回放/预算适配另存；并非框架原始投资组合案例复现，也不是严格只替换库。"},
            {"title": "真实冻结与独立对账", "detail": "各case5个最终种子模型全部保留；每份评估重复指纹一致，独立审计从保存成交重建现金、库存和小时权益。主组合平均五模型目标权重后实际成交。"},
            {"title": "只读汇总", "detail": f"逐一核对保护合同中的{preservation['files_checked']}份既有证据及所有消费输入前后SHA；本步骤不重训、不撮合、不挑种子。"},
        ],
        "caveats": [
            "本年度已被前序策略与框架研究使用；剔除ADA来自此前亏损归因，属于已见样本的事后敏感性研究。模型训练切分不等于研究选择未见测试。",
            "原生PPO与SB3在动作输出、优势处理、更新次数等细节存在差异，不能将收益差异仅归于库本身；SAC/TD3的数据回放、暖启动与更新预算以冻结适配记录为准。",
            "每种算法种子与市场交互预算固定；相同步数不代表相同梯度更新、样本重复使用或算力。无参数搜索、最佳测试种子选择或最佳测试checkpoint选择。",
            "剔除ADA后按历史排名补入成分且重新训练；差值不是单独移除ADA的因果效应。相同编号种子仅描述性配对，不是独立市场重复实验。",
            "历史CMC快照是回溯版本，以日期加两日作可用假设，无法证明原始发布版本；未就绪31完整历史日特征的成分不可买入。",
            "回撤按完整小时收盘权益及初始资金，日图只保留日末；小时内部更深跌幅未捕捉，不能与早期分钟回撤混作同精度。",
            "数量允许小数币，未重建历史数量步长、最小名义金额或盘口深度；现金5%/单币25%是目标约束，BTC/ETH初始各47.5%是被动基准例外。",
            "费用和不利滑点是研究假设；风险暴露不同需同时比较仓位。未改生产交易、风控、凭证、部署或实盘授权。",
        ],
        "metricDefinitions": [
            {"label": "扣费净收益", "definition": "保存的期末现金加持币市值相对10000 USDT的变化，已计每腿模拟费用与滑点；末端退出后净清算值另列。"},
            {"label": "小时最大回撤", "definition": "完整小时收盘权益相对包含初始本金的历史峰值最大跌幅；图表按日末采样。"},
            {"label": "平均仓位", "definition": "每小时持币市值/权益的时间平均；风险仓位差异不能独立归为选币能力。"},
        ],
        "files": [p for p in inputs.files if "/runs/" not in p],
        "dataset_sha256": {v: manifests[(v, "sb3", "ppo")]["frozen_inputs"]["dataset_sha256"] for v in DATASETS},
        "upstream": vendor,
    }


def _readme(summary: pd.DataFrame, algorithms: tuple[str, ...], preservation: dict) -> str:
    lines = ["# ElegantRL原生算法与SB3：历史Top10现货组合对照", "",
             "本报告只读已冻结、完整审计的结果，不重训、不撮合、不挑选种子。原池与剔除ADA后顺位补足10槽位的池分别报告；剔除选择来自此前同一年亏损归因，是事后敏感性研究。", "",
             "训练2023-01-01至2025-07-25；验证90天；测试365天[2025-10-23,2026-10-23)UTC。每种算法/实现/币池均5种子×32768市场交互步，原生更新和数据使用不同，不是严格仅换库。", "",
             "主组合均将五模型的目标权重取平均后在一个账本执行；所有单种子保留。相同成本两档复用同一冻结模型。每日00卖/01买；本金10000 USDT。主成本每腿0.10%手续费+0.02%不利滑点，压力0.20%+0.05%。回撤为完整小时收盘口径。", "",
             "| 资产池 | 实现 | 算法 | 主成本净收益 | 主成本小时回撤 | 压力净收益 | 压力小时回撤 | 主成本平均仓位 |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for variant in DATASETS:
        for implementation, algorithm in (("sb3", "ppo"), ("sb3", "a2c"), *(("elegantrl", a) for a in algorithms)):
            rows = summary.loc[summary.universe_variant.eq(variant) & summary.implementation.eq(implementation) & summary.strategy.eq(algorithm + "_ensemble") & summary.split.eq("test")].set_index("scenario")
            base, stress = rows.loc["base"], rows.loc["stress"]
            lines.append(f"| {variant} | {implementation} | {algorithm.upper()} | {base.net_return:.2%} | {base.max_drawdown:.2%} | {stress.net_return:.2%} | {stress.max_drawdown:.2%} | {base.average_exposure:.2%} |")
    lines += ["", "主结果同时受资产池、目标权重、原生算法与适配、训练轨迹和风险仓位影响，不能解释为框架的普遍优劣或移除ADA的独立因果收益。所有种子共享同一段市场历史。", "",
              f"旧研究动态保护合同共核对{preservation['files_checked']}份既有证据。每池共同4基准仅保留一份，逐个case对照实际四类账本文件SHA，不用复制基准制造不同收益。", "",
              "文件：summary/daily_equity/monthly_returns/seed_statistics/training_metrics/universe_changes/implementation_review.csv；reviewed.json带七类来源；manifest与input_integrity_*.json保存完成、审计、输入及基准核验证据。原case/model目录不变。", "",
              "固定源码、算法与适配说明见implementation_review及上级frozen_spec.json，原生训练并非ElegantRL官方投资组合环境的原样复制。研究结果不构成实盘授权。", ""]
    return "\n".join(lines)


def main(output_dir: Path = OUTPUT) -> None:
    output = output_dir.resolve()
    _check_output(output)
    inputs = Inputs()
    spec = inputs.json(NEW_ROOT / "frozen_spec.json")
    algorithms = tuple(spec["algorithms"])
    if not algorithms or any(a not in ("ppo", "sac", "td3") for a in algorithms) or spec["seeds"] != list(range(5)) or spec["timesteps_per_seed"] != 32768 or spec["parameter_search"] is not False:
        raise ValueError("The frozen native algorithms and bounded five-seed budget are invalid")
    completion = inputs.json(NEW_ROOT / "numeric_completion.json")
    native_cases = _completion_cases(completion, algorithms)
    global_audit = inputs.json(NEW_ROOT / "independent_audit.json")
    require_audit(global_audit, len(native_cases) * 40, "native global")
    if global_audit.get("phase") != "full_native_elegantrl_independent_audit":
        raise ValueError("A preflight cannot replace the completed native financial audit")
    frozen = inputs.json(NEW_ROOT / "training_frozen_inputs.json")
    if frozen.get("protocol_sha256") != inputs.files[(NEW_ROOT / "frozen_spec.json").resolve().as_posix()]:
        raise ValueError("The approved protocol changed after training")
    preservation = _preservation(inputs)
    vendor = _vendor(inputs, spec, frozen)
    studies = {**LEGACY, **{(v, "elegantrl", a): NEW_ROOT / v / a for v in DATASETS for a in algorithms}}
    manifests, audits, actual, parts, source_chains = {}, {}, {}, {name: [] for name in FRAME_NAMES}, []
    for key, folder in studies.items():
        variant, implementation, algorithm = key
        manifest = manifests[key] = inputs.json(folder / "manifest.json")
        audit = audits[key] = inputs.json(folder / "independent_audit.json")
        require_audit(audit, 40, ":".join(key))
        expected = _expected_runs(algorithm)
        if set(manifest["run_results"]) != expected or set(audit["runs"]) != expected:
            raise ValueError("Saved result coverage is incomplete: " + str(key))
        if implementation == "elegantrl" and inputs.files[(folder / "manifest.json").resolve().as_posix()] != native_cases[(variant, algorithm)]["manifest_sha256"]:
            raise ValueError("Completion pins a different native manifest")
        models = {int(m["seed"]): m for m in manifest["models"]}
        if len(manifest["models"]) != 5 or set(models) != set(range(5)) or any(m["timesteps"] != 32768 for m in models.values()):
            raise ValueError("Every final seed must have its complete fixed budget")
        for model in models.values():
            checkpoint = inputs.track(Path(model["model_path"]))
            if inputs.files[checkpoint.as_posix()] != model["model_sha256"]:
                raise ValueError("A frozen model checkpoint changed")
            if implementation == "elegantrl" and (checkpoint.suffix != ".pth" or not model.get("parameter_sha256") or not model.get("actor_parameter_sha256")):
                raise ValueError("Native checkpoint and network digests are required")
        repeats = manifest["repeat_verifications"]
        if len(repeats) != 40 or {r["run_id"] for r in repeats} != expected or not all(r["passed"] for r in repeats):
            raise ValueError("Deterministic-repeat evidence is incomplete")
        if any(r.get("fingerprints") != manifest["run_results"][r["run_id"]]["fingerprints"] for r in repeats):
            raise ValueError("A repeated result differs from its frozen manifest fingerprints")
        if implementation == "elegantrl":
            unchanged = manifest.get("models_unchanged_after_evaluation", [])
            if len(unchanged) != 5 or {r["seed"] for r in unchanged} != set(range(5)) or not all(r.get("parameters_and_checkpoint_unchanged_after_evaluation") is True for r in unchanged):
                raise ValueError("Native network/checkpoint preservation after inference is incomplete")
        for run_id, result in manifest["run_results"].items():
            actual[(key, run_id)] = _run_artifacts(inputs, folder, run_id, result["fingerprints"])
        chain = _verify_source_versions(inputs, manifest, folder)
        if chain:
            source_chains.append(chain)
        inputs.json(folder / "training_frozen_inputs.json")
        for frame_name in FRAME_NAMES:
            frame = inputs.csv(folder / (frame_name + ".csv"))
            if frame_name == "summary":
                if len(frame) != 40 or set(frame.run_id) != expected:
                    raise ValueError("Financial summary is not its complete forty-run manifest")
                for row in frame.to_dict("records"):
                    audited = audit["runs"][row["run_id"]]
                    if any(not math.isfinite(float(row[f])) or abs(row[f] - audited[f]) > 1e-8 for f in ("net_return", "max_drawdown", "fees")):
                        raise ValueError("Summary differs from independent ledger audit")
            elif frame_name == "training_metrics":
                if len(frame) != 5 or sorted(frame.seed.astype(int)) != list(range(5)):
                    raise ValueError("Five saved training records are required")
                for column in ("training_window_metadata", "training_episode_history", "fixed_feature_names", "observation_shape"):
                    if not all(column in m for m in models.values()):
                        raise ValueError("Frozen model records lack train evidence: " + column)
                    frame[column] = frame.seed.map(lambda seed: models[int(seed)][column])
                if not all(isinstance(v, dict) for v in frame.training_window_metadata):
                    raise ValueError("Training metadata must be structured frozen evidence")
                if implementation == "elegantrl":
                    for column in ("parameter_sha256", "actor_parameter_sha256", "gradient_updates", "actor_updates", "replay_samples_drawn", "replay_size", "collector_truncations", "collector_terminals"):
                        frame[column] = frame.seed.map(lambda seed: models[int(seed)].get(column))
            elif frame_name == "seed_statistics" and len(frame) != 4:
                raise ValueError("Expected four split/cost statistics per case")
            parts[frame_name].append(_annotate(frame, variant, implementation, algorithm, frame_name))
    reference = manifests[("original", "sb3", "ppo")]
    reference_hashes = {k.replace("\\", "/"): v for k, v in reference["source_sha256"].items()}
    for key, manifest in manifests.items():
        variant, implementation, algorithm = key
        hashes = {k.replace("\\", "/"): v for k, v in manifest["source_sha256"].items()}
        if any(hashes.get(relative) != reference_hashes[relative] for relative in COMMON_CORE):
            raise ValueError("Studies used different financial implementation sources")
        if any(manifest["config"].get(k) != reference["config"].get(k) for k in COMMON_CONFIG):
            raise ValueError("A common date/cost/capital/budget contract differs")
        dataset = inputs.track(DATASETS[variant])
        if manifest["frozen_inputs"]["dataset_sha256"] != inputs.files[dataset.as_posix()]:
            raise ValueError("A study used a different frozen pool dataset")
        if implementation == "elegantrl":
            if manifest["config"].get("algorithm_parameters") != spec["algorithms"][algorithm]["parameters"] or manifest["config"].get("upstream_commit") != vendor["commit"]:
                raise ValueError("Native model configuration differs from its frozen protocol")
            if any(m.get("commit") != vendor["commit"] or m.get("parameters") != spec["algorithms"][algorithm]["parameters"] for m in manifest["models"]):
                raise ValueError("Native checkpoint metadata differs from the approved source/parameters")
        else:
            original = manifests[("original", "sb3", algorithm)]["config"]
            for parameter in ("algorithm_parameters", "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "clip_range", "ent_coef", "reward", "feature_normalization", "train_episode_days", "parameter_search"):
                if manifest["config"].get(parameter) != original.get(parameter):
                    raise ValueError("A saved SB3 pool pair used different algorithm parameters: " + parameter)
    baseline_checks = []
    for key, manifest in manifests.items():
        variant, implementation, algorithm = key
        if implementation == "sb3" and algorithm == "ppo":
            continue
        reference_key = (variant, "sb3", "ppo")
        for run_id in manifest["run_results"]:
            if not run_id.endswith(tuple("__" + b for b in BASELINES)):
                continue
            if actual[(key, run_id)] != actual[(reference_key, run_id)] or manifest["run_results"][run_id]["fingerprints"] != manifests[reference_key]["run_results"][run_id]["fingerprints"]:
                raise ValueError("Same-pool shared baseline changed across implementations")
            baseline_checks.append({"universe_variant": variant, "implementation": implementation, "algorithm": algorithm.upper(), "run_id": run_id, "manifest_and_actual_four_files_identical": True})
    cross_pool = []
    for run_id in reference["run_results"]:
        if run_id.endswith(("__cash", "__btc_eth_hold")):
            old_key, new_key = ("original", "sb3", "ppo"), ("ex_ada", "sb3", "ppo")
            if any(actual[(old_key, run_id)][f] != actual[(new_key, run_id)][f] for f in ("equity", "fills", "boundary_equity")):
                raise ValueError("The unchanged cross-pool cash/BTCETH reference differs")
            evidence = _compare_weight_economics(inputs, studies[old_key] / "runs" / run_id / "weights.csv", studies[new_key] / "runs" / run_id / "weights.csv")
            cross_pool.append({"run_id": run_id, "equity_fills_boundary_byte_equal": True, "weight_evidence": evidence})
    if len(baseline_checks) != (len(studies) - 2) * 16 or len(cross_pool) != 8:
        raise ValueError("Baseline consistency coverage is incomplete")
    frames = {name: pd.concat(values, ignore_index=True) for name, values in parts.items()}
    frames["universe_changes"] = _universe_changes(inputs)
    review = inputs.json(NEW_ROOT / "sources/implementation_review.json")
    if not review.get("source") or not isinstance(review.get("rows"), list) or not review["rows"]:
        raise ValueError("Pinned implementation and adapter review is required")
    expected_reviews = {("sb3", "PPO"), ("sb3", "A2C"), *(("elegantrl", a.upper()) for a in algorithms)}
    if len(review["rows"]) != len(expected_reviews) or {(r["implementation"], r["algorithm"]) for r in review["rows"]} != expected_reviews:
        raise ValueError("Implementation review does not match actual algorithms")
    if any(r.get("commit") != vendor["commit"] for r in review["rows"] if r["implementation"] == "elegantrl"):
        raise ValueError("Native implementation review pins another upstream version")
    frames["implementation_review"] = pd.DataFrame(review["rows"])
    summary = frames["summary"]
    expected_summary = 2 * ((2 + len(algorithms)) * 24 + 16)
    if len(summary) != expected_summary or not summary.run_id.is_unique or len(frames["training_metrics"]) != len(studies) * 5 or len(frames["seed_statistics"]) != len(studies) * 4:
        raise ValueError("Deduplicated model/summary/statistics coverage is incomplete")
    for (variant, split, scenario), group in summary.groupby(["universe_variant", "split", "scenario"]):
        if len(group.loc[group.algorithm.eq("SHARED")]) != 4:
            raise ValueError("A selected pool/cost/split must contain exactly four shared baselines")
        for implementation, algorithm in (("sb3", "ppo"), ("sb3", "a2c"), *(("elegantrl", a) for a in algorithms)):
            models = group.loc[group.implementation.eq(implementation)]
            if sorted(models.loc[models.strategy.eq(algorithm), "seed"].astype(int)) != list(range(5)) or models.strategy.eq(algorithm + "_ensemble").sum() != 1:
                raise ValueError("Every actual seed and ensemble must be retained")
    source = _source(inputs, manifests, preservation, vendor)
    queries = {}
    for query_id, frame_name in QUERY_MAP.items():
        meta = json.loads(json.dumps(source, ensure_ascii=False))
        meta["files"] = [frame_name + ".csv", "manifest.json", "input_integrity_before.json", "input_integrity_after.json", *source["files"]]
        if frame_name in ("training_metrics", "universe_changes"):
            meta["period"] = {"start": "2023-01-01T00:00:00Z", "end": "2025-07-25T00:00:00Z" if frame_name == "training_metrics" else "2026-10-23T00:00:00Z"}
        queries[query_id] = {"source": meta, "rows": records(frames[frame_name])}
    review_source = json.loads(json.dumps(review["source"], ensure_ascii=False))
    review_source["files"] = ["implementation_review.csv", str(NEW_ROOT / "sources/implementation_review.json"), str(NEW_ROOT / "frozen_spec.json"), *review_source.get("files", [])]
    queries["implementation_review"] = {"source": review_source, "rows": review["rows"]}
    after = inputs.verify()
    output.mkdir(parents=True, exist_ok=True)
    write(output / "input_integrity_before.json", {"sha256": inputs.files})
    write(output / "input_integrity_after.json", {"sha256": after, "passed": True})
    for frame_name, frame in frames.items():
        frame.to_csv(output / (frame_name + ".csv"), index=False, float_format="%.17g", encoding="utf-8-sig")
    evidence_as_of = (pd.Timestamp(reference["config"]["test_end"]).tz_convert("UTC") - pd.Timedelta(hours=1)).date().isoformat()
    native_test = summary.loc[summary.implementation.eq("elegantrl") & summary.split.eq("test") & summary.strategy.str.endswith("_ensemble")]
    report_title = "这365天，ElegantRL组合仍为负收益" if len(native_test) and native_test.net_return.lt(0).all() else "ElegantRL与SB3：同口径收益和回撤"
    write(output / "reviewed.json", {"surface": "report", "status": "reviewed", "buildStatus": "complete",
        "title": report_title, "generatedAt": datetime.now(timezone.utc).isoformat(),
        "report": {"asOf": evidence_as_of, "title": report_title,
                   "studyConfig": {"costs": reference["config"]["costs"], "test_start": reference["config"]["test_start"], "test_end": reference["config"]["test_end"], "hourly_drawdown": True}},
        "filters": [], "queries": queries})
    (output / "README.md").write_text(_readme(summary, algorithms, preservation), encoding="utf-8")
    inputs.verify()
    write(output / "manifest.json", {"assembler": Path(__file__).name, "assembler_source_sha256": sha(Path(__file__)),
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "complete",
        "source_studies": {"__".join(key): str(path) for key, path in studies.items()}, "input_sha256": inputs.files,
        "input_unchanged_after_assembly": True, "original_preservation": preservation, "upstream_source": vendor,
        "shared_baseline_verifications": baseline_checks, "cross_pool_cash_btceth_verifications": cross_pool,
        "historical_display_finalization_chains": source_chains,
        "independent_audit_summaries": {"__".join(key): {field: doc[field] for field in ("status", "checks_total", "checks_failed", "runs_checked")} for key, doc in audits.items()},
        "global_independent_audit_status": global_audit["status"], "numeric_completion_status": completion["status"],
        "row_counts": {name: len(frame) for name, frame in frames.items()},
        "generated_sha256": {name: sha(output / name) for name in OWNED_NAMES},
        "post_hoc_research_extension": True, "numeric_results_changed": False, "production_changes": False})
    print(f"Assembled {len(summary)} saved evaluations, {len(frames['training_metrics'])} models, {len(frames['seed_statistics'])} seed statistics: {output}")


def self_check() -> None:
    """Check identity/time annotation only; no financial sample is published."""
    row = pd.DataFrame([{"run_id": "test__base__sac__seed0", "strategy": "sac", "timestamp": "2025-10-23 00:00:00+00:00"}])
    value = records(_annotate(row, "original", "elegantrl", "sac", "daily_equity"))[0]
    assert value["run_id"] == "elegantrl__original__test__base__sac__seed0"
    assert value["raw_run_id"] == "test__base__sac__seed0" and value["timestamp"] == "2025-10-23T00:00:00.000Z"
    assert value["algorithm"] == "SAC" and value["implementation"] == "elegantrl"
    assert len(_expected_runs("sac")) == len(_expected_runs("td3")) == 40
    print("SELF_CHECK_PASS: global identity, UTC ISO annotation and expected-run schema; no economics executed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    self_check() if args.self_check else main(args.output_dir)
