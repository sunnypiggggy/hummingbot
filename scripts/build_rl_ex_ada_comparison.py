"""Assemble four completed, audited studies without rerunning any economics.

All studies, models, datasets and financial implementations are read-only.
Output is confined to a separate comparison report directory. Missing final
completion/audit evidence is an error, never a reason to publish partial data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NEW_ROOT = ROOT / "results/backtests/crypto_rl_ex_ada_20261024"
OUTPUT = NEW_ROOT / "report"
STUDIES = {
    ("original", "ppo"): ROOT / "results/backtests/crypto_rl_top10_20261024",
    ("original", "a2c"): ROOT / "results/backtests/crypto_rl_a2c_20261024",
    ("ex_ada", "ppo"): NEW_ROOT / "ppo",
    ("ex_ada", "a2c"): NEW_ROOT / "a2c",
}
DATASETS = {"original": ROOT / "data/crypto_rl_top10/dataset.pkl.gz",
            "ex_ada": ROOT / "data/crypto_rl_top10_ex_ada/dataset.pkl.gz"}
FINANCIAL_CORE = ("scripts/backtest_crypto_rl_top10.py", "scripts/crypto_rl_data.py",
                  "scripts/crypto_rl_portfolio_env.py", "scripts/crypto_rl_reporting.py")
DISPLAY_SOURCES = ("scripts/crypto_rl_report_app.py", "scripts/crypto_rl_report_app.jsx", "scripts/crypto_rl_report_app.css")
BASELINES = ("cash", "top10_equal", "top10_cap", "btc_eth_hold")
FRAME_NAMES = ("summary", "daily_equity", "monthly_returns", "seed_statistics", "training_metrics")
QUERY_MAP = {"strategy_summary": "summary", "equity_history": "daily_equity",
             "monthly_returns": "monthly_returns", "seed_statistics": "seed_statistics",
             "training_metrics": "training_metrics", "universe_changes": "universe_changes"}
OWNED_NAMES = tuple(name + ".csv" for name in (*FRAME_NAMES, "universe_changes")) + (
    "reviewed.json", "README.md", "input_integrity_before.json", "input_integrity_after.json")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class Inputs:
    """Pin every consumed file and verify it again before publishing outputs."""

    def __init__(self):
        self.files = {}

    def track(self, path: Path) -> Path:
        path = path.resolve()
        key = path.as_posix()
        digest = sha(path)
        if key in self.files and self.files[key] != digest:
            raise ValueError("An input changed while assembling: " + key)
        self.files[key] = digest
        return path

    def json(self, path: Path):
        return read(self.track(path))

    def csv(self, path: Path):
        return pd.read_csv(self.track(path), float_precision="round_trip")

    def verify(self):
        current = {path: sha(Path(path)) for path in self.files}
        if current != self.files:
            raise ValueError("Read-only study inputs changed during comparison assembly")
        return current


def require_audit(document: dict, runs: int, label: str) -> None:
    if document.get("status") != "passed" or document.get("checks_failed") != 0 or document.get("runs_checked") != runs:
        raise ValueError("Required independent audit is incomplete or failed: " + label)


def _expected_runs(algorithm: str) -> set[str]:
    return {f"{split}__{scenario}__{strategy}" + (f"__seed{seed}" if seed is not None else "")
            for split in ("validation", "test") for scenario in ("base", "stress")
            for strategy, seed in ([(algorithm, s) for s in range(5)]
                                   + [(algorithm + "_ensemble", None)] + [(b, None) for b in BASELINES])}


def annotate(frame: pd.DataFrame, variant: str, algorithm: str, frame_name: str) -> pd.DataFrame:
    frame = frame.copy()
    if frame_name not in ("training_metrics", "seed_statistics") and algorithm == "a2c":
        frame = frame.loc[frame.strategy.isin(("a2c", "a2c_ensemble"))].copy()
    frame["universe_variant"] = variant
    frame["algorithm"] = algorithm.upper()
    if "strategy" in frame:
        frame.loc[~frame.strategy.isin(("ppo", "ppo_ensemble", "a2c", "a2c_ensemble")), "algorithm"] = "SHARED"
    if "run_id" in frame:
        frame["raw_run_id"] = frame.run_id
        frame["run_id"] = variant + "__" + frame.raw_run_id
    if frame_name == "daily_equity":
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="raise")
    return frame


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso", double_precision=15))


def _verify_preservation(inputs: Inputs) -> dict:
    path = NEW_ROOT / "original_research_preservation.json"
    preservation = inputs.json(path)
    if len(preservation["files"]) != 578:
        raise ValueError("Expected the original 578-file preservation contract")
    for relative, expected in preservation["files"].items():
        source = inputs.track(ROOT / relative.replace("\\", "/"))
        if source.stat().st_size != expected["size_bytes"] or inputs.files[source.as_posix()] != expected["sha256"]:
            raise ValueError("An original frozen artifact changed: " + relative)
    for relative in preservation["originally_absent"]:
        if (ROOT / relative.replace("\\", "/")).exists():
            raise ValueError("Unexpected backdated artifact in original study: " + relative)
    return {"files_checked": 578, "originally_absent_checked": len(preservation["originally_absent"]),
            "snapshot_sha256": inputs.files[path.resolve().as_posix()], "passed": True}


def _run_artifacts(inputs: Inputs, folder: Path, run_id: str, expected: dict) -> dict:
    actual = {}
    for field, filename in (("equity", "equity_1h.csv.gz"), ("fills", "fills.csv"),
                            ("weights", "weights.csv"), ("boundary_equity", "boundary_equity.csv")):
        path = inputs.track(folder / "runs" / run_id / filename)
        if field == "equity":
            with gzip.open(path, "rb") as stream:
                actual[field] = hashlib.sha256(stream.read()).hexdigest()
        else:
            actual[field] = inputs.files[path.as_posix()]
        if actual[field] != expected[field]:
            raise ValueError(f"Actual run artifact no longer matches frozen fingerprint: {folder}/{run_id}/{field}")
    return actual


def _verify_source_versions(inputs: Inputs, manifest: dict, folder: Path) -> dict | None:
    hashes = {key.replace("\\", "/"): value for key, value in manifest["source_sha256"].items()}
    actual = {relative: inputs.files[inputs.track(ROOT / relative).as_posix()] for relative in hashes}
    drift = [relative for relative in hashes if actual[relative] != hashes[relative]]
    if not drift:
        return None
    if folder != STUDIES[("original", "ppo")] or any(relative not in DISPLAY_SOURCES for relative in drift):
        raise ValueError("Unknown frozen source drift: " + ", ".join(drift))
    path = folder / "report_finalization.json"
    chain = inputs.json(path)
    if chain.get("status") not in ("display_complete_pending_browser_QA", "display_complete", "complete") or not chain.get("completed_at"):
        raise ValueError("Historical display-only finalization did not complete")
    if chain.get("numeric_results_changed") is not False or chain.get("frozen_financial_sources_preserved") is not True or chain.get("financial_sources_checked") != 4:
        raise ValueError("Display source exception lacks preserved-finance evidence")
    if pd.to_datetime(chain["started_at"], utc=True) > pd.to_datetime(chain["completed_at"], utc=True):
        raise ValueError("Invalid historical display-finalization chronology")
    normalized = {field: {key.replace("\\", "/"): value for key, value in chain[field].items()}
                  for field in ("before_sha256", "at_numeric_run_sha256", "after_sha256")}
    for relative in DISPLAY_SOURCES:
        if normalized["before_sha256"].get(relative) != hashes[relative] or normalized["at_numeric_run_sha256"].get(relative) != hashes[relative] or normalized["after_sha256"].get(relative) != actual[relative]:
            raise ValueError("The display finalization before/numeric/after hash chain is broken: " + relative)
    return {"study": "original__ppo", "chain_path": path.as_posix(), "chain_sha256": inputs.files[path.resolve().as_posix()],
            "status": chain["status"], "completed_at": chain["completed_at"], "display_only_changed_sources": drift,
            "numeric_results_changed": False, "financial_sources_preserved": True,
            "verified_chain": normalized}


def _compare_weight_economics(inputs: Inputs, old_path: Path, new_path: Path) -> dict:
    """Allow pool-name/slot metadata changes only in demonstrably zero rows."""
    def load(path):
        frame = pd.read_csv(inputs.track(path), dtype={"asset_key": str}, float_precision="round_trip")
        frame["date"] = pd.to_datetime(frame.date, utc=True, errors="raise")
        if frame.duplicated(["date", "asset_key"]).any() or frame.asset_key.isna().any():
            raise ValueError("Duplicate/missing identity in baseline weights")
        frame = frame.set_index(["date", "asset_key"]).sort_index()
        if frame.actual_weight.isna().any() or not frame.actual_weight.map(math.isfinite).all():
            raise ValueError("Invalid actual baseline weight")
        if not frame.target_weight.dropna().map(math.isfinite).all():
            raise ValueError("Invalid target baseline weight")
        return frame
    old, new = load(old_path), load(new_path)
    old_only, new_only = old.index.difference(new.index), new.index.difference(old.index)
    columns = ["target_weight", "actual_weight"]
    is_zero = lambda frame: frame[columns].fillna(0).eq(0).all(axis=1)
    if not is_zero(old.loc[old_only]).all() or not is_zero(new.loc[new_only]).all():
        raise ValueError("A removed/added baseline pool identity has nonzero economics")
    old_active = old.loc[~is_zero(old), columns]
    new_active = new.loc[~is_zero(new), columns]
    if not old_active.equals(new_active):
        raise ValueError("Cross-pool baseline nonzero target/actual weights differ")
    common = old.index.intersection(new.index)
    changed_metadata = old.loc[common, ["slot", "symbol"]].ne(new.loc[common, ["slot", "symbol"]]).any(axis=1)
    changed_keys = common[changed_metadata]
    if not is_zero(old.loc[changed_keys]).all() or not is_zero(new.loc[changed_keys]).all():
        raise ValueError("A changed slot/symbol metadata row has nonzero baseline economics")
    return {"economic_weights_equal": True, "comparison_key": ["date_utc", "asset_key"],
            "nonzero_target_or_actual_rows_compared": len(old_active),
            "removed_zero_identity_rows": len(old_only), "added_zero_identity_rows": len(new_only),
            "changed_common_zero_slot_metadata_rows": len(changed_keys),
            "zero_slot_metadata_difference_rows": len(old_only) + len(new_only) + len(changed_keys),
            "all_metadata_differences_have_zero_target_and_actual": True,
            "full_weights_file_byte_equal": inputs.files[old_path.resolve().as_posix()] == inputs.files[new_path.resolve().as_posix()]}


def _universe_changes(inputs: Inputs) -> pd.DataFrame:
    frame = pd.read_csv(inputs.track(NEW_ROOT / "sources/pool_anchor_changes.csv"), dtype=str, keep_default_na=False)
    if len(frame) != 48 or not frame.anchor.is_unique:
        raise ValueError("Expected 48 unique original/new universe selection anchors")
    proofs = inputs.json(NEW_ROOT / "sources/selection_plan.json")
    evidence = {p["anchor"]: p for p in proofs["selection_evidence"]}
    split = lambda value: str(value).split("|") if value else []
    rows = []
    for item in frame.to_dict("records"):
        old_keys, new_keys = split(item["old_asset_keys"]), split(item["new_asset_keys"])
        old, new = split(item["old_symbols"]), split(item["new_symbols"])
        removed, added = split(item["removed_symbols"]), split(item["added_symbols"])
        if len(old) != 10 or len(new) != 10 or "2010" in new_keys or len(set(new_keys)) != 10:
            raise ValueError("Universe change table violates rank-refilled ten-asset contract")
        proof = evidence[pd.Timestamp(item["anchor"]).isoformat()]
        if proof["selected_asset_keys"] != new_keys:
            raise ValueError("Universe table differs from frozen selection evidence")
        added_keys = split(item["added_asset_keys"])
        ranks = {str(p["cmc_id"]): p["rank"] for p in proof["considered"]}
        detail = "; ".join(f"{symbol} (CMC{key}, 历史市值第{ranks[key]}名)" for symbol, key in zip(added, added_keys))
        rows.append({"anchor": pd.Timestamp(item["anchor"]), "original_assets": old, "ex_ada_assets": new,
                     "original_asset_keys": old_keys, "ex_ada_asset_keys": new_keys,
                     "removed_assets": removed, "added_assets": added, "changed": bool(removed or added),
                     "replacement_detail": "移除ADA，按同一资格规则顺位补入 " + detail if detail else "该时点原池未含ADA，成分保持一致"})
    result = pd.DataFrame(rows)
    result["anchor"] = pd.to_datetime(result.anchor, utc=True, errors="raise")
    if int(result.changed.sum()) != 45:
        raise ValueError("Expected 45 ADA membership replacement anchors")
    return result


def _check_output(output: Path) -> None:
    if output != OUTPUT.resolve():
        raise ValueError("Only the separate ex-ADA report directory is writable")
    manifest = output / "manifest.json"
    existing = [name for name in OWNED_NAMES if (output / name).exists()]
    if not manifest.exists():
        if existing:
            raise ValueError("Preserving comparison files without an assembler ownership manifest")
        return
    previous = read(manifest)
    if previous.get("assembler") != Path(__file__).name:
        raise ValueError("Preserving a manifest owned by another task")
    for name, expected in previous.get("generated_sha256", {}).items():
        if sha(output / name) != expected:
            raise ValueError("Preserving independently edited comparison artifact: " + name)


def _financial_source(inputs: Inputs, manifests: dict) -> dict:
    return {
        "label": "原Top10与历史排名补位后剔除ADA重训的PPO/A2C敏感性对照",
        "classification": "modeled", "timezone": "UTC",
        "period": {"start": "2025-07-25T00:00:00Z", "end": "2026-10-23T00:00:00Z"},
        "evidenceFlow": [
            {"title": "同源行情和资金", "detail": "共同19资产的五类小时OHLCV及缺值位置完全一致；CMC2010在历史排名扫描前排除，仍逐时点补足10个合格资产。每次10000 USDT。"},
            {"title": "同算法重新训练", "detail": "原池/新池各PPO与A2C、每算法5种子×32768步；算法参数及现货账本不变。五模型平均目标权重后在一个真实模拟账本执行。"},
            {"title": "四份独立账本审计", "detail": "四研究各40份评估独立重建并对账；每池16个共同基准四类文件逐字节一致。跨池8个现金/BTCETH基准的成交/权益/奖励字节一致，权重按稳定身份核对经济一致，零仓槽位元数据差异另列。"},
            {"title": "冻结证据保护", "detail": "完整审计与数值完成标记齐备后汇总，复核旧578份证据及全部读取输入的前后SHA；仅复制保存数值，不训练、模拟或挑选种子。"},
        ],
        "caveats": [
            "排除ADA是查看本年度结果后的事后研究选择，不能把本次测试称为全新未见样本。训练截止仍早于验证/测试，不代表研究选择过程未见这些行情。",
            "新旧差异包含剔除ADA、按排名补入其他资产、训练资产池变化以及重新学习的共同影响；不是仅移除ADA的因果效应，也不是固定9币或用现金直接替代ADA的反事实。",
            "相同编号种子仅作描述性配对；不是五份独立市场实验。保留全部种子和主组合，未据测试结果挑参数、种子或checkpoint。",
            "PPO/A2C市场交互步相同，优化器、更新频率和计算量不同；A2C是TradeMaster列示算法的SB3实现，非原trainer/env复现。",
            "回撤使用完整小时收盘权益及初始本金；图表按日末采样。小时内更深回撤不可见，不能直接当作早期分钟回撤口径。",
            "历史市值为回溯快照，使用日期加两日可用假设；原始发布版本未核实。至少31个完整先前日特征才可买入，不补造价格。",
            "按小数币数量成交，未重建历史数量步长、最小名义金额或盘口深度。现金5%与单币25%为目标约束，价格漂移后不持续硬钳制；BTC/ETH被动参考初始各47.5%是限仓例外。",
            "研究回测不构成实盘授权；未修改生产交易、风控、凭证或部署。",
        ],
        "metricDefinitions": [
            {"label": "扣费净收益", "definition": "保存的期末现金+持币真实价格估值相对10000 USDT的变化，已扣实际模拟每腿费用及不利滑点；净清算值另列。"},
            {"label": "小时最大回撤", "definition": "完整小时收盘权益相对包含初始本金的历史峰值的最大跌幅；日图保留该小时历史峰值口径。"},
            {"label": "持币仓位", "definition": "保存的每小时持币市值占组合净值比例的时间均值，不能将不同仓位的收益差异只归为选币能力。"},
        ],
        "files": [path for path in inputs.files if not "/runs/" in path],
        "dataset_sha256": {variant: manifests[(variant, "ppo")]["frozen_inputs"]["dataset_sha256"] for variant in DATASETS},
    }


def _readme(summary: pd.DataFrame) -> str:
    lines = ["# 剔除ADA后重训：PPO与A2C历史Top10敏感性对照", "",
             "这是查看原测试结果后发起的敏感性研究。CMC2010在历史排名扫描前排除，各时点仍补足10个符合既定规则的资产。新旧变化包含顺位补位和重新学习，不能解释为单独移除ADA的因果效应。", "",
             "固定测试：[2025-10-23 00:00,2026-10-23 00:00) UTC；每组合10000 USDT。日00基于已闭合日数据判断，00卖、01买。主成本每边手续费0.10%与不利滑点0.02%；压力成本0.20%与0.05%。最大回撤按小时收盘权益计算。", "",
             "| 资产池 | 组合 | 主成本净收益 | 主成本小时回撤 | 压力净收益 | 压力小时回撤 | 主成本费用USDT |", "|---|---|---:|---:|---:|---:|---:|"]
    selected = summary.loc[summary.split.eq("test") & summary.seed.isna()]
    for variant in ("original", "ex_ada"):
        for strategy in ("ppo_ensemble", "a2c_ensemble", *BASELINES):
            match = selected.loc[selected.universe_variant.eq(variant) & selected.strategy.eq(strategy)].set_index("scenario")
            base, stress = match.loc["base"], match.loc["stress"]
            label = "原币池" if variant == "original" else "剔除ADA后补位重训"
            lines.append(f"| {label} | {base['label']} | {base.net_return:.2%} | {base.max_drawdown:.2%} | {stress.net_return:.2%} | {stress.max_drawdown:.2%} | {base.fees:.2f} |")
    lines += ["", "两币池各算法固定5个种子、每个32768步，全部最终模型与单种子结果均保留。主组合平均五模型目标权重后在单一账本成交，不是平均五份回测收益。压力成本只重放，不重新训练。研究过程已经看过同一365天，模型切分的test称谓不表示研究选择未见样本。", "",
              "本报告只读四份已审计研究的保存数值。四份40run去重后共128个全局唯一评估；每币池16个共同基准的四类文件与manifest逐字节一致。跨币池现金/BTCETH的8份参考成交、权益及奖励字节一致，非零目标/实际仓位一致；池中零仓币身份和槽位元数据会变化，该差异在manifest单列，不是经济收益差异。共享金融源SHA与现值一致，旧578份保护证据未变。", "",
              "收益与回撤需同时查看成本、仓位、各种子稳定性及月度结果。本次结果不得外推为ADA永久拖累、剔除任意亏损币即可提高未来收益，或新模型已获实盘授权。", "",
              "## 文件与复现", "",
              "- summary.csv、daily_equity.csv、monthly_returns.csv：128份去重评估及原始run编号。",
              "- seed_statistics.csv：16份已保存种子统计；training_metrics.csv：20个冻结模型，训练窗口元数据由manifest恢复为对象。",
              "- universe_changes.csv：48个历史选择时点的实际移除/补入/完整币池；45个时点发生替补。",
              "- reviewed.json：六类带来源的报告数据；manifest.json和input_integrity_*.json：输入/输出哈希、审计及基准一致性证据。",
              "- ../{ppo,a2c}/：新训练模型、逐笔交易、小时权益及独立审计；原研究目录保留。", "",
              "本汇总不训练、不模拟、不改旧结果。所有数值和审计完成后，从仓库根目录执行：", "",
              "```powershell", "& 'C:\\Users\\sunny\\.cache\\codex-research-envs\\crypto-ppo-20261024\\Scripts\\python.exe' scripts/build_rl_ex_ada_comparison.py", "```", "",
              "原生框架适配说明见 ../../rl_framework_comparison_20261024/README.md；本轮PPO/A2C均为SB3实现。生产系统未改变。", ""]
    return "\n".join(lines)


def main(output_dir: Path = OUTPUT) -> None:
    output = output_dir.resolve()
    _check_output(output)
    inputs = Inputs()
    completion = inputs.json(NEW_ROOT / "numeric_completion.json")
    if completion.get("status") != "complete" or completion.get("evaluations") != 80 or completion.get("repeat_checks") != 80 or completion.get("shared_baselines_match") is not True:
        raise ValueError("The new 80-evaluation numeric study is not complete")
    global_audit = inputs.json(NEW_ROOT / "independent_audit.json")
    require_audit(global_audit, 80, "ex-ADA global")
    if global_audit.get("phase") != "full_independent_audit":
        raise ValueError("A data preflight cannot replace the full independent audit")
    inputs.json(NEW_ROOT / "frozen_spec.json")
    inputs.json(NEW_ROOT / "training_frozen_inputs.json")
    preservation = _verify_preservation(inputs)
    data_summary = inputs.json(NEW_ROOT / "sources/data_change_summary.json")
    if data_summary.get("parent_integrity", {}).get("passed") is not True or len(data_summary["shared_prices"]["common_asset_keys"]) != 19:
        raise ValueError("The shared-price/original-input data protection evidence is incomplete")
    if data_summary["dataset_sha256"] != sha(inputs.track(DATASETS["ex_ada"])) or data_summary["shared_prices"]["parent_dataset_sha256"] != sha(inputs.track(DATASETS["original"])):
        raise ValueError("The data-agent comparison refers to different dataset bytes")
    if set(data_summary["shared_prices"]["matrices"]) != {"hourly_" + field for field in ("open", "high", "low", "close", "volume")}:
        raise ValueError("All five shared OHLCV matrix comparisons are required")
    for item in data_summary["shared_prices"]["matrices"].values():
        if item.get("passed") is not True or item.get("assets") != 19:
            raise ValueError("The common 19-asset OHLCV comparison did not pass")
    manifests, audits, combined, actual = {}, {}, {name: [] for name in FRAME_NAMES}, {}
    for (variant, algorithm), folder in STUDIES.items():
        key = (variant, algorithm)
        manifest = manifests[key] = inputs.json(folder / "manifest.json")
        audit = audits[key] = inputs.json(folder / "independent_audit.json")
        require_audit(audit, 40, variant + ":" + algorithm)
        expected = _expected_runs(algorithm)
        if set(manifest["run_results"]) != expected or set(audit["runs"]) != expected:
            raise ValueError("A forty-run family is incomplete: " + str(key))
        if sorted(m["seed"] for m in manifest["models"]) != list(range(5)) or any(m["timesteps"] != 32768 for m in manifest["models"]):
            raise ValueError("Five fixed 32768-step models are required")
        for model in manifest["models"]:
            path = inputs.track(Path(model["model_path"]))
            if inputs.files[path.as_posix()] != model["model_sha256"]:
                raise ValueError("A frozen model checkpoint changed")
        repeats = manifest["repeat_verifications"]
        if len(repeats) != 40 or {r["run_id"] for r in repeats} != expected or not all(r["passed"] for r in repeats):
            raise ValueError("Missing deterministic-repeat evidence")
        for run_id, result in manifest["run_results"].items():
            actual[(variant, algorithm, run_id)] = _run_artifacts(inputs, folder, run_id, result["fingerprints"])
        for filename in FRAME_NAMES:
            frame = inputs.csv(folder / (filename + ".csv"))
            if filename == "summary":
                if len(frame) != 40 or set(frame.run_id) != expected:
                    raise ValueError("Summary coverage differs from its manifest")
                for row in frame.to_dict("records"):
                    audited = audit["runs"][row["run_id"]]
                    if any(not math.isfinite(float(row[field])) or abs(row[field] - audited[field]) > 1e-8
                           for field in ("net_return", "max_drawdown", "fees")):
                        raise ValueError("Summary does not reconcile with independent audit: " + row["run_id"])
            elif filename == "training_metrics":
                models = {int(m["seed"]): m for m in manifest["models"]}
                if len(frame) != 5 or sorted(frame.seed.astype(int)) != list(range(5)):
                    raise ValueError("Expected five saved training records")
                for column in ("training_window_metadata", "training_episode_history", "fixed_feature_names", "observation_shape"):
                    frame[column] = frame.seed.map(lambda seed: models[int(seed)][column])
                if not all(isinstance(value, dict) for value in frame.training_window_metadata):
                    raise ValueError("Training metadata must be real manifest objects, not CSV repr text")
            elif filename == "seed_statistics" and len(frame) != 4:
                raise ValueError("Expected four split/cost seed statistics per study")
            combined[filename].append(annotate(frame, variant, algorithm, filename))
        inputs.json(folder / "training_frozen_inputs.json")
    for algorithm in ("ppo", "a2c"):
        if sha(STUDIES[("ex_ada", algorithm)] / "manifest.json") != completion["manifest_sha256"][algorithm]:
            raise ValueError("Numeric completion pins a different new manifest")
        if sha(STUDIES[("original", algorithm)] / "manifest.json") != completion["original_manifest_sha256"][algorithm]:
            raise ValueError("Numeric completion pins a different original manifest")
    display_finalization_chains = []
    for key, manifest in manifests.items():
        hashes = {k.replace("\\", "/"): v for k, v in manifest["source_sha256"].items()}
        chain = _verify_source_versions(inputs, manifest, STUDIES[key])
        if chain:
            display_finalization_chains.append(chain)
        for relative in FINANCIAL_CORE:
            if hashes[relative] != {k.replace("\\", "/"): v for k, v in manifests[("original", "ppo")]["source_sha256"].items()}[relative]:
                raise ValueError("The four studies used different financial cores")
        for common in ("train_start", "train_end", "validation_start", "validation_end", "test_start", "test_end",
                       "costs", "capital", "seeds", "timesteps_per_seed", "net_arch", "cash_target_floor", "single_coin_target_cap"):
            if manifest["config"].get(common) != manifests[("original", "ppo")]["config"].get(common):
                raise ValueError("A shared comparison protocol differs: " + common)
        original = manifests[("original", key[1])]
        for parameter in ("algorithm_parameters", "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda",
                          "clip_range", "ent_coef", "reward", "feature_normalization", "train_episode_days", "parameter_search"):
            if manifest["config"].get(parameter) != original["config"].get(parameter):
                raise ValueError("The original/new algorithm parameters differ: " + parameter)
        if manifest["frozen_inputs"]["dataset_sha256"] != sha(inputs.track(DATASETS[key[0]])):
            raise ValueError("A study dataset changed")
    baseline_checks, cross_pool_checks = [], []
    for variant in DATASETS:
        for run_id in manifests[(variant, "ppo")]["run_results"]:
            if not run_id.endswith(tuple("__" + strategy for strategy in BASELINES)):
                continue
            ppo = manifests[(variant, "ppo")]["run_results"][run_id]["fingerprints"]
            a2c = manifests[(variant, "a2c")]["run_results"][run_id]["fingerprints"]
            if ppo != a2c or actual[(variant, "ppo", run_id)] != actual[(variant, "a2c", run_id)]:
                raise ValueError("Within-pool shared baseline differs: " + variant + ":" + run_id)
            baseline_checks.append({"universe_variant": variant, "run_id": run_id, "manifest_and_actual_four_files_identical": True})
    for run_id in manifests[("original", "ppo")]["run_results"]:
        if run_id.endswith(("__cash", "__btc_eth_hold")):
            old, new = actual[("original", "ppo", run_id)], actual[("ex_ada", "ppo", run_id)]
            if any(old[field] != new[field] for field in ("equity", "fills", "boundary_equity")):
                raise ValueError("Cross-pool cash/BTCETH reference changed: " + run_id)
            weight_check = _compare_weight_economics(inputs, STUDIES[("original", "ppo")] / "runs" / run_id / "weights.csv",
                                                     STUDIES[("ex_ada", "ppo")] / "runs" / run_id / "weights.csv")
            cross_pool_checks.append({"run_id": run_id, "economic_results_equal": True,
                                      "equity_fills_boundary_byte_equal": True, "weight_evidence": weight_check})
    if len(baseline_checks) != 32 or len(cross_pool_checks) != 8:
        raise ValueError("Expected 16 shared baselines per pool and eight unchanged cross-pool references")
    frames = {name: pd.concat(parts, ignore_index=True) for name, parts in combined.items()}
    frames["universe_changes"] = _universe_changes(inputs)
    summary = frames["summary"]
    if len(summary) != 128 or not summary.run_id.is_unique or len(frames["seed_statistics"]) != 16 or len(frames["training_metrics"]) != 20:
        raise ValueError("Wrong deduplicated summary/training/statistic coverage")
    for variant, selected in summary.groupby("universe_variant"):
        if len(selected) != 64 or selected.algorithm.value_counts().to_dict() != {"PPO": 24, "A2C": 24, "SHARED": 16}:
            raise ValueError("Each variant needs 64 unique algorithm/baseline results")
        for (split, scenario), group in selected.groupby(["split", "scenario"]):
            if len(group) != 16:
                raise ValueError("Incomplete split/cost group")
            for algorithm in ("ppo", "a2c"):
                if sorted(group.loc[group.strategy.eq(algorithm), "seed"].astype(int)) != list(range(5)) or group.strategy.eq(algorithm + "_ensemble").sum() != 1:
                    raise ValueError("Every seed and ensemble must be retained")
    source = _financial_source(inputs, manifests)
    queries = {}
    for query_id, frame_name in QUERY_MAP.items():
        meta = json.loads(json.dumps(source, ensure_ascii=False))
        meta["files"] = [frame_name + ".csv", "manifest.json", "input_integrity_before.json", "input_integrity_after.json", *source["files"]]
        if frame_name in ("training_metrics", "universe_changes"):
            meta["period"] = {"start": "2023-01-01T00:00:00Z", "end": "2026-10-23T00:00:00Z" if frame_name == "universe_changes" else "2025-07-25T00:00:00Z"}
        queries[query_id] = {"source": meta, "rows": records(frames[frame_name])}
    after = inputs.verify()
    output.mkdir(parents=True, exist_ok=True)
    write(output / "input_integrity_before.json", {"sha256": inputs.files})
    write(output / "input_integrity_after.json", {"sha256": after, "passed": True})
    for name, frame in frames.items():
        frame.to_csv(output / (name + ".csv"), index=False, float_format="%.17g", encoding="utf-8-sig")
    write(output / "reviewed.json", {
        "surface": "report", "status": "reviewed", "buildStatus": "complete",
        "title": "剔除ADA后重训：PPO与A2C历史Top10对照", "generatedAt": datetime.now(timezone.utc).isoformat(),
        "report": {"asOf": "2026-10-24", "title": "剔除ADA后重训：PPO与A2C历史Top10对照",
                   "studyConfig": {"costs": manifests[("original", "ppo")]["config"]["costs"],
                                   "test_start": "2025-10-23T00:00:00Z", "test_end": "2026-10-23T00:00:00Z", "hourly_drawdown": True}},
        "filters": [], "queries": queries})
    (output / "README.md").write_text(_readme(summary), encoding="utf-8")
    inputs.verify()
    write(output / "manifest.json", {
        "assembler": Path(__file__).name, "assembler_source_sha256": sha(Path(__file__)),
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "complete",
        "source_studies": {variant + "__" + algorithm: str(folder) for (variant, algorithm), folder in STUDIES.items()},
        "dataset_sha256": source["dataset_sha256"], "input_sha256": inputs.files,
        "input_unchanged_after_assembly": True, "original_preservation": preservation,
        "shared_baseline_verifications": baseline_checks, "cross_pool_cash_btceth_verifications": cross_pool_checks,
        "historical_display_finalization_chains": display_finalization_chains,
        "independent_audit_summaries": {v + "__" + a: {k: audit[k] for k in ("status", "checks_total", "checks_failed", "runs_checked")}
                                        for (v, a), audit in audits.items()},
        "global_independent_audit_status": global_audit["status"], "numeric_completion_status": completion["status"],
        "row_counts": {name: len(frame) for name, frame in frames.items()},
        "generated_sha256": {name: sha(output / name) for name in OWNED_NAMES},
        "post_hoc_retraining_sensitivity": True, "numeric_results_changed": False, "production_changes": False})
    print(f"Assembled 128 distinct saved runs, 20 models, 16 seed statistics and 48 universe anchors: {output}")


def self_check() -> None:
    """In-memory schema checks only; do not create pretend financial results."""
    sample = pd.DataFrame([{"run_id": "test__base__ppo__seed0", "strategy": "ppo",
                            "timestamp": "2025-10-23 00:00:00+00:00"}])
    frame = annotate(sample, "ex_ada", "ppo", "daily_equity")
    row = records(frame)[0]
    assert row["run_id"] == "ex_ada__test__base__ppo__seed0"
    assert row["raw_run_id"] == "test__base__ppo__seed0"
    assert row["timestamp"] == "2025-10-23T00:00:00.000Z"
    assert row["algorithm"] == "PPO" and row["universe_variant"] == "ex_ada"
    assert len(_expected_runs("ppo")) == len(_expected_runs("a2c")) == 40
    try:
        require_audit({"status": "passed", "checks_failed": 0, "runs_checked": 0}, 80, "synthetic preflight")
    except ValueError:
        pass
    else:
        raise AssertionError("Incomplete audit must fail closed")
    print("SELF_CHECK_PASS: run identity, UTC ISO timestamps, intended-run schema and audit gating")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    self_check() if args.self_check else main(args.output_dir)
