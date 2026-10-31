"""Assemble audited algorithm results and pinned framework source reviews.

No training, model selection, trading simulation or private connection is used.
The original study files are read-only inputs to this new comparison artifact.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.crypto_rl_reporting import clean_json, write_json

OUTPUT = ROOT / "results/backtests/rl_framework_comparison_20261024"
STUDIES = {"ppo": ROOT / "results/backtests/crypto_rl_top10_20261024",
           "a2c": ROOT / "results/backtests/crypto_rl_a2c_20261024"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reviews = read(OUTPUT / "framework_review_rows.json")
    if {row["name"] for row in reviews["rows"]} != {"TensorTrade", "TradeMaster", "EarnHFT", "Market-GAN", "MacroHFT"} or len(reviews["rows"]) != 5:
        raise ValueError("Expected five distinct user-requested framework reviews")
    for item in reviews["source"].get("input_review_files", []):
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("A pinned source review changed: " + item["path"])
    frames = {name: [] for name in ("summary", "daily_equity", "monthly_returns", "seed_statistics", "training_metrics")}
    source_files, audits, baseline_checks = {}, {}, []
    manifests = {name: read(folder / "manifest.json") for name, folder in STUDIES.items()}
    for name, folder in STUDIES.items():
        audits[name] = read(folder / "independent_audit.json")
        if audits[name]["status"] != "passed" or audits[name]["checks_failed"] != 0:
            raise ValueError("An independent financial audit did not pass: " + name)
        for filename in frames:
            path = folder / (filename + ".csv")
            frame = pd.read_csv(path, float_precision="round_trip")
            if filename == "training_metrics":
                records = {row["seed"]: row for row in manifests[name]["models"]}
                frame["training_window_metadata"] = frame.seed.map(lambda seed: records[int(seed)]["training_window_metadata"])
            if filename == "daily_equity":
                frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="raise")
            if filename not in ("training_metrics", "seed_statistics") and name != "ppo":
                frame = frame.loc[frame.strategy.isin([name, name + "_ensemble"])].copy()
            frame["algorithm"] = name.upper()
            if "strategy" in frame:
                frame.loc[~frame.strategy.isin(("ppo", "ppo_ensemble", "a2c", "a2c_ensemble")), "algorithm"] = "SHARED"
            if filename == "summary":
                for row in frame.to_dict("records"):
                    audited = audits[name]["runs"][row["run_id"]]
                    if any(abs(row[field] - audited[field]) > 1e-8
                           for field in ("net_return", "max_drawdown", "fees")):
                        raise ValueError("Current summary differs from independently audited results: " + row["run_id"])
            frames[filename].append(frame)
            source_files[str(path.relative_to(ROOT))] = sha(path)
        for filename in ("manifest.json", "independent_audit.json", "frozen_spec.json", "training_frozen_inputs.json"):
            path = folder / filename
            source_files[str(path.relative_to(ROOT))] = sha(path)
    ppo, a2c = manifests["ppo"], manifests["a2c"]
    if ppo["frozen_inputs"]["dataset_sha256"] != a2c["frozen_inputs"]["dataset_sha256"]:
        raise ValueError("Algorithms used different source datasets")
    if sha(ROOT / "data/crypto_rl_top10/dataset.pkl.gz") != ppo["frozen_inputs"]["dataset_sha256"]:
        raise ValueError("Prepared data changed after the audits")
    shared_files = ("scripts/backtest_crypto_rl_top10.py", "scripts/crypto_rl_data.py",
                    "scripts/crypto_rl_portfolio_env.py", "scripts/crypto_rl_reporting.py")
    ppo_sources = {key.replace("\\", "/"): value for key, value in ppo["source_sha256"].items()}
    a2c_sources = {key.replace("\\", "/"): value for key, value in a2c["source_sha256"].items()}
    for filename in shared_files:
        if ppo_sources[filename] != a2c_sources[filename] or sha(ROOT / filename) != ppo_sources[filename]:
            raise ValueError("Shared financial implementation changed: " + filename)
    for key in ("train_start", "train_end", "validation_start", "validation_end", "test_start", "test_end", "costs", "capital"):
        if ppo["config"].get(key) != a2c["config"].get(key):
            raise ValueError("Different comparison protocol: " + key)
    for name, result in ppo["run_results"].items():
        if not any(name.endswith("__" + strategy) for strategy in ("cash", "top10_equal", "top10_cap", "btc_eth_hold")):
            continue
        match = result["fingerprints"] == a2c["run_results"][name]["fingerprints"]
        baseline_checks.append({"run_id": name, "byte_equal_economic_results": match})
        if not match:
            raise ValueError("Shared baseline differs between algorithm studies: " + name)
        for field in ("equity", "fills", "weights", "boundary_equity"):
            actual = []
            for folder in STUDIES.values():
                path = folder / "runs" / name
                if field == "equity":
                    with gzip.open(path / "equity_1h.csv.gz", "rb") as stream:
                        actual.append(hashlib.sha256(stream.read()).hexdigest())
                else:
                    actual.append(sha(path / (field + ".csv")))
            if not actual[0] == actual[1] == result["fingerprints"][field]:
                raise ValueError("Actual shared baseline artifact differs: " + name + ":" + field)
    if len(baseline_checks) != 16:
        raise ValueError("Expected 16 independently matching shared baseline runs")
    combined = {name: pd.concat(items, ignore_index=True) for name, items in frames.items()}
    summary = combined["summary"]
    if len(summary) != 64 or not summary.run_id.is_unique:
        raise ValueError("Expected 64 unique comparison runs")
    if summary.algorithm.value_counts().to_dict() != {"PPO": 24, "A2C": 24, "SHARED": 16}:
        raise ValueError("Unexpected strategy coverage")
    for (split, scenario), selected in summary.groupby(["split", "scenario"]):
        if len(selected) != 16:
            raise ValueError("Missing comparison group: " + split + ":" + scenario)
        for algorithm in ("ppo", "a2c"):
            seeds = selected.loc[selected.strategy.eq(algorithm), "seed"].astype(int).tolist()
            if sorted(seeds) != [0, 1, 2, 3, 4] or selected.strategy.eq(algorithm + "_ensemble").sum() != 1:
                raise ValueError("Incomplete fixed model family: " + algorithm)
    for name, frame in combined.items():
        frame.to_csv(OUTPUT / (name + ".csv"), index=False, float_format="%.17g", encoding="utf-8-sig")
    queries = {}
    financial_source = {
        "label": "本地审计通过的PPO与A2C统一账本算法对照",
        "classification": "modeled", "timezone": "UTC",
        "period": {"start": "2025-07-25T00:00:00Z", "end": "2026-10-23T00:00:00Z"},
        "evidenceFlow": [
            {"title": "共同市场与资金", "detail": "相同SHA256行情与历史Top10月度池；同一现金／库存账本；每次10000 USDT。"},
            {"title": "固定训练与推理", "detail": "每种算法5种子×32768步，训练止于2025-07-25；验证与测试只做冻结模型确定性推理。主组合平均目标权重后实际执行。"},
            {"title": "独立复核", "detail": "每份评估重复核对指纹，独立审计从逐笔成交重建小时权益；16份共同基准在两研究中逐字节一致。"},
        ],
        "caveats": [
            "A2C是TradeMaster列示算法的SB3实现，未执行TradeMaster原生环境或训练器；这些数值不能称为TradeMaster/TensorTrade/EarnHFT/Market-GAN/MacroHFT原版收益。",
            "两算法市场交互预算相同，优化器、更新频率及算力预算不同；未搜索参数或选最佳种子。",
            "本年度已用于此前规则与PPO研究，不是整个研究过程全新未见的市场样本。",
            "回撤基于完整小时收盘权益；与早先分钟回撤采样不同。市值加权单币限仓后的余款留现金，风险仓位不同。",
            "历史市值快照为回溯版本，采用两日可用滞后，无法证明原始出版版本；小数数量未重建历史下单过滤器。",
        ],
        "metricDefinitions": [
            {"label": "净收益", "definition": "实际模拟成交扣费／滑点后，期末现金加持币市值相对10000 USDT变化；假设末端退出成本另列。"},
            {"label": "小时最大回撤", "definition": "包含初始本金的完整小时收盘权益相对历时峰值的最大跌幅。图表按日末采样，保留小时历史峰值。"},
            {"label": "平均仓位", "definition": "每小时持币市值占该时刻组合净值的平均。"},
        ],
    }
    query_map = {"strategy_summary": "summary", "equity_history": "daily_equity",
                 "monthly_returns": "monthly_returns", "seed_statistics": "seed_statistics",
                 "training_metrics": "training_metrics"}
    for query_id, frame_name in query_map.items():
        source = json.loads(json.dumps(financial_source, ensure_ascii=False))
        source["files"] = [frame_name + ".csv", "manifest.json", *[str(folder / "independent_audit.json") for folder in STUDIES.values()]]
        if frame_name == "training_metrics":
            source["period"] = {"start": "2023-01-01T00:00:00Z", "end": "2025-07-25T00:00:00Z"}
        queries[query_id] = {"source": source, "rows": json.loads(combined[frame_name].to_json(orient="records", date_format="iso"))}
    review_source = dict(reviews["source"])
    review_source["files"] = ["framework_review_rows.json", *reviews.get("review_links", [])]
    queries["framework_comparison"] = {"source": review_source, "rows": reviews["rows"]}
    write_json(OUTPUT / "reviewed.json", {
        "surface": "report", "status": "reviewed", "buildStatus": "complete",
        "title": "强化学习框架适配与Top10组合实测", "generatedAt": datetime.now(timezone.utc).isoformat(),
        "report": {"asOf": "2026-10-24", "title": "强化学习框架适配与Top10组合实测",
                   "studyConfig": {"costs": ppo["config"]["costs"], "test_start": ppo["config"]["test_start"],
                                   "test_end": ppo["config"]["test_end"], "hourly_drawdown": True}},
        "filters": [], "queries": queries,
    })
    write_json(OUTPUT / "manifest.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "source_studies": {name: str(folder) for name, folder in STUDIES.items()},
        "dataset_sha256": ppo["frozen_inputs"]["dataset_sha256"], "input_sha256": source_files,
        "shared_baseline_verifications": baseline_checks, "framework_reviews_sha256": sha(OUTPUT / "framework_review_rows.json"),
        "native_framework_experiments_run": [], "algorithm_controls_run": ["SB3 PPO", "SB3 A2C"],
        "numeric_results_changed": False, "production_changes": False,
    })
    lines = ["# 强化学习框架适配与历史Top10组合实测", "",
             "本报告将框架适配性与实际算法收益分开。五个原项目均完成固定源码审查；本轮实际运行的是统一账本上的SB3 PPO与A2C，不是五项目原版收益复现。", "",
             "| 项目 | 主要任务 | 本轮处理 |", "|---|---|---|"]
    for row in reviews["rows"]:
        lines.append(f"| [{row['name']}]({row['repository']}) | {row['task']} | {row['decision']} |")
    lines += ["", "## 365天扣费结果", "",
              "固定测试：[2025-10-23 00:00,2026-10-23 00:00) UTC；每组合10000 USDT。主成本每边手续费0.10%和滑点0.02%，压力成本0.20%和0.05%。小时收盘权益计算回撤。", "",
              "| 组合 | 主成本净收益 | 主成本小时回撤 | 压力成本净收益 | 压力成本小时回撤 | 主成本平均持币仓位 |",
              "|---|---:|---:|---:|---:|---:|"]
    test = summary.loc[summary.split.eq("test") & summary.seed.isna()]
    for strategy in ("ppo_ensemble", "a2c_ensemble", "top10_equal", "top10_cap", "btc_eth_hold", "cash"):
        selected = test.loc[test.strategy.eq(strategy)].set_index("scenario")
        base, stress = selected.loc["base"], selected.loc["stress"]
        lines.append(f"| {base['label']} | {base.net_return:.2%} | {base.max_drawdown:.2%} | {stress.net_return:.2%} | {stress.max_drawdown:.2%} | {base.average_exposure:.2%} |")
    lines += ["", "两算法各5个事前固定种子、每种子32768个市场交互步，主策略平均五模型的目标权重并在一个账本实际成交。它们的优化器和更新频率不同，不是等算力实验。A2C在已知PPO结果后按任务适配性选入，未按本轮A2C测试结果调参或筛种子。", "",
              "动态市值加权因25%单币目标上限而保留更多现金；收益与回撤差异不能单独归为选币或择时能力。不同算法的平均持币仓位也需同时比较。", "",
              "历史市值为回溯快照、两天可用滞后、月度换池；无法证明原始出版版本。至少31个完整历史日特征才允许买入。名义训练936天，可用128日片段并集覆盖853天；验证90天和测试365天连续评估。旧研究同年数据已被查看，不是全新独立市场样本。小时回撤不能与前两轮分钟回撤直接视为同一采样。", "",
              "## 交付与复现", "",
              "- report.html：可离线查看的交互报告，含五框架适配表、两成本／两窗口、全部种子及源证据。",
              "- summary.csv、daily_equity.csv、monthly_returns.csv、seed_statistics.csv：64份不重复评估的汇总与曲线；16份共同基准两次实跑的成交、权益、权重和奖励字节一致。",
              "- framework_review_rows.json：五个固定commit的结论及官方来源；全部原版运行状态单列。",
              "- manifest.json、verification.json：输入哈希、两项独立金融审计、数据检查及页面验收。",
              "- 原PPO与A2C研究目录保留全部模型、实际训练片段、逐笔成交及小时账本。", "",
              "复现步骤和数据边界见 ../crypto_rl_top10_20261024/METHODOLOGY.md。已有模型重放会重新写对应研究的评估文件，请先保存所需的原产物；哈希不同会失败。无需交易密钥。", "",
              "```powershell",
              "Set-Location 'H:\\PycharmProjects\\hummingbot'",
              "& 'C:\\Users\\sunny\\.cache\\codex-research-envs\\crypto-ppo-20261024\\Scripts\\python.exe' scripts/backtest_crypto_rl_top10.py --evaluate-only --skip-report",
              "& 'C:\\Users\\sunny\\.cache\\codex-research-envs\\crypto-ppo-20261024\\Scripts\\python.exe' scripts/backtest_rl_a2c_comparison.py --evaluate-only",
              "```", "",
              "重新汇总前分别运行 scripts/audit_crypto_rl_top10.py --output-dir 对应研究目录，再运行 scripts/build_rl_framework_comparison.py；图表构建入口为 scripts/rl_framework_comparison_report_app.py 中的 build_report。没有更改生产交易配置、部署服务或下单。", ""]
    (OUTPUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Combined {len(combined['summary'])} distinct evaluation rows; {len(baseline_checks)} identical shared baselines")


if __name__ == "__main__":
    main()
