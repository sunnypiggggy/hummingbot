#!/usr/bin/env python3
"""Build and execute the reader-facing XGBoost v7 research notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


OUTPUT_DIR = Path("results/backtests/xgboost_grid_risk_gate_v7")
NOTEBOOK = OUTPUT_DIR / "xgboost_grid_risk_gate_v7_executed.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build() -> Path:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3"}
    notebook["cells"] = [
        markdown("""
# XGBoost v7 Risk-off 驱动 Grid：180 天锁定结果

## tl;dr

- Grid 仍是交易主体；XGBoost 只暂停对应交易对的普通 BUY，不触发即时卖出。
- 锁定候选净盈利为 **-5.577270 FDUSD**，机制1为 **-16.874115 FDUSD**；虽改善 11.296845 FDUSD，但仍为负收益。
- 拼接最大回撤由 **-12.660523%** 改善至 **-11.290911%**，但组合停止由 1 次增至 2 次。
- 两个指定长期窗口未被 BTC/ETH 全部及时覆盖，所有压力场景均出现停止。
- 最终结论：**NO-GO**，`deployment_allowed=false`。
"""),
        markdown("""
## Context & Methods

### Key Assumptions

- 回放区间：2026-02-01 15:00 至 2026-07-31 15:00 UTC。
- 每对 200 FDUSD，组合储备 20 FDUSD；Maker 0%，风险退出 Taker 0.1%。
- 40 组确定性 XGBoost 参数，同时比较共享/独立架构；长短通道使用 OR 合并为对应交易对的普通 BUY gate。
- 排名目标为 50% Grid 净盈利百分位 + 50% 拼接权益最大回撤百分位。
- 同一 180 天路径及指定窗口参与参数选择，因此属于样本内定向优化，不是全新样本外证据。
"""),
        code("""
from pathlib import Path
import json
import pandas as pd

output_dir = Path.cwd()
if not (output_dir / "summary.json").exists():
    output_dir = Path("results/backtests/xgboost_grid_risk_gate_v7").resolve()

summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
artifact = json.loads((output_dir / "artifact.json").read_text(encoding="utf-8"))
dual = pd.read_csv(output_dir / "dual_channel_search.csv")
pressure = pd.read_csv(output_dir / "pressure_tests.csv")
coverage = pd.read_csv(output_dir / "anchor_window_coverage.csv")
metrics = pd.read_csv(output_dir / "final_metrics.csv")
audit = pd.read_csv(output_dir / "walk_forward_training_audit.csv")
intervals = pd.read_csv(output_dir / "final_risk_intervals.csv")

print(f"Artifact: {output_dir}")
print(f"Evidence: {summary['evidence_status']}")
print(f"Verdict: {summary['verdict']} | deployment_allowed={summary['deployment_allowed']}")
"""),
        markdown("## Data"),
        code("""
data_checks = pd.DataFrame([
    {"check": "dual candidates", "observed": len(dual), "expected": 100, "passed": len(dual) == 100},
    {"check": "eligible dual candidates", "observed": int(dual.eligible.astype(str).str.lower().eq('true').sum()), "expected": 0, "passed": not dual.eligible.astype(str).str.lower().eq('true').any()},
    {"check": "labels mature by cutoff", "observed": bool((audit.last_mature_label_ready_ts <= audit.train_cutoff_ts).all()), "expected": True, "passed": bool((audit.last_mature_label_ready_ts <= audit.train_cutoff_ts).all())},
    {"check": "probability serialization max abs error", "observed": summary['locked_configuration']['serialization_check']['maximum_probability_absolute_error'], "expected": 0.0, "passed": summary['locked_configuration']['serialization_check']['passed']},
    {"check": "model-driven immediate sells", "observed": summary['winner_metrics']['momentum_stop_exits'], "expected": 0, "passed": summary['winner_metrics']['momentum_stop_exits'] == 0},
])
assert data_checks.passed.all(), data_checks
data_checks
"""),
        markdown("## Results"),
        code("""
baseline = summary["baseline"]
winner = summary["winner_metrics"]
headline = pd.DataFrame([
    {"strategy": "Mechanism 1 baseline", "net_pnl_fdusd": baseline["oos_pnl_fdusd"], "stitched_max_drawdown_pct": baseline["stitched_max_drawdown_pct"], "pair_stops": baseline["pair_stop_events"], "portfolio_stops": baseline["portfolio_stop_events"]},
    {"strategy": "Locked XGBoost v7", "net_pnl_fdusd": winner["oos_pnl_fdusd"], "stitched_max_drawdown_pct": winner["stitched_max_drawdown_pct"], "pair_stops": winner["pair_stop_events"], "portfolio_stops": winner["portfolio_stop_events"]},
])
headline["pnl_delta_vs_mechanism1"] = headline.net_pnl_fdusd - baseline["oos_pnl_fdusd"]
headline["drawdown_delta_pp_vs_mechanism1"] = headline.stitched_max_drawdown_pct - baseline["stitched_max_drawdown_pct"]

top = dual.sort_values("rank").iloc[0]
assert abs(top.oos_pnl_fdusd - winner["oos_pnl_fdusd"]) < 1e-9
assert abs(top.stitched_max_drawdown_pct - winner["stitched_max_drawdown_pct"]) < 1e-9
assert summary["verdict"] == "NO-GO" and not summary["deployment_allowed"]
headline.round(6)
"""),
        code("""
acceptance = pd.DataFrame(
    [{"gate": key, "passed": value} for key, value in summary["acceptance"].items()]
)
acceptance
"""),
        code("""
coverage_columns = [name for name in coverage.columns if name.endswith("coverage") or name.endswith("timely")]
coverage[[*coverage.columns[:1], *coverage_columns]].round(4)
"""),
        code("""
pressure[["scenario", "oos_pnl_fdusd", "stitched_max_drawdown_pct", "pair_stop_events", "portfolio_stop_events", "no_stops"]].round(6)
"""),
        code("""
interval_summary = (
    intervals.groupby(["pair", "strategy"], as_index=False)
    .agg(intervals=("start_ts", "size"), risk_off_hours=("duration_hours", "sum"))
)
interval_summary.round(2)
"""),
        markdown("""
## Takeaways

1. XGBoost v7 明显降低了机制1在这条 180 天路径上的亏损与拼接回撤，但没有把 Grid 变为正收益。
2. 组合停止增加、指定长期窗口覆盖失败以及压力测试有停止，均是阻止启用的独立硬条件。
3. 锁定模型只能保持影子信号；在未来独立时间段取得新证据前，不应设置 `deployment_allowed=true`。
4. 交互式进入/退出图见 `xgboost_v7_riskoff_entry_exit_plotly.html`；所有 UTC 事件见 `plotly_dual_entry_exit_events.csv`。
"""),
    ]
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(notebook, timeout=300, kernel_name="python3")
    executed = client.execute(cwd=str(NOTEBOOK.parent.resolve()))
    nbf.write(executed, NOTEBOOK)
    return NOTEBOOK


if __name__ == "__main__":
    print(build())
