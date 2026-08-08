#!/usr/bin/env python3
"""Build the reproducibility notebook, artifact manifest and disabled v8 signal sample."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import nbformat as nbf
import pandas as pd
from nbclient import NotebookClient


OUTPUT = Path("results/backtests/xgboost_roc_sqz_pair_risk_gate_v8")


def utc(timestamp: int) -> str:
    return datetime.fromtimestamp(int(timestamp), timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_signal_sample(lock: dict) -> Path:
    states = pd.read_csv(OUTPUT / "final_risk_states.csv.gz")
    latest = int(states.signal_ts.max())
    pairs = {}
    for pair in ("BTC-FDUSD", "ETH-FDUSD"):
        channels = {}
        for channel in ("long", "short"):
            row = states[(states.pair == pair) & (states.channel == channel)].sort_values("signal_ts").iloc[-1]
            channels[channel] = {
                "probability": float(row.probability),
                "entry_threshold": float(row.entry_threshold),
                "recovery_threshold": float(row.recovery_threshold),
                "risk_off_active": bool(row.risk_off_active),
                "transition": str(row.transition),
                "reason": str(row.reason),
                "last_complete_1h": utc(int(row.signal_ts)),
            }
        pairs[pair] = {
            "channels": channels,
            "active_channels": [name for name, value in channels.items() if value["risk_off_active"]],
            "risk_off_active": True,
            "buy_enabled": False,
            "reason": "fail_closed:locked_model_failed_acceptance",
            "event_id": hashlib.sha256(f"v8|{pair}|{latest}|fail_closed".encode()).hexdigest(),
        }
    payload = {
        "schema": "grid-xgboost-risk-gate-v3",
        "generated_at": utc(latest),
        "valid_until": utc(latest + 150),
        "model_version": lock["model_version"],
        "model_sha256": lock["model_sha256"],
        "feature_schema_sha256": lock["feature_schema_sha256"],
        "data_sha256": lock["training_data_sha256"],
        "source_healthy": True,
        "stale_after_seconds": 150,
        "deployment_allowed": False,
        "mechanism1_runtime_fallback": False,
        "ordinary_buy_gate_only": True,
        "market_sell_action": False,
        "pairs": pairs,
    }
    path = OUTPUT / "grid_xgboost_risk_gate_v3_sample.json"
    write_json(path, payload)
    return path


def build_artifact(summary: dict, lock: dict) -> Path:
    artifact = {
        "schema": "grid-risk-gate-research-artifact-v1",
        "title": "XGBoost ROC/SQZMOM 独立 Risk-off 门：180天 Grid 定向优化",
        "primary_question": "BTC/ETH 独立 XGBoost Risk-off BUY 门能否替代机制1并改善 Grid 盈利和回撤？",
        "period_utc": ["2026-02-01T15:00:00Z", "2026-07-31T15:00:00Z"],
        "evidence_status": summary["evidence_status"],
        "verdict": summary["verdict"],
        "deployment_allowed": summary["deployment_allowed"],
        "headline": {
            "baseline_pnl_fdusd": summary["baseline"]["oos_pnl_fdusd"],
            "candidate_pnl_fdusd": summary["winner_metrics"]["oos_pnl_fdusd"],
            "baseline_stitched_drawdown_pct": summary["baseline"]["stitched_max_drawdown_pct"],
            "candidate_stitched_drawdown_pct": summary["winner_metrics"]["stitched_max_drawdown_pct"],
            "candidate_pair_stops": summary["winner_metrics"]["pair_stop_events"],
            "candidate_portfolio_stops": summary["winner_metrics"]["portfolio_stop_events"],
        },
        "acceptance": summary["acceptance"],
        "search_counts": lock["selection_basis"],
        "sources": [
            "feature_panel.csv.gz", "grid_selections.csv", "baseline_metrics.json",
            "model_screen_40x2pairsx3targetsx8.csv", "single_pair_channel_refined_search.csv",
            "pair_independent_long_short_search.csv", "btc_eth_independent_portfolio_search.csv",
        ],
        "deliverables": {
            "interactive_report": "xgboost_v8_roc_sqz_pair_riskoff_plotly.html",
            "executed_notebook": "xgboost_roc_sqz_pair_risk_gate_v8_executed.ipynb",
            "locked_configuration": "locked_configuration.json",
            "disabled_signal_sample": "grid_xgboost_risk_gate_v3_sample.json",
        },
        "limitations": summary["limitations"],
    }
    path = OUTPUT / "artifact.json"
    write_json(path, artifact)
    return path


def build_notebook() -> Path:
    notebook = nbf.v4.new_notebook()
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.cells = [
        nbf.v4.new_markdown_cell("""# XGBoost ROC/SQZMOM 独立 Risk-off BUY 门\n\n**结论：NO-GO。** 诊断最佳候选改善了180天盈利和拼接回撤，但未覆盖两个指定长期窗口，仍有组合停止，且压力测试均出现停止。模型只允许暂停普通 Grid BUY，不产生卖出动作。"""),
        nbf.v4.new_code_cell("""from pathlib import Path\nimport json\nimport pandas as pd\n\nout = Path.cwd()\nif not (out / 'summary.json').exists():\n    out = Path('results/backtests/xgboost_roc_sqz_pair_risk_gate_v8').resolve()\nsummary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))\nlock = json.loads((out / 'locked_configuration.json').read_text(encoding='utf-8'))\nportfolio = pd.read_csv(out / 'btc_eth_independent_portfolio_search.csv')\npressure = pd.read_csv(out / 'pressure_tests.csv')\naudit = pd.read_csv(out / 'walk_forward_training_audit.csv')\nprint(out)\nprint(summary['verdict'], summary['evidence_status'])"""),
        nbf.v4.new_markdown_cell("## 验证范围与搜索规模"),
        nbf.v4.new_code_cell("""counts = lock['selection_basis']\nchecks = pd.DataFrame([\n    ['40 XGBoost configurations', counts['xgboost_configurations'], 40],\n    ['screen candidates', counts['screen_candidates'], 1920],\n    ['refined candidates', counts['single_refined_candidates'], 1920],\n    ['pair long-short candidates', counts['pair_long_short_candidates'], 200],\n    ['portfolio candidates', len(portfolio), 100],\n], columns=['check','observed','expected'])\nchecks['passed'] = checks.observed == checks.expected\nassert checks.passed.all()\nchecks"""),
        nbf.v4.new_markdown_cell("## 主要结果"),
        nbf.v4.new_code_cell("""baseline, winner = summary['baseline'], summary['winner_metrics']\nheadline = pd.DataFrame([\n    ['机制1', baseline['oos_pnl_fdusd'], baseline['stitched_max_drawdown_pct'], baseline['pair_stop_events'], baseline['portfolio_stop_events']],\n    ['XGBoost诊断最佳', winner['oos_pnl_fdusd'], winner['stitched_max_drawdown_pct'], winner['pair_stop_events'], winner['portfolio_stop_events']],\n], columns=['strategy','net_pnl_fdusd','stitched_max_drawdown_pct','pair_stops','portfolio_stops'])\nheadline"""),
        nbf.v4.new_code_cell("""pd.DataFrame([{'gate': k, 'passed': v} for k,v in summary['acceptance'].items()])"""),
        nbf.v4.new_markdown_cell("## 长期窗口覆盖与独立参数"),
        nbf.v4.new_code_cell("""rows=[]\nfor pair, item in summary['pair_winners'].items():\n    rows.append({\n        'pair': pair, 'long_target': item['long_target'], 'long_model': item['long_model_key'],\n        'short_model': item['short_model_key'], 'feb_coverage': item['feb_03_06_coverage'],\n        'jun_coverage': item['jun_01_06_coverage'], 'anchor_pass': item['anchor_pass'],\n        'long_q': item['long_entry_quantile'], 'short_q': item['short_entry_quantile']})\npd.DataFrame(rows)"""),
        nbf.v4.new_markdown_cell("## 压力测试与无前视检查"),
        nbf.v4.new_code_cell("""assert (audit.last_mature_label_ready_ts <= audit.train_cutoff_ts).all()\nassert (audit.last_calibration_signal_ts < audit.first_test_signal_ts).all()\nassert summary['winner_metrics']['momentum_stop_exits'] == 0\npressure[['scenario','oos_pnl_fdusd','stitched_max_drawdown_pct','pair_stop_events','portfolio_stop_events','no_stops']]"""),
        nbf.v4.new_markdown_cell("""## 决策\n\n不启用。`deployment_allowed=false`，Grid 应 fail-closed 暂停普通 BUY，且不回退机制1。下一步只能在全新、未参与调参的时间段做影子验证；当前180天结果属于样本内定向优化。"""),
    ]
    path = OUTPUT / "xgboost_roc_sqz_pair_risk_gate_v8_executed.ipynb"
    executed = NotebookClient(notebook, timeout=300, kernel_name="python3").execute(cwd=str(OUTPUT.resolve()))
    nbf.write(executed, path)
    return path


def main() -> None:
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    lock = json.loads((OUTPUT / "locked_configuration.json").read_text(encoding="utf-8"))
    print(build_signal_sample(lock))
    print(build_artifact(summary, lock))
    print(build_notebook())


if __name__ == "__main__":
    main()
