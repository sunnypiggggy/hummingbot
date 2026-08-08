#!/usr/bin/env python3
"""Build v9 research manifest, executed notebook, and disabled Grid contract sample."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import nbformat as nbf
import pandas as pd
from nbclient import NotebookClient


OUT = Path("results/backtests/xgboost_regime_spike_pair_risk_gate_v9")
ARTIFACT_TITLE = "XGBoost v9 独立长期趋势/短期插针 Risk-off BUY门"
PLOT_FILENAME = "xgboost_v9_regime_spike_pair_riskoff_plotly.html"
NOTEBOOK_FILENAME = "xgboost_regime_spike_pair_risk_gate_v9_executed.ipynb"
NOTEBOOK_TITLE = "XGBoost v9：长期趋势与1h插针Risk-off"
MODEL_LABEL = "XGBoost v9"


def utc(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), timezone.utc).isoformat().replace("+00:00", "Z")


def dump(name: str, value: dict) -> Path:
    path = OUT / name
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def signal(lock: dict) -> Path:
    states = pd.read_csv(OUT / "final_risk_states.csv.gz")
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
            "active_channels": [key for key, value in channels.items() if value["risk_off_active"]],
            "risk_off_active": True,
            "buy_enabled": False,
            "reason": "fail_closed:locked_model_failed_acceptance",
            "event_id": hashlib.sha256(f"v9|{pair}|{latest}|fail_closed".encode()).hexdigest(),
        }
    return dump("grid_xgboost_risk_gate_v3_sample.json", {
        "schema": "grid-xgboost-risk-gate-v3",
        "generated_at": utc(latest), "valid_until": utc(latest + 150),
        "model_version": lock["model_version"], "model_sha256": lock["model_sha256"],
        "feature_schema_sha256": lock["feature_schema_sha256"],
        "data_sha256": lock["training_data_sha256"], "source_healthy": True,
        "stale_after_seconds": 150, "deployment_allowed": False,
        "mechanism1_runtime_fallback": False, "ordinary_buy_gate_only": True,
        "market_sell_action": False, "pairs": pairs,
    })


def artifact(summary: dict, lock: dict) -> Path:
    return dump("artifact.json", {
        "schema": "grid-risk-gate-research-artifact-v1",
        "title": ARTIFACT_TITLE,
        "period_utc": ["2026-02-01T15:00:00Z", "2026-07-31T15:00:00Z"],
        "evidence_status": summary["evidence_status"], "verdict": summary["verdict"],
        "deployment_allowed": summary["deployment_allowed"],
        "feature_contract": lock["selection_basis"]["feature_contract"],
        "baseline": summary["baseline"], "winner_metrics": summary["winner_metrics"],
        "acceptance": summary["acceptance"], "pair_winners": summary["pair_winners"],
        "deliverables": {
            "plotly": PLOT_FILENAME,
            "notebook": NOTEBOOK_FILENAME,
            "lock": "locked_configuration.json", "signal": "grid_xgboost_risk_gate_v3_sample.json",
        },
        "limitations": summary["limitations"],
    })


def notebook() -> Path:
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.cells = [
        nbf.v4.new_markdown_cell(f"# {NOTEBOOK_TITLE}\n\n**结论：NO-GO。** 精选特征未同时覆盖两个长期窗口，仍有组合停止，不能驱动真实Grid BUY。"),
        nbf.v4.new_code_cell("""from pathlib import Path\nimport json, pandas as pd\nout=Path.cwd()\nif not (out/'summary.json').exists(): out=Path('results/backtests/xgboost_regime_spike_pair_risk_gate_v9').resolve()\nsummary=json.loads((out/'summary.json').read_text())\nlock=json.loads((out/'locked_configuration.json').read_text())\nportfolio=pd.read_csv(out/'btc_eth_independent_portfolio_search.csv')\npressure=pd.read_csv(out/'pressure_tests.csv')\naudit=pd.read_csv(out/'walk_forward_training_audit.csv')\nprint(summary['verdict'], summary['evidence_status'])"""),
        nbf.v4.new_markdown_cell("## 搜索规模与无前视"),
        nbf.v4.new_code_cell("""assert len(pd.read_csv(out/'model_screen_40x2pairsx3targetsx8.csv'))==1920\nassert len(pd.read_csv(out/'single_pair_channel_refined_search.csv'))==1920\nassert len(pd.read_csv(out/'pair_independent_long_short_search.csv'))==200\nassert len(portfolio)==100\nassert (audit.last_mature_label_ready_ts<=audit.train_cutoff_ts).all()\nassert (audit.last_calibration_signal_ts<audit.first_test_signal_ts).all()\nlock['selection_basis']['feature_contract']"""),
        nbf.v4.new_markdown_cell("## Grid结果"),
        nbf.v4.new_code_cell(f"""b,w=summary['baseline'],summary['winner_metrics']\npd.DataFrame([['机制1',b['oos_pnl_fdusd'],b['stitched_max_drawdown_pct'],b['pair_stop_events'],b['portfolio_stop_events']],['{MODEL_LABEL}',w['oos_pnl_fdusd'],w['stitched_max_drawdown_pct'],w['pair_stop_events'],w['portfolio_stop_events']]],columns=['strategy','pnl_fdusd','stitched_dd_pct','pair_stops','portfolio_stops'])"""),
        nbf.v4.new_markdown_cell("## 重点窗口"),
        nbf.v4.new_code_cell("""pd.DataFrame([{'pair':p,'feb_coverage':v['feb_03_06_coverage'],'jun_coverage':v['jun_01_06_coverage'],'timely_jun':v['jun_01_06_timely'],'anchor_pass':v['anchor_pass'],'overlap':v['active_jaccard']} for p,v in summary['pair_winners'].items()])"""),
        nbf.v4.new_markdown_cell("## 压力测试与决策"),
        nbf.v4.new_code_cell("""assert summary['verdict']=='NO-GO' and not summary['deployment_allowed']\npressure[['scenario','oos_pnl_fdusd','stitched_max_drawdown_pct','pair_stop_events','portfolio_stop_events','no_stops']]"""),
        nbf.v4.new_markdown_cell("模型只暂停对应交易对普通BUY，不产生卖出动作。当前180天同时用于调参和验收，属于样本内定向优化。"),
    ]
    path = OUT / NOTEBOOK_FILENAME
    executed = NotebookClient(nb, timeout=300, kernel_name="python3").execute(cwd=str(OUT.resolve()))
    nbf.write(executed, path)
    return path


def main() -> None:
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    lock = json.loads((OUT / "locked_configuration.json").read_text(encoding="utf-8"))
    print(signal(lock)); print(artifact(summary, lock)); print(notebook())


if __name__ == "__main__":
    main()
