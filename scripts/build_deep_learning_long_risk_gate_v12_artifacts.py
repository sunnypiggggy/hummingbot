#!/usr/bin/env python3
"""Build the disabled v12 hybrid signal, notebook, manifest and Plotly report."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nbformat as nbf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from nbclient import NotebookClient
from plotly.subplots import make_subplots
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


PAIRS = ("BTC-FDUSD", "ETH-FDUSD")
ANCHORS = (
    ("2月3–7日", pd.Timestamp("2026-02-03T00:00:00Z"), pd.Timestamp("2026-02-07T00:00:00Z")),
    ("6月1–7日", pd.Timestamp("2026-06-01T00:00:00Z"), pd.Timestamp("2026-06-07T00:00:00Z")),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, value: Any) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def utc(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), timezone.utc).isoformat().replace("+00:00", "Z")


def locked_prediction(out: Path, pair: str, lock: dict[str, Any]) -> pd.DataFrame:
    config_id = lock["pair_winners"][pair]["config_id"]
    return pd.read_csv(out / "prediction_cache" / f"{pair}__{config_id}__seed42.csv.gz")


def write_calibration_metrics(out: Path, lock: dict[str, Any]) -> tuple[Path, Path]:
    metrics, calibration = [], []
    for pair in PAIRS:
        frame = locked_prediction(out, pair, lock)
        for head, target in (("p72", "target_72h"), ("p120", "target_120h"), ("pmean", "target_72h")):
            valid = frame[[head, target]].dropna()
            y, probability = valid[target].astype(int), valid[head].clip(1e-7, 1 - 1e-7)
            metrics.append({
                "pair": pair, "head": head, "target": target, "rows": len(valid),
                "positive_rate": float(y.mean()),
                "roc_auc": float(roc_auc_score(y, probability)) if y.nunique() > 1 else np.nan,
                "average_precision": float(average_precision_score(y, probability)),
                "log_loss": float(log_loss(y, probability, labels=[0, 1])),
                "brier_score": float(brier_score_loss(y, probability)),
            })
            bins = pd.qcut(probability.rank(method="first"), 10, labels=False, duplicates="drop")
            for bin_id, indexes in valid.groupby(bins).groups.items():
                values = valid.loc[indexes]
                calibration.append({
                    "pair": pair, "head": head, "target": target, "bin": int(bin_id),
                    "rows": len(values), "mean_probability": float(values[head].mean()),
                    "observed_rate": float(values[target].mean()),
                })
    metrics_path, calibration_path = out / "classification_calibration_metrics.csv", out / "calibration_bins.csv"
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    pd.DataFrame(calibration).to_csv(calibration_path, index=False)
    return metrics_path, calibration_path


def signal_contract(out: Path, lock: dict[str, Any]) -> Path:
    states = pd.read_csv(out / "final_risk_states.csv.gz")
    latest = int(states.signal_ts.max())
    pairs = {}
    for pair in PAIRS:
        prediction = locked_prediction(out, pair, lock).sort_values("signal_ts").iloc[-1]
        channels = {}
        for channel in ("long", "short"):
            row = states[(states.pair == pair) & (states.channel == channel)].sort_values("signal_ts").iloc[-1]
            winner = lock["pair_winners"][pair]
            item = {
                "model_type": winner["architecture"] if channel == "long" else "xgboost-v11-fixed-short",
                "probability": float(row.probability), "entry_threshold": float(row.entry_threshold),
                "recovery_threshold": float(row.recovery_threshold),
                "risk_off_active": bool(row.risk_off_active), "reason": str(row.reason),
                "sequence_cutoff": utc(int(row.signal_ts) - (300 if channel == "long" else 3600)),
            }
            if channel == "long":
                item.update({"p72": float(prediction.p72), "p120": float(prediction.p120),
                             "combined_probability": float(prediction.pmean),
                             "selected_probability_head": winner["head"]})
            channels[channel] = item
        pairs[pair] = {
            "channels": channels, "risk_off_active": True, "buy_enabled": False,
            "active_channels": [name for name, value in channels.items() if value["risk_off_active"]],
            "event_id": hashlib.sha256(f"v12|{pair}|{latest}|disabled".encode()).hexdigest(),
            "reason": "research_contract_not_deployment_authorized; fail_closed",
        }
    value = {
        "schema": "grid-hybrid-risk-gate-v1", "model_version": lock["model_version"],
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "valid_until": utc(latest + 150), "source_healthy": True,
        "feature_schema_sha256": lock["feature_schema_sha256"],
        "v11_short_lock_sha256": lock["v11_short_lock_sha256"],
        "research_gate_passed": False, "deployment_allowed": False,
        "market_sell_action": False, "stop_excess_inventory": False,
        "mechanism1_fallback_allowed": False, "pairs": pairs,
        "evidence_status": lock["evidence_status"],
    }
    return dump(out / "grid_hybrid_risk_gate_v1_sample.json", value)


def build_report(out: Path, summary: dict[str, Any]) -> Path:
    states = pd.read_csv(out / "final_risk_states.csv.gz")
    events = pd.read_csv(out / "final_risk_events.csv")
    intervals = pd.read_csv(out / "final_risk_intervals.csv")
    comparison = pd.read_csv(out / "version_comparison.csv")
    pair_search = pd.read_csv(out / "pair_long_candidate_search.csv")
    attribution = pd.read_csv(out / "permutation_feature_attribution.csv")
    calibration = pd.read_csv(out / "classification_calibration_metrics.csv")
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=.035,
                        subplot_titles=("BTC价格", "BTC长期/短期概率", "ETH价格", "ETH长期/短期概率"),
                        row_heights=[.28, .22, .28, .22])
    colors = {"long": "#d95f02", "short": "#2563eb"}
    for pair_index, pair in enumerate(PAIRS):
        price_row, probability_row = pair_index * 2 + 1, pair_index * 2 + 2
        price = pd.read_csv(Path("data/backtesting_candles") / f"binance_{pair}_5m.csv")
        price = price[(price.timestamp >= int(pd.Timestamp("2026-02-01T15:00:00Z").timestamp()))
                      & (price.timestamp < int(pd.Timestamp("2026-07-31T15:00:00Z").timestamp()))].copy()
        price["time"] = pd.to_datetime(price.timestamp, unit="s", utc=True)
        fig.add_trace(go.Scatter(x=price.time, y=price.close, name=f"{pair[:3]} px",
                                 line={"color": "#263238", "width": 1}), row=price_row, col=1)
        group = states[states.pair.eq(pair)].copy()
        group["time"] = pd.to_datetime(group.signal_ts, unit="s", utc=True)
        for channel in ("long", "short"):
            item = group[group.channel.eq(channel)]
            channel_code = "L" if channel == "long" else "S"
            fig.add_trace(go.Scatter(x=item.time, y=item.probability, name=f"{pair[:3]} {channel_code} p",
                                     line={"color": colors[channel], "width": 1.3}), row=probability_row, col=1)
            fig.add_trace(go.Scatter(x=item.time, y=item.entry_threshold, name=f"{pair[:3]} {channel_code} thr",
                                     line={"color": colors[channel], "width": 1, "dash": "dot"}), row=probability_row, col=1)
            transition = events[(events.pair == pair) & (events.channel == channel)].copy()
            if not transition.empty:
                transition["time"] = pd.to_datetime(transition.timestamp, unit="s", utc=True)
                symbols = ["triangle-down" if value == "enter" else "circle-open" for value in transition.event]
                fig.add_trace(go.Scatter(x=transition.time, y=transition.probability, mode="markers",
                                         marker={"color": colors[channel], "size": 7, "symbol": symbols},
                                         name=f"{pair[:3]} {channel_code} evt"), row=probability_row, col=1)
    shapes, shape_channels = [], []
    for item in intervals.itertuples(index=False):
        pair_index = 0 if item.pair == "BTC-FDUSD" else 1
        for row in (pair_index * 2 + 1, pair_index * 2 + 2):
            shapes.append({"type": "rect", "xref": "x" if row == 1 else f"x{row}",
                           "yref": "y domain" if row == 1 else f"y{row} domain",
                           "x0": pd.to_datetime(item.start_ts, unit="s", utc=True),
                           "x1": pd.to_datetime(item.end_ts, unit="s", utc=True), "y0": 0, "y1": 1,
                           "fillcolor": colors[item.channel], "opacity": .07,
                           "line": {"width": 0}, "layer": "below"})
            shape_channels.append(item.channel)
    for _label, left, right in ANCHORS:
        shapes.append({"type": "rect", "xref": "x", "yref": "paper", "x0": left, "x1": right,
                       "y0": 0, "y1": 1, "fillcolor": "rgba(0,0,0,0)",
                       "line": {"color": "#6b7280", "width": 1}})
        shape_channels.append("anchor")
    fig.update_layout(height=1050, shapes=shapes, hovermode="x unified", template="plotly_white",
                      title="深度学习v12长期Risk-off + 固定XGBoost v11短期Risk-off",
                      legend={"orientation": "h", "y": 1.08, "x": 0, "font": {"size": 11}},
                      margin={"l": 60, "r": 30, "t": 130, "b": 50})
    fig.update_yaxes(title_text="FDUSD", row=1, col=1); fig.update_yaxes(range=[0, 1], row=2, col=1)
    fig.update_yaxes(title_text="FDUSD", row=3, col=1); fig.update_yaxes(range=[0, 1], row=4, col=1)
    chart = fig.to_html(full_html=False, include_plotlyjs=True, div_id="hybrid-risk-chart")
    architecture = pair_search.sort_values("rank").groupby(["pair", "architecture"], as_index=False).first()
    top_attr = attribution.sort_values("mean_absolute_probability_change", ascending=False).groupby("pair", as_index=False).head(12)
    metrics = summary["winner_metrics"]
    html = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>深度学习v12长期Risk-off研究</title><style>body{{font-family:Arial,sans-serif;margin:16px;color:#172033;max-width:100%;overflow-x:hidden}} .note{{padding:12px;border-left:4px solid #d95f02;background:#f8fafc}} .switches{{position:sticky;top:4px;z-index:5;background:white;padding:10px;border:1px solid #ddd}} #hybrid-risk-chart,.plotly-graph-div{{width:100%!important;max-width:100%!important}} table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:6px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}} .scroll{{overflow-x:auto}} @media(max-width:600px){{body{{margin:8px}}h1{{font-size:21px}}.note{{font-size:14px}}.switches{{font-size:14px}}}}</style></head><body>
<h1>深度学习v12长期Risk-off研究</h1><div class='note'><b>结论：{summary['verdict']}</b>。净收益 {metrics['oos_pnl_fdusd']:.6f} FDUSD，拼接最大回撤 {metrics['stitched_max_drawdown_pct']:.6f}%，单对/组合停止 {metrics['pair_stop_events']}/{metrics['portfolio_stop_events']}。模型只暂停普通BUY，不产生卖出。180天及重点窗口用于硬筛选，属于样本内定向优化，deployment_allowed=false。</div>
<div class='switches'>阴影：<label><input id='toggle-long' type='checkbox' checked> 深度学习长期</label>　<label><input id='toggle-short' type='checkbox' checked> XGBoost短期</label></div>
{chart}<h2>旧版本与v12对比</h2><div class='scroll'>{comparison.to_html(index=False, border=0)}</div>
<h2>TCN、GRU与Transformer各自最佳候选</h2><div class='scroll'>{architecture.to_html(index=False, border=0)}</div>
<h2>分类与校准指标</h2><div class='scroll'>{calibration.to_html(index=False, border=0)}</div>
<h2>模型置换归因</h2><div class='scroll'>{top_attr.to_html(index=False, border=0)}</div>
<script>const channels={json.dumps(shape_channels)},desktopTitle=document.getElementById('hybrid-risk-chart').layout.title.text;function apply(){{const l=document.getElementById('toggle-long').checked,s=document.getElementById('toggle-short').checked;const plot=document.getElementById('hybrid-risk-chart');const shapes=plot.layout.shapes.map((x,i)=>Object.assign({{}},x,{{visible:channels[i]==='anchor'||(channels[i]==='long'&&l)||(channels[i]==='short'&&s)}}));Plotly.relayout(plot,{{shapes}});}}function responsivePlot(){{const plot=document.getElementById('hybrid-risk-chart'),mobile=window.innerWidth<=600;Plotly.relayout(plot,mobile?{{height:1320,'legend.orientation':'v','legend.font.size':9,'legend.x':0,'legend.xanchor':'left','legend.y':1.01,'legend.yanchor':'bottom','margin.l':44,'margin.r':8,'margin.t':285,'margin.b':40,'title.text':''}}:{{height:1050,'legend.orientation':'h','legend.font.size':11,'legend.x':0,'legend.xanchor':'left','legend.y':1.08,'legend.yanchor':'auto','margin.l':60,'margin.r':30,'margin.t':130,'margin.b':50,'title.text':desktopTitle,'title.font.size':17}});Plotly.Plots.resize(plot);}}document.getElementById('toggle-long').addEventListener('change',apply);document.getElementById('toggle-short').addEventListener('change',apply);window.addEventListener('resize',responsivePlot);requestAnimationFrame(responsivePlot);</script>
</body></html>"""
    path = out / "deep_learning_v12_hybrid_riskoff_plotly.html"
    path.write_text(html, encoding="utf-8")
    return path


def build_notebook(out: Path) -> Path:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb["cells"] = [
        nbf.v4.new_markdown_cell("# 深度学习v12长期Risk-off实验\n\n180天结果为样本内定向优化证据。"),
        nbf.v4.new_code_cell("import json,pandas as pd\nfrom pathlib import Path\nOUT=Path('.')\nsummary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))\nsummary['winner_metrics']"),
        nbf.v4.new_code_cell("pd.read_csv(OUT/'version_comparison.csv')[['version','oos_pnl_fdusd','stitched_max_drawdown_pct','pair_stop_events','portfolio_stop_events']]"),
        nbf.v4.new_code_cell("pd.read_csv(OUT/'classification_calibration_metrics.csv')"),
        nbf.v4.new_code_cell("pd.read_csv(OUT/'seed_stability_grid_metrics.csv')"),
        nbf.v4.new_code_cell("pd.read_csv(OUT/'pressure_tests.csv')"),
        nbf.v4.new_code_cell("assert summary['deployment_allowed'] is False\nassert summary['evidence_status']=='full_180d_in_sample_anchor_targeted_optimization'\nsummary['acceptance']"),
    ]
    executed = NotebookClient(nb, timeout=300, kernel_name="python3").execute(cwd=str(out.resolve()))
    path = out / "deep_learning_long_risk_gate_v12_executed.ipynb"
    nbf.write(executed, path)
    return path


def build_all(out: Path, _v11_dir: Path) -> dict[str, str]:
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    lock = json.loads((out / "locked_configuration.json").read_text(encoding="utf-8"))
    metric_paths = write_calibration_metrics(out, lock)
    signal = signal_contract(out, lock)
    report_path = build_report(out, summary)
    notebook_path = build_notebook(out)
    material = [out / name for name in (
        "summary.json", "locked_configuration.json", "environment_lock.json", "deep_model_configurations.csv",
        "sequence_data_quality.csv", "walk_forward_training_audit_seed42.csv",
        "pair_long_candidate_search.csv", "portfolio_search.csv", "pressure_tests.csv",
        "seed_stability_grid_metrics.csv", "permutation_feature_attribution.csv",
        "time_block_attribution.csv", "version_comparison.csv", "sequence_feature_contract.csv",
        "seed_stability_training.json", "final_risk_states.csv.gz", "final_risk_events.csv",
        "final_risk_intervals.csv", "final_equity_curve.csv.gz", "final_trade_events.csv.gz",
        "final_stop_events.csv", "final_weekly_results.csv",
    )] + sorted((out / "models").glob("*.pt")) + list(metric_paths) + [signal, report_path, notebook_path]
    artifact = {
        "schema": "deep-learning-v12-research-artifact-v1", "model_version": lock["model_version"],
        "verdict": summary["verdict"], "deployment_allowed": False,
        "evidence_status": summary["evidence_status"],
        "artifacts": [{"path": str(path.relative_to(out)), "bytes": path.stat().st_size,
                       "sha256": sha256(path)} for path in material],
    }
    manifest = dump(out / "artifact.json", artifact)
    result = {"signal": str(signal), "report": str(report_path),
              "notebook": str(notebook_path), "manifest": str(manifest)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    build_all(Path("results/backtests/deep_learning_long_risk_gate_v12"),
              Path("results/backtests/xgboost_feature_selected_pair_risk_gate_v11"))
