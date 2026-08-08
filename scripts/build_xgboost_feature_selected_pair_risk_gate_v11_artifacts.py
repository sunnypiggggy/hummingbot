#!/usr/bin/env python3
"""Build the disabled v11 research signal, notebook, manifest, and report addendum."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import nbformat as nbf
import pandas as pd
import plotly.express as px
from nbclient import NotebookClient


OUT = Path("results/backtests/xgboost_feature_selected_pair_risk_gate_v11")
REPORT = "xgboost_v11_feature_selected_riskoff_plotly.html"
NOTEBOOK = "xgboost_feature_selected_pair_risk_gate_v11_executed.ipynb"


def utc(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def build_signal(lock: dict) -> Path:
    states = pd.read_csv(OUT / "final_risk_states.csv.gz")
    stability = pd.read_csv(OUT / "feature_stability.csv")
    latest_ts = int(states.signal_ts.max())
    pairs = {}
    for pair in ("BTC-FDUSD", "ETH-FDUSD"):
        channels = {}
        locked = lock["pair_winners"][pair]
        for channel in ("long", "short"):
            row = states[(states.pair == pair) & (states.channel == channel)].sort_values(
                "signal_ts"
            ).iloc[-1]
            model_key = str(locked[f"{channel}_model_key"])
            target, _, config_id = model_key.split("|")
            model_features = pd.read_csv(OUT / "xgboost_v11_gain_feature_importance.csv")
            model_features = model_features[
                (model_features.pair == pair) & (model_features.channel == channel)
            ].sort_values("gain", ascending=False).feature.tolist()
            frequencies = stability[
                (stability.pair == pair) & (stability.target == target)
                & stability.feature.isin(model_features)
            ].set_index("feature")["selection_frequency"].to_dict()
            channels[channel] = {
                "target": target,
                "config_id": config_id,
                "features": model_features,
                "selection_frequency": {key: float(frequencies.get(key, 0)) for key in model_features},
                "probability": float(row.probability),
                "entry_threshold": float(row.entry_threshold),
                "recovery_threshold": float(row.recovery_threshold),
                "risk_off_active": bool(row.risk_off_active),
                "last_complete_1h": utc(int(row.signal_ts) - 3600),
                "state_advanced_at": utc(int(row.signal_ts)),
                "reason": str(row.reason),
            }
        pairs[pair] = {
            "channels": channels,
            "risk_off_active": True,
            "buy_enabled": False,
            "active_channels": [name for name, value in channels.items() if value["risk_off_active"]],
            "event_id": hashlib.sha256(f"v11|{pair}|{latest_ts}|fail-closed".encode()).hexdigest(),
            "reason": "locked_research_candidate_failed_acceptance; fail_closed",
        }
    signal = {
        "schema": "grid-xgboost-risk-gate-v4",
        "model_version": lock["model_version"],
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "valid_until": utc(latest_ts + 150),
        "last_complete_1h": utc(latest_ts - 3600),
        "source_healthy": True,
        "research_gate_passed": bool(lock.get("research_gate_passed", False)),
        "deployment_allowed": False,
        "feature_schema_sha256": lock["feature_schema_sha256"],
        "model_sha256": lock["model_sha256"],
        "training_data_sha256": lock["training_data_sha256"],
        "grid_sequence_sha256": lock["grid_sequence_sha256"],
        "mechanism1_fallback_allowed": False,
        "market_sell_action": False,
        "stop_excess_inventory": False,
        "pairs": pairs,
        "evidence_status": "full_180d_in_sample_targeted_revalidation",
    }
    return dump(OUT / "grid_xgboost_risk_gate_v4_sample.json", signal)


def enhance_report() -> Path:
    path = OUT / REPORT
    html = path.read_text(encoding="utf-8")
    comparison = pd.read_csv(OUT / "previous_version_comparison.csv")
    stability = pd.read_csv(OUT / "feature_stability.csv")
    ablation = pd.read_csv(OUT / "drop_column_grid_ablation.csv")
    fig1 = px.bar(
        comparison, x="version", y="oos_pnl_fdusd", color="version",
        title="机制1、XGBoost v8/v9、LightGBM v10与v11净收益对比",
    )
    top = stability.sort_values(
        ["pair", "channel", "selection_frequency"], ascending=[True, True, False]
    ).groupby(["pair", "channel"], as_index=False).head(8)
    fig2 = px.bar(
        top, x="feature", y="selection_frequency", color="pair", facet_row="channel",
        hover_data=["positive_permutation_frequency", "median_permutation", "median_gain"],
        title="特征时序稳定性（每对、每通道前8）",
    )
    section = """
<section id="v11-audit" style="max-width:1500px;margin:24px auto;padding:18px;background:#111827;color:#e5e7eb;border-radius:12px">
<h2>v11筛选与旧版本审计</h2>
<p><b>结论：NO-GO。</b> 这是使用同一180天路径与重点窗口调参的样本内定向再验证，deployment_allowed=false。</p>
{comparison}
{fig1}
{fig2}
<h3>锁定模型逐特征drop-column完整Grid回放</h3>
{ablation}
</section>
""".format(
        comparison=comparison.to_html(index=False, classes="comparison", border=0),
        fig1=fig1.to_html(full_html=False, include_plotlyjs=False),
        fig2=fig2.to_html(full_html=False, include_plotlyjs=False),
        ablation=ablation.sort_values("grid_composite_contribution", ascending=False).to_html(
            index=False, classes="ablation", border=0
        ),
    )
    if '<section id="v11-audit"' in html:
        html = html.split('<section id="v11-audit"', 1)[0] + "</body></html>"
    html = html.replace("</body>", section + "</body>")
    path.write_text(html, encoding="utf-8")
    return path


def build_notebook() -> Path:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# XGBoost v11 长短期Risk-off特征筛选审计\n\n"
            "本Notebook只读取锁定产物并复核结论；180天区间属于样本内定向再验证。"
        ),
        nbf.v4.new_code_cell(
            "import json, pandas as pd\nfrom pathlib import Path\n"
            "OUT=Path('.')\nsummary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))\n"
            "comparison=pd.read_csv(OUT/'previous_version_comparison.csv')\n"
            "comparison[['version','oos_pnl_fdusd','stitched_max_drawdown_pct','pair_stop_events','portfolio_stop_events']]"
        ),
        nbf.v4.new_code_cell(
            "stability=pd.read_csv(OUT/'feature_stability.csv')\n"
            "stability.sort_values(['pair','channel','selection_frequency'],ascending=[True,True,False]).groupby(['pair','channel']).head(8)"
        ),
        nbf.v4.new_code_cell(
            "ablation=pd.read_csv(OUT/'drop_column_grid_ablation.csv')\n"
            "ablation.sort_values('grid_composite_contribution',ascending=False)"
        ),
        nbf.v4.new_code_cell(
            "assert summary['verdict']=='NO-GO'\n"
            "assert summary['deployment_allowed'] is False\n"
            "assert not summary['acceptance']['BTC_anchor_pass']\n"
            "assert not summary['acceptance']['ETH_anchor_pass']\n"
            "summary['winner_metrics']"
        ),
    ]
    path = OUT / NOTEBOOK
    executed = NotebookClient(nb, timeout=300, kernel_name="python3").execute(cwd=str(OUT.resolve()))
    nbf.write(executed, path)
    return path


def build_manifest(paths: list[Path], summary: dict) -> Path:
    required = [
        OUT / "summary.json", OUT / "locked_configuration.json",
        OUT / "selected_feature_subsets.json", OUT / "feature_stability.csv",
        OUT / "feature_selection_fold_audit.csv", OUT / "feature_correlation_clusters.csv",
        OUT / "drop_column_grid_ablation.csv", OUT / "pressure_tests.csv",
        OUT / "previous_version_comparison.csv", *paths,
    ]
    manifest = {
        "schema": "xgboost-v11-research-artifact-v1",
        "model_version": "xgboost-feature-selected-pair-risk-gate-v11",
        "verdict": summary["verdict"],
        "deployment_allowed": False,
        "evidence_status": summary["evidence_status"],
        "artifacts": [
            {"path": str(path.relative_to(OUT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in dict.fromkeys(required)
        ],
    }
    return dump(OUT / "artifact.json", manifest)


def main() -> None:
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    lock = json.loads((OUT / "locked_configuration.json").read_text(encoding="utf-8"))
    signal = build_signal(lock)
    report = enhance_report()
    notebook = build_notebook()
    manifest = build_manifest([signal, report, notebook], summary)
    print(json.dumps({"signal": str(signal), "report": str(report), "notebook": str(notebook), "manifest": str(manifest)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
