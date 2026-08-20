#!/usr/bin/env python3
"""Compare long-only and bilateral live-DCA semantics under the v22 gate.

This is an offline ablation.  It never reads credentials or submits orders.
The long-only arm removes the SELL executor while leaving the BUY executor,
total bot capital, fills, fees, exits, and v22 handling unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtest_dca_live_local import BAR_SECONDS, SIDE_BUDGET, TOTAL_BUDGET, load_window
from backtest_dca_momentum_guard import run_pair_guarded
from dca_live_common import LIVE_EXECUTOR_REFRESH_SECONDS, LIVE_TIME_LIMIT_SECONDS


PAIRS = {"BTC-USDT": "BTCUSDT", "ETH-USDT": "ETHUSDT"}
PAIR_MAP = {"BTC-USDT": "BTC-FDUSD", "ETH-USDT": "ETH-FDUSD"}
SCENARIOS = {
    "long_only": {"label": "只做多", "active_sides": ("BUY",)},
    "bilateral": {"label": "双边交易", "active_sides": ("BUY", "SELL")},
}
DEFAULT_END = "2026-08-20T00:00:00Z"
DEFAULT_CACHE = Path("data/backtesting_candles/dca_sell_gate_360d")
DEFAULT_V22_STATES = Path(
    "results/backtests/binance_ai_grid_presets_360d/v22_gate_states.csv.gz"
)
DEFAULT_OUTPUT = Path("results/backtests/dca_long_only_vs_bilateral_v22_360d")
REQUIRED_V22_COLUMNS = {
    "pair", "signal_ts", "fold", "probability", "entry_threshold",
    "risk_off_active", "recommended_buy_enabled", "transition", "reason",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--days", type=int, default=360)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--v22-states", type=Path, default=DEFAULT_V22_STATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--risk-slippage-bps", type=float, default=2.0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false", "1", "0"}).all():
        raise ValueError("v22 boolean column contains unsupported values")
    return normalized.isin({"true", "1"})


def load_v22_states(path: Path) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    states = pd.read_csv(path)
    missing = REQUIRED_V22_COLUMNS - set(states.columns)
    if missing:
        raise ValueError(f"v22 state file is missing {sorted(missing)}")
    states["signal_ts"] = pd.to_numeric(states["signal_ts"], errors="raise").astype("int64")
    states["risk_off_active"] = bool_series(states["risk_off_active"])
    states["recommended_buy_enabled"] = bool_series(states["recommended_buy_enabled"])
    coverage: dict[str, dict[str, int]] = {}
    for source_pair in PAIR_MAP.values():
        pair_states = states[states.pair.eq(source_pair)].sort_values("signal_ts")
        if pair_states.empty:
            raise ValueError(f"v22 states are missing {source_pair}")
        if pair_states.signal_ts.duplicated().any():
            raise ValueError(f"v22 states contain duplicate hours for {source_pair}")
        gaps = pair_states.signal_ts.diff().dropna()
        if not gaps.eq(3600).all():
            raise ValueError(f"v22 states are not hourly-contiguous for {source_pair}")
        if (pair_states.risk_off_active == pair_states.recommended_buy_enabled).any():
            raise ValueError(f"v22 risk and recommendation semantics disagree for {source_pair}")
        coverage[source_pair] = {
            "start": int(pair_states.signal_ts.min()),
            "end": int(pair_states.signal_ts.max()) + 3600,
            "hours": int(len(pair_states)),
            "folds": int(pair_states.fold.nunique()),
        }
    return states.sort_values(["pair", "signal_ts"]).reset_index(drop=True), coverage


def build_v22_gate(
    frame: pd.DataFrame,
    states: pd.DataFrame,
    dca_pair: str,
) -> tuple[pd.Series, pd.DataFrame, dict[str, int]]:
    source_pair = PAIR_MAP[dca_pair]
    source = states[states.pair.eq(source_pair)].sort_values("signal_ts").copy()
    coverage = {
        "start": int(source.signal_ts.min()),
        "end": int(source.signal_ts.max()) + 3600,
    }
    targets = pd.DataFrame({"timestamp": frame.timestamp.astype("int64")})
    joined = pd.merge_asof(
        targets.sort_values("timestamp"),
        source[["signal_ts", "risk_off_active", "recommended_buy_enabled"]],
        left_on="timestamp", right_on="signal_ts", direction="backward",
    )
    available = joined.timestamp.ge(coverage["start"]) & joined.timestamp.lt(coverage["end"])
    recommended = joined.recommended_buy_enabled.map(
        lambda value: bool(value) if pd.notna(value) else False
    )
    gate = (available & recommended).astype(bool)
    audit = pd.DataFrame({
        "timestamp": joined.timestamp.astype("int64"),
        "v22_available": available.astype(bool),
        "v22_risk_off": (available & ~recommended).astype(bool),
        "v22_gate_enabled": gate,
    })
    return pd.Series(gate.to_numpy(), index=frame.index, dtype=bool), audit, coverage


def boolean_intervals(
    timestamps: pd.Series,
    values: pd.Series,
    *,
    end_ts: int,
) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for timestamp, active in zip(timestamps.astype("int64"), values.astype(bool)):
        if active and start is None:
            start = int(timestamp)
        elif not active and start is not None:
            intervals.append((start, int(timestamp)))
            start = None
    if start is not None:
        intervals.append((start, end_ts))
    return intervals


def add_v22_regions(
    fig: go.Figure,
    *,
    row_start: int,
    state: pd.DataFrame,
    window_start: int,
    window_end: int,
    coverage: Mapping[str, int],
) -> None:
    unavailable = []
    if window_start < int(coverage["start"]):
        unavailable.append((window_start, min(window_end, int(coverage["start"]))))
    if int(coverage["end"]) < window_end:
        unavailable.append((max(window_start, int(coverage["end"])), window_end))
    risk_off = boolean_intervals(
        state.signal_ts,
        state.risk_off_active,
        end_ts=int(coverage["end"]),
    )
    for row in range(row_start, row_start + 4):
        for begin, finish in unavailable:
            if begin >= finish:
                continue
            fig.add_vrect(
                x0=pd.to_datetime(begin, unit="s", utc=True),
                x1=pd.to_datetime(finish, unit="s", utc=True),
                fillcolor="#D95F5F", opacity=0.10,
                line=dict(color="#A4262C", width=1, dash="dash"),
                layer="below", row=row, col=1,
            )
        for begin, finish in risk_off:
            begin, finish = max(begin, window_start), min(finish, window_end)
            if begin >= finish:
                continue
            fig.add_vrect(
                x0=pd.to_datetime(begin, unit="s", utc=True),
                x1=pd.to_datetime(finish, unit="s", utc=True),
                fillcolor="#E6A23C", opacity=0.12,
                line=dict(color="#A96500", width=1),
                layer="below", row=row, col=1,
            )


def make_plot(
    frames: Mapping[str, pd.DataFrame],
    curves: Mapping[tuple[str, str], pd.DataFrame],
    states: pd.DataFrame,
    coverage: Mapping[str, Mapping[str, int]],
    output: Path,
    *,
    start_ts: int,
    end_ts: int,
) -> None:
    titles: list[str] = []
    for pair in PAIRS:
        titles.extend((
            f"{pair} 价格",
            f"{pair} 单机器人连续权益",
            f"{pair} 回撤",
            f"{PAIR_MAP[pair]} v22 周度概率 / fold-local 阈值",
        ))
    fig = make_subplots(
        rows=8, cols=1, shared_xaxes=True, vertical_spacing=0.018,
        row_heights=[0.13, 0.17, 0.10, 0.10, 0.13, 0.17, 0.10, 0.10],
        subplot_titles=titles,
    )
    palette = {"long_only": "#2468C9", "bilateral": "#D97706"}
    dashes = {"long_only": "solid", "bilateral": "dash"}
    for pair_index, pair in enumerate(PAIRS):
        row_start = pair_index * 4 + 1
        hourly_price = (
            frames[pair].assign(datetime=pd.to_datetime(frames[pair].timestamp, unit="s", utc=True))
            .set_index("datetime")["close"].resample("1h").last()
        )
        fig.add_trace(go.Scatter(
            x=hourly_price.index, y=hourly_price, name=f"{pair} 价格",
            mode="lines", line=dict(color="#4B5563", width=1),
            legendgroup=f"price-{pair}", showlegend=pair_index == 0,
            hovertemplate="%{x}<br>价格 %{y:,.4f}<extra></extra>",
        ), row=row_start, col=1)
        for scenario, spec in SCENARIOS.items():
            curve = curves[(scenario, pair)]
            hourly = curve.resample("1h").last()
            fig.add_trace(go.Scatter(
                x=hourly.index, y=hourly.equity,
                name=spec["label"], legendgroup=scenario,
                showlegend=pair_index == 0,
                mode="lines", line=dict(color=palette[scenario], width=2, dash=dashes[scenario]),
                hovertemplate=f"%{{x}}<br>{spec['label']} %{{y:+.4f}} USDT<extra></extra>",
            ), row=row_start + 1, col=1)
            drawdown = (hourly.equity / hourly.equity.cummax() - 1) * 100
            fig.add_trace(go.Scatter(
                x=hourly.index, y=drawdown,
                name=f"{spec['label']}回撤", legendgroup=scenario,
                showlegend=False, mode="lines",
                line=dict(color=palette[scenario], width=1.6, dash=dashes[scenario]),
                hovertemplate=f"%{{x}}<br>{spec['label']}回撤 %{{y:.3f}}%<extra></extra>",
            ), row=row_start + 2, col=1)
        source_pair = PAIR_MAP[pair]
        state = states[states.pair.eq(source_pair)].copy()
        state["datetime"] = pd.to_datetime(state.signal_ts, unit="s", utc=True)
        fig.add_trace(go.Scatter(
            x=state.datetime, y=state.probability,
            name="v22概率", legendgroup="v22-probability", showlegend=pair_index == 0,
            mode="lines", line=dict(color="#7C3AED", width=1.4),
            customdata=state[["fold", "reason"]],
            hovertemplate="%{x}<br>概率 %{y:.5f}<br>fold %{customdata[0]}<br>%{customdata[1]}<extra></extra>",
        ), row=row_start + 3, col=1)
        fig.add_trace(go.Scatter(
            x=state.datetime, y=state.entry_threshold,
            name="v22逐周阈值", legendgroup="v22-threshold", showlegend=pair_index == 0,
            mode="lines", line=dict(color="#111827", width=1.2, dash="dot"),
            hovertemplate="%{x}<br>阈值 %{y:.5f}<extra></extra>",
        ), row=row_start + 3, col=1)
        transitions = state[state.transition.isin(("enter", "recover"))]
        if not transitions.empty:
            colors = transitions.transition.map({"enter": "#A96500", "recover": "#2468C9"})
            symbols = transitions.transition.map({"enter": "triangle-down", "recover": "triangle-up"})
            fig.add_trace(go.Scatter(
                x=transitions.datetime, y=transitions.probability,
                name="v22进入/恢复", legendgroup="v22-events", showlegend=pair_index == 0,
                mode="markers", marker=dict(color=colors, symbol=symbols, size=9, line=dict(width=1, color="#111827")),
                text=transitions.transition.map({"enter": "进入Risk-Off", "recover": "恢复Risk-On"}),
                hovertemplate="%{x}<br>%{text}<br>概率 %{y:.5f}<extra></extra>",
            ), row=row_start + 3, col=1)
        add_v22_regions(
            fig, row_start=row_start, state=state,
            window_start=start_ts, window_end=end_ts,
            coverage=coverage[source_pair],
        )

    fig.add_trace(go.Scatter(
        x=[None], y=[None], name="v22 Risk-Off 阴影",
        mode="lines", line=dict(color="#A96500", width=8),
        legendgroup="v22-risk-off",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], name="v22无签名周（Fail-Closed）",
        mode="lines", line=dict(color="#A4262C", width=3, dash="dash"),
        legendgroup="v22-unavailable",
    ), row=1, col=1)
    fig.update_yaxes(title_text="USDT", row=1, col=1)
    fig.update_yaxes(title_text="权益 USDT", row=2, col=1)
    fig.update_yaxes(title_text="%", row=3, col=1)
    fig.update_yaxes(title_text="概率", row=4, col=1)
    fig.update_yaxes(title_text="USDT", row=5, col=1)
    fig.update_yaxes(title_text="权益 USDT", row=6, col=1)
    fig.update_yaxes(title_text="%", row=7, col=1)
    fig.update_yaxes(title_text="概率", row=8, col=1)
    date_start = pd.to_datetime(start_ts, unit="s", utc=True)
    date_end = pd.to_datetime(end_ts, unit="s", utc=True)
    range_keys = {f"xaxis{'' if index == 1 else index}.range": [date_start, date_end] for index in range(1, 9)}
    jan_feb = {f"xaxis{'' if index == 1 else index}.range": ["2026-01-01", "2026-03-01"] for index in range(1, 9)}
    may_june = {f"xaxis{'' if index == 1 else index}.range": ["2026-05-01", "2026-07-01"] for index in range(1, 9)}
    fig.update_layout(
        title=dict(
            text=("DCA 只做多 vs 双边交易：360天 v22 风控对照"
                  "<br><sup>每机器人190 USDT；BUY侧均为95 USDT。只做多组关闭SELL并保留95 USDT现金；"
                  "v22覆盖外Fail-Closed，无FOMC/ROC/SQZMOM独立门</sup>"),
            x=0.02,
        ),
        template="plotly_white", height=2300, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.015, xanchor="left", x=0),
        margin=dict(l=75, r=35, t=145, b=60),
        updatemenus=[dict(
            type="buttons", direction="right", x=0.0, y=1.055, showactive=True,
            buttons=[
                dict(label="完整360天", method="relayout", args=[range_keys]),
                dict(label="2026年1–2月", method="relayout", args=[jan_feb]),
                dict(label="2026年5–6月", method="relayout", args=[may_june]),
            ],
        )],
    )
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1)
    fig.write_html(output, include_plotlyjs=True, full_html=True, config={"responsive": True})
    html = output.read_text(encoding="utf-8")
    if re.search(r"<script[^>]+src=[\"']https?://", html, flags=re.IGNORECASE):
        raise AssertionError("Plotly artifact unexpectedly loads an external script")


def main() -> int:
    args = parse_args()
    if args.days != 360:
        raise ValueError("this comparison is fixed to 360 days")
    end = datetime.fromisoformat(args.end.replace("Z", "+00:00")).astimezone(timezone.utc)
    start = end - timedelta(days=args.days)
    start_ts, end_ts = int(start.timestamp()), int(end.timestamp())
    states, coverage = load_v22_states(args.v22_states)
    frames = {
        pair: load_window(args.cache_dir / f"{symbol}_5m.csv", start, end)
        for pair, symbol in PAIRS.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    trade_parts: list[pd.DataFrame] = []
    curve_parts: list[pd.DataFrame] = []
    curves: dict[tuple[str, str], pd.DataFrame] = {}
    gates: dict[str, pd.DataFrame] = {}
    for pair, frame in frames.items():
        gate, gate_audit, _ = build_v22_gate(frame, states, pair)
        gates[pair] = gate_audit
        for scenario, spec in SCENARIOS.items():
            active_sides = tuple(spec["active_sides"])
            summary, trades, curve = run_pair_guarded(
                frame, gate, pair, args.fee_rate, args.risk_slippage_bps,
                refresh_seconds=LIVE_EXECUTOR_REFRESH_SECONDS,
                time_limit_seconds=LIVE_TIME_LIMIT_SECONDS,
                guarded_sides=active_sides,
                flatten_on_risk_off=True,
                active_sides=active_sides,
            )
            summary.update({
                "scenario": scenario,
                "scenario_label": spec["label"],
                "initial_equity_quote": TOTAL_BUDGET,
                "buy_side_budget_quote": SIDE_BUDGET,
                "idle_cash_quote": SIDE_BUDGET if scenario == "long_only" else 0.0,
                "v22_unavailable_hours": float((~gate_audit.v22_available).sum() * BAR_SECONDS / 3600),
                "v22_risk_off_hours": float(gate_audit.v22_risk_off.sum() * BAR_SECONDS / 3600),
            })
            summaries.append(summary)
            if not trades.empty:
                trades.insert(0, "scenario", scenario)
                trade_parts.append(trades)
            curve = curve.copy()
            curve["drawdown_pct"] = (curve.equity / curve.equity.cummax() - 1) * 100
            curve["v22_available"] = gate_audit.v22_available.to_numpy()
            curve["v22_risk_off"] = gate_audit.v22_risk_off.to_numpy()
            curves[(scenario, pair)] = curve
            exported = curve.reset_index(drop=True)
            exported.insert(0, "pair", pair)
            exported.insert(0, "scenario", scenario)
            curve_parts.append(exported)

    pair_summary = pd.DataFrame(summaries)
    aggregate_rows = []
    for scenario, spec in SCENARIOS.items():
        pair_curves = [curves[(scenario, pair)].equity.rename(pair) for pair in PAIRS]
        combined = pd.concat(pair_curves, axis=1)
        combined_equity = combined.sum(axis=1)
        aggregate_rows.append({
            "scenario": scenario,
            "scenario_label": spec["label"],
            "initial_equity_quote": TOTAL_BUDGET * len(PAIRS),
            "final_equity_quote": float(combined_equity.iloc[-1]),
            "net_pnl_quote": float(combined_equity.iloc[-1] - TOTAL_BUDGET * len(PAIRS)),
            "return_pct": float((combined_equity.iloc[-1] / (TOTAL_BUDGET * len(PAIRS)) - 1) * 100),
            "max_drawdown_pct": float((combined_equity / combined_equity.cummax() - 1).min() * 100),
            "fees_quote": float(pair_summary.loc[pair_summary.scenario.eq(scenario), "fees_quote"].sum()),
            "positioned_executors": int(pair_summary.loc[pair_summary.scenario.eq(scenario), "positioned_executors"].sum()),
            "risk_flatten_positions": int(pair_summary.loc[pair_summary.scenario.eq(scenario), "risk_flatten_positions"].sum()),
        })
    aggregate = pd.DataFrame(aggregate_rows)
    bilateral_pnl = float(aggregate.loc[aggregate.scenario.eq("bilateral"), "net_pnl_quote"].iloc[0])
    long_pnl = float(aggregate.loc[aggregate.scenario.eq("long_only"), "net_pnl_quote"].iloc[0])
    aggregate["delta_pnl_vs_long_only"] = aggregate.net_pnl_quote - long_pnl

    pair_summary.to_csv(args.output_dir / "pair_summary.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate_summary.csv", index=False)
    pd.concat(trade_parts, ignore_index=True).to_csv(
        args.output_dir / "positioned_executors.csv.gz", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    pd.concat(curve_parts, ignore_index=True).to_csv(
        args.output_dir / "equity_curves_5m.csv.gz", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    states.to_csv(
        args.output_dir / "v22_states.csv.gz", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    plot_path = args.output_dir / "dca_long_only_vs_bilateral_v22_360d_plotly.html"
    make_plot(
        frames, curves, states, coverage, plot_path,
        start_ts=start_ts, end_ts=end_ts,
    )
    audit = {
        "schema": "dca-long-only-vs-bilateral-v22-360d-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "offline_only": True,
        "orders_submitted": False,
        "deployment_allowed": False,
        "window": {"start": start.isoformat(), "end_exclusive": end.isoformat(), "days": args.days},
        "data": {pair: {"rows": int(len(frame)), "first_ts": int(frame.timestamp.min()),
                         "last_ts": int(frame.timestamp.max())} for pair, frame in frames.items()},
        "comparison": {
            "long_only": "BUY executor unchanged; SELL executor removed; 95 USDT SELL-side reserve stays cash",
            "bilateral": "same BUY executor plus SELL executor backed by 95 USDT-equivalent managed base inventory",
            "capital_per_bot_quote": TOTAL_BUDGET,
            "side_budget_quote": SIDE_BUDGET,
            "isolated_variable": "presence of SELL executor",
        },
        "execution": {
            "refresh_seconds": LIVE_EXECUTOR_REFRESH_SECONDS,
            "time_limit_seconds_from_first_fill": LIVE_TIME_LIMIT_SECONDS,
            "fee_rate_per_entry_and_exit": args.fee_rate,
            "fee_asset": "USDT",
            "bnb_fee_used": False,
            "v22_risk_exit_slippage_bps": args.risk_slippage_bps,
        },
        "v22": {
            "source": str(args.v22_states),
            "mapping": PAIR_MAP,
            "coverage": coverage,
            "risk_off_action": "block active strategy directions and flatten open executors",
            "outside_signed_coverage": "UNAVAILABLE_FAIL_CLOSED_NO_FALLBACK",
            "fomc_enabled": False,
            "roc_sqzmom_independent_gate_enabled": False,
        },
        "result": {
            "bilateral_incremental_pnl_vs_long_only": bilateral_pnl - long_pnl,
            "aggregate": aggregate.to_dict(orient="records"),
        },
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    manifest = {
        "schema": "dca-long-only-vs-bilateral-v22-360d-manifest-v1",
        "offline_only": True,
        "orders_submitted": False,
        "input_hashes": {
            "v22_states": sha256_file(args.v22_states),
            **{pair: sha256_file(args.cache_dir / f"{symbol}_5m.csv") for pair, symbol in PAIRS.items()},
        },
        "output_hashes": {
            path.name: sha256_file(path)
            for path in sorted(args.output_dir.iterdir())
            if path.is_file() and path.name != "manifest.json"
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(aggregate.round(4).to_string(index=False))
    print(json.dumps({
        "output": str(args.output_dir.resolve()),
        "plotly": str(plot_path.resolve()),
        "orders_submitted": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
