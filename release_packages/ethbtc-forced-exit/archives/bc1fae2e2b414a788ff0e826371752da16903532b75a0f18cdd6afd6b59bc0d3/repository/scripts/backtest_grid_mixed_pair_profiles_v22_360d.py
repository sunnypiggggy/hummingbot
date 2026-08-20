#!/usr/bin/env python3
"""True 420-FDUSD mixed-profile Grid replay using live movement semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline.offline import get_plotlyjs
from plotly.subplots import make_subplots

from backtest_binance_short_sideways_long_vs_bidirectional_360d import (
    END_TS,
    PAIR_CAPITAL,
    START_TS,
    Preset,
    expand_v22_gate,
    load_inputs,
    simulate_arm,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/backtests/grid_btc_medium_eth_long_v22_360d"
PAIR_NAMES = ("BTC-FDUSD", "ETH-FDUSD")
PORTFOLIO_RESERVE = 20.0

FIXED = Preset("fixed_current", "现网固定6%", 0.06, 10, 0.006)
MIXED = {
    "BTC-FDUSD": Preset(
        "medium_sideways", "BTC中短期横盘", 0.12698379475402316, 18, 0.004,
    ),
    "ETH-FDUSD": Preset(
        "long_volatility", "ETH长期波动", 0.5246511596640915, 18, 0.014179761072002472,
    ),
}
ARMS = {
    "live_fixed_5_25": {
        "label": "现网固定参数（5.25 FDUSD）",
        "presets": {pair: FIXED for pair in PAIR_NAMES},
        "minimums": {pair: 5.25 for pair in PAIR_NAMES},
    },
    "live_fixed_10": {
        "label": "现网固定参数（10 FDUSD敏感性）",
        "presets": {pair: FIXED for pair in PAIR_NAMES},
        "minimums": {pair: 10.0 for pair in PAIR_NAMES},
    },
    "btc_medium_eth_long": {
        "label": "BTC中短期横盘／ETH长期波动",
        "presets": MIXED,
        "minimums": {pair: 10.0 for pair in PAIR_NAMES},
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarise(arm: str, states, equity: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in PAIR_NAMES:
        frame = equity[equity.pair == pair]
        values = frame.equity.to_numpy()
        drawdown = values / np.maximum.accumulate(values) - 1.0
        state = states[pair]
        rows.append({
            "arm": arm,
            "label": ARMS[arm]["label"],
            "scope": pair,
            "profile": state.preset.preset_id,
            "initial_equity_fdusd": PAIR_CAPITAL,
            "final_equity_fdusd": float(values[-1]),
            "net_pnl_fdusd": float(values[-1] - PAIR_CAPITAL),
            "max_drawdown_pct": float(drawdown.min() * 100.0),
            "buy_fills": state.maker_buys,
            "sell_fills": state.normal_sells + state.take_profit_sells,
            "maker_fill_rate_pct": (
                (state.maker_buys + state.normal_sells + state.take_profit_sells)
                / state.maker_orders * 100.0 if state.maker_orders else 0.0
            ),
            "minimum_order_fdusd": state.minimum_order,
            "minimum_placed_order_fdusd": (
                None if np.isinf(state.min_order_notional) else state.min_order_notional
            ),
            "grid_moves": state.grid_moves,
            "forced_exits": state.forced_exits,
            "reentries": state.reentries,
            "risk_off_hours": state.blocked_bars * 5.0 / 60.0,
            "fees_fdusd": state.fees,
            "risk_execution_cost_fdusd": state.risk_execution_cost,
            "ending_base": state.base,
        })

    pivot = equity.pivot(index="timestamp", columns="pair", values="equity")
    combined = pivot.sum(axis=1).to_numpy() + PORTFOLIO_RESERVE
    combined_dd = combined / np.maximum.accumulate(combined) - 1.0
    rows.append({
        "arm": arm,
        "label": ARMS[arm]["label"],
        "scope": "PORTFOLIO",
        "profile": "mixed" if arm == "btc_medium_eth_long" else "fixed_shared",
        "initial_equity_fdusd": 420.0,
        "final_equity_fdusd": float(combined[-1]),
        "net_pnl_fdusd": float(combined[-1] - 420.0),
        "max_drawdown_pct": float(combined_dd.min() * 100.0),
        "buy_fills": sum(state.maker_buys for state in states.values()),
        "sell_fills": sum(state.normal_sells + state.take_profit_sells for state in states.values()),
        "maker_fill_rate_pct": 0.0,
        "minimum_order_fdusd": min(state.minimum_order for state in states.values()),
        "minimum_placed_order_fdusd": min(state.min_order_notional for state in states.values()),
        "grid_moves": sum(state.grid_moves for state in states.values()),
        "forced_exits": sum(state.forced_exits for state in states.values()),
        "reentries": sum(state.reentries for state in states.values()),
        "risk_off_hours": sum(state.blocked_bars for state in states.values()) * 5.0 / 60.0,
        "fees_fdusd": sum(state.fees for state in states.values()),
        "risk_execution_cost_fdusd": sum(state.risk_execution_cost for state in states.values()),
        "ending_base": None,
    })
    return rows


def write_plotly(output: Path, equity: pd.DataFrame) -> Path:
    figures: list[tuple[str, go.Figure]] = []
    colors = {"live_fixed_5_25": "#64748b", "live_fixed_10": "#2563eb", "btc_medium_eth_long": "#d97706"}
    for scope in (*PAIR_NAMES, "PORTFOLIO"):
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            subplot_titles=("价格", "连续权益", "峰值回撤（%）"))
        for arm, metadata in ARMS.items():
            frame = equity[equity.arm == arm]
            if scope == "PORTFOLIO":
                wide = frame.pivot(index="timestamp", columns="pair", values="equity")
                series = wide.sum(axis=1) + PORTFOLIO_RESERVE
                timestamp = series.index
                price = None
            else:
                pair_frame = frame[frame.pair == scope].set_index("timestamp")
                series = pair_frame.equity
                timestamp = series.index
                price = pair_frame.price
            sample = np.arange(len(series)) % 12 == 0
            dt = pd.to_datetime(np.asarray(timestamp)[sample], unit="s", utc=True)
            values = series.to_numpy()[sample]
            peak = np.maximum.accumulate(series.to_numpy())[sample]
            if price is not None and arm == "live_fixed_5_25":
                fig.add_trace(go.Scatter(x=dt, y=price.to_numpy()[sample], name=f"{scope}价格",
                                         line=dict(color="#111827", width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=dt, y=values, name=metadata["label"],
                                     line=dict(color=colors[arm], width=1.4)), row=2, col=1)
            fig.add_trace(go.Scatter(x=dt, y=(values / peak - 1.0) * 100.0,
                                     name=f'{metadata["label"]}回撤', showlegend=False,
                                     line=dict(color=colors[arm], width=1.1)), row=3, col=1)
        fig.update_layout(title=f"{scope} · v22口径 · 360天混合Grid回放", height=920,
                          template="plotly_white", hovermode="x unified",
                          legend=dict(orientation="h", y=1.05), margin=dict(l=65, r=25, t=105, b=45))
        fig.update_yaxes(title_text="FDUSD", row=2, col=1)
        fig.update_yaxes(title_text="%", row=3, col=1)
        figures.append((scope, fig))

    buttons = []
    sections = []
    for index, (label, fig) in enumerate(figures):
        buttons.append(f'<button onclick="showTab({index},this)" class="tab {"active" if index == 0 else ""}">{label}</button>')
        spec = fig.to_json(remove_uids=True).replace("</", "<\\/")
        sections.append(
            f'<section id="section-{index}" class="section {"active" if index == 0 else ""}">'
            f'<div id="plot-{index}"></div><script id="spec-{index}" type="application/json">{spec}</script></section>'
        )
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BTC中短期横盘／ETH长期波动</title><style>body{{font-family:Arial,'Microsoft YaHei',sans-serif;margin:0;background:#f5f7fa}}header,nav{{padding:14px 20px;background:white;border-bottom:1px solid #ddd}}h1{{font-size:22px;margin:0 0 8px}}p{{margin:4px 0;color:#475569}}nav{{position:sticky;top:0;z-index:3;overflow-x:auto;white-space:nowrap}}button{{padding:8px 12px;margin:2px;border:1px solid #cbd5e1;border-radius:7px;background:white}}button.active{{background:#1f2937;color:white}}.section{{display:none;margin:12px;background:white}}.section.active{{display:block}}</style><script>{get_plotlyjs()}</script></head><body><header><h1>BTC中短期横盘／ETH长期波动 Grid：360天 v22 口径</h1><p>2025-08-24 00:00 → 2026-08-19 00:00 UTC；组合初始权益420 FDUSD；FOMC不参与。</p><p>采用现网语义：价格越过Grid边界后再偏移1.5%才移动；Maker费0%，风险退出按Taker 0.1%和2bp滑点。</p></header><nav>{''.join(buttons)}</nav>{''.join(sections)}<script>const done=new Set();function draw(i){{if(done.has(i))return;const s=JSON.parse(document.getElementById('spec-'+i).textContent);Plotly.newPlot('plot-'+i,s.data,s.layout,{{responsive:true,displaylogo:false}});done.add(i)}}function showTab(i,b){{document.querySelectorAll('.section,.tab').forEach(x=>x.classList.remove('active'));document.getElementById('section-'+i).classList.add('active');b.classList.add('active');draw(i)}}draw(0);</script></body></html>"""
    path = output / "mixed_grid_v22_360d_plotly.html"
    path.write_text(html, encoding="utf-8")
    return path


def run(output: Path) -> None:
    candles, gate, filters, evidence = load_inputs()
    gate_arrays = {pair: expand_v22_gate(candles[pair], gate, pair) for pair in PAIR_NAMES}
    summaries: list[dict[str, Any]] = []
    equities: list[pd.DataFrame] = []
    events: list[pd.DataFrame] = []
    safety: dict[str, Any] = {}
    for arm, metadata in ARMS.items():
        states, equity, arm_events = simulate_arm(
            candles, gate_arrays, filters, "bidirectional", metadata["presets"], "protected_v22",
            minimum_order_by_pair=metadata["minimums"],
            movement_semantics="boundary_plus_threshold",
            portfolio_reserve=PORTFOLIO_RESERVE,
        )
        equity["arm"] = arm
        if not arm_events.empty:
            arm_events["arm"] = arm
            arm_events["scope"] = "protected_v22"
            events.append(arm_events)
        summaries.extend(summarise(arm, states, equity))
        equities.append(equity)
        placed = arm_events[arm_events.event_type == "ORDER_PLACED"] if not arm_events.empty else arm_events
        buys = arm_events[arm_events.event_type == "BUY"] if not arm_events.empty else arm_events
        blocked_buys = 0
        if not buys.empty:
            gate_lookup = equity[["timestamp", "pair", "v22_state"]].drop_duplicates()
            blocked_buys = int((
                buys[["timestamp", "pair"]].merge(gate_lookup, on=["timestamp", "pair"], how="left").v22_state
                != "RISK_ON"
            ).sum())
        safety[arm] = {
            "negative_inventory_observations": int((equity.base < -1e-12).sum()),
            "orders_below_configured_minimum": int(sum(
                (placed[placed.pair == pair].notional < metadata["minimums"][pair] - 1e-9).sum()
                for pair in PAIR_NAMES
            )),
            "v22_blocked_buy_events": blocked_buys,
        }
        if any(safety[arm].values()):
            raise AssertionError(f"Safety gate failed for {arm}: {safety[arm]}")

    output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summaries)
    all_equity = pd.concat(equities, ignore_index=True)
    all_events = pd.concat(events, ignore_index=True)
    summary.to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    all_equity.to_csv(output / "continuous_equity.csv.gz", index=False,
                      compression={"method": "gzip", "mtime": 0})
    all_events.to_csv(output / "trade_and_risk_events.csv.gz", index=False,
                      compression={"method": "gzip", "mtime": 0})
    parameter_rows = []
    for arm, metadata in ARMS.items():
        for pair in PAIR_NAMES:
            preset = metadata["presets"][pair]
            parameter_rows.append({
                "arm": arm, "pair": pair, "profile": preset.preset_id,
                "total_range_pct": preset.total_range * 100.0,
                "half_range_pct": preset.total_range * 50.0,
                "grid_levels": preset.levels, "side_levels": preset.side_levels,
                "actual_step_pct": preset.actual_step * 100.0,
                "take_profit_pct": preset.take_profit * 100.0,
                "minimum_order_fdusd": metadata["minimums"][pair],
                "move_threshold_beyond_boundary_pct": preset.move_threshold * 100.0,
                "move_cooldown_minutes": 30, "order_refresh_hours": 2,
            })
    pd.DataFrame(parameter_rows).to_csv(output / "parameter_mapping.csv", index=False, encoding="utf-8-sig")
    plot = write_plotly(output, all_equity)
    report = {
        "schema": "grid-btc-medium-eth-long-v22-360d-v1",
        "offline_only": True,
        "deployment_allowed": False,
        "window": {"start": START_TS, "end_exclusive": END_TS},
        "execution": {"portfolio_capital_fdusd": 420, "pair_capital_fdusd": 200,
                      "portfolio_reserve_fdusd": 20, "maker_fee": 0,
                      "taker_fee": 0.001, "taker_slippage": 0.0002,
                      "movement_semantics": "boundary_plus_1.5pct", "fomc_enabled": False,
                      "v22_enabled": True, "bnb_fee_used": False},
        "safety": safety,
        "evidence": evidence,
        "summary": summaries,
    }
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (output / "manifest.json").write_text(
        json.dumps({"schema": "grid-mixed-profile-manifest-v1", "artifacts": artifacts}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"Plotly: {plot}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.output.resolve())


if __name__ == "__main__":
    main()
