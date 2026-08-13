#!/usr/bin/env python3
"""Two-stage FDUSD inventory-exit search with a locked 60-day holdout."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fdusd_live_grid_optimizer import DAY_SECONDS, local_day_start, rolling_validation_windows, select_candidate
from validate_grid_live import (
    Candidate,
    InventoryExitPolicy,
    crash_candles,
    read_cache,
    simulate,
    slice_window,
    technical_buy_gate_timeline,
)


PAIRS = ("BTC-FDUSD", "ETH-FDUSD")
INITIAL_APPROVED = Candidate(0.03, 0.006, 0.006, 0.015, 1800)
CAPS = (10, 15, 20, 25, 30, 40, 50)
HOLDS_HOURS = (12, 24, 36, 48, 72)
MIDDLE_FLOORS = (0.0, 0.001, 0.002, 0.003)
STAGE_SPLITS = ((0.25, 0.50), (1 / 3, 2 / 3), (0.50, 0.75))
SCORE_WEIGHTS = {
    "pnl_rank": 0.40,
    "drawdown_rank": 0.25,
    "portfolio_stop_rank": 0.20,
    "pair_stop_rank": 0.15,
}


def policy_id(policy: InventoryExitPolicy) -> str:
    return (
        f"cap{policy.max_extra_inventory_quote:g}_hold{policy.max_hold_seconds // 3600}h_"
        f"mid{policy.stage_one_profit_rate:.3f}_"
        f"split{policy.stage_one_fraction:.4f}-{policy.stage_two_fraction:.4f}"
    )


def inventory_policy_space() -> list[InventoryExitPolicy]:
    policies = [
        InventoryExitPolicy(
            max_extra_inventory_quote=cap,
            max_hold_seconds=hours * 3600,
            stage_one_profit_rate=middle,
            stage_two_profit_rate=0.0,
            stage_one_fraction=first,
            stage_two_fraction=second,
        )
        for cap, hours, middle, (first, second) in itertools.product(
            CAPS, HOLDS_HOURS, MIDDLE_FLOORS, STAGE_SPLITS
        )
    ]
    if len(policies) != 420 or len({policy_id(policy) for policy in policies}) != 420:
        raise RuntimeError("Inventory policy search space must contain 420 unique policies.")
    return policies


def holdout_windows(start_ts: int, end_ts: int) -> list[tuple[int, int, int, int]]:
    windows = []
    test_start = start_ts
    while test_start + 7 * DAY_SECONDS <= end_ts:
        windows.append((
            test_start - 14 * DAY_SECONDS, test_start,
            test_start, test_start + 7 * DAY_SECONDS,
        ))
        test_start += 7 * DAY_SECONDS
    return windows


def stop_metrics(result: dict, curve: pd.DataFrame, trades: list[dict], test_end: int) -> dict:
    pair_stops = [trade for trade in trades if trade["reason"] == "pair_breaker_flatten"]
    pair_hours = sum(max(test_end - int(trade["timestamp"]), 0) / 3600 for trade in pair_stops)
    portfolio_hours = 0.0
    if result["liquidated"] and not curve.empty:
        portfolio_hours = max(test_end - int(curve.timestamp.iloc[-1]), 0) / 3600
    return {
        "pair_stop_events": len(pair_stops),
        "pair_stop_hours": pair_hours,
        "portfolio_stop_events": int(bool(result["liquidated"])),
        "portfolio_stop_hours": portfolio_hours,
    }


def simulate_window(candles, candidate, gate, policy, start, end,
                    maker_fee=0.0, taker_fee=0.001, slippage=0.0):
    trades: list[dict] = []
    result, curve, pairs = simulate(
        slice_window(candles, start, end), candidate,
        maker_fee=maker_fee, taker_fee=taker_fee, slippage=slippage,
        order_refresh_seconds=7200, technical_buy_gate=gate,
        trade_log=trades, risk_breakers_enabled=True,
        cost_floor_enabled=True, inventory_exit_policy=policy,
    )
    stops = stop_metrics(result, curve, trades, end)
    return result, curve, pairs, trades, stops


def aggregate_rows(rows: list[dict], pair_rows: list[dict]) -> dict:
    pair_frame = pd.DataFrame(pair_rows)
    return {
        "oos_pnl_fdusd": sum(float(row["net_pnl_quote"]) for row in rows),
        "positive_folds": sum(float(row["net_pnl_quote"]) > 0 for row in rows),
        "worst_fold_pnl": min(float(row["net_pnl_quote"]) for row in rows),
        "worst_drawdown_pct": min(float(row["max_drawdown_pct"]) for row in rows) * 100,
        "pair_stop_events": sum(int(row["pair_stop_events"]) for row in rows),
        "pair_stop_hours": sum(float(row["pair_stop_hours"]) for row in rows),
        "portfolio_stop_events": sum(int(row["portfolio_stop_events"]) for row in rows),
        "portfolio_stop_hours": sum(float(row["portfolio_stop_hours"]) for row in rows),
        "forced_exits": sum(int(row["forced_inventory_exits"]) for row in rows),
        "trades": sum(int(row["trades"]) for row in rows),
        "fees_fdusd": sum(float(row["fees_quote"]) for row in rows),
        "btc_pnl_fdusd": float(pair_frame[pair_frame.pair == "BTC-FDUSD"].net_pnl_quote.sum()),
        "eth_pnl_fdusd": float(pair_frame[pair_frame.pair == "ETH-FDUSD"].net_pnl_quote.sum()),
    }


def fixed_policy_evaluation(candles, gate, windows, policy) -> tuple[dict, list[dict], list[dict]]:
    weekly, pair_rows = [], []
    for fold, (_, _, test_start, test_end) in enumerate(windows, 1):
        result, _, pairs, _, stops = simulate_window(
            candles, INITIAL_APPROVED, gate, policy, test_start, test_end
        )
        weekly.append({"fold": fold, "test_start": test_start, "test_end": test_end,
                       **result, **stops})
        pair_rows.extend({"fold": fold, "pair": pair, **metrics} for pair, metrics in pairs.items())
    return aggregate_rows(weekly, pair_rows), weekly, pair_rows


def add_balanced_score(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["pnl_rank"] = ranked.oos_pnl_fdusd.rank(method="average", pct=True)
    ranked["drawdown_rank"] = ranked.worst_drawdown_pct.rank(method="average", pct=True)
    ranked["portfolio_stop_rank"] = ranked.portfolio_stop_hours.rank(
        method="average", pct=True, ascending=False
    )
    ranked["pair_stop_rank"] = ranked.pair_stop_hours.rank(
        method="average", pct=True, ascending=False
    )
    ranked["balanced_score"] = sum(
        ranked[column] * weight for column, weight in SCORE_WEIGHTS.items()
    )
    return ranked.sort_values(
        ["balanced_score", "oos_pnl_fdusd", "worst_drawdown_pct", "policy_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True).assign(rank=lambda data: data.index + 1)


def policy_row(policy: InventoryExitPolicy) -> dict:
    return {
        "policy_id": policy_id(policy),
        "max_extra_inventory_quote": policy.max_extra_inventory_quote,
        "max_hold_hours": policy.max_hold_seconds // 3600,
        "middle_profit_floor": policy.stage_one_profit_rate,
        "final_profit_floor": policy.stage_two_profit_rate,
        "stage_one_fraction": policy.stage_one_fraction,
        "stage_two_fraction": policy.stage_two_fraction,
    }


def approved_candidate(selected: Candidate, eligible_count: int,
                       previous: Candidate) -> tuple[Candidate, bool]:
    """Mirror live deployment: keep the approved parameters when none qualify."""
    if eligible_count == 0:
        return previous, True
    return selected, False


def walk_forward_policy(candles, gate, windows, policy, initial_candidate,
                        keep_candidate_rows=False, collect_details=False):
    previous = initial_candidate
    weekly, pair_rows, candidate_rows, selections = [], [], [], []
    equity_rows, trade_rows, stop_rows = [], [], []
    cumulative = 0.0
    for fold, (train_start, train_end, test_start, test_end) in enumerate(windows, 1):
        selected, evaluations = select_candidate(
            slice_window(candles, train_start, train_end), 0.0,
            taker_fee=0.001, require_eligible=False,
            technical_buy_gate=gate, cost_floor_enabled=True,
            inventory_exit_policy=policy,
        )
        eligible_count = int(evaluations.attrs.get("eligible_count", 0))
        selected, retained = approved_candidate(selected, eligible_count, previous)
        if not retained:
            previous = selected
        selections.append(selected)
        if keep_candidate_rows:
            evaluations.insert(0, "policy_id", policy_id(policy) if policy else "online")
            evaluations.insert(1, "fold", fold)
            evaluations.insert(2, "train_start", train_start)
            evaluations.insert(3, "train_end", train_end)
            candidate_rows.extend(evaluations.to_dict("records"))
        result, curve, pairs, trades, stops = simulate_window(
            candles, selected, gate, policy, test_start, test_end
        )
        weekly.append({
            "fold": fold, "train_start": train_start, "train_end": train_end,
            "test_start": test_start, "test_end": test_end,
            "training_eligible_candidates": eligible_count,
            "parameters_retained": retained, **asdict(selected), **result, **stops,
        })
        pair_rows.extend({"fold": fold, "pair": pair, **metrics} for pair, metrics in pairs.items())
        if collect_details:
            for item in curve.itertuples(index=False):
                equity_rows.append({
                    "timestamp": int(item.timestamp), "fold": fold,
                    "cumulative_oos_pnl": cumulative + float(item.equity) - 420.0,
                    "fold_drawdown_pct": float(item.drawdown_pct) * 100,
                })
            for trade in trades:
                trade_rows.append({"fold": fold, **trade})
            for trade in trades:
                if trade["reason"] == "pair_breaker_flatten":
                    stop_rows.append({
                        "fold": fold, "scope": trade["pair"], "kind": "pair_stop",
                        "start_ts": int(trade["timestamp"]), "end_ts": test_end,
                    })
            if result["liquidated"] and not curve.empty:
                stop_rows.append({
                    "fold": fold, "scope": "PORTFOLIO", "kind": "portfolio_stop",
                    "start_ts": int(curve.timestamp.iloc[-1]), "end_ts": test_end,
                })
        cumulative += float(result["net_pnl_quote"])
    return {
        "summary": aggregate_rows(weekly, pair_rows),
        "weekly": weekly, "pairs": pair_rows, "candidate_rows": candidate_rows,
        "selections": selections, "final_candidate": previous,
        "equity": equity_rows, "trades": trade_rows, "stops": stop_rows,
    }


def replay_policy(candles, gate, windows, policy, selections):
    weekly, pair_rows, equity_rows, trade_rows, stop_rows = [], [], [], [], []
    cumulative = 0.0
    for fold, ((_, _, test_start, test_end), candidate) in enumerate(zip(windows, selections), 1):
        result, curve, pairs, trades, stops = simulate_window(
            candles, candidate, gate, policy, test_start, test_end
        )
        weekly.append({"fold": fold, "test_start": test_start, "test_end": test_end,
                       **asdict(candidate), **result, **stops})
        pair_rows.extend({"fold": fold, "pair": pair, **metrics} for pair, metrics in pairs.items())
        for item in curve.itertuples(index=False):
            equity_rows.append({
                "timestamp": int(item.timestamp), "fold": fold,
                "cumulative_oos_pnl": cumulative + float(item.equity) - 420.0,
                "fold_drawdown_pct": float(item.drawdown_pct) * 100,
            })
        trade_rows.extend({"fold": fold, **trade} for trade in trades)
        for trade in trades:
            if trade["reason"] == "pair_breaker_flatten":
                stop_rows.append({"fold": fold, "scope": trade["pair"], "kind": "pair_stop",
                                  "start_ts": int(trade["timestamp"]), "end_ts": test_end})
        if result["liquidated"] and not curve.empty:
            stop_rows.append({"fold": fold, "scope": "PORTFOLIO", "kind": "portfolio_stop",
                              "start_ts": int(curve.timestamp.iloc[-1]), "end_ts": test_end})
        cumulative += float(result["net_pnl_quote"])
    return {"summary": aggregate_rows(weekly, pair_rows), "weekly": weekly,
            "pairs": pair_rows, "equity": equity_rows, "trades": trade_rows,
            "stops": stop_rows}


def stress_tests(candles, gate, candidate, policy, end_ts) -> pd.DataFrame:
    start = end_ts - 7 * DAY_SECONDS
    base = slice_window(candles, start, end_ts)
    scenarios = [
        ("base", base, 0.0, 0.001, 0.0, gate),
        ("taker_fee_150pct", base, 0.0, 0.0015, 0.0, gate),
        ("slippage_005pct", base, 0.0, 0.001, 0.0005, gate),
        ("slippage_010pct", base, 0.0, 0.001, 0.001, gate),
    ]
    crashed = crash_candles(base, 0.15)
    crash_history = {
        pair: pd.concat([candles[pair][candles[pair].timestamp < start], crashed[pair]], ignore_index=True)
        for pair in PAIRS
    }
    scenarios.append((
        "15pct_one_day_drop", crashed, 0.0, 0.001, 0.001,
        technical_buy_gate_timeline(crash_history["BTC-FDUSD"]),
    ))
    rows = []
    for name, window, maker, taker, slippage, scenario_gate in scenarios:
        trades: list[dict] = []
        result, curve, pairs = simulate(
            window, candidate, maker_fee=maker, taker_fee=taker, slippage=slippage,
            order_refresh_seconds=7200, technical_buy_gate=scenario_gate,
            trade_log=trades, risk_breakers_enabled=True, cost_floor_enabled=True,
            inventory_exit_policy=policy,
        )
        stops = stop_metrics(result, curve, trades, end_ts)
        rows.append({"scenario": name, **result, **stops,
                     "pair_stop_triggered": any(value["liquidations"] for value in pairs.values())})
    return pd.DataFrame(rows)


def deployment_gates(summary: dict, pair_rows: list[dict], stress: pd.DataFrame) -> dict:
    pair_frame = pd.DataFrame(pair_rows)
    return {
        "holdout_oos_positive": summary["oos_pnl_fdusd"] > 0,
        "no_holdout_portfolio_stop": summary["portfolio_stop_events"] == 0,
        "no_holdout_pair_stop": summary["pair_stop_events"] == 0,
        "holdout_drawdown_within_6pct": summary["worst_drawdown_pct"] >= -6.0,
        "each_pair_fold_loss_within_6_fdusd": bool((pair_frame.net_pnl_quote >= -6.0).all()),
        "stress_no_portfolio_stop": not bool(stress.liquidated.astype(bool).any()),
        "stress_no_pair_stop": not bool(stress.pair_stop_triggered.astype(bool).any()),
    }


def build_plot(stage1, stage2, development, holdout, stops, stress, gates, policy, split):
    fig = make_subplots(
        rows=4, cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None], [{"colspan": 2}, None],
               [{"type": "table", "colspan": 2}, None]],
        row_heights=[0.24, 0.24, 0.20, 0.32], vertical_spacing=0.075,
        subplot_titles=("粗筛参数热力图", "收益—停止时长 Pareto", "累计盈亏（开发集与样本外分别从零开始）",
                        "60天样本外停止交易区域", "最终参数、压力测试与部署门槛"),
    )
    best_cells = stage1.sort_values("balanced_score").groupby(
        ["max_extra_inventory_quote", "max_hold_hours"], as_index=False
    ).tail(1)
    pivot = best_cells.pivot(index="max_hold_hours", columns="max_extra_inventory_quote",
                             values="balanced_score")
    fig.add_trace(go.Heatmap(
        x=pivot.columns, y=pivot.index, z=pivot.values, text=pivot.values,
        texttemplate="%{text:.2f}", colorscale="Blues", showscale=False,
        hovertemplate="库存上限 %{x} FDUSD<br>最长持有 %{y}h<br>最佳综合分 %{z:.3f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=stage1.pair_stop_hours + stage1.portfolio_stop_hours,
        y=stage1.oos_pnl_fdusd, mode="markers", name="420组粗筛",
        marker={"color": stage1.worst_drawdown_pct, "colorscale": "Blues", "size": 6,
                "showscale": False},
        customdata=stage1[["policy_id", "balanced_score", "worst_drawdown_pct"]],
        hovertemplate=("%{customdata[0]}<br>停止时长 %{x:.1f}h<br>收益 %{y:+.2f}"
                       "<br>最差回撤 %{customdata[2]:.2f}%<br>综合分 %{customdata[1]:.3f}<extra></extra>"),
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=stage2.pair_stop_hours + stage2.portfolio_stop_hours,
        y=stage2.oos_pnl_fdusd, mode="markers+text", name="联合搜索前10",
        marker={"color": "#D97706", "size": 9, "symbol": "diamond"},
        text=stage2["rank"], textposition="top center",
        customdata=stage2[["policy_id", "balanced_score"]],
        hovertemplate="排名 %{text}<br>%{customdata[0]}<br>停止时长 %{x:.1f}h<br>收益 %{y:+.2f}<extra></extra>",
    ), row=1, col=2)

    colors = {"development_online": "#9CA3AF", "development_new": "#D97706",
              "holdout_online": "#374151", "holdout_new": "#2563EB"}
    display_labels = {"development_online": "开发集·线上", "development_new": "开发集·新机制",
                      "holdout_online": "样本外·线上", "holdout_new": "样本外·新机制"}
    for label, frame in development + holdout:
        values = pd.DataFrame(frame)
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(values.timestamp, unit="s", utc=True), y=values.cumulative_oos_pnl,
            mode="lines", name=display_labels[label], line={"color": colors[label], "width": 2,
                                             "dash": "dash" if label.startswith("development") else "solid"},
            hovertemplate=("%{x|%Y-%m-%d}<br>%{y:+.2f} FDUSD<extra>"
                           + display_labels[label] + "</extra>"),
        ), row=2, col=1)
    split_time = pd.to_datetime(split, unit="s", utc=True)
    fig.add_shape(type="line", xref="x3", yref="y3 domain",
                  x0=split_time, x1=split_time, y0=0, y1=1,
                  line={"color": "#6B7280", "width": 1.2, "dash": "dash"})

    scope_order = ["技术门：暂停新买单", "线上：BTC", "线上：ETH", "线上：组合",
                   "新机制：BTC", "新机制：ETH", "新机制：组合"]
    kind_colors = {"technical_pause": "#9CA3AF", "pair_stop": "#D97706",
                   "portfolio_stop": "#B91C1C"}
    kind_names = {"technical_pause": "技术门暂停买入", "pair_stop": "单对停止",
                  "portfolio_stop": "组合停止"}
    shown = set()
    for item in stops.itertuples(index=False):
        fig.add_trace(go.Scatter(
            x=[item.start, item.end], y=[item.scope, item.scope], mode="lines",
            line={"color": kind_colors[item.kind], "width": 11},
            name=kind_names[item.kind], legendgroup=item.kind, showlegend=item.kind not in shown,
            hovertemplate="%{y}<br>%{x|%Y-%m-%d %H:%M UTC}<extra></extra>",
        ), row=3, col=1)
        shown.add(item.kind)
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(scope_order)), row=3, col=1)

    winner = stage2.iloc[0]
    rows = [
        ("最优库存参数", policy_id(policy), "锁定60天后不再调参"),
        ("开发集综合排名", "1 / 10", f"综合分 {winner.balanced_score:.3f}"),
        ("样本外收益", f"{holdout[1][1][-1]['cumulative_oos_pnl']:+.2f} FDUSD" if holdout[1][1] else "n/a",
         "部署要求 > 0"),
        ("压力测试", f"{sum(stress.pair_stop_triggered.astype(bool))} 个场景触发单对停止",
         "要求全部不触发"),
        ("部署结论", "GO" if all(gates.values()) else "NO-GO",
         "; ".join(key for key, passed in gates.items() if not passed) or "全部通过"),
    ]
    fig.add_trace(go.Table(
        header={"values": ["项目", "结果", "说明"], "fill_color": "#E5E7EB",
                "align": "left", "font": {"color": "#111827", "size": 12}},
        cells={"values": list(zip(*rows)), "fill_color": "#FFFFFF", "align": "left",
               "font": {"color": "#374151", "size": 11}, "height": 28},
        columnwidth=[0.22, 0.34, 0.44],
    ), row=4, col=1)
    fig.update_layout(
        title={"text": "FDUSD 库存退出机制参数搜索<br><sup>前120天开发｜后60天锁定样本外｜全部线上机制启用，宏观门关闭</sup>",
               "x": 0.02, "xanchor": "left"},
        template="plotly_white", height=1900, margin={"l": 90, "r": 65, "t": 155, "b": 40},
        font={"family": "Arial, Microsoft YaHei, sans-serif", "color": "#1F2937"},
        legend={"orientation": "h", "y": 1.055, "x": 0.28}, hovermode="closest",
    )
    fig.update_xaxes(title_text="停止时长（pair-hours + portfolio-hours）", row=1, col=2)
    fig.update_yaxes(title_text="开发集样本外收益 FDUSD", zeroline=True, row=1, col=2)
    fig.update_yaxes(title_text="累计收益 FDUSD", zeroline=True, row=2, col=1)
    return fig


def technical_pause_rows(gate, start, end):
    rows, paused_at = [], None
    for timestamp in range(start, end, 300):
        paused = not bool(gate.get(timestamp, False))
        if paused and paused_at is None:
            paused_at = timestamp
        elif not paused and paused_at is not None:
            rows.append({"scope": "技术门：暂停新买单", "kind": "technical_pause",
                         "start_ts": paused_at, "end_ts": timestamp})
            paused_at = None
    if paused_at is not None:
        rows.append({"scope": "技术门：暂停新买单", "kind": "technical_pause",
                     "start_ts": paused_at, "end_ts": end})
    return rows


def labeled_stops(rows, label):
    mapped = []
    for row in rows:
        suffix = {"BTC-FDUSD": "BTC", "ETH-FDUSD": "ETH", "PORTFOLIO": "组合"}[row["scope"]]
        mapped.append({**row, "scope": f"{label}：{suffix}"})
    return mapped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/backtests/fdusd_inventory_exit_parameter_search"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-policies", type=int, help="Test-only coarse-search limit.")
    args = parser.parse_args()
    candles = {pair: read_cache(args.cache_dir / f"binance_{pair}_5m.csv") for pair in PAIRS}
    if any(frame.empty for frame in candles.values()):
        raise RuntimeError("BTC/ETH FDUSD candle cache is required.")
    common_end = min(int(frame.timestamp.max()) for frame in candles.values()) + 300
    end_ts = local_day_start(common_end)
    holdout_start = end_ts - 60 * DAY_SECONDS
    development_start = holdout_start - 120 * DAY_SECONDS
    if any(int(frame.timestamp.min()) > development_start for frame in candles.values()):
        raise RuntimeError("Candle cache does not cover the aligned 180-day range.")
    development_windows = rolling_validation_windows(development_start, holdout_start)
    final_windows = holdout_windows(holdout_start, end_ts)
    if not development_windows or not final_windows or development_windows[-1][3] > holdout_start:
        raise RuntimeError("Invalid development/holdout split.")
    gate = technical_buy_gate_timeline(candles["BTC-FDUSD"])
    policies = inventory_policy_space()
    if args.max_policies:
        policies = policies[:args.max_policies]

    stage1_rows = []
    for index, policy in enumerate(policies, 1):
        summary, _, _ = fixed_policy_evaluation(candles, gate, development_windows, policy)
        stage1_rows.append({**policy_row(policy), **summary})
        if index % 25 == 0 or index == len(policies):
            print(f"STAGE1 {index}/{len(policies)}", flush=True)
    stage1 = add_balanced_score(pd.DataFrame(stage1_rows))
    shortlisted = [
        next(policy for policy in policies if policy_id(policy) == identifier)
        for identifier in stage1.head(min(args.top_k, len(stage1))).policy_id
    ]

    stage2_rows, stage2_details, all_candidate_rows = [], {}, []
    for index, policy in enumerate(shortlisted, 1):
        detail = walk_forward_policy(
            candles, gate, development_windows, policy, INITIAL_APPROVED,
            keep_candidate_rows=True, collect_details=False,
        )
        identifier = policy_id(policy)
        stage2_details[identifier] = detail
        all_candidate_rows.extend(detail["candidate_rows"])
        stage2_rows.append({**policy_row(policy), **detail["summary"],
                            "retained_parameter_folds": sum(row["parameters_retained"] for row in detail["weekly"])})
        print(f"STAGE2 {index}/{len(shortlisted)}", flush=True)
    stage2 = add_balanced_score(pd.DataFrame(stage2_rows))
    winner_id = str(stage2.iloc[0].policy_id)
    winner_policy = next(policy for policy in shortlisted if policy_id(policy) == winner_id)
    winner_dev_selection = stage2_details[winner_id]
    winner_development = replay_policy(
        candles, gate, development_windows, winner_policy, winner_dev_selection["selections"]
    )
    online_development = walk_forward_policy(
        candles, gate, development_windows, None, INITIAL_APPROVED, collect_details=True
    )

    online_holdout = walk_forward_policy(
        candles, gate, final_windows, None, online_development["final_candidate"], collect_details=True
    )
    winner_holdout = walk_forward_policy(
        candles, gate, final_windows, winner_policy,
        winner_dev_selection["final_candidate"], collect_details=True
    )
    stress = stress_tests(
        candles, gate, winner_holdout["final_candidate"], winner_policy, end_ts
    )
    gates = deployment_gates(winner_holdout["summary"], winner_holdout["pairs"], stress)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage1.to_csv(args.output_dir / "stage1_all_420_policies.csv", index=False)
    stage2.to_csv(args.output_dir / "stage2_top10_joint_search.csv", index=False)
    pd.DataFrame(all_candidate_rows).to_csv(args.output_dir / "stage2_grid_candidate_evaluations.csv", index=False)
    comparison_weekly, comparison_pairs, comparison_equity, comparison_trades = [], [], [], []
    for period, label, detail in (
        ("development", "online", online_development),
        ("development", "new", winner_development),
        ("holdout", "online", online_holdout),
        ("holdout", "new", winner_holdout),
    ):
        comparison_weekly.extend({"period": period, "scenario": label, **row} for row in detail["weekly"])
        comparison_pairs.extend({"period": period, "scenario": label, **row} for row in detail["pairs"])
        comparison_equity.extend({"period": period, "scenario": label, **row} for row in detail["equity"])
        comparison_trades.extend({"period": period, "scenario": label, **row} for row in detail["trades"])
    pd.DataFrame(comparison_weekly).to_csv(args.output_dir / "weekly_results.csv", index=False)
    pd.DataFrame(comparison_pairs).to_csv(args.output_dir / "pair_results.csv", index=False)
    pd.DataFrame(comparison_equity).to_csv(args.output_dir / "equity_curves.csv", index=False)
    pd.DataFrame(comparison_trades).to_csv(args.output_dir / "trades.csv", index=False)
    stress.to_csv(args.output_dir / "stress_tests.csv", index=False)

    stop_rows = technical_pause_rows(gate, holdout_start, final_windows[-1][3])
    stop_rows += labeled_stops(online_holdout["stops"], "线上")
    stop_rows += labeled_stops(winner_holdout["stops"], "新机制")
    stops = pd.DataFrame(stop_rows)
    stops["start"] = pd.to_datetime(stops.start_ts, unit="s", utc=True)
    stops["end"] = pd.to_datetime(stops.end_ts, unit="s", utc=True)
    stops.to_csv(args.output_dir / "stopped_trading_regions.csv", index=False)

    payload = {
        "range": {"development_start": development_start, "holdout_start": holdout_start,
                  "end_ts": end_ts, "development_folds": len(development_windows),
                  "holdout_folds": len(final_windows)},
        "macro_gate_enabled": False,
        "score_weights": SCORE_WEIGHTS,
        "winner_policy": asdict(winner_policy),
        "winner_development": winner_development["summary"],
        "online_holdout": online_holdout["summary"],
        "winner_holdout": winner_holdout["summary"],
        "deployment_gates": gates,
        "deployment_allowed": all(gates.values()),
        "failed_gates": [key for key, value in gates.items() if not value],
        "oci_mutated": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    development_plot = [
        ("development_online", online_development["equity"]),
        ("development_new", winner_development["equity"]),
    ]
    holdout_plot = [
        ("holdout_online", online_holdout["equity"]),
        ("holdout_new", winner_holdout["equity"]),
    ]
    figure = build_plot(stage1, stage2, development_plot, holdout_plot, stops, stress,
                        gates, winner_policy, holdout_start)
    figure.write_html(args.output_dir / "fdusd_inventory_exit_parameter_search.html",
                      include_plotlyjs=True, full_html=True,
                      config={"displaylogo": False, "responsive": True})
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
