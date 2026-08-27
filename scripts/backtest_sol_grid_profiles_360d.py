#!/usr/bin/env python3
"""SOL-FDUSD Binance AI Grid profiles with isolated weekly risk evidence.

The runner is offline-only.  It compares short/medium-sideways Grid profiles,
trains a fold-local SOL-only XGBoost gate, and also replays the prospective
BTC+ETH+SOL 620 FDUSD portfolio without writing any live selection or runtime
state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline.offline import get_plotlyjs
from plotly.subplots import make_subplots
import requests
from sklearn.metrics import average_precision_score, roc_auc_score

import backtest_binance_short_sideways_long_vs_bidirectional_360d as engine
import sol_grid_weekly_risk_v22 as sol_v22


ROOT = Path(__file__).resolve().parents[1]
START = pd.Timestamp("2025-08-31T00:00:00Z")
END = pd.Timestamp("2026-08-26T00:00:00Z")
START_TS = int(START.timestamp())
END_TS = int(END.timestamp())
ROWS = 103_680
OUTPUT = ROOT / "results/backtests/sol_fdusd_binance_ai_profiles_360d"
PAIR = "SOL-FDUSD"
PAIR_CAPITAL = 200.0
PORTFOLIO_RESERVE = 20.0
PAIR_LOSS = 6.0
PORTFOLIO_LOSS = 36.0
MAKER_FEE = 0.0
TAKER_FEE = 0.001
SLIPPAGE = 0.0002

PROFILES = {
    "short_sideways": engine.Preset(
        "short_sideways", "短期横盘", 0.07695669969152726, 18, 0.004
    ),
    "medium_sideways": engine.Preset(
        "medium_sideways", "中短期横盘", 0.12698379475402316, 18, 0.004
    ),
}
MIXED_PROFILES = {
    "BTC-FDUSD": engine.Preset(
        "medium_sideways", "BTC中短期横盘", 0.12698379475402316, 18, 0.004
    ),
    "ETH-FDUSD": engine.Preset(
        "long_volatility", "ETH长期波动", 0.5246511596640915, 18,
        0.014179761072002472,
    ),
}

CJK_FONT = Path("C:/Windows/Fonts/msyh.ttc")
if CJK_FONT.exists():
    font_manager.fontManager.addfont(str(CJK_FONT))
    plt.rcParams["font.family"] = font_manager.FontProperties(
        fname=str(CJK_FONT)
    ).get_name()
plt.rcParams["axes.unicode_minus"] = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def private_fee_evidence() -> dict[str, Any]:
    path = Path(os.getenv(
        "GRID_PRIVATE_PREFLIGHT_PATH",
        str(ROOT / "results/oci_grid_report/private_preflight.json"),
    ))
    result: dict[str, Any] = {
        "path": str(path), "verified_for_sol": False,
        "maker_fee": MAKER_FEE, "taker_fee": TAKER_FEE,
    }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        age = max(0, int(time.time()) - int(payload.get("fetched_at", 0)))
        pairs = {str(pair) for pair in payload.get("pairs", [])}
        checks = payload.get("checks", {})
        verified = bool(
            payload.get("private_preflight_complete") is True
            and checks.get("commission_verified") is True
            and payload.get("profile") == "binance_live_grid_fdusd_400"
            and PAIR in pairs
            and age <= 8 * 86400
            and float(payload.get("maker_fee")) == MAKER_FEE
            and float(payload.get("taker_fee")) == TAKER_FEE
        )
        result.update({
            "verified_for_sol": verified,
            "source_age_seconds": age,
            "source_pairs": sorted(pairs),
            "private_preflight_complete": bool(payload.get("private_preflight_complete")),
            "commission_verified": bool(checks.get("commission_verified")),
            "reason": "verified" if verified else "fresh SOL fee preflight is missing",
        })
    except (OSError, ValueError, TypeError) as exc:
        result["reason"] = f"fee evidence unavailable: {type(exc).__name__}: {exc}"
    return result


def read_cache(pair: str) -> pd.DataFrame:
    path = ROOT / "data/backtesting_candles" / f"{pair.replace('-', '')}_5m.csv"
    frame = pd.read_csv(path)
    timestamp = pd.to_numeric(frame["timestamp"], errors="raise")
    if timestamp.max() > 10_000_000_000:
        timestamp = timestamp // 1000
    frame["timestamp"] = timestamp.astype("int64")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame = frame[(frame.timestamp >= START_TS) & (frame.timestamp < END_TS)]
    frame = frame.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
    validate_candles(pair, frame)
    return frame[["timestamp", "open", "high", "low", "close", "volume"]]


def read_training_cache() -> pd.DataFrame:
    path = ROOT / "data/backtesting_candles/SOLFDUSD_5m.csv"
    frame = pd.read_csv(path)
    timestamp = pd.to_numeric(frame["timestamp"], errors="raise")
    if timestamp.max() > 10_000_000_000:
        timestamp = timestamp // 1000
    frame["timestamp"] = timestamp.astype("int64")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")


def validate_candles(pair: str, frame: pd.DataFrame) -> None:
    if len(frame) != ROWS:
        raise ValueError(f"{pair}: expected {ROWS} rows, got {len(frame)}")
    if not np.all(np.diff(frame.timestamp.to_numpy()) == 300):
        raise ValueError(f"{pair}: five-minute continuity failure")
    if frame.timestamp.duplicated().any():
        raise ValueError(f"{pair}: duplicated timestamps")
    if not (
        (frame.high >= frame[["open", "close"]].max(axis=1))
        & (frame.low <= frame[["open", "close"]].min(axis=1))
        & (frame.low > 0)
        & (frame.volume >= 0)
    ).all():
        raise ValueError(f"{pair}: OHLCV integrity failure")


def exchange_filter(pair: str) -> tuple[engine.ExchangeFilter, dict[str, Any]]:
    payload = requests.get(
        "https://api.binance.com/api/v3/exchangeInfo",
        params={"symbol": pair.replace("-", "")}, timeout=30,
    ).json()["symbols"][0]
    filters = {value["filterType"]: value for value in payload["filters"]}
    notional = filters.get("NOTIONAL") or filters["MIN_NOTIONAL"]
    result = engine.ExchangeFilter(
        tick_size=float(filters["PRICE_FILTER"]["tickSize"]),
        step_size=float(filters["LOT_SIZE"]["stepSize"]),
        minimum_notional=float(notional["minNotional"]),
    )
    return result, payload


def expand_gate(frame: pd.DataFrame, hourly: pd.DataFrame) -> np.ndarray:
    lookup = hourly.set_index("signal_ts")["model_signal"]
    signal_ts = ((frame.timestamp.to_numpy(np.int64) // 3600) * 3600)
    mapped = pd.Series(signal_ts).map(lookup)
    return mapped.fillna("UNAVAILABLE").to_numpy(object)


def existing_v22_gate(frame: pd.DataFrame, pair: str) -> np.ndarray:
    path = ROOT / "results/backtests/binance_ai_grid_presets_360d/v22_gate_states.csv.gz"
    evidence = pd.read_csv(path)
    pair_rows = evidence[evidence.pair == pair].copy()
    pair_rows["model_signal"] = np.where(pair_rows.risk_off_active, "RISK_OFF", "RISK_ON")
    return expand_gate(frame, pair_rows[["signal_ts", "model_signal"]])


def run_portfolio(
    candles: dict[str, pd.DataFrame], gates: dict[str, np.ndarray],
    filters: dict[str, engine.ExchangeFilter], profiles: dict[str, engine.Preset],
    risk_gate_enabled: bool, breakers_enabled: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, engine.PairState]]:
    engine.START_TS = START_TS
    engine.TAKER_FEE = TAKER_FEE
    engine.TAKER_SLIPPAGE = SLIPPAGE
    pairs = tuple(candles)
    states = {
        pair: engine.initialise_state(
            pair, "bidirectional", profiles[pair], filters[pair],
            float(candles[pair].iloc[0].close), 10.0,
        ) for pair in pairs
    }
    pair_halted_until = {pair: -1 for pair in pairs}
    portfolio_halted_until = -1
    portfolio_peak = len(pairs) * PAIR_CAPITAL + PORTFOLIO_RESERVE
    portfolio_baseline = portfolio_peak
    rows: list[dict[str, Any]] = []
    for bar in range(ROWS):
        prices = {pair: float(candles[pair].iloc[bar].close) for pair in pairs}
        timestamp = int(candles[pairs[0]].iloc[bar].timestamp)
        for pair in pairs:
            state = states[pair]
            signal = gates[pair][bar] if risk_gate_enabled else "RISK_ON"
            state.last_gate = signal
            blocked = risk_gate_enabled and (
                signal != "RISK_ON" or bar < pair_halted_until[pair]
                or bar < portfolio_halted_until
            )
            if blocked:
                state.healthy_cycles = 0
                state.blocked_bars += 1
                if state.active or state.orders:
                    engine.force_exit(state, prices[pair], timestamp, f"gate={signal}")
            else:
                state.healthy_cycles += 1
                if not state.active and state.healthy_cycles >= 3:
                    engine.reenter(state, prices[pair], timestamp, bar)
            if state.active:
                engine.process_maker_fills(state, candles[pair].iloc[bar], bar)
                half = state.preset.total_range / 2
                moved = (
                    prices[pair] > state.center * (1 + half) * (1 + state.preset.move_threshold)
                    or prices[pair] < state.center * (1 - half) * (1 - state.preset.move_threshold)
                ) and bar - state.last_move_bar >= 6
                refreshed = bar - state.last_refresh_bar >= 24
                if moved or refreshed:
                    engine.place_grid(state, bar, prices[pair])
                    if moved:
                        state.grid_moves += 1
                        state.last_move_bar = bar
        equities = {pair: states[pair].equity(prices[pair]) for pair in pairs}
        if risk_gate_enabled and breakers_enabled:
            for pair in pairs:
                state = states[pair]
                state.peak_equity = max(state.peak_equity, equities[pair])
                if state.active and (
                    state.cycle_baseline - equities[pair] >= PAIR_LOSS
                    or equities[pair] / state.peak_equity - 1 <= -0.03
                ):
                    pair_halted_until[pair] = bar + 72
                    engine.force_exit(state, prices[pair], timestamp, "pair_breaker")
            portfolio_equity = sum(states[pair].equity(prices[pair]) for pair in pairs) + PORTFOLIO_RESERVE
            portfolio_peak = max(portfolio_peak, portfolio_equity)
            if bar >= portfolio_halted_until and (
                portfolio_baseline - portfolio_equity >= PORTFOLIO_LOSS
                or portfolio_equity / portfolio_peak - 1 <= -0.06
            ):
                portfolio_halted_until = bar + 144
                for pair in pairs:
                    engine.force_exit(states[pair], prices[pair], timestamp, "portfolio_breaker")
                portfolio_baseline = sum(states[pair].equity(prices[pair]) for pair in pairs) + PORTFOLIO_RESERVE
                portfolio_peak = portfolio_baseline
        for pair in pairs:
            state = states[pair]
            state.exposure_sum += state.base * prices[pair]
            rows.append({
                "timestamp": timestamp, "pair": pair, "profile": profiles[pair].preset_id,
                "scope": (
                    "protected" if breakers_enabled else
                    "v22_only" if risk_gate_enabled else "parameter_only"
                ),
                "price": prices[pair], "equity": state.equity(prices[pair]),
                "quote": state.quote, "base": state.base,
                "model_signal": state.last_gate, "active": state.active,
            })
    events = pd.DataFrame([event for state in states.values() for event in state.events])
    return pd.DataFrame(rows), events, states


def metrics(equity: pd.DataFrame, states: dict[str, engine.PairState], run: str) -> list[dict[str, Any]]:
    result = []
    for pair, state in states.items():
        values = equity[equity.pair == pair].equity.to_numpy(float)
        drawdown = values / np.maximum.accumulate(values) - 1
        result.append({
            "run": run, "pair": pair, "profile": state.preset.preset_id,
            "net_pnl_fdusd": float(values[-1] - PAIR_CAPITAL),
            "return_pct": float((values[-1] / PAIR_CAPITAL - 1) * 100),
            "max_drawdown_pct": float(drawdown.min() * 100),
            "maker_fills": state.maker_buys + state.normal_sells + state.take_profit_sells,
            "maker_orders": state.maker_orders,
            "maker_fill_rate_pct": float(
                (state.maker_buys + state.normal_sells + state.take_profit_sells)
                / state.maker_orders * 100 if state.maker_orders else 0
            ),
            "minimum_order_fdusd": None if math.isinf(state.min_order_notional) else state.min_order_notional,
            "grid_moves": state.grid_moves, "forced_exits": state.forced_exits,
            "reentries": state.reentries, "fees_fdusd": state.fees,
            "blocked_hours": state.blocked_bars * 5 / 60,
            "ending_base": state.base, "ending_quote": state.quote,
        })
    portfolio = equity.groupby("timestamp", as_index=False).equity.sum()
    portfolio["equity"] += PORTFOLIO_RESERVE
    values = portfolio.equity.to_numpy(float)
    result.append({
        "run": run, "pair": "PORTFOLIO", "profile": "mixed",
        "net_pnl_fdusd": float(values[-1] - (len(states) * 200 + 20)),
        "return_pct": float((values[-1] / (len(states) * 200 + 20) - 1) * 100),
        "max_drawdown_pct": float((values / np.maximum.accumulate(values) - 1).min() * 100),
    })
    return result


def build_plot(
    equities: dict[str, pd.DataFrame], gate: pd.DataFrame, output: Path,
    bundle: dict[str, Any], weekly_audit: pd.DataFrame, summary: pd.DataFrame,
) -> None:
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.028,
                        row_heights=[0.22, 0.17, 0.22, 0.22, 0.17],
                        subplot_titles=("SOL-FDUSD价格", "SOL-v22周度概率与fold-local阈值",
                                        "参数纯效果权益", "SOL-v22及现行风控生效权益", "回撤（%）"))
    colors = {"short_sideways": "#2563eb", "medium_sideways": "#d97706"}
    for profile, color in colors.items():
        for scope, row in (("parameter_only", 2), ("v22_only", 3), ("protected", 3)):
            frame = equities[f"{profile}:{scope}"]
            hourly = frame.copy()
            hourly["datetime"] = pd.to_datetime(hourly.timestamp, unit="s", utc=True)
            hourly = hourly.set_index("datetime").resample("1h").last().dropna().reset_index()
            fig.add_trace(go.Scatter(
                x=hourly.datetime, y=hourly.equity, mode="lines",
                name=f"{PROFILES[profile].label}-{scope}", line=dict(color=color,
                dash={"parameter_only": "solid", "v22_only": "solid", "protected": "dash"}[scope]),
                legendgroup=profile,
            ), row=row + 1, col=1)
            if scope in {"v22_only", "protected"}:
                dd = (hourly.equity / hourly.equity.cummax() - 1) * 100
                fig.add_trace(go.Scatter(
                    x=hourly.datetime, y=dd, mode="lines", name=f"{PROFILES[profile].label}回撤",
                    line=dict(color=color), legendgroup=profile, showlegend=False,
                ), row=5, col=1)
    price = equities["short_sideways:parameter_only"].copy()
    price["datetime"] = pd.to_datetime(price.timestamp, unit="s", utc=True)
    price = price.set_index("datetime").resample("1h").last().dropna().reset_index()
    fig.add_trace(go.Scatter(x=price.datetime, y=price.price, name="SOL价格",
                             line=dict(color="#374151")), row=1, col=1)
    hourly_gate = gate.copy()
    hourly_gate["datetime"] = pd.to_datetime(hourly_gate.signal_ts, unit="s", utc=True)
    fig.add_trace(go.Scatter(
        x=hourly_gate.datetime, y=hourly_gate.probability, name="周模型OOS概率",
        line=dict(color="#7c3aed", width=1.1), customdata=hourly_gate[["fold"]],
        hovertemplate="概率=%{y:.5f}<br>Fold=%{customdata[0]}<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=hourly_gate.datetime, y=hourly_gate.entry_threshold, name="本周校准阈值",
        line=dict(color="#ea580c", width=1.1, shape="hv"),
        hovertemplate="阈值=%{y:.5f}<extra></extra>",
    ), row=2, col=1)
    changes = hourly_gate.model_signal.ne(hourly_gate.model_signal.shift()).cumsum()
    for _, block in hourly_gate.groupby(changes):
        if block.model_signal.iloc[0] != "RISK_OFF":
            continue
        for row in (1, 2, 4, 5):
            fig.add_vrect(x0=block.datetime.iloc[0], x1=block.datetime.iloc[-1] + pd.Timedelta(hours=1),
                          fillcolor="rgba(190,24,93,0.08)", line_color="#be185d",
                          line_width=1, layer="below", row=row, col=1)
    fig.update_layout(
        title="SOL-FDUSD 独立v22周度Walk-forward｜360天离线验证（每周重训模型与阈值）",
        template="plotly_white", height=1550, hovermode="x unified",
        legend=dict(orientation="h", y=1.03),
    )
    fig.update_yaxes(range=[0, 1], row=2, col=1)
    fig.update_xaxes(
        rangeselector=dict(buttons=[
            dict(count=360, label="360天", step="day", stepmode="backward"),
            dict(count=60, label="近60天", step="day", stepmode="backward"),
            dict(step="all", label="全部"),
        ]), row=5, col=1,
    )
    plot = fig.to_html(
        full_html=False, include_plotlyjs=False, config={"responsive": True},
        div_id="sol-grid-profiles-360d",
    )
    valid = gate.target.notna()
    auc = roc_auc_score(gate.loc[valid, "target"], gate.loc[valid, "probability"])
    ap = average_precision_score(gate.loc[valid, "target"], gate.loc[valid, "probability"])
    risk_hours = int(gate.risk_off_active.sum())
    fold_table = weekly_audit[[
        "fold", "test_start_utc", "test_end_utc", "threshold", "best_tree_count",
        "mature_rows", "calibration_rows", "model_sha256",
    ]].copy()
    fold_table.columns = [
        "周", "样本外开始", "样本外结束", "本周阈值", "树数", "成熟训练样本",
        "校准样本", "模型哈希",
    ]
    fold_table["本周阈值"] = fold_table["本周阈值"].map(lambda value: f"{value:.6f}")
    fold_table["模型哈希"] = fold_table["模型哈希"].str.slice(0, 16) + "…"
    comparison = summary[(summary.pair == PAIR) & summary.run.isin([
        "short_sideways:parameter_only", "short_sideways:v22_only", "short_sideways:protected",
        "medium_sideways:parameter_only", "medium_sideways:v22_only", "medium_sideways:protected",
    ])][["run", "net_pnl_fdusd", "max_drawdown_pct", "maker_fills", "forced_exits", "blocked_hours"]].copy()
    comparison.columns = ["方案", "净收益FDUSD", "最大回撤%", "Maker成交", "强制退出", "受限小时"]
    for column in ("净收益FDUSD", "最大回撤%", "受限小时"):
        comparison[column] = comparison[column].map(lambda value: f"{float(value):.4f}")
    header = f"""
    <section class='report'>
      <h1>SOL-FDUSD 独立 v22 周度 Walk-forward：360天离线报告</h1>
      <p class='warning'>OFFLINE ONLY / NO-GO：本报告不授权 OCI 或实盘交易。</p>
      <div class='cards'>
        <div><b>周模型</b><span>{len(bundle['weeks'])} 个独立 XGBoost</span></div>
        <div><b>样本外覆盖</b><span>{len(gate):,} 小时，无缺周</span></div>
        <div><b>Risk-Off</b><span>{risk_hours:,} 小时（{risk_hours / len(gate) * 100:.2f}%）</span></div>
        <div><b>OOS ROC-AUC</b><span>{auc:.4f}</span></div>
        <div><b>OOS AP</b><span>{ap:.4f}（基准阳性率 {gate.loc[valid, 'target'].mean():.4f}）</span></div>
        <div><b>泄漏检查</b><span>最晚成熟标签 ≤ 每周截止：通过</span></div>
      </div>
      <h2>每周更新的参数</h2>
      <p>每周重新训练树权重与早停树数，使用最近14天的成熟校准样本生成 fold-local 98.5%分位阈值，并生成独立模型哈希。固定不搜索的部分是 xgb_34 超参数、15项 SOL-only 特征和 v22 状态机规则；BTC/ETH 数据、模型和状态均未参与。</p>
      <p><b>模型族：</b>{bundle['model_version']}　<b>血缘哈希：</b><code>{bundle['model_lineage_sha256']}</code></p>
      <h2>Grid 回放结果</h2>
      {comparison.to_html(index=False, border=0, classes='data')}
      <details><summary>查看53周训练审计（模型哈希、阈值、训练/校准样本）</summary>
      {fold_table.to_html(index=False, border=0, classes='data')}
      </details>
    </section>
    """
    style = """
    <style>
    body{margin:0;background:#f8fafc;color:#172033;font-family:'Microsoft YaHei','Noto Sans CJK SC',sans-serif}
    .report{max-width:1400px;margin:20px auto 0;padding:24px;background:#fff;border:1px solid #e2e8f0;border-radius:16px}
    h1{margin:0 0 10px;font-size:28px} h2{margin-top:24px;font-size:20px}.warning{color:#b91c1c;font-weight:700}
    .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}.cards div{padding:14px;border:1px solid #dbe4ef;border-radius:10px;background:#f8fafc}.cards b,.cards span{display:block}.cards span{margin-top:6px;font-size:16px}
    table.data{width:100%;border-collapse:collapse;font-size:13px}table.data th,table.data td{padding:8px;border-bottom:1px solid #e5e7eb;text-align:left}table.data th{background:#eef2ff;position:sticky;top:0}details{margin-top:18px}summary{cursor:pointer;font-weight:700}code{word-break:break-all}
    </style>
    """
    output.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' "
        "content='width=device-width,initial-scale=1'>" + style + "<script>" + get_plotlyjs()
        + "</script></head><body>" + header + plot + "</body></html>", encoding="utf-8",
    )


def png_windows(
    equities: dict[str, pd.DataFrame], gate: pd.DataFrame, output: Path,
) -> list[Path]:
    windows = {
        "360d": (START, END),
        "2026_01_02": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-03-01T00:00:00Z")),
        "2026_05_06": (pd.Timestamp("2026-05-01T00:00:00Z"), pd.Timestamp("2026-07-01T00:00:00Z")),
    }
    paths = []
    colors = {"short_sideways": "#2563eb", "medium_sideways": "#d97706"}
    for name, (start, end) in windows.items():
        fig, axes = plt.subplots(3, 1, figsize=(9, 15), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1.2, 1.2]})
        base = equities["short_sideways:parameter_only"].copy()
        base["datetime"] = pd.to_datetime(base.timestamp, unit="s", utc=True)
        base = base[(base.datetime >= start) & (base.datetime < end)].iloc[::12]
        axes[0].plot(base.datetime, base.price, color="#374151", linewidth=1.1, label="SOL价格")
        gate_view = gate.copy()
        gate_view["datetime"] = pd.to_datetime(gate_view.signal_ts, unit="s", utc=True)
        gate_view = gate_view[(gate_view.datetime >= start) & (gate_view.datetime < end)]
        blocks = gate_view.model_signal.ne(gate_view.model_signal.shift()).cumsum()
        first_risk_label = True
        first_entry_label = True
        first_recovery_label = True
        for _, block in gate_view.groupby(blocks):
            signal = str(block.model_signal.iloc[0])
            boundary = block.datetime.iloc[0]
            if signal == "RISK_OFF":
                finish = block.datetime.iloc[-1] + pd.Timedelta(hours=1)
                for axis in (axes[0], axes[2]):
                    axis.axvspan(
                        boundary, finish, color="#be185d", alpha=.08,
                        label="SOL周模型 Risk-Off" if first_risk_label else None,
                    )
                axes[0].axvline(
                    boundary, color="#be185d", linewidth=.8, linestyle="--",
                    label="进入 Risk-Off" if first_entry_label else None,
                )
                first_risk_label = False
                first_entry_label = False
            elif boundary > start:
                axes[0].axvline(
                    boundary, color="#15803d", linewidth=.8, linestyle=":",
                    label="恢复 Risk-On" if first_recovery_label else None,
                )
                first_recovery_label = False
        for profile, color in colors.items():
            for scope, axis, style in (
                ("parameter_only", axes[1], "-"), ("v22_only", axes[2], "-"),
                ("protected", axes[2], "--"),
            ):
                frame = equities[f"{profile}:{scope}"].copy()
                frame["datetime"] = pd.to_datetime(frame.timestamp, unit="s", utc=True)
                frame = frame[(frame.datetime >= start) & (frame.datetime < end)].iloc[::12]
                axis.plot(frame.datetime, frame.equity, color=color, linestyle=style,
                          linewidth=1.2, label=f"{PROFILES[profile].label}-{scope}")
        axes[0].set_title("SOL-FDUSD价格", loc="left")
        axes[1].set_title("参数纯效果：单机器人权益（FDUSD）", loc="left")
        axes[2].set_title("SOL-v22仅模型（实线）/叠加现行熔断（虚线）：权益（FDUSD）", loc="left")
        for axis in axes:
            axis.grid(alpha=.18)
            axis.legend(loc="upper left")
        fig.suptitle(f"SOL-FDUSD Grid 双参数对照｜{start.date()}–{end.date()} UTC", fontsize=16)
        fig.tight_layout(rect=(0, 0, 1, .97))
        path = output / f"sol_grid_profiles_{name}_mobile.png"
        # Telegram evidence validation is deliberately exact: 9x15 inches at
        # 160 dpi must remain 1440x2400.  ``bbox_inches='tight'`` crops a few
        # pixels and makes an otherwise valid report fail closed.
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    candles = {pair: read_cache(pair) for pair in ("BTC-FDUSD", "ETH-FDUSD", "SOL-FDUSD")}
    filters = {pair: exchange_filter(pair)[0] for pair in candles}
    gate_frame, bundle, weekly_audit, feature_importance = sol_v22.train_weekly_bundle(
        read_training_cache(), START, END,
    )
    public_bundle = sol_v22.public_bundle_metadata(bundle)
    gates = {
        "BTC-FDUSD": existing_v22_gate(candles["BTC-FDUSD"], "BTC-FDUSD"),
        "ETH-FDUSD": existing_v22_gate(candles["ETH-FDUSD"], "ETH-FDUSD"),
        "SOL-FDUSD": expand_gate(candles["SOL-FDUSD"], gate_frame),
    }
    all_equity, all_events, summary = [], [], []
    sol_equities: dict[str, pd.DataFrame] = {}
    for profile, preset in PROFILES.items():
        for scope, risk_gate_enabled, breakers_enabled in (
            ("parameter_only", False, False),
            ("v22_only", True, False),
            ("protected", True, True),
        ):
            equity, events, states = run_portfolio(
                {PAIR: candles[PAIR]}, {PAIR: gates[PAIR]}, {PAIR: filters[PAIR]},
                {PAIR: preset}, risk_gate_enabled, breakers_enabled,
            )
            key = f"{profile}:{scope}"
            equity["run"] = key
            events["run"] = key if not events.empty else None
            sol_equities[key] = equity
            all_equity.append(equity); all_events.append(events)
            summary.extend(metrics(equity, states, key))
        mixed_profiles = {**MIXED_PROFILES, PAIR: preset}
        equity, events, states = run_portfolio(
            candles, gates, filters, mixed_profiles, True, True,
        )
        key = f"mixed:{profile}:protected"
        equity["run"] = key
        events["run"] = key if not events.empty else None
        all_equity.append(equity); all_events.append(events)
        summary.extend(metrics(equity, states, key))
    equity_frame = pd.concat(all_equity, ignore_index=True)
    event_frames = [frame for frame in all_events if not frame.empty]
    event_frame = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    summary_frame = pd.DataFrame(summary)
    deterministic_gzip = {"method": "gzip", "mtime": 0}
    equity_frame.to_csv(
        args.output / "continuous_equity_5m.csv.gz", index=False,
        compression=deterministic_gzip,
    )
    event_frame.to_csv(
        args.output / "trades_and_events.csv.gz", index=False,
        compression=deterministic_gzip,
    )
    summary_frame.to_csv(args.output / "summary.csv", index=False)
    gate_frame.to_csv(
        args.output / "sol_weekly_gate_states.csv.gz", index=False,
        compression=deterministic_gzip,
    )
    joblib.dump(bundle, args.output / "sol_v22_weekly_model_bundle.joblib", compress=0)
    (args.output / "sol_weekly_model_bundle.json").write_text(
        json.dumps(public_bundle, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    weekly_audit.to_csv(args.output / "sol_v22_weekly_training_audit.csv", index=False)
    feature_importance.to_csv(args.output / "sol_v22_weekly_feature_importance.csv", index=False)
    build_plot(
        sol_equities, gate_frame, args.output / "sol_grid_profiles_360d.html",
        public_bundle, weekly_audit, summary_frame,
    )
    pngs = png_windows(sol_equities, gate_frame, args.output)
    candidate_identity = hashlib.sha256(json.dumps({
        "model": public_bundle,
        "profiles": {key: vars(value) for key, value in PROFILES.items()},
        "window": [START.isoformat(), END.isoformat()],
        "summary": summary,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )).hexdigest()
    windows = ("360d", "2026_jan_feb", "2026_may_june")
    fee_evidence = private_fee_evidence()
    manifest = {
        "schema": "sol-grid-mobile-evidence-v1",
        "identity_sha256": candidate_identity,
        "model_sha256": bundle["model_lineage_sha256"],
        "evidence_complete": True,
        "activation_eligible": bool(fee_evidence["verified_for_sol"]),
        "hard_gates": {"sol_account_fee_verified": bool(fee_evidence["verified_for_sol"])},
        "images": [
            {
                "path": path.name,
                "window": window,
                "pair": PAIR,
                "sha256": sha256_file(path),
            }
            for path, window in zip(pngs, windows)
        ],
    }
    (args.output / "sol_grid_evidence_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    evidence = {
        "schema": "sol-grid-profile-backtest-v1", "offline_only": True,
        "deployment_allowed": False, "first_live_activation_requires_manual_approval": True,
        "window": {"start": START.isoformat(), "end_exclusive": END.isoformat(), "bars": ROWS},
        "data_quality": {
            pair: {
                "bars": int(len(frame)),
                "first_timestamp": int(frame.timestamp.iloc[0]),
                "last_timestamp": int(frame.timestamp.iloc[-1]),
                "interval_seconds": 300,
                "continuous": bool(np.all(np.diff(frame.timestamp.to_numpy()) == 300)),
            }
            for pair, frame in candles.items()
        },
        "capital": {"pair": 200, "portfolio": 620, "reserve": 20},
        "fees": {
            "maker": MAKER_FEE, "taker": TAKER_FEE, "slippage": SLIPPAGE,
            "offline_assumption": True, "evidence": fee_evidence,
        },
        "fomc_included": False, "profiles": {key: vars(value) for key, value in PROFILES.items()},
        "candidate_identity_sha256": candidate_identity,
        "model": public_bundle, "summary": summary,
        "artifacts": {},
    }
    for path in [args.output / "summary.csv", args.output / "continuous_equity_5m.csv.gz",
                 args.output / "trades_and_events.csv.gz", args.output / "sol_weekly_gate_states.csv.gz",
                 args.output / "sol_weekly_model_bundle.json",
                 args.output / "sol_v22_weekly_model_bundle.joblib",
                 args.output / "sol_v22_weekly_training_audit.csv",
                 args.output / "sol_v22_weekly_feature_importance.csv",
                 args.output / "sol_grid_evidence_manifest.json",
                 args.output / "sol_grid_profiles_360d.html", *pngs]:
        evidence["artifacts"][path.name] = sha256_file(path)
    (args.output / "result.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    print(summary_frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
