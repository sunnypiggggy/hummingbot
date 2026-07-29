#!/usr/bin/env python3
"""Validate and stage the quote-only BTC/ETH FDUSD live grid without deploying it."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from decimal import Decimal
from html import escape
from pathlib import Path

import pandas as pd
import yaml

from fdusd_live_grid_optimizer import run_walk_forward, select_candidate
from grid_live_common import (
    PORTFOLIOS,
    ACTIVE_SELECTION_SCHEMA_VERSION,
    FDUSD_BUDGET,
    build_fdusd_bootstrap_plan,
    build_live_config,
    effective_take_profit,
    validate_live_config,
)
from validate_grid_live import (
    INTERVAL_SECONDS,
    crash_candles,
    load_candles,
    public_market_state,
    simulate,
    slice_window,
    technical_buy_gate_timeline,
)


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def candle_quality(pair: str, frame: pd.DataFrame, start_ts: int, end_ts: int) -> dict:
    window = frame[(frame.timestamp >= start_ts) & (frame.timestamp < end_ts)].copy()
    expected_rows = (end_ts - start_ts) // INTERVAL_SECONDS
    duplicate_timestamps = int(window.timestamp.duplicated().sum())
    timestamps = window.timestamp.sort_values()
    gaps = timestamps.diff().dropna()
    invalid_ohlc = int((
        (window.low > window.high)
        | (window.open <= 0)
        | (window.high <= 0)
        | (window.low <= 0)
        | (window.close <= 0)
        | (window.high < window[["open", "close"]].max(axis=1))
        | (window.low > window[["open", "close"]].min(axis=1))
    ).sum())
    coverage = len(window) / expected_rows if expected_rows else 0.0
    max_gap_seconds = int(gaps.max()) if not gaps.empty else 0
    return {
        "pair": pair,
        "grain_seconds": INTERVAL_SECONDS,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "rows": len(window),
        "expected_rows": expected_rows,
        "coverage": coverage,
        "duplicate_timestamps": duplicate_timestamps,
        "invalid_ohlc_rows": invalid_ohlc,
        "max_gap_seconds": max_gap_seconds,
        "passed": (
            coverage >= 0.995
            and duplicate_timestamps == 0
            and invalid_ohlc == 0
            and max_gap_seconds <= INTERVAL_SECONDS * 3
        ),
    }


def report_html(path: Path, payload: dict, weekly: pd.DataFrame,
                pair_metrics: pd.DataFrame, stress: pd.DataFrame) -> None:
    def table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "<p>无数据</p>"
        return frame.to_html(index=False, border=0, classes="data", float_format=lambda value: f"{value:.6f}")

    cards = "".join(
        f"<div class='metric'><span>{escape(str(key))}</span><strong>{escape(str(value))}</strong></div>"
        for key, value in payload["overview"].items()
    )
    decision_class = "go" if payload["validation_decision"] == "CONDITIONAL GO" else "no-go"
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FDUSD 双币 Grid 实盘滑窗验证</title>
<style>
body{{font-family:Arial,"Microsoft YaHei",sans-serif;margin:0;background:#f5f7fa;color:#17212b}}
main{{max-width:1360px;margin:auto;padding:24px}}h1,h2{{letter-spacing:0}}h1{{font-size:30px}}
.banner{{background:#fff;border-left:5px solid #b42318;padding:16px;margin:18px 0}}
.banner.go{{border-color:#067647}}.metrics{{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:10px}}
.metric{{background:#fff;border:1px solid #d8dee5;border-radius:6px;padding:12px}}
.metric span{{display:block;color:#5f6b76;font-size:13px}}.metric strong{{font-size:20px}}
section{{margin:24px 0}}.table-wrap{{overflow:auto;background:#fff;border:1px solid #d8dee5}}
table.data{{border-collapse:collapse;width:100%;font-size:12px}}.data th,.data td{{padding:7px;border-bottom:1px solid #e5e9ed;text-align:right;white-space:nowrap}}
.data th{{background:#eef2f5;position:sticky;top:0}}code{{background:#e9edf1;padding:2px 5px}}
.flow{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}}.step{{background:#fff;border:1px solid #d8dee5;padding:12px;border-radius:6px}}
@media(max-width:760px){{main{{padding:14px}}.metrics{{grid-template-columns:1fr 1fr}}.flow{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>FDUSD 双币 Grid 实盘滑窗验证</h1>
<div class="banner {decision_class}"><strong>{escape(payload["validation_decision"])}</strong>
<p>{escape(payload["decision_reason"])}</p></div>
<div class="metrics">{cards}</div>
<section><h2>仅 FDUSD 启动流程</h2><div class="flow">
<div class="step"><strong>1. 余额</strong><p>最低 420 FDUSD，建议 440 FDUSD。</p></div>
<div class="step"><strong>2. 私有预检</strong><p>费率、权限、IP、精度、深度和 test-order。</p></div>
<div class="step"><strong>3. 建仓</strong><p>各用不超过 100 FDUSD 买入 BTC 与 ETH。</p></div>
<div class="step"><strong>4. 建立基线</strong><p>记录实际成交数量和成本，部分成交即阻断。</p></div>
<div class="step"><strong>5. 人工授权</strong><p>只有 GO 且再次批准后才能启用。</p></div>
</div></section>
<section><h2>滚动样本外周结果</h2><div class="table-wrap">{table(weekly)}</div></section>
<section><h2>逐交易对样本外结果</h2><div class="table-wrap">{table(pair_metrics)}</div></section>
<section><h2>压力测试</h2><div class="table-wrap">{table(stress)}</div></section>
<section><h2>部署门禁</h2><ul>
<li><code>GRID_LIVE_TRADING_ENABLED=false</code>，本报告不会创建订单或 bot。</li>
<li>当前配置标记 <code>bootstrap_completed=false</code>，执行策略会拒绝下单。</li>
<li>账户实际费率和私有权限预检未完成前，部署许可固定为 false。</li>
<li>参数切换必须先撤销本实例挂单，并保留库存、成本、峰值权益和熔断状态。</li>
</ul></section>
</main></body></html>"""
    path.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=182)
    parser.add_argument("--end-ts", type=int)
    parser.add_argument("--maker-fee", type=float, default=0.0,
                        help="Verified account maker fee as a decimal fraction.")
    parser.add_argument("--taker-fee", type=float, default=0.001)
    parser.add_argument("--private-fee-verified", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/backtests/grid_live_fdusd_400_walk_forward"))
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    end_ts = int(args.end_ts or time.time()) // INTERVAL_SECONDS * INTERVAL_SECONDS
    start_ts = end_ts - args.days * 86400
    portfolio = PORTFOLIOS["FDUSD"]
    prices, symbols = public_market_state(portfolio.pairs)
    # Eight warm-up days provide at least 40 complete 4h bars before the
    # requested validation period. Data-quality checks still cover only the
    # exact requested period.
    candles = {
        pair: load_candles(pair, start_ts - 8 * 86400, end_ts, args.cache_dir, not args.no_download)
        for pair in portfolio.pairs
    }
    technical_gate = technical_buy_gate_timeline(candles["BTC-FDUSD"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    quality = pd.DataFrame([
        candle_quality(pair, frame, start_ts, end_ts)
        for pair, frame in candles.items()
    ])
    quality.to_csv(args.output_dir / "data_quality.csv", index=False)

    evaluations, weekly, pair_metrics = run_walk_forward(
        candles,
        args.maker_fee,
        start_ts,
        end_ts,
        technical_buy_gate=technical_gate,
    )
    evaluations.to_csv(args.output_dir / "candidate_evaluations.csv", index=False)
    weekly.to_csv(args.output_dir / "weekly_selections.csv", index=False)
    pair_metrics.to_csv(args.output_dir / "pair_metrics.csv", index=False)

    initial_training = {
        pair: frame[(frame.timestamp >= end_ts - 30 * 86400) & (frame.timestamp < end_ts)].reset_index(drop=True)
        for pair, frame in candles.items()
    }
    recommended, initial_candidates = select_candidate(
        initial_training,
        args.maker_fee,
        require_eligible=False,
        technical_buy_gate=technical_gate,
    )
    initial_eligible_count = int(initial_candidates.attrs.get("eligible_count", 0))
    initial_candidates.to_csv(args.output_dir / "initial_30d_candidates.csv", index=False)

    # Stress the latest seven-day window so the synthetic final-day shock is
    # actually reached. Running stress over the full 182 days can liquidate the
    # strategy before the injected shock and produce a false duplicate result.
    stress_window_days = 7
    stress_candles = slice_window(
        candles, end_ts - stress_window_days * 86400, end_ts
    )
    stress_rows = []
    for label, fee_multiplier, slippage in (
        ("base", 1.0, 0.0),
        ("fee_150pct", 1.5, 0.0),
        ("slippage_005pct", 1.0, 0.0005),
        ("slippage_010pct", 1.0, 0.001),
    ):
        result, _, pair_stats = simulate(
            stress_candles, recommended, args.maker_fee * fee_multiplier,
            taker_fee=args.taker_fee * fee_multiplier, slippage=slippage,
            technical_buy_gate=technical_gate,
        )
        stress_rows.append({
            "scenario": label,
            "pair_stop_triggered": any(value["liquidations"] for value in pair_stats.values()),
            **result,
        })
    crashed_stress_candles = crash_candles(stress_candles)
    crashed_history = {}
    stress_start = end_ts - stress_window_days * 86400
    for pair, frame in candles.items():
        history = frame[frame.timestamp < stress_start]
        crashed_history[pair] = pd.concat(
            [history, crashed_stress_candles[pair]], ignore_index=True
        )
    crash_technical_gate = technical_buy_gate_timeline(crashed_history["BTC-FDUSD"])
    crash_result, _, crash_pairs = simulate(
        crashed_stress_candles, recommended, args.maker_fee,
        taker_fee=args.taker_fee, slippage=0.001,
        technical_buy_gate=crash_technical_gate,
    )
    stress_rows.append({
        "scenario": "15pct_one_day_drop",
        "pair_stop_triggered": any(value["liquidations"] for value in crash_pairs.values()),
        **crash_result,
    })
    stress = pd.DataFrame(stress_rows)
    stress.to_csv(args.output_dir / "stress_tests.csv", index=False)

    total_oos_pnl = float(weekly.net_pnl_quote.sum())
    worst_dd = float(weekly.max_drawdown_pct.min())
    per_pair_total = pair_metrics.groupby("pair", as_index=False).net_pnl_quote.sum()
    pair_loss_floor = -float(FDUSD_BUDGET.pair_budget) * 0.03
    gates = {
        "candle_data_quality_passed": bool(quality.passed.all()),
        "oos_positive": total_oos_pnl > 0,
        "max_drawdown_within_6pct": worst_dd >= -0.06,
        "each_pair_fold_loss_within_3pct": bool(
            (pair_metrics.net_pnl_quote >= pair_loss_floor).all()
        ),
        "no_oos_portfolio_liquidation": not bool(weekly.liquidated.any()),
        "no_oos_pair_stop": not bool(pair_metrics.liquidations.astype(bool).any()),
        "stress_within_live_limits": not bool(
            stress.liquidated.astype(bool).any() or stress.pair_stop_triggered.astype(bool).any()
        ),
        "stress_scenarios_are_distinct": bool(
            stress.loc[stress.scenario == "base", "net_pnl_quote"].iloc[0]
            != stress.loc[stress.scenario == "15pct_one_day_drop", "net_pnl_quote"].iloc[0]
        ),
        "every_training_window_has_eligible_candidate": not bool(weekly.diagnostic_fallback.any()),
        "initial_30d_has_eligible_candidate": initial_eligible_count > 0,
        "private_fee_verified": bool(args.private_fee_verified),
    }
    quantitative_go = all(value for key, value in gates.items() if key != "private_fee_verified")
    conditional_go = quantitative_go and gates["private_fee_verified"]
    decision = "CONDITIONAL GO" if conditional_go else "NO-GO"
    if not quantitative_go:
        reason = "滚动样本外或压力测试未通过安全门槛，禁止部署。"
    else:
        reason = "量化门槛通过，但仍需完成私有账户权限、余额、IP、费率和 test-order 预检。"

    decimal_prices = {pair: Decimal(str(value)) for pair, value in prices.items()}
    bootstrap = build_fdusd_bootstrap_plan(decimal_prices)
    config = build_live_config(
        portfolio,
        decimal_prices,
        Decimal(str(args.maker_fee)),
        trading_enabled=False,
        bootstrap_from_quote=True,
        bootstrap_completed=False,
    )
    config.update({
        "grid_range": recommended.half_range * 2,
        "grid_levels": recommended.levels,
        "take_profit": float(effective_take_profit(
            Decimal(str(args.maker_fee)), Decimal(str(recommended.take_profit))
        )),
        "move_threshold": recommended.move_threshold,
        "min_grid_move_seconds": recommended.move_cooldown_seconds,
        "active_parameter_version": "initial-30d-v1",
    })
    validate_live_config(config)
    config_path = args.output_dir / portfolio.config_name
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    active_selection = {
        "schema_version": ACTIVE_SELECTION_SCHEMA_VERSION,
        "parameter_version": "initial-30d-v1",
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "valid_from": end_ts,
        "training_window": {
            "lookback_days": 30,
            "start_ts": end_ts - 30 * 86400,
            "end_ts": end_ts,
        },
        "trading_pairs": list(portfolio.pairs),
        "maker_fee": args.maker_fee,
        "parameters": {
            "half_range": recommended.half_range,
            "minimum_spread": recommended.min_spread,
            "take_profit": recommended.take_profit,
            "move_threshold": recommended.move_threshold,
            "levels": recommended.levels,
            "min_grid_move_seconds": recommended.move_cooldown_seconds,
        },
    }
    write_json(args.output_dir / "active_selection.json", active_selection)

    payload = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "validation_decision": decision,
        "decision_reason": reason,
        "period": {"start_ts": start_ts, "end_ts": end_ts, "days": args.days},
        "gates": gates,
        "recommended_candidate": {**asdict(recommended), "levels": recommended.levels},
        "initial_eligible_candidates": initial_eligible_count,
        "maker_fee_used": args.maker_fee,
        "taker_fee_used": args.taker_fee,
        "stress_window_days": stress_window_days,
        "technical_buy_gate": {
            "schema_version": "grid-technical-buy-gate-v3",
            "signal_pair": "BTC-FDUSD",
            "interval": "4h",
            "roc_length": 12,
            "sqzmom_length": 20,
            "trigger": "ROC48 <= -8% AND SQZMOM <= -3%",
            "action": "cancel and suppress BUY only; preserve SELL",
            "recovery": "first maroon SQZMOM bar while risk-off is active",
        },
        "actual_fee_required": True,
        "private_preflight_complete": False,
        "deployment_allowed": False,
        "trading_enabled": False,
        "bootstrap": bootstrap,
        "exchange_filters": {
            pair: {
                "status": symbols[pair].get("status"),
                "filters": symbols[pair].get("filters", []),
            }
            for pair in portfolio.pairs
        },
        "data_quality": quality.to_dict(orient="records"),
        "overview": {
            "结论": decision,
            "样本外收益": f"{total_oos_pnl:+.2f} FDUSD",
            "最差周内回撤": f"{worst_dd:.2%}",
            "周数": len(weekly),
            "候选数": 81,
            "实盘开关": "关闭",
        },
    }
    write_json(args.output_dir / "validation_result.json", payload)
    reservations = {
        "version": 2,
        "generated_at": payload["generated_at"],
        "portfolio": "FDUSD",
        "profile": portfolio.profile_name,
        "bot_name": portfolio.bot_name,
        "deployment_allowed": False,
        "trading_enabled": False,
        "bootstrap_completed": False,
        "quote_only_required": {"FDUSD": str(FDUSD_BUDGET.capital_limit)},
        "recommended_account_balance": {"FDUSD": bootstrap["recommended_balance"]},
        "bootstrap": bootstrap,
        "config": {
            "path": str(config_path),
            "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        },
        "active_selection": str(args.output_dir / "active_selection.json"),
        "validation_decision": decision,
        "private_preflight_complete": False,
    }
    write_json(args.output_dir / "capital_reservations.json", reservations)
    (args.output_dir / "validation_result.md").write_text(
        "# FDUSD Live Grid Walk-Forward Validation\n\n"
        f"- decision: **{decision}**\n"
        f"- reason: {reason}\n"
        f"- OOS aggregate PnL: {total_oos_pnl:+.2f} FDUSD\n"
        f"- worst fold drawdown: {worst_dd:.2%}\n"
        f"- recommended candidate: `{asdict(recommended)}`\n"
        "- strategy budget: 400 FDUSD; minimum account balance: 420 FDUSD; recommended account balance: 440 FDUSD\n"
        "- private account preflight: pending\n"
        "- bootstrap completed: false\n"
        "- live deployment: disabled\n",
        encoding="utf-8",
    )
    report_html(args.output_dir / "report.html", payload, weekly, per_pair_total, stress)
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 0 if decision == "CONDITIONAL GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
