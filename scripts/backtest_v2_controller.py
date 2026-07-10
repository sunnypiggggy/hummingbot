#!/usr/bin/env python
"""
Generic V2 controller backtest helper for controllers under ./controllers.

Examples:
    python scripts/backtest_v2_controller.py --controller-type directional_trading --controller-name macd_bb_v1 --trading-pair BTC-USDT
    python scripts/backtest_v2_controller.py --controller-type market_making --controller-name pmm_dynamic --connector binance --trading-pair BTC-USDT
"""

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hummingbot.strategy_v2.backtesting.backtesting_engine_base import BacktestingEngineBase  # noqa: E402
from hummingbot.strategy_v2.backtesting.backtesting_result import BacktestingResult  # noqa: E402


def build_config(args):
    config_data = {
        "id": f"backtest_{args.controller_name}",
        "controller_name": args.controller_name,
        "controller_type": args.controller_type,
        "connector_name": args.connector,
        "trading_pair": args.trading_pair,
        "total_amount_quote": args.amount,
        "leverage": args.leverage,
        "stop_loss": str(args.stop_loss),
        "take_profit": str(args.take_profit),
        "time_limit": args.time_limit,
        "cooldown_time": args.cooldown_time,
    }
    if args.controller_name in {
        "bollinger_v1", "bollinger_v2", "bollingrid", "macd_bb_v1", "supertrend_v1", "pmm_dynamic"
    }:
        config_data.update({
            "interval": args.interval,
            "candles_connector": args.candles_connector or args.connector,
            "candles_trading_pair": args.candles_trading_pair or args.trading_pair,
        })
    if args.controller_type == "directional_trading":
        config_data.update({
            "max_executors_per_side": args.max_executors_per_side,
        })
        if args.controller_name in {"bollinger_v1", "bollinger_v2", "bollingrid", "macd_bb_v1"}:
            config_data.update({
                "bb_length": args.bb_length,
                "bb_std": args.bb_std,
                "bb_long_threshold": args.bb_long_threshold,
                "bb_short_threshold": args.bb_short_threshold,
            })
        if args.controller_name == "macd_bb_v1":
            config_data.update({
                "macd_fast": args.macd_fast,
                "macd_slow": args.macd_slow,
                "macd_signal": args.macd_signal,
            })
        if args.controller_name == "supertrend_v1":
            config_data.update({
                "length": args.supertrend_length,
                "multiplier": args.supertrend_multiplier,
                "percentage_threshold": args.percentage_threshold,
            })
    if args.controller_type == "market_making":
        config_data.update({
            "buy_spreads": args.buy_spreads,
            "sell_spreads": args.sell_spreads,
            "buy_amounts_pct": args.buy_amounts_pct,
            "sell_amounts_pct": args.sell_amounts_pct,
            "executor_refresh_time": args.executor_refresh_time,
            "skip_rebalance": True,
        })
        if args.controller_name == "pmm_dynamic":
            config_data.update({
                "macd_fast": args.macd_fast,
                "macd_slow": args.macd_slow,
                "macd_signal": args.macd_signal,
                "natr_length": args.natr_length,
            })
        if args.controller_name == "dman_maker_v2":
            config_data.update({
                "dca_spreads": args.dca_spreads,
                "dca_amounts": args.dca_amounts,
            })
    return BacktestingEngineBase.get_controller_config_instance_from_dict(config_data, controllers_module="controllers")


async def main():
    parser = argparse.ArgumentParser(description="Backtest a V2 controller")
    parser.add_argument("--controller-type", required=True, choices=["directional_trading", "market_making"])
    parser.add_argument("--controller-name", required=True)
    parser.add_argument("--connector", default=None)
    parser.add_argument("--trading-pair", default="BTC-USDT")
    parser.add_argument("--candles-connector", default=None)
    parser.add_argument("--candles-trading-pair", default=None)
    parser.add_argument("--days", type=float, default=30)
    parser.add_argument("--amount", type=int, default=1000)
    parser.add_argument("--interval", default="3m")
    parser.add_argument("--resolution", default="5m")
    parser.add_argument("--leverage", type=int, default=1)
    parser.add_argument("--stop-loss", type=float, default=0.03)
    parser.add_argument("--take-profit", type=float, default=0.02)
    parser.add_argument("--time-limit", type=int, default=2700)
    parser.add_argument("--cooldown-time", type=int, default=300)
    parser.add_argument("--max-executors-per-side", type=int, default=2)
    parser.add_argument("--bb-length", type=int, default=100)
    parser.add_argument("--bb-std", type=float, default=2.0)
    parser.add_argument("--bb-long-threshold", type=float, default=0.0)
    parser.add_argument("--bb-short-threshold", type=float, default=1.0)
    parser.add_argument("--macd-fast", type=int, default=21)
    parser.add_argument("--macd-slow", type=int, default=42)
    parser.add_argument("--macd-signal", type=int, default=9)
    parser.add_argument("--supertrend-length", type=int, default=20)
    parser.add_argument("--supertrend-multiplier", type=float, default=4.0)
    parser.add_argument("--percentage-threshold", type=float, default=0.01)
    parser.add_argument("--buy-spreads", default="1,2,4")
    parser.add_argument("--sell-spreads", default="1,2,4")
    parser.add_argument("--buy-amounts-pct", default="")
    parser.add_argument("--sell-amounts-pct", default="")
    parser.add_argument("--executor-refresh-time", type=int, default=300)
    parser.add_argument("--natr-length", type=int, default=14)
    parser.add_argument("--dca-spreads", default="0.01,0.02,0.04,0.08")
    parser.add_argument("--dca-amounts", default="0.1,0.2,0.4,0.8")
    args = parser.parse_args()
    if args.connector is None:
        args.connector = "binance_perpetual" if args.controller_type == "directional_trading" else "binance"

    end_ts = int(time.time())
    start_ts = end_ts - int(args.days * 24 * 3600)
    config = build_config(args)
    engine = BacktestingEngineBase()
    print(
        f"Running backtest: {args.controller_name} | {args.connector} {args.trading_pair} | "
        f"{args.days}d @ {args.resolution}"
    )
    result = await engine.run_backtesting(
        config, start_ts, end_ts,
        backtesting_resolution=args.resolution,
        trade_cost=0.0002,
    )
    r = result["results"]
    print(f"Net PnL: {r['net_pnl_quote']:.4f} USDT ({r['net_pnl'] * 100:.2f}%)")
    print(f"Sharpe: {r['sharpe_ratio']:.4f}")
    print(f"Max drawdown: {r['max_drawdown_pct']:.4%}")
    print(f"Executors: {r['total_executors']} | With position: {r['total_executors_with_position']}")
    print(f"Close types: {r['close_types']}")
    print(BacktestingResult(result, config).get_results_summary())


if __name__ == "__main__":
    asyncio.run(main())
