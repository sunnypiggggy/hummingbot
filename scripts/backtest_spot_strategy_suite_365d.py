#!/usr/bin/env python3
"""Reproduce five isolated, cash-funded spot strategy experiments.

No live configuration, credentials, order endpoint, or live risk gate is used.
Run from the repository root with the hummingbot Conda Python environment.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.spot_strategy_suite_data import load_market_data, fetch_exchange_filters
from scripts.spot_strategy_suite_engine import (
    STRATEGY_IDS, aggregate_candles, generate_signals, rotation_targets,
    simulate_single, simulate_rotation, simulate_buy_hold,
)
from scripts.spot_strategy_suite_reporting import (
    combine_sleeves, make_reviewed_snapshot, result_fingerprints, save_result,
    summarize, write_json, write_readout,
)

PERIODS = {"donchian": 60, "cluc": 5, "rsi2": 60, "macd_ema": 60}


def utc_boundary(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    if ts != ts.floor("min"):
        raise ValueError("start/end must be UTC minute boundaries")
    return ts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    parser.add_argument("--start", default="2025-10-23T00:00:00Z")
    parser.add_argument("--end", default="2026-10-23T00:00:00Z")
    parser.add_argument("--capital", type=float, default=10000.0, help="total capital per strategy group, USDT")
    parser.add_argument("--warmup-days", type=int, default=210)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--stress-fee-bps", type=float, default=20.0)
    parser.add_argument("--stress-slippage-bps", type=float, default=5.0)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/spot_strategy_suite_365d")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/backtests/spot_strategy_suite_365d_20261023")
    parser.add_argument("--offline", action="store_true", help="require validated cached data and filter snapshot")
    parser.add_argument("--skip-repeat-check", action="store_true", help="skip deterministic rerun verification (recorded)")
    parser.add_argument("--skip-report", action="store_true", help="write numerical artifacts without rebuilding the optional report app")
    args = parser.parse_args()
    symbols = tuple(s.strip().upper() for s in args.symbols.split(","))
    if set(symbols) != {"BTCUSDT", "ETHUSDT"} or len(symbols) != 2:
        parser.error("this fixed experiment requires BTCUSDT,ETHUSDT")
    symbols = ("BTCUSDT", "ETHUSDT")
    start, end = utc_boundary(args.start), utc_boundary(args.end)
    if start >= end or args.capital <= 0 or args.warmup_days < 200:
        parser.error("require start<end, positive capital, and at least 200 warmup days")
    if any(v < 0 or v >= 10000 for v in (args.fee_bps, args.slippage_bps, args.stress_fee_bps, args.stress_slippage_bps)):
        parser.error("cost bps must be in [0,10000)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {"symbols": symbols, "start": start.isoformat(), "end": end.isoformat(),
              "capital": args.capital, "sleeve_capital": args.capital / 2, "warmup_days": args.warmup_days,
              "allocation": .95, "periods_minutes": PERIODS, "protective_stop": .05,
              "costs": {"base": {"fee": args.fee_bps / 10000, "slippage": args.slippage_bps / 10000},
                        "stress": {"fee": args.stress_fee_bps / 10000, "slippage": args.stress_slippage_bps / 10000}},
              "strategy_rules": {
                  "donchian": "1h close>prior20 high AND close>SMA200; exit close<prior20 low",
                  "cluc": "5m close<EMA50 AND close<.985*BB(typical_price,20,2,ddof1).lower AND volume<prior30 mean*20; exit close>BBmid; net ROI1%, SL5%",
                  "rsi2": "1h close>SMA200 AND RSI2<=10; exit close>SMA5; long-only adaptation",
                  "macd_ema": "1h close>EMA100 AND MACD12/26/9>signal; exit close<EMA100 AND MACD<signal; state not cross",
                  "rotation": "daily ratio ETHUSDT/BTCUSDT>EMA20 select ETH else BTC; selected close>SMA200 else USDT; equality BTC; no rebalance when unchanged",
              }, "baseline": "same 95% initial cash allocation, no rebalancing, fees/slippage included",
              "risk_variant": "5% stop from actual entry price; wait for strictly later strategy boundary to re-enter",
              "cost_classification": "research scenarios, not measured historical account rates",
              "accounting": "quote-paid fees, base/quote nonnegative; no leverage/no BNB fee discount",
              "equity_timestamp_semantics": "minute open label; MTM at minute close",
              "repeat_check": not args.skip_repeat_check}
    write_json(args.output_dir / "config.json", config)
    print("Loading and validating public spot candles...", flush=True)
    frames, data_manifest = load_market_data(symbols, start, end, args.warmup_days, args.cache_dir, args.offline)
    filter_path = args.cache_dir / "exchange_filters_current.json"
    if args.offline:
        if not filter_path.exists():
            raise RuntimeError("Offline replay needs the saved exchange filter snapshot")
        saved = json.loads(filter_path.read_text(encoding="utf-8"))
        filters, filter_meta = saved["filters"], saved["metadata"]
    elif filter_path.exists():
        saved = json.loads(filter_path.read_text(encoding="utf-8"))
        filters, filter_meta = saved["filters"], saved["metadata"]
    else:
        filters, filter_meta = fetch_exchange_filters(symbols)
        write_json(filter_path, {"filters": filters, "metadata": filter_meta})
    signals = {}
    for symbol in symbols:
        for strategy in STRATEGY_IDS:
            bars = aggregate_candles(frames[symbol], PERIODS[strategy])
            signals[(symbol, strategy)] = generate_signals(bars, strategy)
    daily_frames = {s: aggregate_candles(frames[s], 1440) for s in symbols}
    targets = rotation_targets(daily_frames)
    summary_rows, histories, months, run_manifest = [], [], [], {}
    replay_verifications = []

    def record(result: dict, strategy: str, pair: str, scenario: str, variant: str, repeat=None):
        run_id = f"{scenario}__{variant}__{strategy}__{pair}"
        fingerprint = save_result(args.output_dir / "runs" / run_id, result)
        if repeat is not None and not args.skip_repeat_check:
            repeated = result_fingerprints(repeat())
            if repeated != fingerprint:
                raise AssertionError(f"Non-deterministic result: {run_id}")
            replay_verifications.append({"run_id": run_id, "passed": True, "fingerprints": fingerprint})
        row, history, monthly = summarize(result, run_id=run_id, strategy=strategy, pair=pair,
                                           scenario=scenario, variant=variant, start=start, end=end)
        summary_rows.append(row)
        histories.append(history)
        months.append(monthly)
        run_manifest[run_id] = {"fingerprints": fingerprint, "metadata": result["metadata"]}
        print(f"{run_id}: return={row['net_return']:.2%} DD={row['max_drawdown']:.2%} closed={row['closed_trades']}", flush=True)

    for scenario, costs in config["costs"].items():
        for strategy in STRATEGY_IDS:
            cluc_cache = None
            for variant, stop in (("original", None), ("stop5", .05)):
                sleeves = []
                for symbol in symbols:
                    kwargs = dict(symbol=symbol, strategy=strategy, start=start, end=end, capital=args.capital / 2,
                                  **costs, protective_stop=stop, filters=filters[symbol])
                    run = lambda s=symbol, kw=kwargs: simulate_single(frames[s], signals[(s, strategy)], **kw)
                    if strategy == "cluc" and variant == "stop5":
                        result = cluc_cache[symbol]
                        repeat = None
                    else:
                        result = run()
                        repeat = run
                    record(result, strategy, symbol, scenario, variant, repeat)
                    sleeves.append(result)
                combined = combine_sleeves(sleeves)
                record(combined, strategy, "PORTFOLIO", scenario, variant)
                if strategy == "cluc" and variant == "original":
                    cluc_cache = dict(zip(symbols, sleeves))
        for variant, stop in (("original", None), ("stop5", .05)):
            kwargs = dict(start=start, end=end, capital=args.capital, **costs, protective_stop=stop, filters=filters)
            run = lambda kw=kwargs: simulate_rotation(frames, targets, **kw)
            record(run(), "rotation", "PORTFOLIO", scenario, variant, run)
        benchmark_sleeves = []
        for symbol in symbols:
            kwargs = dict(symbol=symbol, start=start, end=end, capital=args.capital / 2, **costs, filters=filters[symbol])
            run = lambda s=symbol, kw=kwargs: simulate_buy_hold(frames[s], **kw)
            result = run()
            record(result, "buy_hold", symbol, scenario, "benchmark", run)
            benchmark_sleeves.append(result)
        record(combine_sleeves(benchmark_sleeves), "buy_hold_equal", "PORTFOLIO", scenario, "benchmark")
        idx = benchmark_sleeves[0]["equity"].index
        empty = benchmark_sleeves[0]
        cash = {"equity": pd.DataFrame({"equity": args.capital, "cash": args.capital, "base_value": 0., "exposure": 0.}, index=idx),
                "fills": empty["fills"].iloc[:0].copy(), "trades": empty["trades"].iloc[:0].copy(),
                "metadata": {"initial_capital": args.capital, "final_cash": args.capital, "final_positions": {}, "fees": 0,
                             "total_notional": 0, "slippage_cost": 0, "final_liquidation_equity": args.capital, "unrealized_pnl": 0}}
        record(cash, "cash", "PORTFOLIO", scenario, "benchmark")
    summary, history, monthly = pd.DataFrame(summary_rows), pd.concat(histories, ignore_index=True), pd.concat(months, ignore_index=True)
    summary.to_csv(args.output_dir / "summary.csv", index=False, float_format="%.12g", encoding="utf-8-sig")
    history.to_csv(args.output_dir / "daily_equity.csv", index=False, float_format="%.12g", encoding="utf-8-sig")
    monthly.to_csv(args.output_dir / "monthly_returns.csv", index=False, float_format="%.12g", encoding="utf-8-sig")
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "TA-Lib", "requests")}
    code_files = [Path(__file__), *sorted((ROOT / "scripts").glob("spot_strategy_suite_*.py")),
                  *sorted((ROOT / "scripts").glob("spot_strategy_suite_*.jsx")),
                  *sorted((ROOT / "scripts").glob("spot_strategy_suite_*.css"))]
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "config": config, "python": sys.version,
                "platform": platform.platform(), "versions": versions, "git_head": git_head,
                "research_code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files},
                "data": data_manifest, "exchange_filters": {"filters": filters, "metadata": filter_meta},
                "reproduction_argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                "offline_replay": "add --offline; preserve cache plus original archives/checksums and source evidence",
                "run_results": run_manifest, "repeat_verifications": replay_verifications,
                "cluc_stop5_reused": True, "live_trading_authorized": False}
    write_json(args.output_dir / "manifest.json", manifest)
    write_json(args.output_dir / "reviewed.json", make_reviewed_snapshot(summary, history, monthly, manifest=manifest))
    write_readout(args.output_dir / "README.md", summary, config)
    if not args.skip_report:
        from scripts.spot_strategy_suite_report_app import build_report
        report = build_report(args.output_dir)
        write_json(args.output_dir / "report_build.json", report)
    print(f"Complete: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
