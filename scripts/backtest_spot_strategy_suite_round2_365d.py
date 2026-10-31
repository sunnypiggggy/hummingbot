#!/usr/bin/env python3
"""Run five frozen new spot hypotheses on the first round's exact market cache.

Research only. Does not access private APIs, credentials, live configuration or
production state. Run from the repository root in the hummingbot environment.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
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
from scripts.spot_strategy_round2 import IDS, LABELS, PERIODS, RULES, generate_signals, simulate_strategy
from scripts.spot_strategy_suite_data import load_market_data
from scripts.spot_strategy_suite_engine import aggregate_candles, simulate_buy_hold
from scripts import spot_strategy_suite_reporting as reporting


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capital", type=float, default=10000.)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/spot_strategy_suite_365d")
    parser.add_argument("--previous-dir", type=Path, default=ROOT / "results/backtests/spot_strategy_suite_365d_20261023")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/backtests/spot_strategy_suite_round2_365d_20261024")
    parser.add_argument("--offline", action="store_true", help="explicitly document local-cache-only execution (always enforced)")
    parser.add_argument("--workers", type=int, default=3, help="independent process workers; economics are unchanged")
    parser.add_argument("--skip-repeat-check", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args()
    if args.capital <= 0 or not 1 <= args.workers <= 4:
        parser.error("capital must be positive; workers must be between 1 and 4")
    if args.output_dir.resolve() == args.previous_dir.resolve():
        parser.error("round two must not overwrite round one")
    start, end = pd.Timestamp("2025-10-23T00:00Z"), pd.Timestamp("2026-10-23T00:00Z")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.output_dir / "sources/sources.json"
    if not source_path.is_file():
        raise RuntimeError("Freeze reviewed source evidence in output/sources/sources.json before numerical runs")
    previous_manifest_path = args.previous_dir / "manifest.json"
    previous = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
    previous_summary = pd.read_csv(args.previous_dir / "summary.csv")
    original_files = [previous_manifest_path, args.previous_dir / "summary.csv", args.previous_dir / "report.html",
                      *[ROOT / p for p in previous["research_code_sha256"]]]
    preservation = {str(p): sha(p) for p in original_files}
    config = {"round": 2, "symbols": ["BTCUSDT", "ETHUSDT"], "start": start.isoformat(), "end": end.isoformat(),
              "capital": args.capital, "sleeve_capital": args.capital / 2, "warmup_days": 210,
              "allocation": .95, "periods_minutes": PERIODS, "strategy_rules": RULES,
              "strategy_labels": LABELS, "protective_stop": .05,
              "costs": {"base": {"fee": .001, "slippage": .0002}, "stress": {"fee": .002, "slippage": .0005}},
              "baseline": "same 95% initial cash allocation including entry fees; no rebalancing; both coins independent equal-capital sleeves",
              "risk_variant": "original no added stop/ROI, stop5 = 5% from actual entry fill; re-entry strictly later strategy signal, momentum waits next Monday",
              "cash_yield": 0, "accounting": "quote-paid taker fees; cash/base nonnegative, no borrowing/pyramiding/BNB fee discounts",
              "equity_timestamp_semantics": "minute open label; MTM at minute close",
              "cost_classification": "research assumptions, not historical account actual rates",
              "repeat_check": not args.skip_repeat_check,
              "selection_warning": "These new hypotheses were selected after viewing round one's results on this same year. No optimization this round, but not an independent out-of-sample test."}
    for name in ("start", "end", "capital", "warmup_days", "costs"):
        if config[name] != previous["config"][name]:
            raise ValueError(f"Round-one comparison requires identical {name}")
    reporting.LABELS.update(LABELS)  # process-local display registry; no old source edit
    frozen = {"frozen_at": datetime.now(timezone.utc).isoformat(), "config": config,
              "strategy_code_sha256": sha(ROOT / "scripts/spot_strategy_round2.py"), "sources_sha256": sha(source_path),
              "numerical_results_observed": False, "parameter_search": False}
    frozen_path = args.output_dir / "frozen_spec.json"
    if frozen_path.exists():
        existing = json.loads(frozen_path.read_text(encoding="utf-8"))
        for name in ("config", "strategy_code_sha256", "sources_sha256"):
            if existing.get(name) != frozen[name]:
                raise RuntimeError(f"Frozen specification changed ({name}); preserve this output and use a new experiment directory")
        frozen = existing
    else:
        reporting.write_json(frozen_path, frozen)
    reporting.write_json(args.output_dir / "config.json", config)
    print("Validating the unchanged round-one public market cache...", flush=True)
    symbols = config["symbols"]
    frames, data_manifest = load_market_data(symbols, start, end, 210, args.cache_dir, True)
    for symbol in symbols:
        if data_manifest["symbols"][symbol]["normalized_cache_sha256"] != previous["data"]["symbols"][symbol]["normalized_cache_sha256"]:
            raise AssertionError(f"Market data changed: {symbol}")
    filter_path = args.cache_dir / "exchange_filters_current.json"
    saved = json.loads(filter_path.read_text(encoding="utf-8"))
    if saved["filters"] != previous["exchange_filters"]["filters"]:
        raise AssertionError("Exchange filter snapshot differs from round one")
    filters = saved["filters"]
    signals, bar_cache = {}, {}
    for symbol in symbols:
        for strategy in IDS:
            key = symbol, PERIODS[strategy]
            if key not in bar_cache:
                bar_cache[key] = aggregate_candles(frames[symbol], PERIODS[strategy])
            signals[symbol, strategy] = generate_signals(bar_cache[key], strategy)
            signals[symbol, strategy].to_csv(args.output_dir / f"signals_{strategy}_{symbol}.csv", index_label="timestamp")
    summary_rows, histories, months, runs, repeats = [], [], [], {}, []

    from scripts.spot_strategy_round2_worker import initialize_worker, run_family
    tasks = []
    for scenario, costs in config["costs"].items():
        common = dict(out=str(args.output_dir.resolve()), symbols=symbols,
                      start=start.isoformat(), end=end.isoformat(), capital=args.capital,
                      costs=costs, filters=filters, scenario=scenario,
                      repeat_check=not args.skip_repeat_check)
        for strategy in IDS:
            for variant, stop in (("original", None), ("stop5", .05)):
                tasks.append(dict(common, mode="strategy", strategy=strategy, variant=variant, stop=stop))
        tasks.append(dict(common, mode="benchmark"))
    cache_paths = {symbol: data_manifest["symbols"][symbol]["normalized_cache"] for symbol in symbols}
    print(f"Running {len(tasks)} independent families with {args.workers} worker(s); frozen rules unchanged...", flush=True)

    def absorb(batch):
        summary_rows.extend(batch["summary_rows"])
        histories.extend(batch["histories"])
        months.extend(batch["months"])
        runs.update(batch["runs"])
        repeats.extend(batch["repeat_verifications"])
        for row in batch["summary_rows"]:
            print(f"{row['run_id']}: return={row['net_return']:.2%} DD={row['max_drawdown']:.2%} closed={row['closed_trades']}", flush=True)

    if args.workers == 1:
        initialize_worker(cache_paths)
        for task in tasks:
            absorb(run_family(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker, initargs=(cache_paths,)) as pool:
            # map yields in fixed input order even when independent workers finish out of order.
            for batch in pool.map(run_family, tasks):
                absorb(batch)
    summary, history, monthly = pd.DataFrame(summary_rows), pd.concat(histories, ignore_index=True), pd.concat(months, ignore_index=True)
    benchmark_matches = []
    for row in summary[summary.variant.eq("benchmark")].to_dict("records"):
        old = previous_summary.loc[previous_summary.run_id.eq(row["run_id"])].iloc[0]
        for field in ("final_equity", "net_return", "max_drawdown", "fees"):
            if abs(row[field] - old[field]) > max(1e-7, abs(row[field]) * 1e-10):
                raise AssertionError(f"Baseline drift: {row['run_id']}/{field}")
        benchmark_matches.append(row["run_id"])
    for filename, frame in (("summary.csv", summary), ("daily_equity.csv", history), ("monthly_returns.csv", monthly)):
        frame.to_csv(args.output_dir / filename, index=False, float_format="%.12g", encoding="utf-8-sig")
    old_groups = previous_summary.loc[previous_summary.pair.eq("PORTFOLIO") & previous_summary.variant.ne("benchmark")].copy()
    comparison = pd.concat([old_groups.assign(round=1), summary.loc[summary.pair.eq("PORTFOLIO") & summary.variant.ne("benchmark")].assign(round=2)], ignore_index=True)
    comparison.to_csv(args.output_dir / "comparison10.csv", index=False, float_format="%.12g", encoding="utf-8-sig")
    code_files = [Path(__file__), *sorted((ROOT / "scripts").glob("spot_strategy_round2*.*")),
                  ROOT / "scripts/spot_strategy_suite_engine.py", ROOT / "scripts/spot_strategy_suite_data.py", ROOT / "scripts/spot_strategy_suite_reporting.py"]
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "config": config, "frozen_spec": frozen,
                "python": sys.version, "platform": platform.platform(),
                "versions": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "TA-Lib", "requests")},
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "research_code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in code_files if p.is_file()},
                "data": data_manifest, "exchange_filters": saved, "source_evidence_sha256": sha(source_path),
                "reproduction_argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                "run_results": runs, "repeat_verifications": repeats, "baseline_matches_round1": benchmark_matches,
                "previous_evidence": preservation, "execution_workers": args.workers, "live_trading_authorized": False}
    if any(sha(path) != expected for path, expected in preservation.items()):
        raise AssertionError("Round-one source or result was modified")
    manifest["previous_evidence_preserved"] = True
    reporting.write_json(args.output_dir / "manifest.json", manifest)
    reviewed = reporting.make_reviewed_snapshot(summary, history, monthly, manifest=manifest)
    reviewed["title"] = reviewed["report"]["title"] = "第二轮五组现货策略：收益与回撤比较"
    for query in reviewed["queries"].values():
        query["source"] = copy.deepcopy(query["source"])
        query["source"]["evidenceFlow"][2]["detail"] = "本轮五组各10,000 USDT，BTC/ETH各5,000独立现金与库存；原规则、统一5%保护、两档研究成本。"
        query["source"]["caveats"].append(config["selection_warning"])
    previous_source = copy.deepcopy(reviewed["queries"]["strategy_summary"]["source"])
    previous_source["files"] = [str(args.previous_dir / "summary.csv"), str(args.previous_dir / "manifest.json")]
    previous_source["label"] = "第一轮固定报告的组合结果，只读对照"
    reviewed["queries"]["previous_summary"] = {"rows": json.loads(old_groups.to_json(orient="records")), "source": previous_source}
    reporting.write_json(args.output_dir / "reviewed.json", reviewed)
    if not args.skip_report:
        from scripts.spot_strategy_round2_report_app import build_report
        reporting.write_json(args.output_dir / "report_build.json", build_report(args.output_dir))
    print(f"Complete: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
