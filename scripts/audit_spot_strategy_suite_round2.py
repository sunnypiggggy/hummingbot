#!/usr/bin/env python3
"""Reconstruct round-two outputs without strategy, engine or metric helpers."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/backtests/spot_strategy_suite_round2_365d_20261024"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_signals(candles, strategy):
    """Independent formulas from complete raw-minute aggregation; no TA-Lib."""
    close = candles.close

    def atr(period):
        true_range = pd.concat([candles.high - candles.low,
                                (candles.high - close.shift()).abs(),
                                (candles.low - close.shift()).abs()], axis=1).max(axis=1)
        values = np.full(len(candles), np.nan)
        if len(candles) > period:
            values[period] = true_range.iloc[1:period + 1].mean()
            for j in range(period + 1, len(candles)):
                values[j] = (values[j - 1] * (period - 1) + true_range.iloc[j]) / period
        return pd.Series(values, index=candles.index)

    def ema(period):
        values = np.full(len(candles), np.nan)
        if len(candles) >= period:
            values[period - 1] = close.iloc[:period].mean()
            alpha = 2 / (period + 1)
            for j in range(period, len(candles)):
                values[j] = values[j - 1] + alpha * (close.iloc[j] - values[j - 1])
        return pd.Series(values, index=candles.index)

    if strategy == "ibs":
        spread = candles.high - candles.low
        value = ((close - candles.low) / spread).where(spread > 0, .5)
        enter, leave = value <= .2, value >= .8
    elif strategy == "absolute_momentum":
        change = close / close.shift(180) - 1
        decision = (candles.index.dayofweek == 0) & (candles.index.hour == 0) & (candles.index.minute == 0)
        enter, leave = (change > 0) & decision, (change <= 0) & decision
    elif strategy == "mfi":
        typical = (candles.high + candles.low + close) / 3
        raw = typical * candles.volume
        difference = typical.diff()
        positive = raw.where(difference > 0, 0).rolling(14).sum()
        negative = raw.where(difference < 0, 0).rolling(14).sum()
        total = positive + negative
        value = (100 * positive / total).where(total >= 1, 0.)
        value.iloc[:14] = np.nan
        enter = (value.shift() >= 20) & (value < 20)
        leave = (value.shift() <= 80) & (value > 80)
    elif strategy == "squeeze":
        mid = close.rolling(20).mean()
        band = 2 * close.rolling(20).std(ddof=0)
        center, width = ema(20), 1.5 * atr(20)
        compressed = (mid + band < center + width) & (mid - band > center - width)
        momentum = close - close.shift(20)
        enter = compressed.shift(fill_value=False) & ~compressed & (close > mid + band) & (momentum > 0)
        leave = (close < mid) | (momentum <= 0)
    elif strategy == "supertrend":
        volatility = atr(10)
        entries, exits = [], []
        upper = lower = None
        up = False
        for j in range(len(candles)):
            if pd.isna(volatility.iloc[j]):
                entries.append(False)
                exits.append(False)
                continue
            middle = (candles.high.iloc[j] + candles.low.iloc[j]) / 2
            raw_upper, raw_lower = middle + 3 * volatility.iloc[j], middle - 3 * volatility.iloc[j]
            if upper is None:
                upper, lower, up = raw_upper, raw_lower, False
            else:
                if raw_upper < upper or close.iloc[j - 1] > upper:
                    upper = raw_upper
                if raw_lower > lower or close.iloc[j - 1] < lower:
                    lower = raw_lower
                up = close.iloc[j] >= lower if up else close.iloc[j] > upper
            entries.append(bool(up))
            exits.append(not bool(up))
        enter = pd.Series(entries, index=candles.index)
        leave = pd.Series(exits, index=candles.index)
    else:
        raise ValueError(strategy)
    return pd.DataFrame({"enter": enter.fillna(False).astype(bool), "exit": leave.fillna(False).astype(bool)})


def audit(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    summary = pd.read_csv(output / "summary.csv")
    histories = pd.read_csv(output / "daily_equity.csv")
    monthly = pd.read_csv(output / "monthly_returns.csv")
    start, end = pd.Timestamp(config["start"]), pd.Timestamp(config["end"])
    expected_index = pd.date_range(start, end, freq="min", inclusive="left")
    symbols = config["symbols"]
    failures, runs, sleeve_equities = [], {}, {}
    checks = 0
    max_error = {"cash": 0.0, "base_value": 0.0, "equity": 0.0}

    def check(ok, message):
        nonlocal checks
        checks += 1
        if not bool(ok):
            failures.append(message)

    previous_manifests = [Path(path) for path in manifest.get("previous_evidence", {})
                          if Path(path).name == "manifest.json"]
    previous_dir = previous_manifests[0].parent if len(previous_manifests) == 1 else None
    check(previous_dir is not None, "one unambiguous first-round evidence directory")
    for path, expected in manifest.get("previous_evidence", {}).items():
        check(digest(path) == expected, "unchanged first-round evidence: " + path)

    market, full_market = {}, {}
    for symbol in symbols:
        spec = manifest["data"]["symbols"][symbol]
        path = Path(spec["normalized_cache"])
        check(digest(path) == spec["normalized_cache_sha256"], symbol + ": normalized source hash")
        whole = pd.read_pickle(path)
        frame = whole.loc[(whole.index >= start) & (whole.index < end)]
        check(frame.index.equals(expected_index), symbol + ": continuous evaluation minutes")
        check(int((whole.index < start).sum()) == config["warmup_days"] * 1440, symbol + ": excluded warmup")
        market[symbol] = frame
        full_market[symbol] = whole
    signal_cache = {}
    for strategy in config["periods_minutes"]:
        for symbol in symbols:
            frame = pd.read_csv(output / f"signals_{strategy}_{symbol}.csv", index_col="timestamp")
            frame.index = pd.to_datetime(frame.index, utc=True)
            grouped = full_market[symbol].resample(f"{config['periods_minutes'][strategy]}min", origin="epoch",
                                                   label="right", closed="left")
            bars = grouped.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            bars = bars.loc[grouped.size().eq(config["periods_minutes"][strategy])]
            independent = reference_signals(bars, strategy)
            check(frame.index.equals(independent.index), f"{strategy}/{symbol}: closed raw-minute aggregation")
            check(np.array_equal(frame[["enter", "exit"]].to_numpy(), independent[["enter", "exit"]].to_numpy()),
                  f"{strategy}/{symbol}: all frozen signals independently recalculated")
            signal_cache[strategy, symbol] = frame

    for record in summary.to_dict("records"):
        run_id = record["run_id"]
        folder = output / "runs" / run_id
        meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        eq = pd.read_csv(folder / "equity_1m.csv.gz", index_col="timestamp")
        eq.index = pd.to_datetime(eq.index, utc=True)
        fills, trades = pd.read_csv(folder / "fills.csv"), pd.read_csv(folder / "trades.csv")
        if record["variant"] == "benchmark" and previous_dir is not None:
            for name in ("fills.csv", "trades.csv", "equity_1m.csv.gz"):
                check(digest(folder / name) == digest(previous_dir / "runs" / run_id / name),
                      run_id + ": byte-identical first-round benchmark " + name)
        fills["timestamp"] = pd.to_datetime(fills.timestamp, utc=True)
        for col in ("entry_time", "exit_time"):
            trades[col] = pd.to_datetime(trades[col], utc=True)
        initial = float(record["initial_capital"])
        expected_capital = config["capital"] if record["pair"] == "PORTFOLIO" else config["sleeve_capital"]
        check(initial == expected_capital == meta["initial_capital"], run_id + ": capital")
        check(eq.index.equals(expected_index), run_id + ": exact minute window")
        check((eq.cash >= -.001).all() and (eq.base_value >= -.001).all(), run_id + ": nonnegative serialized balances")
        check(np.isfinite(eq.to_numpy()).all(), run_id + ": finite values")
        family_strategy = "buy_hold" if record["strategy"] == "buy_hold_equal" else record["strategy"]
        family = record["scenario"], record["variant"], family_strategy
        if record["pair"] in symbols:
            sleeve_equities.setdefault(family, {})[record["pair"]] = eq[["cash", "base_value", "equity"]].to_numpy()
        elif record["strategy"] != "cash":
            sleeves = sleeve_equities.pop(family, {})
            check(set(sleeves) == set(symbols), run_id + ": two complete independent sleeves")
            if set(sleeves) == set(symbols):
                difference = np.abs(sum(sleeves.values()) - eq[["cash", "base_value", "equity"]].to_numpy())
                check(float(difference.max()) < .001, run_id + ": every-minute portfolio equals sum of sleeves")
        costs = config["costs"][record["scenario"]]
        fee_rate, slip = costs["fee"], costs["slippage"]
        cash_delta = np.zeros(len(eq))
        qty_delta = {symbol: np.zeros(len(eq)) for symbol in symbols}
        owned = {}
        sleeve_cash = {symbol: float(config["sleeve_capital"]) for symbol in symbols}
        calculated_trades, last_stop = [], {}
        fill_positions = expected_index.get_indexer(fills.timestamp)
        check((fill_positions >= 0).all(), run_id + ": fills within evaluation")
        fee_total = notional_total = slippage_total = 0.0
        violations = []
        for offset, fill in zip(fill_positions, fills.itertuples(index=False)):
            symbol, quantity, price = fill.symbol, float(fill.qty), float(fill.price)
            notional, fee = quantity * price, float(fill.fee)
            if abs(fee - notional * fee_rate) > 1e-6:
                violations.append("fee")
            if quantity <= 0:
                violations.append("quantity")
            if fill.side == "BUY":
                if symbol in owned:
                    violations.append("pyramiding")
                available = sleeve_cash[symbol]
                step = float(manifest["exchange_filters"]["filters"][symbol].get("step_size", 0))
                target_quantity = available * .95 / ((1 + fee_rate) * price)
                if step:
                    target_quantity = np.floor(target_quantity / step + 1e-10) * step
                if abs(quantity - target_quantity) > max(1e-9, abs(quantity) * 1e-9):
                    violations.append("95% budget/lot rounding")
                if fill.timestamp <= last_stop.get(symbol, start - pd.Timedelta(minutes=1)):
                    violations.append("same-boundary stop re-entry")
                reference = float(market[symbol].open.iloc[offset])
                if abs(price - reference * (1 + slip)) > 1e-6:
                    violations.append("entry price/slippage")
                cash_delta[offset] -= notional + fee
                sleeve_cash[symbol] -= notional + fee
                qty_delta[symbol][offset] += quantity
                owned[symbol] = quantity, price, fee, fill.timestamp
                slippage_total += quantity * (price - reference)
            elif fill.side == "SELL":
                if symbol not in owned:
                    violations.append("sell unowned coin")
                    continue
                q, entry_price, entry_fee, entry_time = owned.pop(symbol)
                if abs(quantity - q) > 1e-9:
                    violations.append("inventory mismatch")
                reference = float(market[symbol].open.iloc[offset])
                if fill.reason == "stop_loss":
                    barrier = entry_price * .95
                    if market[symbol].low.iloc[offset] > barrier + 1e-6:
                        violations.append("unreached stop")
                    reference = min(reference, barrier)
                    last_stop[symbol] = fill.timestamp
                    if record["variant"] != "stop5":
                        violations.append("native stop in original")
                if abs(price - reference * (1 - slip)) > 1e-6:
                    violations.append("exit price/slippage")
                cash_delta[offset] += notional - fee
                sleeve_cash[symbol] += notional - fee
                qty_delta[symbol][offset] -= quantity
                calculated_trades.append({"pnl": notional - fee - (q * entry_price + entry_fee),
                                          "holding_minutes": (fill.timestamp - entry_time).total_seconds() / 60})
                slippage_total += quantity * (reference - price)
            else:
                violations.append("unknown side")
            if record["variant"] in ("original", "stop5") and fill.reason != "stop_loss":
                signal = signal_cache[record["strategy"], symbol]
                if fill.timestamp not in signal.index:
                    violations.append("no closed decision boundary")
                else:
                    s = signal.loc[fill.timestamp]
                    if fill.side == "BUY" and not (bool(s.enter) and not bool(s.exit)):
                        violations.append("entry violates saved signal")
                    if fill.side == "SELL" and not bool(s.exit):
                        violations.append("exit violates saved signal")
                period = config["periods_minutes"][record["strategy"]]
                if (fill.timestamp.value // 60_000_000_000) % period:
                    violations.append("bar alignment")
                if record["strategy"] == "absolute_momentum" and fill.timestamp.dayofweek != 0:
                    violations.append("weekly alignment")
            fee_total += fee
            notional_total += notional
        check(not violations, run_id + ": fill economics/timing " + str(sorted(set(violations))))
        cash = initial + cash_delta.cumsum()
        base_value = np.zeros(len(eq))
        for symbol in symbols:
            inventory = qty_delta[symbol].cumsum()
            check(inventory.min() >= -1e-9, run_id + ": owned " + symbol)
            base_value += inventory * market[symbol].close.to_numpy()
        equity = cash + base_value
        for key, rebuilt in (("cash", cash), ("base_value", base_value), ("equity", equity)):
            error = float(np.max(np.abs(rebuilt - eq[key].to_numpy())))
            max_error[key] = max(max_error[key], error)
            check(error < .001, run_id + ": every-minute " + key)
        pnl = np.array([t["pnl"] for t in calculated_trades], dtype=float)
        open_pnl = sum(q * float(market[s].close.iloc[-1]) - (q * p + f) for s, (q, p, f, _) in owned.items())
        final = float(equity[-1])
        drawdown = equity / np.maximum.accumulate(np.r_[initial, equity])[1:] - 1
        series = pd.Series(equity, index=expected_index)
        daily = series.resample("D").last()
        daily_r = daily.pct_change()
        daily_r.iloc[0] = daily.iloc[0] / initial - 1
        sd = daily_r.std(ddof=1)
        sharpe = daily_r.mean() / sd * np.sqrt(365) if sd > 1e-15 else np.nan
        days = (end - start).total_seconds() / 86400
        cagr = (final / initial) ** (365 / days) - 1
        gains, losses = pnl[pnl > 0], pnl[pnl < 0]
        observed = {
            "final_equity": final, "net_pnl": final - initial, "net_return": final / initial - 1,
            "max_drawdown": float(drawdown.min()), "sharpe_daily": sharpe,
            "fees": fee_total, "slippage_cost": slippage_total, "turnover": notional_total / initial,
            "realized_pnl": pnl.sum(), "unrealized_pnl": open_pnl, "closed_trades": len(calculated_trades),
            "exposure": float(np.mean(base_value / equity)), "final_cash": float(cash[-1]),
            "net_liquidation_equity": float(cash[-1] + base_value[-1] * (1 - slip) * (1 - fee_rate)),
            "net_liquidation_return": float(cash[-1] + base_value[-1] * (1 - slip) * (1 - fee_rate)) / initial - 1,
            "cagr": cagr, "calmar": cagr / abs(drawdown.min()) if drawdown.min() < -1e-15 else np.nan,
            "win_rate": float((pnl > 0).mean()) if len(pnl) else np.nan,
            "profit_factor": float(gains.sum() / -losses.sum()) if len(losses) else np.nan,
            "payoff_ratio": float(gains.mean() / -losses.mean()) if len(gains) and len(losses) else np.nan,
            "average_holding_hours": float(np.mean([t["holding_minutes"] for t in calculated_trades]) / 60) if calculated_trades else np.nan,
        }
        for key, actual in observed.items():
            expected = record[key]
            tolerance = .001 if key in ("final_equity", "net_pnl", "fees", "slippage_cost", "realized_pnl", "unrealized_pnl", "final_cash", "net_liquidation_equity") else 1e-8
            check((pd.isna(actual) and pd.isna(expected)) or abs(actual - expected) < tolerance, run_id + ": summary " + key)
        check(len(trades) == len(calculated_trades), run_id + ": serialized closed count")
        check(abs(float(trades.pnl.sum()) - float(pnl.sum())) < .001, run_id + ": serialized closed PNL")
        check(abs(final - initial - pnl.sum() - open_pnl) < .001, run_id + ": realized/open PNL bridge")
        check(set(meta["final_positions"]) == set(owned), run_id + ": final owned coin set")
        check(all(abs(meta["final_positions"][s] - v[0]) < 1e-9 for s, v in owned.items()), run_id + ": final owned quantity")
        history = histories.loc[histories.run_id == run_id]
        check(len(history) == len(daily) + 1, run_id + ": initial plot point and all days")
        check(np.max(np.abs(history.equity.to_numpy() - np.r_[initial, daily.to_numpy()])) < .001, run_id + ": daily plot/table")
        month = monthly.loc[monthly.run_id == run_id]
        month_ends = series.resample("ME").last().to_numpy()
        month_returns = month_ends / np.r_[initial, month_ends[:-1]] - 1
        check(len(month) == len(month_ends) and np.max(np.abs(month["return"].to_numpy() - month_returns)) < 1e-8, run_id + ": monthly returns")
        runs[run_id] = {"rows": len(eq), "fills": len(fills), "closed_trades": len(trades),
                        "net_return": observed["net_return"], "max_drawdown": observed["max_drawdown"],
                        "fees": fee_total, "reconstructed_final_cash": float(cash[-1])}
        print(f"AUDIT {run_id}: {len(fills)} fills reconstructed", flush=True)
    return {"created_at": datetime.now(timezone.utc).isoformat(), "status": "passed" if not failures else "failed",
            "independent_of_strategy_engine_and_metric_helpers": True,
            "study_start": str(start), "study_end_exclusive": str(end),
            "expected_minutes": len(expected_index), "runs_checked": len(runs),
            "checks_total": checks, "checks_failed": len(failures), "failures": failures,
            "max_absolute_reconstruction_error_usdt": max_error,
            "config_sha256": digest(output / "config.json"), "runs": runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(args.output_dir.resolve())
    path = args.output_dir / "independent_audit.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "runs_checked", "checks_total", "checks_failed", "max_absolute_reconstruction_error_usdt")}, ensure_ascii=False))
    return int(result["checks_failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
