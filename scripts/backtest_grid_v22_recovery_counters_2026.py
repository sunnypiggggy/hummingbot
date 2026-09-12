#!/usr/bin/env python3
"""Reproducible, offline-only 2026 v22 recovery ablation; no network clients.

Run from the repository root. Input downloads are deliberately a separate,
read-only acquisition step. Never loads credentials, alters releases or trains.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import xgboost_long_risk_gate_v22 as v22
from grid_live_common import clip_quantized_buy_levels, clip_quantized_sell_levels
from grid_v22_recovery_ablation import CounterState, rollover, step
from xgboost_v22_io import sha256_file

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/backtests/grid_v22_recovery_counter_ablation_2026"
START = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp())
STRUCTURE = ["roc_48h_4h", "sqzmom_pct_4h", "di_spread", "ema20_slope_atr_12h", "below_ema20_ratio_72h"]
PARAMS = {
    "BTC-FDUSD": {"range": .12698379475402316, "levels": 18, "tp": .004},
    "ETH-FDUSD": {"range": .5246511596640915, "levels": 18, "tp": .014179761072002472},
}
ARMS = {"A": "A 现行共用计数", "B": "B 独立连续计数"}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")


def utc(ts: int) -> str:
    return pd.Timestamp(int(ts), unit="s", tz="UTC").isoformat()


def save_csv(path: Path, frame: pd.DataFrame) -> None:
    # gzip mtime is otherwise non-deterministic.
    opts = {"method": "gzip", "mtime": 0} if path.suffix == ".gz" else None
    frame.to_csv(path, index=False, float_format="%.17g", compression=opts, lineterminator="\n")


def validate_frame(frame: pd.DataFrame, pair: str) -> None:
    if frame.empty or frame.timestamp.duplicated().any() or not frame.timestamp.diff().dropna().eq(300).all():
        raise ValueError(f"{pair}: missing/duplicate/out-of-order 5m candles")
    x = frame[["open", "high", "low", "close", "volume"]]
    if not np.isfinite(x).all().all() or not (x.iloc[:, :4] > 0).all().all() or not (x.volume >= 0).all():
        raise ValueError(f"{pair}: invalid OHLCV")
    if not ((frame.high >= frame[["open", "close"]].max(axis=1)) &
            (frame.low <= frame[["open", "close"]].min(axis=1)) & (frame.high >= frame.low)).all():
        raise ValueError(f"{pair}: broken OHLC bounds")


def load_inputs(inputs: Path, as_of: str):
    lock = json.loads((inputs / "shadow_lock.json").read_text(encoding="utf-8"))
    if sha256_file(inputs / "model.joblib") != lock["model_sha256"]:
        raise ValueError("model package SHA256 mismatch")
    bundle = joblib.load(inputs / "model.joblib")  # Trusted, hash-verified own OCI artifact only.
    v22.validate_weekly_bundle(bundle)
    for key in ("feature_schema_sha256", "strategy_schema_sha256"):
        if bundle[key] != lock[key]:
            raise ValueError(f"lock {key} mismatch")
    if sha256_file(inputs / "original_training_panel.csv.gz") != lock["training_panel_sha256"]:
        raise ValueError("original frozen training panel hash mismatch")
    remote_source = (inputs / "oci_xgboost_long_risk_gate_v22.py").read_bytes().replace(b"\r\n", b"\n")
    if remote_source != (ROOT / "scripts/xgboost_long_risk_gate_v22.py").read_bytes().replace(b"\r\n", b"\n"):
        raise ValueError("local A reference differs from copied OCI gate source")
    audits, windows = [], []
    for pair in v22.PAIRS:
        sequence = []
        for week in bundle["pairs"][pair]["weeks"]:
            actual = hashlib.sha256(bytes(week["model"].get_booster().save_raw(raw_format="ubj"))).hexdigest()
            if actual != week["model_sha256"]:
                raise ValueError(f"{pair} fold {week['fold']} booster hash mismatch")
            if week["last_label_ready_ts"] > week["train_cutoff"] or week["development_last_ts"] >= week["calibration_first_ts"]:
                raise ValueError(f"{pair} training/calibration temporal leakage")
            sequence.append((week["test_start"], week["test_end"], week["fold"]))
            audits.append({"pair": pair, **{k: val for k, val in week.items() if k != "model"}})
        windows.append(sequence)
    if windows[0] != windows[1]:
        raise ValueError("BTC/ETH signed coverage differs")
    candles = {}
    for pair in v22.PAIRS:
        frame = pd.read_csv(inputs / f"binance_{pair}_5m.csv")
        frame.timestamp = pd.to_numeric(frame.timestamp).astype("int64")
        validate_frame(frame, pair)
        candles[pair] = frame
    end = min(int(pd.Timestamp(as_of).timestamp()), int(lock["effective_end"]),
              *[int(x.timestamp.max()) + 300 for x in candles.values()]) // 86400 * 86400
    if end <= START:
        raise ValueError("no complete 2026 UTC day covered")
    filters = json.loads((inputs / "exchange_filters.json").read_text(encoding="utf-8"))["pairs"]
    return lock, bundle, candles, filters, audits, end


def predict_shared(bundle, candles, lock, end):
    print("Building causal features and shared weekly probabilities", flush=True)
    panel = v22.build_inference_panel(candles)
    outputs = []
    for pair in v22.PAIRS:
        frame = panel[(panel.pair == pair) & (panel.signal_ts >= lock["effective_start"]) &
                      (panel.signal_ts < end)].copy().reset_index(drop=True)
        required_ts = np.arange(lock["effective_start"], end, 3600, dtype=np.int64)
        if not np.array_equal(frame.signal_ts.to_numpy(np.int64), required_ts):
            raise ValueError(f"{pair}: common inference panel hourly coverage incomplete")
        frame["probability"] = np.nan
        frame["threshold"] = np.nan
        frame["fold"] = -1
        frame["week_start"] = 0
        for week in bundle["pairs"][pair]["weeks"]:
            mask = frame.signal_ts.between(week["test_start"], week["test_end"], inclusive="left")
            if mask.any():
                # Never set model params: even n_jobs changes would alter booster hash.
                frame.loc[mask, "probability"] = week["model"].predict_proba(frame.loc[mask, list(v22.FEATURES[pair])])[:, 1]
                frame.loc[mask, "threshold"] = week["entry_threshold"]
                frame.loc[mask, "fold"] = week["fold"]
                frame.loc[mask, "week_start"] = week["test_start"]
        if frame[["probability", "threshold"]].isna().any().any() or not frame.probability.between(0, 1).all():
            raise ValueError("missing/invalid weekly inference; no fallback")
        outputs.append(frame)
    return pd.concat(outputs, ignore_index=True)


def replay_signals(shared):
    rows, final, parity = [], {}, 0
    for pair in v22.PAIRS:
        states = {a: CounterState() for a in ARMS}
        reference = CounterState()
        for r in shared[shared.pair == pair].itertuples(index=False):
            args = dict(pair=pair, probability=float(r.probability), threshold=float(r.threshold),
                        ts=int(r.signal_ts), structure_ts=int(r.last_complete_4h_ts),
                        structure=tuple(float(getattr(r, k)) for k in STRUCTURE),
                        fold=int(r.fold), boundary=int(r.week_start))
            for arm in ARMS:
                rows.append(step(states[arm], arm=arm, **args))
            rollover(reference, int(r.fold), int(r.week_start))
            reference.gate, _ = v22.advance_gate(
                pair=pair, probability=args["probability"], entry_threshold=args["threshold"],
                signal_ts=args["ts"], last_complete_4h_ts=args["structure_ts"],
                structure=args["structure"], state=reference.gate)
            if v22.state_to_dict(reference.gate) != v22.state_to_dict(states["A"].gate):
                raise AssertionError("A != unmodified production gate")
            parity += 1
        final[pair] = {a: s.dump() for a, s in states.items()}
    return pd.DataFrame(rows), final, parity


@dataclass
class Book:
    pair: str
    quote: float
    base: float
    cost: float
    initial_base: float
    center: float
    active: bool = True
    peak: float = 200.
    baseline: float = 200.
    until: int = 0
    healthy: int = 0
    pending: tuple | None = None
    reenter_pending: bool = False
    excess_since: int | None = None
    last_refresh: int = -10**12
    last_move: int = -10**12
    generation: int = 0
    orders: list = field(default_factory=list)
    moves: int = 0
    submitted: int = 0

    def equity(self, price):
        return self.quote + self.base * price


def quantize(x, step, up=False):
    return float((Decimal(str(x)) / Decimal(str(step))).to_integral_value(
        rounding=ROUND_UP if up else ROUND_DOWN) * Decimal(str(step)))


def build_orders(book, price, ts, filters):
    """Use actual runtime quantized level builders, TP/cost floor, excess cap."""
    p = PARAMS[book.pair]
    f = filters[book.pair]
    half = p["range"] / 2
    if ((price > book.center * (1 + half) * 1.015 or price < book.center * (1 - half) * .985)
            and ts - book.last_move >= 1800):
        book.center, book.last_move = price, ts
        book.moves += 1
    levels = [book.center * (1 - half + p["range"] * i / (p["levels"] - 1)) for i in range(p["levels"])]
    lower, upper = [Decimal(str(x)) for x in levels if x < price], [Decimal(str(x)) for x in levels if x > price]
    extra, deficit = max(0., book.base - book.initial_base), max(0., book.initial_base - book.base)
    budget = max(0., min(book.quote, 100., deficit * price + 10. - extra * price))
    amt = lambda x: Decimal(str(quantize(float(x), f["step_size"])))
    px = lambda x: Decimal(str(quantize(float(x), f["tick_size"])))
    floor_profit = p["tp"] if book.excess_since is None or ts - book.excess_since < 86400 else 0.
    cost_floor = book.cost / book.base * (1 + floor_profit) if book.base > 0 else 0.
    buys = clip_quantized_buy_levels(lower, Decimal(str(budget)), Decimal("10"), px, amt,
                                    amount_step=Decimal(str(f["step_size"])), minimum_amount=Decimal(str(f["step_size"])))
    sells = clip_quantized_sell_levels(upper, Decimal(str(max(0., min(book.base, book.initial_base)))), Decimal("10"),
                                      lambda x: Decimal(str(quantize(max(float(x), price * (1 + p["tp"]), cost_floor), f["tick_size"], True))), amt)
    book.generation += 1
    book.orders = []
    for side, orders in (("BUY", buys), ("SELL", sells)):
        for index, (limit, qty) in enumerate(orders):
            assert float(limit * qty) >= 10. - 1e-8
            assert (float(limit) < price if side == "BUY" else float(limit) > price)
            book.orders.append((side, float(limit), float(qty), ts, f"{book.pair}:{book.generation}:{side}:{index}"))
    assert len({(o[0], o[1]) for o in book.orders}) == len(book.orders)
    assert sum(o[2] for o in book.orders if o[0] == "SELL") <= book.base + 1e-10
    assert sum(o[1] * o[2] for o in book.orders if o[0] == "BUY") <= book.quote + 1e-8
    book.last_refresh = ts
    book.submitted += len(book.orders)


def execute(book, side, qty, price, ts, reason, trades, *, taker=False, signal_ts=None, signal_price=None, order_id=""):
    if qty <= 0:
        return
    fill = price * (1 + (.0002 if side == "BUY" else -.0002)) if taker else price
    fee = fill * qty * .001 if taker else 0.
    if side == "BUY":
        assert fill * qty + fee <= book.quote + 1e-8, "overspend"
        book.quote -= fill * qty + fee
        book.base += qty
        book.cost += fill * qty + fee
    else:
        assert qty <= book.base + 1e-12, "oversell"
        book.cost *= max(0., (book.base - qty) / book.base)
        book.base -= qty
        book.quote += fill * qty - fee
    trades.append(dict(pair=book.pair, ts=ts, side=side, quantity=qty, price=fill, notional=fill*qty,
                       fee=fee, slippage_cost=abs(fill-price)*qty, kind="TAKER" if taker else "MAKER",
                       reason=reason, order_id=order_id, signal_ts=signal_ts, signal_price=signal_price,
                       signal_after_loss=(signal_price-fill)*qty+fee if side == "SELL" and signal_price is not None else 0.))
    assert book.base >= -1e-12 and book.quote >= -1e-8 and book.cost >= -1e-8


def can_reenter(book, price, filters):
    # Rebuild approximately 100 FDUSD baseline including retained dust.
    target = max(0., 100. - book.base * price)
    if target < filters[book.pair]["minimum_notional"]:
        return True
    return book.quote >= target * 1.0012


def reenter(book, price, ts, filters, trades):
    if not can_reenter(book, price, filters):
        return False
    target = max(0., 100. - book.base * price)
    qty = quantize(target / (price * 1.0002 * 1.001), filters[book.pair]["step_size"])
    if qty * price >= filters[book.pair]["minimum_notional"]:
        execute(book, "BUY", qty, price, ts, "REENTRY", trades, taker=True, signal_ts=ts-300)
    book.initial_base = book.base
    book.baseline = book.peak = book.equity(price)
    book.center = price
    book.active, book.healthy, book.reenter_pending = True, 0, False
    book.excess_since, book.last_refresh = None, -10**12
    return True


def replay_grid(candles, signals, filters, arm, start, end):
    arrays = {p: x[(x.timestamp >= start) & (x.timestamp < end)][["timestamp", "open", "high", "low", "close"]].to_numpy()
              for p, x in candles.items()}
    if any(len(x) != (end-start)//300 for x in arrays.values()):
        raise ValueError("execution candle window incomplete")
    gates, gate_times = {}, {}
    for p in v22.PAIRS:
        h = signals[(signals.pair == p) & (signals.arm == arm)].sort_values("signal_ts")
        available = h.signal_ts.to_numpy(np.int64) + 300
        idx = np.searchsorted(available, arrays[p][:, 0].astype(np.int64), side="right") - 1
        if (idx < 0).any() or ((arrays[p][:, 0] - available[idx]) >= 3600).any():
            raise ValueError("missing usable hourly gate; no silent fallback")
        gates[p] = h.risk_off.to_numpy(bool)[idx]
        gate_times[p] = h.signal_ts.to_numpy(np.int64)[idx]
    books = {}
    for p in v22.PAIRS:
        price = arrays[p][0, 1]
        base = quantize(100/price, filters[p]["step_size"])
        books[p] = Book(p, 200-base*price, base, base*price, base, price)
    trades, events, equity = [], [], []
    portfolio_peak = portfolio_baseline = 420.
    portfolio_until, portfolio_halted, portfolio_ready = 0, False, False
    for i in range(len(arrays[v22.PAIRS[0]])):
        ts = int(arrays[v22.PAIRS[0]][i, 0])
        opens = {p: float(arrays[p][i, 1]) for p in v22.PAIRS}
        closes = {p: float(arrays[p][i, 4]) for p in v22.PAIRS}
        # Gate events generated on the completed hour apply no earlier than +5m.
        for p, b in books.items():
            if gates[p][i] and b.active and b.pending is None:
                signal_time = int(gate_times[p][i])
                price_index = (signal_time-start)//300-1
                signal_price = float(arrays[p][price_index,4]) if price_index >= 0 else opens[p]
                b.pending = ("V22_RISK_OFF", signal_time, signal_price, 0)
                b.reenter_pending = False
            if b.pending is not None:
                reason, st, sp, cooldown = b.pending
                b.orders.clear()  # Pair-only cancellation, before any new candle fills.
                qty = quantize(b.base, filters[p]["step_size"])
                if qty*opens[p] >= filters[p]["minimum_notional"]:
                    execute(b, "SELL", qty, opens[p], ts, reason, trades, taker=True, signal_ts=st, signal_price=sp)
                b.active, b.pending, b.healthy = False, None, 0
                b.until = max(b.until, ts + cooldown)
                b.reenter_pending = False
                b.excess_since = None
                events.append(dict(pair=p, ts=ts, kind="EXIT_COMPLETE", reason=reason,
                                   signal_ts=st, dust_value=b.base*opens[p], cooldown_until=b.until))
        if portfolio_ready:
            ready = all(not gates[p][i] and not b.pending and ts >= b.until and can_reenter(b, opens[p], filters)
                        for p, b in books.items())
            if ready:
                for p, b in books.items():
                    assert reenter(b, opens[p], ts, filters, trades)
                    events.append(dict(pair=p, ts=ts, kind="REENTRY", reason="PORTFOLIO_RECOVERED"))
                portfolio_baseline = portfolio_peak = 20 + sum(b.equity(opens[p]) for p, b in books.items())
                portfolio_halted, portfolio_ready = False, False
        if not portfolio_halted:
            for p, b in books.items():
                if b.reenter_pending:
                    if not gates[p][i] and ts >= b.until and reenter(b, opens[p], ts, filters, trades):
                        events.append(dict(pair=p, ts=ts, kind="REENTRY", reason="PAIR_RECOVERED"))
                    else:
                        b.reenter_pending = False
        # Only orders already placed on a preceding completed candle may fill.
        for p, b in books.items():
            high, low = arrays[p][i, 2:4]
            refreshed = False
            if b.active:
                remaining = []
                # Intrabar ordering is unavailable: deterministic OHLC path,
                # same in both arms, no intrabar re-use/replenishment of capital.
                ordered = sorted(b.orders, key=lambda o: (o[0] != ("BUY" if closes[p] >= opens[p] else "SELL"),
                                                          -o[1] if o[0] == "BUY" else o[1]))
                for side, limit, qty, created, oid in ordered:
                    touched = low <= limit if side == "BUY" else high >= limit
                    if touched and created <= ts:
                        execute(b, side, qty, limit, ts+300, "GRID", trades, order_id=oid)
                        refreshed = True
                    else:
                        remaining.append((side, limit, qty, created, oid))
                b.orders = remaining
            extra = max(0., b.base-b.initial_base)
            if extra <= 1e-12:
                b.excess_since = None
            elif b.excess_since is None:
                b.excess_since = ts+300
            # Extra inventory exit is scheduled at this close, filled next open.
            if b.active and b.excess_since is not None and ts+300-b.excess_since >= 48*3600 and extra*closes[p] >= 10:
                # Persist intent as a separate partial-inventory task.
                b.orders.clear()
                events.append(dict(pair=p, ts=ts+300, kind="EXCESS_EXIT_REQUEST", reason="POSITION_48H"))
                b.excess_since = -1  # sentinel consumed at next open below
            value = b.equity(closes[p])
            b.peak = max(b.peak, value)
            if b.active:
                reason = "STRATEGY_LOSS" if value <= b.baseline-6 else "STRATEGY_DRAWDOWN" if value <= b.peak*.97 else ""
                if reason:
                    b.pending = (reason, ts+300, closes[p], 6*3600)
                    b.orders.clear()
            if not b.active:
                healthy = not gates[p][i] and ts+300 >= max(b.until, portfolio_until) and not b.orders
                b.healthy = b.healthy+1 if healthy else 0
                if b.healthy >= 3 and not portfolio_halted:
                    b.reenter_pending = True
            # Build at CLOSE; these prices cannot fill on the candle just used.
            if b.active and not b.pending and b.excess_since != -1:
                half = PARAMS[p]["range"]/2
                moved = ((closes[p] > b.center*(1+half)*1.015 or closes[p] < b.center*(1-half)*.985)
                         and ts+300-b.last_move >= 1800)
                if refreshed or moved or ts+300-b.last_refresh >= 7200:
                    b.orders.clear()
                    build_orders(b, closes[p], ts+300, filters)
        total = 20 + sum(b.equity(closes[p]) for p, b in books.items())
        portfolio_peak = max(portfolio_peak, total)
        if not portfolio_halted and (total <= portfolio_baseline-24 or total <= portfolio_peak*.94):
            reason = "PORTFOLIO_LOSS" if total <= portfolio_baseline-24 else "PORTFOLIO_DRAWDOWN"
            portfolio_halted, portfolio_until = True, ts+300+12*3600
            for p, b in books.items():
                b.pending = (reason, ts+300, closes[p], 12*3600)
                b.reenter_pending = False
                b.orders.clear()
        if portfolio_halted and ts+300 >= portfolio_until and all(b.healthy >= 3 for b in books.values()):
            portfolio_ready = True
        for p, b in books.items():
            equity.append(dict(arm=arm, pair=p, ts=ts+300, price=closes[p], equity=b.equity(closes[p]),
                               quote=b.quote, base=b.base, cost=b.cost, cycle_peak=b.peak, cycle_baseline=b.baseline,
                               risk_off=bool(gates[p][i]), active=b.active, orders=len(b.orders),
                               phase="EXITING" if b.pending else "ACTIVE" if b.active else
                               "COOLDOWN" if ts+300 < max(b.until, portfolio_until) else "REENTRY"))
        # Partial-inventory actions execute at next OPEN, never using that next
        # candle's high/low/close. Full-risk exits supersede these intents.
        if i+1 < len(arrays[v22.PAIRS[0]]):
            for p, b in books.items():
                if b.excess_since == -1 and b.pending is None and not gates[p][i+1]:
                    qty = quantize(max(0., b.base-b.initial_base), filters[p]["step_size"])
                    next_price = float(arrays[p][i+1, 1])
                    if qty*next_price >= filters[p]["minimum_notional"]:
                        execute(b, "SELL", qty, next_price, ts+300, "POSITION_48H", trades, taker=True,
                                signal_ts=ts+300, signal_price=closes[p])
                    b.excess_since = None
                    b.last_refresh = -10**12
    return pd.DataFrame(equity), pd.DataFrame(trades), pd.DataFrame(events), books


def max_dd(values, initial):
    values = np.r_[initial, np.asarray(values, dtype=float)]
    return float(np.max(1-values/np.maximum.accumulate(values))*100)


def summarize(eq, trades, events, signals, books, start, end):
    summary = {}
    for p in (*v22.PAIRS, "PORTFOLIO"):
        own = eq if p == "PORTFOLIO" else eq[eq.pair == p]
        path = own.groupby("ts").equity.sum()+20 if p == "PORTFOLIO" else own.equity
        initial = 420 if p == "PORTFOLIO" else 200
        t = trades if p == "PORTFOLIO" else trades[trades.pair == p]
        e = events if p == "PORTFOLIO" else events[events.pair == p]
        recovery = t[t.reason == "REENTRY"]
        summary[p] = dict(final_equity=float(path.iloc[-1]), pnl=float(path.iloc[-1]-initial),
                          return_pct=float((path.iloc[-1]/initial-1)*100), max_drawdown_pct=max_dd(path, initial),
                          trades=len(t), maker_fills=int(t.kind.eq("MAKER").sum()), fees=float(t.fee.sum()),
                          exits=int(e.kind.eq("EXIT_COMPLETE").sum()), reentries=int(e.kind.eq("REENTRY").sum()),
                          excess_exits=int(t.reason.eq("POSITION_48H").sum()),
                          reentry_cost=float((recovery.fee+recovery.slippage_cost).sum()),
                          exit_cost=float((t[t.side.eq("SELL") & t.kind.eq("TAKER")].fee +
                                           t[t.side.eq("SELL") & t.kind.eq("TAKER")].slippage_cost).sum()),
                          grid_moves=sum(b.moves for k,b in books.items() if p == "PORTFOLIO" or k == p))
        if p != "PORTFOLIO":
            h = signals[(signals.pair == p) & (signals.signal_ts >= start) & (signals.signal_ts < end)]
            off = h.risk_off.to_numpy(bool)
            longest = run = 0
            for flag in off:
                run = run+1 if flag else 0
                longest = max(longest, run)
            recovered = h[h.transition == "recover"]
            entries = h.loc[h.transition == "enter", "signal_ts"].to_numpy()
            summary[p].update(risk_off_hours=int(off.sum()), longest_risk_off_hours=longest,
                              ordinary_recoveries=int(recovered.recovery_path.eq("ordinary").sum()),
                              strong_recoveries=int(recovered.recovery_path.str.startswith("strong").sum()),
                              retrigger_within_48h=sum(bool(((entries > ts) & (entries <= ts+48*3600)).any())
                                                       for ts in recovered.signal_ts),
                              stopped_hours=float((~own.active).sum()/12))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=OUTPUT/"inputs")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--as-of", default="2026-09-06T00:00:00Z")
    parser.add_argument("--reuse-shared", type=Path, help="Verified own shared_predictions CSV, for exact rerun")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock, bundle, candles, filters, audit, end = load_inputs(args.inputs, args.as_of)
    print(f"Window {utc(START)} -> {utc(end)}; {int((end-START)/86400)} days", flush=True)
    diagnostic_end = min(int(lock["effective_end"]), *[int(x.timestamp.max())+300 for x in candles.values()])//3600*3600
    shared = predict_shared(bundle, candles, lock, diagnostic_end)
    # Predictions are always recomputed: --reuse-shared only verifies equality.
    if args.reuse_shared:
        previous = pd.read_csv(args.reuse_shared, float_precision="round_trip")
        pd.testing.assert_frame_equal(shared.reset_index(drop=True), previous[shared.columns], check_dtype=False)
    save_csv(args.output/"shared_predictions.csv.gz", shared)
    signals, final, parity = replay_signals(shared)
    save_csv(args.output/"hourly_signals.csv.gz", signals)
    write_json(args.output/"final_gate_states.json", final)
    write_json(args.output/"weekly_model_audit.json", audit)
    results, equities, all_trades = {}, [], []
    for arm in ARMS:
        print(f"Replaying {arm}", flush=True)
        eq, trades, events, books = replay_grid(candles, signals, filters, arm, START, end)
        equities.append(eq)
        all_trades.append(trades.assign(arm=arm))
        results[arm] = summarize(eq, trades, events, signals[signals.arm == arm], books, START, end)
        save_csv(args.output/f"{arm}_equity_5m.csv.gz", eq)
        save_csv(args.output/f"{arm}_trades.csv", trades)
        save_csv(args.output/f"{arm}_execution_events.csv", events)
    config = dict(params=PARAMS, initial_pair_capital=200, reserve=20, minimum_order=10,
                  refresh_seconds=7200, move_threshold=.015, move_cooldown_seconds=1800,
                  movement_semantics="boundary_plus_threshold", extra_inventory_quote_cap=10,
                  cost_profit_protection_hours=24, excess_inventory_timeout_hours=48,
                  maker_fee=0, taker_fee=.001, slippage=.0002, bnb_fee=False,
                  strategy_loss=6, strategy_drawdown=.03, portfolio_loss=24, portfolio_drawdown=.06,
                  strategy_cooldown_hours=6, portfolio_cooldown_hours=12, reentry_health_bars=3,
                  initial_inventory="100 FDUSD per pair, valued at first open, no initial fee; no warmup PnL",
                  v22_execution_delay_seconds=300, fomc=False, parameter_optimization=False, momentum_tp=False,
                  weekly_risk_on_evidence_reset=True, start=utc(START), end_exclusive=utc(end))
    write_json(args.output/"parameter_snapshot.json", config)
    inputs = {x.name: sha256_file(x) for x in sorted(args.inputs.iterdir()) if x.is_file()}
    codes = {str(x.relative_to(ROOT)): sha256_file(x) for x in [Path(__file__), ROOT/"scripts/grid_v22_recovery_ablation.py",
             ROOT/"scripts/render_grid_v22_recovery_ablation.py", ROOT/"scripts/validate_grid_v22_recovery_ablation.py",
             ROOT/"scripts/xgboost_long_risk_gate_v22.py", ROOT/"scripts/xgboost_long_risk_gate_v22_features.py", ROOT/"scripts/grid_live_common.py"]}
    report = dict(experiment="grid_v22_recovery_counter_ablation_2026", offline_only=True, deployment_allowed=False,
                  label="历史模型离线反事实实验", start=utc(START), end_exclusive=utc(end),
                  days=(end-START)//86400, five_minute_rows_per_pair=(end-START)//300,
                  diagnostic_end_exclusive=utc(diagnostic_end),
                  warmup_start=utc(lock["effective_start"]), a_reference_parity_points=parity,
                  a_reference_mismatches=0, identical_shared_model_inputs=True, results=results,
                  input_sha256=inputs, code_sha256=codes,
                  training_evidence="Original frozen training panel SHA256 verified; per-week booster hashes and training/calibration timestamps verified. Appended-week raw training-candle snapshots not independently reconstructed; no retraining.",
                  runtime_versions={k:importlib.metadata.version(k) for k in ("numpy","pandas","xgboost","joblib","scikit-learn","plotly","matplotlib")},
                  source_acquisition="Read-only OCI public model/candles/gate source copied to inputs; source matches local gate after CRLF normalization",
                  approximations=["5-minute OHLC touch fill, deterministic path, no queue/latency modelling",
                                  "Three healthy 5-minute cycles approximate Guard recovery cycles",
                                  "All risk exits at next available open; no intrabar tick-level protection",
                                  "Frozen exchange filter snapshot applied to all dates, not reconstructed historical filter changes",
                                  "Current per-pair parameters held constant; not a replay of historical live settings"])
    write_json(args.output/"summary.json", report)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    from render_grid_v22_recovery_ablation import render
    render(args.output, report, signals, pd.concat(equities, ignore_index=True), end)
    write_json(args.output/"artifact_manifest.json", {x.name:sha256_file(x) for x in sorted(args.output.iterdir())
                                                    if x.is_file() and x.name != "artifact_manifest.json"})


if __name__ == "__main__":
    main()
