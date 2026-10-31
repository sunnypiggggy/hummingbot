#!/usr/bin/env python3
"""Build a read-only V3 DCA performance snapshot and seven-day trade chart."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sqlite3
import logging
import multiprocessing
import signal
import threading
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import requests
from PIL import Image, ImageDraw, ImageFont

try:
    from runtime_endpoints import OFFICIAL_BINANCE_API, binance_api_base
except ModuleNotFoundError:
    from live_guard.runtime_endpoints import OFFICIAL_BINANCE_API, binance_api_base

try:
    from telegram_notifications import (
        RuntimeErrorChannel, TelegramChannelClient, TelegramOutbox, append_event, build_event,
        format_event, render_mobile_profit_card,
    )
except ModuleNotFoundError:
    from live_guard.telegram_notifications import (
        RuntimeErrorChannel, TelegramChannelClient, TelegramOutbox, append_event, build_event,
        format_event, render_mobile_profit_card,
    )
try:
    from telegram_parameter_report import build_parameter_attachments
except ModuleNotFoundError:
    from live_guard.telegram_parameter_report import build_parameter_attachments
try:
    from trading_status import evaluate_status, gate_row
except ModuleNotFoundError:
    from live_guard.trading_status import evaluate_status, gate_row
try:
    from management_parameters import ManagementParameterPublisher
except ModuleNotFoundError:
    from live_guard.management_parameters import ManagementParameterPublisher

try:
    from dca_live_common import LIVE_PAIRS, STRATEGY_BUDGET_QUOTE, side_budget, adjustment_timestamp_seconds, base_fee_correction
except ModuleNotFoundError:  # Repository import; container copies it to /app.
    from scripts.dca_live_common import LIVE_PAIRS, STRATEGY_BUDGET_QUOTE, side_budget, adjustment_timestamp_seconds, base_fee_correction


try:
    import report_cards
    from report_cards import (
        MAX_DATA_AGE_SECONDS, MAX_GUARD_STATUS_AGE_SECONDS, SHANGHAI,
        dca_report_infrastructure_status, dust_usdt_text, inventory_exposure_quote,
    )
except ModuleNotFoundError:
    from live_guard import report_cards
    from live_guard.report_cards import (
        MAX_DATA_AGE_SECONDS, MAX_GUARD_STATUS_AGE_SECONDS, SHANGHAI,
        dca_report_infrastructure_status, dust_usdt_text, inventory_exposure_quote,
    )

BINANCE_API = OFFICIAL_BINANCE_API
SCALE = Decimal("1000000")
WINDOW_DAYS = 7
MAX_PUBLIC_FILLS = 2_000
CHART_SIZE = (1280, 1000)


def _low_priority_report_worker() -> None:
    for name in ("XGBOOST_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "2"
    if hasattr(os, "nice"):
        os.nice(10)


def _run_parameter_job(event: dict[str, Any], release_root: str,
                       output_root: str) -> list[dict[str, Any]]:
    return build_parameter_attachments(
        event, release_root=Path(release_root), output_root=Path(output_root),
    )


def timestamp_seconds(value: Any) -> float:
    timestamp = float(value)
    while timestamp > 10_000_000_000:
        timestamp /= 1000
    return timestamp


def decimal_value(value: Any) -> Decimal:
    return Decimal(int(value or 0)) / SCALE


def normalize_side(value: Any) -> str:
    text = str(value).upper()
    if text in {"1", "BUY", "TRADETYPE.BUY"}:
        return "BUY"
    if text in {"2", "SELL", "TRADETYPE.SELL"}:
        return "SELL"
    return "UNKNOWN"


def row_metrics(rows: Iterable[tuple[Any, ...]]) -> dict[str, Decimal]:
    cashflow = Decimal("0")
    net_base = Decimal("0")
    fees = Decimal("0")
    trades = Decimal("0")
    buys = Decimal("0")
    sells = Decimal("0")
    for row in rows:
        side, raw_price, raw_amount, raw_fee, *_ = row
        price = decimal_value(raw_price)
        amount = decimal_value(raw_amount)
        fee = decimal_value(raw_fee)
        normalized = normalize_side(side)
        if normalized == "BUY":
            cashflow -= price * amount
            net_base += amount
            buys += 1
        elif normalized == "SELL":
            cashflow += price * amount
            net_base -= amount
            sells += 1
        else:
            continue
        base_fee, quote_rebate = base_fee_correction(row)
        net_base -= base_fee
        cashflow += quote_rebate
        fees += fee
        trades += 1
    return {
        "cashflow_quote": cashflow,
        "net_base": net_base,
        "fees_quote": fees,
        "trades": trades,
        "buys": buys,
        "sells": sells,
    }


def apply_emergency_adjustments(
    metrics: dict[str, Decimal], adjustments: Iterable[dict[str, Any]]
) -> None:
    for adjustment in adjustments:
        metrics["cashflow_quote"] += Decimal(
            str(adjustment.get("quote_cashflow", "0"))
        )
        metrics["net_base"] += Decimal(str(adjustment.get("base_delta", "0")))
        metrics["fees_quote"] += Decimal(str(adjustment.get("fee_quote", "0")))
        metrics["trades"] += 1
        side = str(adjustment.get("side", "")).upper()
        if side == "BUY":
            metrics["buys"] += 1
        elif side == "SELL":
            metrics["sells"] += 1


def calculate_pair_report(
    *,
    bot_name: str,
    pair: str,
    rows: list[tuple[Any, ...]],
    candles: list[dict[str, Any]],
    database_age_seconds: float,
    now: datetime,
    max_public_fills: int = MAX_PUBLIC_FILLS,
    emergency_adjustments: list[dict[str, Any]] | None = None,
    managed_base: Decimal = Decimal("0"),
    managed_base_cost_quote: Decimal = Decimal("0"),
) -> dict[str, Any]:
    now_ts = now.timestamp()
    window_start = now - timedelta(days=WINDOW_DAYS)
    window_start_ts = window_start.timestamp()
    normalized_candles = sorted(
        (
            {
                "timestamp": timestamp_seconds(item["timestamp"]),
                "close": str(Decimal(str(item["close"]))),
            }
            for item in candles
        ),
        key=lambda item: item["timestamp"],
    )
    latest_candle = normalized_candles[-1] if normalized_candles else None
    market_age = (
        max(0.0, now_ts - latest_candle["timestamp"])
        if latest_candle is not None
        else None
    )
    market_healthy = market_age is not None and market_age <= MAX_DATA_AGE_SECONDS
    mark = Decimal(latest_candle["close"]) if latest_candle else None
    ordered_rows = sorted(rows, key=lambda item: timestamp_seconds(item[4]))
    window_rows = [
        row
        for row in ordered_rows
        if window_start_ts <= timestamp_seconds(row[4]) <= now_ts
    ]
    emergency_adjustments = list(emergency_adjustments or [])
    window_adjustments = [
        item
        for item in emergency_adjustments
        if window_start_ts
        <= adjustment_timestamp_seconds(item["recorded_at"])
        <= now_ts
    ]
    all_metrics = row_metrics(ordered_rows)
    apply_emergency_adjustments(all_metrics, emergency_adjustments)

    all_time_mtm = (
        all_metrics["cashflow_quote"]
        - all_metrics["fees_quote"]
        + (managed_base + all_metrics["net_base"]) * mark
        - managed_base_cost_quote
        if market_healthy and mark is not None
        else None
    )
    owned_base = managed_base + all_metrics["net_base"]
    reconciliation_status = (
        "RECONCILED" if owned_base >= 0 else "EQUITY_UNRECONCILED"
    )
    if managed_base_cost_quote > 0 and owned_base < 0:
        # A replay that consumes more than the explicitly managed inventory is
        # not evidence of a short position on Spot. Preserve the audit fields
        # but withhold equity until ownership is reconciled.
        all_time_mtm = None
    cash_balance = (
        managed_base_cost_quote
        + all_metrics["cashflow_quote"]
        - all_metrics["fees_quote"]
    )
    inventory_mtm = owned_base * mark if mark is not None else None
    def period_mtm(hours: int) -> tuple[Decimal | None, dict[str, Decimal]]:
        start_ts = now_ts - hours * 3600
        earlier_rows = [row for row in ordered_rows if timestamp_seconds(row[4]) < start_ts]
        period_rows = [row for row in ordered_rows if start_ts <= timestamp_seconds(row[4]) <= now_ts]
        earlier_adjustments = [item for item in emergency_adjustments
                               if adjustment_timestamp_seconds(item["recorded_at"]) < start_ts]
        period_adjustments = [item for item in emergency_adjustments
                              if start_ts <= adjustment_timestamp_seconds(item["recorded_at"]) <= now_ts]
        earlier = row_metrics(earlier_rows)
        period = row_metrics(period_rows)
        apply_emergency_adjustments(earlier, earlier_adjustments)
        apply_emergency_adjustments(period, period_adjustments)
        start_candle = next((item for item in normalized_candles
                             if item["timestamp"] >= start_ts), None)
        if not market_healthy or mark is None or start_candle is None:
            return None, period
        start_mark = Decimal(start_candle["close"])
        starting_position = managed_base + earlier["net_base"]
        ending_position = starting_position + period["net_base"]
        value = (period["cashflow_quote"] - period["fees_quote"]
                 + ending_position * mark - starting_position * start_mark)
        return value, period

    four_hour_mtm, four_hour_metrics = period_mtm(4)
    twenty_four_hour_mtm, twenty_four_hour_metrics = period_mtm(24)
    seven_day_mtm, seven_day_metrics = period_mtm(WINDOW_DAYS * 24)

    fills = []
    for side, raw_price, raw_amount, raw_fee, raw_timestamp, *_ in window_rows:
        normalized = normalize_side(side)
        if normalized == "UNKNOWN":
            continue
        price = decimal_value(raw_price)
        amount = decimal_value(raw_amount)
        fills.append(
            {
                "timestamp": timestamp_seconds(raw_timestamp),
                "side": normalized,
                "price": str(price),
                "amount": str(amount),
                "quote": str(price * amount),
                "fee_quote": str(decimal_value(raw_fee)),
            }
        )
    for adjustment in window_adjustments:
        amount = Decimal(str(adjustment.get("executed_qty", "0")))
        quote = abs(Decimal(str(adjustment.get("quote_cashflow", "0"))))
        fills.append(
            {
                "timestamp": adjustment_timestamp_seconds(adjustment["recorded_at"]),
                "side": str(adjustment["side"]).upper(),
                "price": str(quote / amount if amount > 0 else Decimal("0")),
                "amount": str(amount),
                "quote": str(quote),
                "fee_quote": str(adjustment.get("fee_quote", "0")),
                "emergency": True,
            }
        )
    fills.sort(key=lambda item: item["timestamp"])
    truncated = len(fills) > max_public_fills
    public_fills = fills[-max_public_fills:]
    all_timestamps = [timestamp_seconds(row[4]) for row in ordered_rows] + [
        adjustment_timestamp_seconds(item["recorded_at"])
        for item in emergency_adjustments
    ]
    first_fill = min(all_timestamps) if all_timestamps else None
    last_fill = max(all_timestamps) if all_timestamps else None

    return {
        "bot_name": bot_name,
        "trading_pair": pair,
        "database_available": True,
        "market_data_healthy": market_healthy,
        "mark_price": str(mark) if mark is not None else None,
        "market_data_age_seconds": market_age,
        "database_age_seconds": max(0.0, database_age_seconds),
        "position": {
            "scope": "strategy_owned_inventory",
            "net_base": str(all_metrics["net_base"]),
            "managed_base": str(managed_base),
            "owned_base": str(managed_base + all_metrics["net_base"]),
            "tradable_base": str(max(owned_base, Decimal("0"))),
            "reconciliation_status": reconciliation_status,
            "market_value_quote": (
                str((managed_base + all_metrics["net_base"]) * mark)
                if mark is not None else None
            ),
        },
        "profit": {
            "valuation": "cashflow + owned_base * mark - recorded_fees - managed_base_cost",
            "all_time_mtm_quote": (
                str(all_time_mtm) if all_time_mtm is not None else None
            ),
            "cash_balance_quote": str(cash_balance),
            "owned_inventory_mtm_quote": (
                str(inventory_mtm) if inventory_mtm is not None else None
            ),
            "equity_source": "cash_plus_strategy_owned_inventory_mtm_v1",
            "seven_day_mtm_quote": (
                str(seven_day_mtm) if seven_day_mtm is not None else None
            ),
            "four_hour_mtm_quote": (
                str(four_hour_mtm) if four_hour_mtm is not None else None
            ),
            "twenty_four_hour_mtm_quote": (
                str(twenty_four_hour_mtm) if twenty_four_hour_mtm is not None else None
            ),
            "all_time_cashflow_quote": str(all_metrics["cashflow_quote"]),
            "seven_day_cashflow_quote": str(seven_day_metrics["cashflow_quote"]),
            "four_hour_cashflow_quote": str(four_hour_metrics["cashflow_quote"]),
            "twenty_four_hour_cashflow_quote": str(twenty_four_hour_metrics["cashflow_quote"]),
            "all_time_fees_quote": str(all_metrics["fees_quote"]),
            "seven_day_fees_quote": str(seven_day_metrics["fees_quote"]),
            "four_hour_fees_quote": str(four_hour_metrics["fees_quote"]),
            "twenty_four_hour_fees_quote": str(twenty_four_hour_metrics["fees_quote"]),
        },
        "trades": {
            "all_time": int(all_metrics["trades"]),
            "all_time_buys": int(all_metrics["buys"]),
            "all_time_sells": int(all_metrics["sells"]),
            "seven_day": int(seven_day_metrics["trades"]),
            "seven_day_buys": int(seven_day_metrics["buys"]),
            "seven_day_sells": int(seven_day_metrics["sells"]),
            "first_fill_at": first_fill,
            "last_fill_at": last_fill,
            "fills_7d": public_fills,
            "truncated": truncated,
        },
        "_candles": normalized_candles,
    }


def unavailable_pair_report(
    *,
    bot_name: str,
    pair: str,
    candles: list[dict[str, Any]],
    now: datetime,
) -> dict[str, Any]:
    """Represent an uncreated/stopped V3 bot without inventing zero PnL."""
    report = calculate_pair_report(
        bot_name=bot_name,
        pair=pair,
        rows=[],
        candles=candles,
        database_age_seconds=0,
        now=now,
    )
    report["database_available"] = False
    report["database_age_seconds"] = None
    report["position"]["net_base"] = None
    report["position"]["market_value_quote"] = None
    for field in (
        "all_time_mtm_quote",
        "seven_day_mtm_quote",
        "four_hour_mtm_quote",
        "twenty_four_hour_mtm_quote",
        "all_time_cashflow_quote",
        "seven_day_cashflow_quote",
        "four_hour_cashflow_quote",
        "twenty_four_hour_cashflow_quote",
        "all_time_fees_quote",
        "seven_day_fees_quote",
        "four_hour_fees_quote",
        "twenty_four_hour_fees_quote",
    ):
        report["profit"][field] = None
    for field in (
        "all_time",
        "all_time_buys",
        "all_time_sells",
        "seven_day",
        "seven_day_buys",
        "seven_day_sells",
    ):
        report["trades"][field] = None
    return report


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _draw_triangle(
    draw: ImageDraw.ImageDraw, x: int, y: int, *, up: bool, color: str
) -> None:
    points = (
        [(x, y - 9), (x - 8, y + 7), (x + 8, y + 7)]
        if up
        else [(x, y + 9), (x - 8, y - 7), (x + 8, y - 7)]
    )
    draw.polygon(points, fill=color, outline="#ffffff")


def render_chart(report: dict[str, Any], output: Path) -> None:
    image = Image.new("RGB", CHART_SIZE, "#f5f7fa")
    draw = ImageDraw.Draw(image)
    title_font = _font(28)
    label_font = _font(18)
    small_font = _font(15)
    draw.text((50, 24), "DCA V3 - Last 7 Days Trades", fill="#172033", font=title_font)
    draw.text(
        (50, 62),
        f"Generated UTC: {report['generated_at']}  BUY: green  SELL: red",
        fill="#64748b",
        font=small_font,
    )

    for index, bot in enumerate(report["bots"]):
        top = 105 + index * 430
        left, right = 90, 1230
        bottom = top + 355
        candles = bot.get("_candles", [])
        fills = bot["trades"]["fills_7d"]
        prices = [float(item["close"]) for item in candles] + [
            float(item["price"]) for item in fills
        ]
        draw.rounded_rectangle(
            (45, top - 12, 1245, bottom + 55),
            radius=10,
            fill="#ffffff",
            outline="#dce3ea",
        )
        draw.text(
            (65, top),
            (
                f"{bot['trading_pair']}  position={bot['position']['net_base']}  "
                f"all-time PnL={bot['profit']['all_time_mtm_quote']} USDT  "
                f"7d PnL={bot['profit']['seven_day_mtm_quote']} USDT"
            ),
            fill="#172033",
            font=label_font,
        )
        plot_top = top + 45
        if not prices or not candles:
            draw.text(
                (left, plot_top + 120),
                "Market data unavailable",
                fill="#b91c1c",
                font=label_font,
            )
            continue
        low, high = min(prices), max(prices)
        padding = max((high - low) * 0.08, high * 0.001, 1e-9)
        low -= padding
        high += padding
        start = report["window"]["start_at_epoch"]
        end = report["window"]["end_at_epoch"]

        def point(timestamp: float, price: float) -> tuple[int, int]:
            x = left + int((timestamp - start) / max(1.0, end - start) * (right - left))
            y = plot_top + int((high - price) / max(1e-12, high - low) * (bottom - plot_top))
            return max(left, min(right, x)), max(plot_top, min(bottom, y))

        for grid in range(5):
            y = plot_top + int(grid / 4 * (bottom - plot_top))
            draw.line((left, y, right, y), fill="#e5e7eb", width=1)
            value = high - grid / 4 * (high - low)
            draw.text((50, y - 8), f"{value:,.2f}", fill="#64748b", font=small_font)
        line_points = [
            point(float(item["timestamp"]), float(item["close"])) for item in candles
        ]
        if len(line_points) > 1:
            draw.line(line_points, fill="#2563eb", width=3, joint="curve")
        for fill in fills:
            x, y = point(float(fill["timestamp"]), float(fill["price"]))
            _draw_triangle(
                draw,
                x,
                y,
                up=fill["side"] == "BUY",
                color="#16a34a" if fill["side"] == "BUY" else "#dc2626",
            )
        draw.text(
            (left, bottom + 12),
            datetime.fromtimestamp(start, timezone.utc).strftime("%Y-%m-%d"),
            fill="#64748b",
            font=small_font,
        )
        end_text = datetime.fromtimestamp(end, timezone.utc).strftime("%Y-%m-%d")
        draw.text((right - 90, bottom + 12), end_text, fill="#64748b", font=small_font)

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, "PNG", optimize=True)


def read_economic_adjustments(path: Path, bot_name: str, pair: str, database: Path, legacy: list) -> list:
    """Read shared fill facts without opening the ledger for writes or migration."""
    export = path.parent / "economic_fills_readonly.json"
    if not path.is_file() and not export.is_file():
        return list(legacy)
    if export.is_file():
        payload = json.loads(export.read_text(encoding="utf-8"))
        expected = payload.pop("sha256", "")
        actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
        if payload.get("schema") != "inventory-economic-fills-v1" or expected != actual:
            raise ValueError("economic fill export integrity check failed")
        if not 0 <= time.time() - float(payload.get("generated_at", 0)) <= 180:
            raise ValueError("economic fill export is stale")
        facts = [row for row in payload["fills"] if row["scope"] == f"dca:{bot_name}" and row["pair"] == pair]
    else:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10) as connection:
            connection.row_factory = sqlite3.Row
            if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='economic_fills'").fetchone():
                return list(legacy)
            facts = connection.execute(
                "SELECT f.*,j.created_at AS executed_at FROM economic_fills f JOIN liquidation_jobs j ON j.job_id=f.job_id "
                "WHERE f.scope=? AND f.pair=?", (f"dca:{bot_name}", pair),
            ).fetchall()
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=10) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(TradeFill)")}
        normal_ids = {str(row[0]) for row in connection.execute("SELECT exchange_trade_id FROM TradeFill WHERE symbol=?", (pair,))} if "exchange_trade_id" in columns else set()
    legacy_orders = {str(row.get("order_id", "")) for row in legacy}
    result = list(legacy)
    for fill in facts:
        if fill["trade_id"] in normal_ids or fill["order_id"] in legacy_orders:
            continue
        quantity, price, fee = (Decimal(fill[key]) for key in ("quantity", "price", "commission"))
        base, quote = pair.split("-", 1)
        if fee and fill["commission_asset"] not in {base, quote}:
            raise ValueError("third-asset fee valuation unavailable")
        result.append({
            "recorded_at": datetime.fromtimestamp(fill["executed_at"], timezone.utc).isoformat(),
            "side": fill["side"], "order_id": fill["order_id"], "trade_id": fill["trade_id"],
            "base_delta": str((quantity if fill["side"] == "BUY" else -quantity) - (fee if fill["commission_asset"] == base else 0)),
            "quote_cashflow": str(quantity * price * (-1 if fill["side"] == "BUY" else 1)),
            "fee_quote": str(fee if fill["commission_asset"] == quote else 0),
            "accounting_source": "shared_economic_trade_fact",
        })
    return result


class DcaLiveReportCollector:
    def __init__(
        self,
        bots_path: Path,
        output_dir: Path,
        *,
        candle_reader: Callable[[str, datetime, datetime], list[dict[str, Any]]] | None = None,
    ):
        self.bots_path = bots_path
        self.output_dir = output_dir
        self.guard_state_path = output_dir / "guard_state.json"
        self.inventory_status_path = Path(os.getenv(
            "ACCOUNT_INVENTORY_LEDGER_PATH", "/workspace/account-inventory"
        )) / "account_inventory_status.json"
        self.candle_reader = candle_reader or self._binance_candles

    def _economic_adjustments(self, bot_name: str, pair: str, database: Path, legacy: list) -> list:
        return read_economic_adjustments(self.inventory_status_path.parent / "account_inventory.sqlite", bot_name, pair, database, legacy)

    def _database(self, bot_name: str) -> Path | None:
        exact = self.bots_path / "instances" / bot_name / "data" / f"{bot_name}.sqlite"
        if exact.exists():
            return exact
        candidates = sorted(
            (self.bots_path / "instances").glob(f"{bot_name}*/data/*.sqlite"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    @staticmethod
    def _rows(database: Path, pair: str) -> list[tuple[Any, ...]]:
        connection = sqlite3.connect(
            f"file:{database}?mode=ro", uri=True, timeout=30
        )
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            return connection.execute(
                "SELECT trade_type, price, amount, trade_fee_in_quote, timestamp, trade_fee, symbol "
                "FROM TradeFill WHERE symbol = ? ORDER BY timestamp, rowid",
                (pair,),
            ).fetchall()
        finally:
            connection.close()

    @staticmethod
    def _binance_candles(
        pair: str, start: datetime, end: datetime
    ) -> list[dict[str, Any]]:
        response = requests.get(
            f"{binance_api_base()}/api/v3/klines",
            params={
                "symbol": pair.replace("-", ""),
                "interval": "15m",
                "startTime": int(start.timestamp() * 1000),
                "endTime": int(end.timestamp() * 1000),
                "limit": 1000,
            },
            timeout=20,
        )
        response.raise_for_status()
        return [
            {"timestamp": item[0], "close": item[4]} for item in response.json()
        ]

    def collect(self, *, now: datetime | None = None) -> dict[str, Any]:
        now = now or datetime.now(timezone.utc)
        window_start = now - timedelta(days=WINDOW_DAYS)
        bots: list[dict[str, Any]] = []
        warnings: list[str] = []
        try:
            guard_state = (
                json.loads(self.guard_state_path.read_text(encoding="utf-8"))
                if self.guard_state_path.exists()
                else {}
            )
        except (OSError, ValueError, TypeError) as exc:
            guard_state = {}
            warnings.append(f"guard emergency adjustments unavailable: {exc}")
        for pair, spec in LIVE_PAIRS.items():
            database = self._database(spec.bot_name)
            if database is None:
                warnings.append(f"{spec.bot_name}: database not found")
                try:
                    candles = self.candle_reader(pair, window_start, now)
                except Exception as exc:
                    candles = []
                    warnings.append(
                        f"{spec.bot_name}: market data {type(exc).__name__}: {exc}"
                    )
                bots.append(
                    unavailable_pair_report(
                        bot_name=spec.bot_name,
                        pair=pair,
                        candles=candles,
                        now=now,
                    )
                )
                continue
            try:
                bot_state = guard_state.get("bots", {}).get(spec.bot_name, {})
                managed_base = Decimal(str(
                    bot_state.get("managed_base_target", "0")
                ))
                bot = calculate_pair_report(
                    bot_name=spec.bot_name,
                    pair=pair,
                    rows=self._rows(database, pair),
                    candles=self.candle_reader(pair, window_start, now),
                    database_age_seconds=max(
                        0.0, time.time() - database.stat().st_mtime
                    ),
                    now=now,
                    emergency_adjustments=self._economic_adjustments(
                        spec.bot_name, pair, database, bot_state.get("emergency_adjustments", [])
                    ),
                    managed_base=managed_base,
                    managed_base_cost_quote=side_budget(),
                )
                bots.append(bot)
            except Exception as exc:
                warnings.append(f"{spec.bot_name}: {type(exc).__name__}: {exc}")
                bots.append(
                    unavailable_pair_report(
                        bot_name=spec.bot_name,
                        pair=pair,
                        candles=[],
                        now=now,
                    )
                )

        all_time_values = [
            Decimal(bot["profit"]["all_time_mtm_quote"])
            for bot in bots
            if bot["profit"]["all_time_mtm_quote"] is not None
        ]
        seven_day_values = [
            Decimal(bot["profit"]["seven_day_mtm_quote"])
            for bot in bots
            if bot["profit"]["seven_day_mtm_quote"] is not None
        ]
        report = {
            "schema_version": 3,
            "policy_version": "dca-macro-v3",
            "generated_at": now.isoformat(),
            "window": {
                "days": WINDOW_DAYS,
                "start_at": window_start.isoformat(),
                "end_at": now.isoformat(),
                "start_at_epoch": window_start.timestamp(),
                "end_at_epoch": now.timestamp(),
            },
            "position_scope": "strategy_owned_inventory",
            "portfolio": {
                "all_time_mtm_quote": (
                    str(sum(all_time_values, Decimal("0")))
                    if len(all_time_values) == len(LIVE_PAIRS)
                    else None
                ),
                "seven_day_mtm_quote": (
                    str(sum(seven_day_values, Decimal("0")))
                    if len(seven_day_values) == len(LIVE_PAIRS)
                    else None
                ),
            },
            "bots": bots,
            "warnings": warnings,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        chart_path = self.output_dir / "dca_trade_chart_7d.png"
        chart_temp = chart_path.with_suffix(".png.tmp")
        render_chart(report, chart_temp)
        chart_bytes = chart_temp.read_bytes()
        report["chart_sha256"] = hashlib.sha256(chart_bytes).hexdigest()
        report["report_id"] = hashlib.sha256(
            json.dumps(
                {
                    "generated_at": report["generated_at"],
                    "chart_sha256": report["chart_sha256"],
                    "bots": [
                        {
                            "name": bot["bot_name"],
                            "trades": bot["trades"]["all_time"],
                            "last": bot["trades"]["last_fill_at"],
                        }
                        for bot in bots
                    ],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:24]
        public_report = {
            **report,
            "bots": [
                {key: value for key, value in bot.items() if key != "_candles"}
                for bot in bots
            ],
        }
        report_path = self.output_dir / "dca_trade_report_v3.json"
        report_temp = report_path.with_suffix(".json.tmp")
        report_temp.write_text(
            json.dumps(public_report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(chart_temp, chart_path)
        os.replace(report_temp, report_path)
        return public_report


def healthcheck(output_dir: Path) -> bool:
    report_path = output_dir / "dca_trade_report_v3.json"
    chart_path = output_dir / "dca_trade_chart_7d.png"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        generated = datetime.fromisoformat(report["generated_at"])
        if generated.tzinfo is None:
            return False
        age = (datetime.now(timezone.utc) - generated).total_seconds()
        base_report_healthy = (
            report.get("schema_version") == 3
            and len(report.get("bots", [])) == len(LIVE_PAIRS)
            and age <= MAX_DATA_AGE_SECONDS
            and hashlib.sha256(chart_path.read_bytes()).hexdigest()
            == report.get("chart_sha256")
        )
        if not base_report_healthy:
            return False
        runtime_error_path = output_dir / "report_runtime_error_state.json"
        if runtime_error_path.exists():
            runtime_error_state = json.loads(runtime_error_path.read_text(encoding="utf-8"))
            components = runtime_error_state.get("components", {})
            for component in ("report_cycle", "profit_report_generation"):
                if bool(components.get(component, {}).get("active")):
                    return False
        return True
    except Exception:
        return False


class ParameterReportWorker:
    """Persistent one-process queue for expensive 360-day evidence rendering."""

    def __init__(self, *, root: Path, release_root: Path,
                 outbox: TelegramOutbox) -> None:
        self.root = root
        self.release_root = release_root
        self.outbox = outbox
        self.jobs = root / "jobs"
        self.results = root / "parameters"
        self.receipts = root / "evidence_receipts"
        self.jobs.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)
        self.receipts.mkdir(parents=True, exist_ok=True)
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=1, initializer=_low_priority_report_worker,
            # MQTT owns background sockets/locks; fork must not inherit them.
            **({"mp_context": multiprocessing.get_context("spawn")}
               if _cardputer_mqtt_requested() else {}),
        )
        self.future: concurrent.futures.Future | None = None
        self.active_path: Path | None = None
        self.active_event: dict[str, Any] | None = None

    @staticmethod
    def _requires_report(event: Mapping[str, Any]) -> bool:
        return str(event.get("details", {}).get("report_request", "")) in {
            "v22_png_windows", "v22_360d", "grid_360d",
        }

    def schedule(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        if not self._requires_report(event):
            return []
        event_id = str(event["event_id"])
        path = self.jobs / f"{event_id}.json"
        terminal = [self.jobs / f"{event_id}.{suffix}" for suffix in ("done", "failed")]
        if not path.exists() and not any(item.exists() for item in terminal):
            temporary = path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            os.replace(temporary, path)
        return []

    def _finish(self) -> None:
        assert self.future is not None and self.active_path is not None
        assert self.active_event is not None
        event = self.active_event
        event_id = str(event["event_id"])
        try:
            attachments = self.future.result()
            for number, attachment in enumerate(attachments, 1):
                path = Path(str(attachment["path"]))
                self.outbox.enqueue(
                    event_id=event_id,
                    kind=str(attachment.get("kind", "document")),
                    text=str(attachment.get("caption") or f"{event.get('pair')} 附件 {number}"),
                    file_path=path,
                    file_sha256=str(attachment.get("sha256", "")),
                )
            if any(attachment.get("evidence_complete") is False for attachment in attachments):
                failure = build_event(
                    source="dca-live-report", strategy=str(event.get("strategy", "")),
                    bot=str(event.get("bot", "")), pair=str(event.get("pair", "")),
                    mechanism="parameter_update", transition="REPORT_EVIDENCE_MISSING",
                    reason="360天或重点窗口PNG证据缺失；未伪造图片，参数流程沿用原有门槛",
                    severity="critical", action="review_missing_png_evidence",
                    release_sha256=str(event.get("release_sha256", "")),
                    model_sha256=str(event.get("model_sha256", "")),
                    parameter_sha256=str(event.get("parameter_sha256", "")),
                    correlation_id=f"{event_id}:evidence_missing",
                    details={"evidence_complete": False, "source_event_id": event_id},
                )
                self.outbox.enqueue(
                    event_id=str(failure["event_id"]), kind="message", text=format_event(failure),
                )
            marker = self.active_path.with_suffix(".done")
            marker.write_text(json.dumps({
                "event_id": event_id, "completed_at": datetime.now(timezone.utc).isoformat(),
                "release_sha256": str(event.get("release_sha256", "")),
                "model_sha256": str(event.get("model_sha256", "")),
                "parameter_sha256": str(event.get("parameter_sha256", "")),
                "report_request": str(event.get("details", {}).get("report_request", "")),
                "attachments": attachments,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            failure = build_event(
                source="dca-live-report", strategy=str(event.get("strategy", "")),
                bot=str(event.get("bot", "")), pair=str(event.get("pair", "")),
                mechanism="parameter_update", transition="REPORT_EVIDENCE_MISSING",
                reason=("360天或重点窗口PNG证据生成失败；参数更新仍仅受原有交易门槛控制："
                        f"{type(exc).__name__}: {exc}"),
                severity="critical", action="notify_missing_evidence_without_fabrication",
                release_sha256=str(event.get("release_sha256", "")),
                model_sha256=str(event.get("model_sha256", "")),
                parameter_sha256=str(event.get("parameter_sha256", "")),
                correlation_id=f"{event_id}:evidence_missing",
                details={"evidence_complete": False, "source_event_id": event_id},
            )
            self.outbox.enqueue(
                event_id=str(failure["event_id"]), kind="message", text=format_event(failure),
            )
            marker = self.active_path.with_suffix(".failed")
            marker.write_text(json.dumps({
                "event_id": event_id, "failed_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        self.active_path.unlink(missing_ok=True)
        self.future = None
        self.active_path = None
        self.active_event = None

    def finalize_delivery_receipts(self) -> int:
        """Persist a hash-bound receipt only after Telegram accepted every item."""
        written = 0
        for marker in sorted(self.jobs.glob("*.done")):
            local_receipt = marker.with_suffix(".receipt")
            if local_receipt.exists():
                continue
            job = json.loads(marker.read_text(encoding="utf-8"))
            event_id = str(job.get("event_id", ""))
            delivery = self.outbox.event_delivery(event_id)
            if not delivery["all_sent"]:
                continue
            attachments = list(job.get("attachments", []))
            request = str(job.get("report_request", ""))
            expected_photos = 12 if request in {"v22_png_windows", "v22_360d"} else 6
            photos = [row for row in attachments if row.get("kind") == "photo"]
            if len(photos) != expected_photos or any(
                row.get("evidence_complete") is not True for row in attachments
            ):
                continue
            identity = str(job.get("release_sha256") or job.get("parameter_sha256") or "")
            if len(identity) != 64:
                continue
            payload = {
                "schema": "telegram-evidence-delivery-receipt-v1",
                "identity_sha256": identity,
                "source_event_id": event_id,
                "release_sha256": str(job.get("release_sha256", "")),
                "model_sha256": str(job.get("model_sha256", "")),
                "parameter_sha256": str(job.get("parameter_sha256", "")),
                "report_request": request,
                "expected_photo_count": expected_photos,
                "photo_sha256": [str(row["sha256"]) for row in photos],
                "delivered_at": datetime.now(timezone.utc).isoformat(),
                "telegram_message_ids": [
                    str(row.get("telegram_message_id") or "") for row in delivery["items"]
                ],
            }
            payload["delivery_receipt_sha256"] = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            target = self.receipts / f"{identity}.json"
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, target)
            local_receipt.write_text(str(target), encoding="utf-8")
            written += 1
        return written

    def poll(self) -> dict[str, Any]:
        if self.future is not None and self.future.done():
            self._finish()
        if self.future is None:
            paths = sorted(self.jobs.glob("*.json"), key=lambda item: item.stat().st_mtime)
            if paths:
                self.active_path = paths[0]
                self.active_event = json.loads(self.active_path.read_text(encoding="utf-8"))
                self.future = self.executor.submit(
                    _run_parameter_job, self.active_event,
                    str(self.release_root), str(self.results),
                )
        return {
            "active": self.future is not None,
            "pending": len(list(self.jobs.glob("*.json"))),
            "event_id": str((self.active_event or {}).get("event_id", "")),
        }


class UnifiedTelegramReporting:
    """One outbound channel publisher inside the existing report service."""

    def __init__(self, *, bots_path: Path, dca_state: Path) -> None:
        self.bots_path = bots_path
        self.dca_state = dca_state
        self.grid_state = Path(os.getenv("GRID_LIVE_STATE_PATH", "/workspace/grid"))
        self.inventory_status_path = Path(os.getenv(
            "ACCOUNT_INVENTORY_LEDGER_PATH", "/workspace/account-inventory"
        )) / "account_inventory_status.json"
        self.release_root = Path(os.getenv(
            "ETHBTC_RELEASE_FAMILY_PATH", "/workspace/releases"
        ))
        self.output = dca_state / "telegram"
        self.output.mkdir(parents=True, exist_ok=True)
        self.events = dca_state / "telegram_events.jsonl"
        channel_id = os.getenv("TELEGRAM_NOTIFY_CHANNEL_ID", "")
        self.outbox = TelegramOutbox(
            self.output / "telegram_outbox.sqlite", channel_id=channel_id,
        )
        self.parameter_worker = ParameterReportWorker(
            root=self.output, release_root=self.release_root, outbox=self.outbox,
        )
        self.management_parameters = ManagementParameterPublisher(
            bots_path=self.bots_path,
            dca_state=self.dca_state,
            grid_state=self.grid_state,
            release_root=self.release_root,
            approval_root=Path(os.getenv(
                "MODEL_APPROVAL_PUBLIC_ROOT", "/workspace/model-approvals",
            )),
            output=self.output,
        )
        self.enabled = os.getenv("TELEGRAM_NOTIFY_ENABLED", "false").lower() == "true"
        self.profit_enabled = os.getenv(
            "TELEGRAM_PROFIT_REPORT_ENABLED", "true"
        ).lower() == "true"
        self.client = None
        if self.enabled:
            self.client = TelegramChannelClient(
                Path(os.getenv(
                    "TELEGRAM_NOTIFY_BOT_TOKEN_FILE",
                    "/run/secrets/telegram_notify_bot_token",
                )),
                channel_id,
            )

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}

    def _publish_current_runtime_errors(self, now: datetime) -> dict[str, Any]:
        errors: list[dict[str, Any]] = []
        sources = (
            ("grid", self.grid_state / "runtime_error_state.json",
             "grid-live-fdusd-400", "BTC-FDUSD,ETH-FDUSD"),
            ("dca", self.dca_state / "runtime_error_state.json",
             "dca-live-btcusdt-200,dca-live-ethusdt-200", "BTC-USDT,ETH-USDT"),
            ("report", self.dca_state / "report_runtime_error_state.json",
             "dca-live-report", "BTC-FDUSD,ETH-FDUSD,BTC-USDT,ETH-USDT"),
        )
        for source, path, bot, pair in sources:
            state = self._load(path)
            components = state.get("components", {}) if isinstance(state, Mapping) else {}
            if not isinstance(components, Mapping):
                continue
            for component, raw in components.items():
                if not isinstance(raw, Mapping) or not raw.get("active"):
                    continue
                details = raw.get("details") if isinstance(raw.get("details"), Mapping) else {}
                errors.append({
                    "active": True,
                    "source": source,
                    "component": str(component),
                    "bot": details.get("bot") or bot,
                    "pair": details.get("pair") or pair,
                    "summary": str(raw.get("summary") or "未知运行错误"),
                    "severity": str(raw.get("severity") or "warning"),
                    "first_seen_at": details.get("execution_first_failure_at") or raw.get("first_seen_at"),
                    "last_seen_at": raw.get("last_seen_at"),
                    "occurrences": int(raw.get("occurrences") or 1),
                    "trading_impact": str(raw.get("trading_impact") or ""),
                    "action": str(raw.get("action") or "automatic_retry"),
                    "reason": details.get("reason"),
                    "residual_orders": details.get("residual_orders", []),
                    "reconciliation_checks": details.get("checks"),
                    "release_condition": details.get("release_condition") or (
                        "订单、成交、余额及活动执行器核验通过后按原冷却恢复；其他风控门仍须放行"
                        if str(component).startswith("exit_reconciliation:") else None
                    ),
                })
        payload = {
            "schema": "management-current-runtime-errors-v1",
            "generated_at": now.astimezone(timezone.utc).isoformat(),
            "errors": sorted(errors, key=lambda row: (
                str(row.get("severity")), str(row.get("source")), str(row.get("component"))
            )),
        }
        temporary = self.output / "current_runtime_errors.json.tmp"
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.output / "current_runtime_errors.json")
        return payload

    def _inventory_dust(self, asset: str) -> dict[str, Any] | None:
        status = self._load(self.inventory_status_path)
        row = status.get("assets", {}).get(asset, {})
        if row.get("inventory_phase") != "DUST":
            return None
        return {
            "quantity": row.get("unattributed"),
            "estimated_notional": row.get("estimated_notional"),
            "minimum_notional": row.get("minimum_notional"),
            "reason": row.get("dust_reason") or "below_exchange_minimum",
        }

    def _inventory_row(self, asset: str) -> dict[str, Any]:
        status = self._load(self.inventory_status_path)
        row = status.get("assets", {}).get(asset, {})
        deficit = Decimal(str(row.get("ownership_deficit") or "0")) > 0
        healthy = bool(status.get("sources_healthy", status.get("healthy", False)))
        return gate_row(
            "inventory_ownership_gate", state="BLOCK" if deficit else "ALLOW",
            buy_enabled=not deficit and healthy, sell_enabled=not deficit and healthy,
            health="HEALTHY" if healthy and not deficit else "FAILED",
            reason=("ownership_deficit" if deficit else
                    row.get("confirmation_block_reason") or "ownership_reconciled"),
            source="account_inventory_status",
        )

    @staticmethod
    def _reported_cutover_phase(
        cutover: Mapping[str, Any], contract: Mapping[str, Any], now: datetime,
    ) -> str | None:
        return report_cards.reported_cutover_phase(cutover, contract, now)

    @staticmethod
    def _breaker_rows(mechanisms: Mapping[str, Any], recovery: Mapping[str, Any]) -> list[dict[str, Any]]:
        return report_cards.breaker_rows(mechanisms, recovery)

    def _dca_cards(self, report: Mapping[str, Any], now: datetime) -> list[dict[str, Any]]:
        guard = self._load(self.dca_state / "guard_state.json")
        v22_observation = guard.get("v22_observation", {})
        dca_read_telemetry = guard.get("read_retry_telemetry", {}).get(
            "binance_public", {}
        )
        inventory_status = self._load(self.inventory_status_path)
        cutover = self._load(self.grid_state / "v22_cutover_status.json")
        return report_cards.build_dca_cards(
            report, now, guard=guard, v22_observation=v22_observation,
            dca_read_telemetry=dca_read_telemetry, inventory_status=inventory_status,
            cutover=cutover,
            profit_history=lambda *args, **kwargs: self.outbox.profit_history(*args, **kwargs),
            inventory_dust=self._inventory_dust, inventory_row=self._inventory_row,
            breaker_rows=self._breaker_rows, reported_cutover_phase=self._reported_cutover_phase,
        )

    def _grid_trade_counts(self, database: str, pair: str) -> tuple[int | None, int | None, float | None]:
        if not database or not Path(database).is_file():
            return None, None, None
        try:
            connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=10)
            connection.execute("PRAGMA query_only=ON")
            rows = connection.execute(
                "SELECT trade_type,trade_fee_in_quote FROM TradeFill WHERE symbol=?", (pair,),
            ).fetchall()
            connection.close()
            buys = sum(normalize_side(row[0]) == "BUY" for row in rows)
            sells = sum(normalize_side(row[0]) == "SELL" for row in rows)
            fees = sum(float(decimal_value(row[1])) for row in rows)
            return buys, sells, fees
        except Exception:
            return None, None, None

    def _grid_transport_summary(self, now: datetime) -> dict[str, Any]:
        state = self._load(self.grid_state / "runtime_error_state.json")
        cutoff = now.timestamp() - 4 * 60 * 60
        read_rows = [
            row for row in state.get("history", [])
            if isinstance(row, Mapping)
            and str(row.get("component") or "").startswith("guard_read:")
            and bool(row.get("suppressed_as_transient"))
            and float(row.get("recovered_at", 0) or 0) >= cutoff
        ]
        trade_rows = [
            row for row in state.get("history", [])
            if isinstance(row, Mapping)
            and str(row.get("component") or "").startswith("trade_sync:")
            and float(row.get("recovered_at", 0) or 0) >= cutoff
        ]
        components = state.get("components", {})
        pending = sum(
            1 for name, row in components.items()
            if str(name).startswith("trade_sync:") and bool(row.get("active"))
        ) if isinstance(components, Mapping) else 0
        return {
            "window_hours": 4,
            "recovered_episodes": len(read_rows),
            "retry_attempts": sum(
                int(row.get("occurrences", 1) or 1) for row in read_rows
            ),
            "last_reason": (
                str(read_rows[-1].get("summary") or "") if read_rows else None
            ),
            "trade_sync_recovered": len(trade_rows),
            "trade_sync_longest_recovery_seconds": max(
                (float(row.get("duration_seconds", 0) or 0) for row in trade_rows),
                default=0.0,
            ),
            "trade_sync_pending": pending,
            "trading_impact": "none",
        }

    def _grid_cards(self, now: datetime) -> list[dict[str, Any]]:
        state = self._load(self.grid_state / "guard_state.json")
        v22_observation = state.get("v22_observation", {})
        grid_read_telemetry = state.get("read_retry_telemetry", {})
        macro = self._load(self.grid_state / "macro_gate.json")
        bot_name = "grid-live-fdusd-400"
        bot = state.get("bots", {}).get(bot_name, {})
        latest = bot.get("latest", {})
        gate = state.get("xgboost_risk_gate", {})
        cutover = self._load(self.grid_state / "v22_cutover_status.json")
        runtime_candidates = sorted(
            self.bots_path.glob(f"instances/{bot_name}*/data/live_grid_runtime_state.json"),
            key=lambda path: path.stat().st_mtime, reverse=True,
        )
        runtime = self._load(runtime_candidates[0]) if runtime_candidates else {}
        inventory_status = self._load(self.inventory_status_path)
        transport_summary = self._grid_transport_summary(now)
        runtime_buy_ids = {str(value) for value in runtime.get("buy_order_ids", [])}
        runtime_sell_ids = {str(value) for value in runtime.get("sell_order_ids", [])}
        active_pair_parameters = runtime.get("active_pair_parameters", {})
        return report_cards.build_grid_cards(
            now, state=state, v22_observation=v22_observation,
            grid_read_telemetry=grid_read_telemetry, macro=macro, bot_name=bot_name,
            latest=latest, gate=gate, cutover=cutover, runtime=runtime,
            inventory_status=inventory_status,
            transport_summary=transport_summary, runtime_buy_ids=runtime_buy_ids,
            runtime_sell_ids=runtime_sell_ids, active_pair_parameters=active_pair_parameters,
            profit_history=lambda *args, **kwargs: self.outbox.profit_history(*args, **kwargs),
            trade_counts=self._grid_trade_counts, inventory_dust=self._inventory_dust,
            inventory_row=self._inventory_row, clock=time.time,
            breaker_rows=self._breaker_rows, reported_cutover_phase=self._reported_cutover_phase,
        )

    def update_snapshots(self, report: Mapping[str, Any], now: datetime) -> list[dict[str, Any]]:
        robots = [*self._grid_cards(now), *self._dca_cards(report, now)]
        for item in robots:
            current = item.get("profit", {}).get("all_time_mtm_quote")
            for hours, key in ((4, "four_hour_mtm_quote"),
                               (24, "twenty_four_hour_mtm_quote"),
                               (168, "seven_day_mtm_quote")):
                if item["strategy"] == "grid" or item["profit"].get(key) is None:
                    item["profit"][key] = self.outbox.mtm_delta(
                        item["strategy"], item["pair"], hours,
                        None if current is None else float(current),
                    )
            self.outbox.record_profit(item, observed_at=now.timestamp())
        status_contract = {
            "schema": "grid-dca-trading-status-collection-v1",
            "generated_at": now.astimezone(timezone.utc).isoformat(),
            "robots": [item["trading_status"] for item in robots],
        }
        temporary = self.output / ".trading_status.json.tmp"
        temporary.write_text(
            json.dumps(status_contract, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        os.replace(temporary, self.output / "trading_status.json")
        # Directory-mounted exports follow atomic replacements; single-file bind
        # mounts pin an obsolete inode. Preserve original source timestamps.
        contract_dir = self.output / "guard_contracts"
        contract_dir.mkdir(parents=True, exist_ok=True)
        for name, source in (
            ("grid_guard_state.json", self.grid_state / "guard_state.json"),
            ("dca_guard_state.json", self.dca_state / "guard_state.json"),
            ("account_inventory_status.json", self.inventory_status_path),
        ):
            try:
                value = json.loads(source.read_text(encoding="utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("invalid contract")
                temp = contract_dir / (name + ".tmp")
                temp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
                os.replace(temp, contract_dir / name)
            except Exception:
                logging.getLogger(__name__).warning("Management contract export unavailable: %s", name)
        try:
            try:
                from management_trading import publish
            except ModuleNotFoundError:
                from live_guard.management_trading import publish
            publish(self.bots_path, self.output, status_contract, now.timestamp())
        except Exception:
            logging.getLogger(__name__).warning("Read-only management trading summary unavailable")
        return robots

    def _queue_profit_report(self, robots: list[dict[str, Any]], slot: str,
                             now: datetime) -> None:
        attachments = []
        for item in robots:
            history = self.outbox.profit_history(item["strategy"], item["pair"])
            item["equity_series"] = [row["equity"] for row in history if row["equity"] is not None]
            item["drawdown_series"] = [row["drawdown_pct"] for row in history
                                       if row["drawdown_pct"] is not None]
            path = self.output / "profit" / slot[:10] / (
                f"{slot[11:13]}_{item['strategy']}_{item['pair'].replace('-', '').lower()}.png"
            )
            render_mobile_profit_card(item, path)
            attachments.append({"path": str(path), "kind": "photo",
                                "caption": f"{item['strategy'].upper()} {item['pair']}｜单机器人收益"})
        event = build_event(
            source="dca-live-report", strategy="grid+dca", bot="4 robots",
            pair="BTC-FDUSD,ETH-FDUSD,BTC-USDT,ETH-USDT",
            mechanism="profit_report", transition="PROFIT_REPORT",
            reason="北京时间每4小时收益报告", severity="info",
            action="read_only_report", correlation_id=slot,
            attachments=attachments,
            details={"slot": slot, "robots": [{key: value for key, value in item.items()
                                                if key not in {"equity_series", "drawdown_series"}}
                                               for item in robots]},
        )
        append_event(self.events, event)

    def archive_risk_sources(self, now: datetime) -> None:
        # Archive original producer records before notification flags, formatting
        # or attachment generation can filter/mutate them. Never gate trading.
        history_sources = [self.events, self.dca_state / "risk_audit.jsonl",
                           self.grid_state / "telegram_events.jsonl",
                           self.grid_state / "risk_audit.jsonl",
                           Path("/workspace/stock-reports/telegram_events.jsonl"),
                           Path(os.getenv("MACRO_AUDIT_PATH", "/workspace/macro/audit.jsonl"))]
        history_sources.extend(self.bots_path.glob("instances/*/data/telegram_events.jsonl"))
        try:
            if not hasattr(self, "risk_history"):
                try:
                    from risk_history import RiskHistory
                except ModuleNotFoundError:
                    from live_guard.risk_history import RiskHistory
                self.risk_history = RiskHistory(self.output, now=now.timestamp(), sources=history_sources)
            for source in history_sources:
                self.risk_history.ingest(source, now=now.timestamp())
            # Inventory Guard already durably publishes pending inventory events
            # into its JSONL before acknowledging them, regardless of Telegram.
            # Archive that original stream above; do not open its live WAL DB
            # through a read-only mount (which may need writable shared memory).
        except Exception:
            logging.getLogger(__name__).exception("Read-only risk archive unavailable; trading unaffected")

    def cycle(self, report: Mapping[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
        now = now or datetime.now(timezone.utc)
        self.archive_risk_sources(now)
        robots = self.update_snapshots(report, now)
        try:
            # Guard timestamps describe the gates; candle/database ages do not.
            from_source = {}
            for strategy, root in (("grid", self.grid_state), ("dca", self.dca_state)):
                state = self._load(root / "guard_state.json")
                from_source[strategy] = (state.get("last_success_at") or state.get("generated_at")
                                         or state.get("updated_at"))
            self.risk_history.sample(robots, now=now.timestamp(), source_times=from_source)
            self.risk_history.maintain(now=now.timestamp())
            self.risk_history.publish(now=now.timestamp())
        except Exception:
            logging.getLogger(__name__).exception("Risk snapshot history unavailable; trading unaffected")
        mqtt_reporter = getattr(self, "mqtt_reporter", None)
        if mqtt_reporter is not None:
            try:
                mqtt_reporter.notify_report_ready()
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "Cardputer MQTT refresh unavailable (%s); report continues",
                    type(exc).__name__,
                )
        try:
            if not hasattr(self, "simple_audit"):
                try:
                    from simple_audit import SimpleAudit
                except ModuleNotFoundError:
                    from live_guard.simple_audit import SimpleAudit
                self.simple_audit = SimpleAudit(self.output)
            self.simple_audit.cycle(
                grid=self._load(self.grid_state / "guard_state.json"),
                dca=self._load(self.dca_state / "guard_state.json"),
                inventory=self._load(self.inventory_status_path),
                history_connection=self.outbox.connection, api_base=binance_api_base(),
            )
        except Exception:
            logging.getLogger(__name__).warning("Read-only simple audit unavailable")
        current_errors = self._publish_current_runtime_errors(now)
        parameter_catalog = self.management_parameters.publish(
            [item["trading_status"] for item in robots], now=now,
        )
        due, slot = self.outbox.slot_due(now=now)
        profit_report_error = ""
        profit_report_deferred_pairs = []
        if self.enabled and self.profit_enabled and due:
            profit_report_deferred_pairs = [
                str(bot["trading_pair"]) for bot in report.get("bots", [])
                if bot.get("market_data_healthy") is not True
            ]
            # At a candle boundary the normal collector may briefly lack a
            # fresh mark. Keep this slot due for the next existing cycle;
            # missing historical window anchors must not delay delivery.
            if not profit_report_deferred_pairs:
                try:
                    self._queue_profit_report(robots, slot, now)
                    self.outbox.mark_slot(slot)
                except Exception as exc:
                    # A presentation failure must not block risk-event ingestion,
                    # the persistent outbox, or parameter evidence delivery.
                    profit_report_error = f"{type(exc).__name__}: {exc}"
        sources = [self.events, self.grid_state / "telegram_events.jsonl"]
        sources.append(Path("/workspace/stock-reports/telegram_events.jsonl"))
        sources.extend(self.bots_path.glob("instances/*/data/telegram_events.jsonl"))
        for source in sources:
            self.outbox.ingest(Path(source), attachment_builder=self.parameter_worker.schedule)
        worker = self.parameter_worker.poll()
        sent = self.outbox.drain(self.client) if self.enabled and self.client is not None else 0
        evidence_receipts = self.parameter_worker.finalize_delivery_receipts()
        return {"enabled": self.enabled, "sent": sent, **self.outbox.health(),
                "slot": slot, "parameter_worker": worker,
                "parameter_catalog_sha256": parameter_catalog["catalog_sha256"],
                "active_runtime_errors": len(current_errors["errors"]),
                "evidence_receipts_written": evidence_receipts,
                "profit_report_deferred_pairs": profit_report_deferred_pairs,
                "profit_report_error": profit_report_error}


def _cardputer_mqtt_requested() -> bool:
    return os.getenv("CARDPUTER_MQTT_ENABLED", "false").strip().lower() in {
        "true", "1", "yes", "on",
    }


def _start_cardputer_mqtt(output_dir: Path):
    if not _cardputer_mqtt_requested():
        return None
    try:
        from cardputer_monitor.mqtt_reporter import MqttReportPublisher
        reporter = MqttReportPublisher.from_environment(
            output_dir / "telegram", logger=logging.getLogger(__name__),
        )
        reporter.start()
        return reporter
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Cardputer MQTT initialization unavailable (%s); report continues",
            type(exc).__name__,
        )
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bots-path",
        type=Path,
        default=Path(os.getenv("BOTS_PATH", "/workspace/bots")),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.getenv("DCA_LIVE_STATE_PATH", "/workspace/state")),
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=int(os.getenv("DCA_LIVE_REPORT_INTERVAL", "60")),
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--healthcheck", action="store_true")
    args = parser.parse_args()
    if args.healthcheck:
        return 0 if healthcheck(args.output_dir) else 1
    collector = DcaLiveReportCollector(args.bots_path, args.output_dir)
    telegram = UnifiedTelegramReporting(bots_path=args.bots_path, dca_state=args.output_dir)
    runtime_errors = RuntimeErrorChannel(
        event_path=args.output_dir / "telegram_events.jsonl",
        state_path=args.output_dir / "report_runtime_error_state.json",
        source="dca-live-report", strategy="grid+dca", bot="report-service",
        pair="BTC-FDUSD,ETH-FDUSD,BTC-USDT,ETH-USDT",
    )
    reporter = _start_cardputer_mqtt(args.output_dir)
    telegram.mqtt_reporter = reporter
    stopping = threading.Event()
    previous_signals = {}
    def request_stop(_signum, _frame):
        stopping.set()
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_signals[signum] = signal.signal(signum, request_stop)
    try:
        return _run_report_loop(args, collector, telegram, runtime_errors, stopping)
    finally:
        if reporter is not None:
            try:
                reporter.stop(timeout=2)
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "Cardputer MQTT shutdown unavailable (%s)", type(exc).__name__,
                )
        for signum, previous in previous_signals.items():
            signal.signal(signum, previous)


def _run_report_loop(args, collector, telegram, runtime_errors, stopping) -> int:
    while not stopping.is_set():
        cycle_succeeded = False
        try:
            telegram.archive_risk_sources(datetime.now(timezone.utc))
            report = collector.collect()
            telegram_status = telegram.cycle(report)
            if report.get("warnings"):
                runtime_errors.failure(
                    "report_data_sources", "; ".join(map(str, report["warnings"])),
                    severity="warning",
                    trading_impact="收益报告部分数据降级；不影响交易或风控决策。",
                    action="continue_degraded_report_and_retry_sources",
                )
            else:
                runtime_errors.recovered(
                    "report_data_sources",
                    trading_status="收益报告数据源恢复完整；交易始终未受影响",
                )
            if int(telegram_status.get("retrying", 0)) > 0:
                runtime_errors.failure(
                    "telegram_delivery",
                    str(telegram_status.get("last_error") or "Telegram delivery retry pending"),
                    severity="warning",
                    trading_impact="频道消息进入持久化重试队列；不影响交易或风控决策。",
                    action="persistent_outbox_retry",
                    details={
                        "pending": telegram_status.get("pending"),
                        "max_attempts": telegram_status.get("max_attempts"),
                    },
                )
            else:
                runtime_errors.recovered(
                    "telegram_delivery",
                    trading_status="Telegram 持久化队列发送恢复正常；交易始终未受影响",
                )
            profit_report_error = str(telegram_status.get("profit_report_error") or "")
            if profit_report_error:
                runtime_errors.failure(
                    "profit_report_generation", profit_report_error,
                    severity="warning",
                    trading_impact="四小时收益PNG暂未生成；风控事件和其他频道消息继续发送。",
                    action="retry_current_profit_slot_without_blocking_outbox",
                )
            else:
                runtime_errors.recovered(
                    "profit_report_generation",
                    trading_status="四小时收益PNG生成恢复；未补发多个历史时段",
                )
            runtime_errors.recovered(
                "report_cycle", trading_status="报告与通知恢复；交易始终不受报告服务影响",
            )
            cycle_succeeded = True
            mqtt_health = {}
            reporter = getattr(telegram, "mqtt_reporter", None)
            if reporter is not None:
                try:
                    mqtt_health = reporter.health()
                except Exception:
                    mqtt_health = {"status": "unavailable"}
            print(
                json.dumps(
                    {
                        "generated_at": report["generated_at"],
                        "report_id": report["report_id"],
                        "bots": len(report["bots"]),
                        "warnings": report["warnings"],
                        "telegram": telegram_status,
                        "cardputer_mqtt": mqtt_health,
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            runtime_errors.failure(
                "report_cycle", exc, severity="warning",
                trading_impact="报告或频道发送链路降级；不影响 Grid/DCA 交易与风控决策。",
                action="persistent_outbox_retry",
            )
            print(
                json.dumps(
                    {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                ),
                flush=True,
            )
        if args.once:
            if not cycle_succeeded:
                return 1
            if _cardputer_mqtt_requested():
                reporter = getattr(telegram, "mqtt_reporter", None)
                try:
                    return 0 if reporter is not None and reporter.flush_once(timeout=30) else 1
                except Exception:
                    return 1
            return 0
        stopping.wait(max(60, args.interval))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
