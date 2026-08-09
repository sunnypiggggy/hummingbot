#!/usr/bin/env python3
"""Build a read-only V3 DCA performance snapshot and seven-day trade chart."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from zoneinfo import ZoneInfo

import requests
from PIL import Image, ImageDraw, ImageFont

try:
    from telegram_notifications import (
        TelegramChannelClient, TelegramOutbox, append_event, build_event,
        format_event, render_mobile_profit_card,
    )
except ModuleNotFoundError:
    from live_guard.telegram_notifications import (
        TelegramChannelClient, TelegramOutbox, append_event, build_event,
        format_event, render_mobile_profit_card,
    )
try:
    from telegram_parameter_report import build_parameter_attachments
except ModuleNotFoundError:
    from live_guard.telegram_parameter_report import build_parameter_attachments

try:
    from dca_live_common import LIVE_PAIRS, STRATEGY_BUDGET_QUOTE
except ModuleNotFoundError:  # Repository import; container copies it to /app.
    from scripts.dca_live_common import LIVE_PAIRS, STRATEGY_BUDGET_QUOTE


BINANCE_API = "https://api.binance.com"
SCALE = Decimal("1000000")
WINDOW_DAYS = 7
MAX_PUBLIC_FILLS = 2_000
MAX_DATA_AGE_SECONDS = 900
CHART_SIZE = (1280, 1000)
SHANGHAI = ZoneInfo("Asia/Shanghai")


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
    for side, raw_price, raw_amount, raw_fee, *_ in rows:
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
    before_rows = [
        row for row in ordered_rows if timestamp_seconds(row[4]) < window_start_ts
    ]
    window_rows = [
        row
        for row in ordered_rows
        if window_start_ts <= timestamp_seconds(row[4]) <= now_ts
    ]
    emergency_adjustments = list(emergency_adjustments or [])
    before_adjustments = [
        item
        for item in emergency_adjustments
        if datetime.fromisoformat(str(item["recorded_at"])).timestamp()
        < window_start_ts
    ]
    window_adjustments = [
        item
        for item in emergency_adjustments
        if window_start_ts
        <= datetime.fromisoformat(str(item["recorded_at"])).timestamp()
        <= now_ts
    ]
    all_metrics = row_metrics(ordered_rows)
    before_metrics = row_metrics(before_rows)
    seven_day_metrics = row_metrics(window_rows)
    apply_emergency_adjustments(all_metrics, emergency_adjustments)
    apply_emergency_adjustments(before_metrics, before_adjustments)
    apply_emergency_adjustments(seven_day_metrics, window_adjustments)

    all_time_mtm = (
        all_metrics["cashflow_quote"]
        - all_metrics["fees_quote"]
        + all_metrics["net_base"] * mark
        if market_healthy and mark is not None
        else None
    )
    def period_mtm(hours: int) -> tuple[Decimal | None, dict[str, Decimal]]:
        start_ts = now_ts - hours * 3600
        earlier_rows = [row for row in ordered_rows if timestamp_seconds(row[4]) < start_ts]
        period_rows = [row for row in ordered_rows if start_ts <= timestamp_seconds(row[4]) <= now_ts]
        earlier_adjustments = [item for item in emergency_adjustments
                               if datetime.fromisoformat(str(item["recorded_at"])).timestamp() < start_ts]
        period_adjustments = [item for item in emergency_adjustments
                              if start_ts <= datetime.fromisoformat(str(item["recorded_at"])).timestamp() <= now_ts]
        earlier = row_metrics(earlier_rows)
        period = row_metrics(period_rows)
        apply_emergency_adjustments(earlier, earlier_adjustments)
        apply_emergency_adjustments(period, period_adjustments)
        start_candle = next((item for item in normalized_candles
                             if item["timestamp"] >= start_ts), None)
        if not market_healthy or mark is None or start_candle is None:
            return None, period
        start_mark = Decimal(start_candle["close"])
        ending_position = earlier["net_base"] + period["net_base"]
        value = (period["cashflow_quote"] - period["fees_quote"]
                 + ending_position * mark - earlier["net_base"] * start_mark)
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
                "timestamp": datetime.fromisoformat(
                    str(adjustment["recorded_at"])
                ).timestamp(),
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
        datetime.fromisoformat(str(item["recorded_at"])).timestamp()
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
            "scope": "strategy_owned_inventory_delta",
            "net_base": str(all_metrics["net_base"]),
            "market_value_quote": (
                str(all_metrics["net_base"] * mark) if mark is not None else None
            ),
        },
        "profit": {
            "valuation": "cashflow + net_base * mark - recorded_fees",
            "all_time_mtm_quote": (
                str(all_time_mtm) if all_time_mtm is not None else None
            ),
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
        self.candle_reader = candle_reader or self._binance_candles

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
                "SELECT trade_type, price, amount, trade_fee_in_quote, timestamp "
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
            f"{BINANCE_API}/api/v3/klines",
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
                bot = calculate_pair_report(
                    bot_name=spec.bot_name,
                    pair=pair,
                    rows=self._rows(database, pair),
                    candles=self.candle_reader(pair, window_start, now),
                    database_age_seconds=max(
                        0.0, time.time() - database.stat().st_mtime
                    ),
                    now=now,
                    emergency_adjustments=guard_state.get("bots", {})
                    .get(spec.bot_name, {})
                    .get("emergency_adjustments", []),
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
            "position_scope": "strategy_owned_inventory_delta",
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
        return (
            report.get("schema_version") == 3
            and len(report.get("bots", [])) == len(LIVE_PAIRS)
            and age <= MAX_DATA_AGE_SECONDS
            and hashlib.sha256(chart_path.read_bytes()).hexdigest()
            == report.get("chart_sha256")
        )
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
        self.jobs.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=1, initializer=_low_priority_report_worker,
        )
        self.future: concurrent.futures.Future | None = None
        self.active_path: Path | None = None
        self.active_event: dict[str, Any] | None = None

    @staticmethod
    def _requires_report(event: Mapping[str, Any]) -> bool:
        return str(event.get("details", {}).get("report_request", "")) in {
            "v22_360d", "grid_360d",
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
                    reason="360天PNG证据缺失或哈希不匹配；未伪造图片，参数流程沿用原有门槛",
                    severity="critical", action="review_attached_missing-evidence_pdf",
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
                "attachments": attachments,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            failure = build_event(
                source="dca-live-report", strategy=str(event.get("strategy", "")),
                bot=str(event.get("bot", "")), pair=str(event.get("pair", "")),
                mechanism="parameter_update", transition="REPORT_EVIDENCE_MISSING",
                reason=("360天PNG/PDF证据生成失败；参数更新仍仅受原有交易门槛控制："
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

    def _dca_cards(self, report: Mapping[str, Any], now: datetime) -> list[dict[str, Any]]:
        guard = self._load(self.dca_state / "guard_state.json")
        output = []
        for bot in report.get("bots", []):
            pair = str(bot["trading_pair"])
            state = guard.get("bots", {}).get(bot["bot_name"], {})
            recovery = state.get("recovery", {})
            executor_counts = state.get("latest", {}).get("executor_counts", {})
            aggregate = guard.get("gate_aggregate", {}).get("bots", {}).get(bot["bot_name"], {})
            all_time = bot.get("profit", {}).get("all_time_mtm_quote")
            initial_equity = float(STRATEGY_BUDGET_QUOTE)
            equity = None if all_time is None else initial_equity + float(all_time)
            history = self.outbox.profit_history("dca", pair, days=370)
            historic_peak = max([
                initial_equity,
                *(row["equity"] for row in history if row["equity"] is not None),
            ])
            peak = max(historic_peak, float(state.get("peak_equity", historic_peak)),
                       equity or historic_peak)
            drawdown = None if equity is None or peak <= 0 else (peak - equity) / peak * 100
            technical = guard.get("v22_observation", {}).get("event_ids", {}).get(pair, "")
            data_age = max(bot.get("market_data_age_seconds") or 0,
                           bot.get("database_age_seconds") or 0)
            warnings = list(report.get("warnings", []))
            if not bot.get("database_available"):
                warnings.append(f"{bot['bot_name']}: SQLite无可信数据")
            if not bot.get("market_data_healthy"):
                warnings.append(f"{pair}: Binance行情过期或缺失")
            if data_age > MAX_DATA_AGE_SECONDS:
                warnings.append(f"{pair}: 数据年龄{data_age:.0f}秒")
            item = {
                "strategy": "dca", "bot": bot["bot_name"], "pair": pair,
                "quote_asset": "USDT", "generated_at_bjt": now.astimezone(SHANGHAI).isoformat(),
                "data_age_seconds": data_age,
                "data_sources": ["Hummingbot SQLite TradeFill", "Binance 15m OHLCV",
                                 "dca-live-guard state"],
                "profit": dict(bot.get("profit", {})), "equity": equity,
                "peak_equity": peak, "drawdown_pct": drawdown,
                "owned_base": bot.get("position", {}).get("net_base"),
                "fees_quote": bot.get("profit", {}).get("all_time_fees_quote"),
                "buys": bot.get("trades", {}).get("all_time_buys"),
                "sells": bot.get("trades", {}).get("all_time_sells"),
                "phase": recovery.get("phase", "ACTIVE"),
                "active_runtime": {
                    "orders": executor_counts.get("open_orders"),
                    "buy_executors": executor_counts.get("active_buy_executors"),
                    "sell_executors": executor_counts.get("active_sell_executors"),
                },
                "v22_gate": (
                    "放行" if aggregate.get("v22_buy_enabled") is True else
                    "关闭" if aggregate.get("v22_buy_enabled") is False else
                    "event:" + str(technical)[:8] if technical else "无可信数据"
                ),
                "fomc_gate": (
                    f"BUY={'开' if aggregate.get('fomc_buy_enabled') else '关'} / "
                    f"SELL={'开' if aggregate.get('fomc_sell_enabled') else '关'}"
                    if aggregate else "无可信数据"
                ),
                "warnings": warnings,
            }
            output.append(item)
        return output

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

    def _grid_cards(self, now: datetime) -> list[dict[str, Any]]:
        state = self._load(self.grid_state / "guard_state.json")
        macro = self._load(self.grid_state / "macro_gate.json")
        bot_name = "grid-live-fdusd-400"
        bot = state.get("bots", {}).get(bot_name, {})
        latest = bot.get("latest", {})
        gate = state.get("xgboost_risk_gate", {})
        output = []
        runtime_candidates = sorted(
            self.bots_path.glob(f"instances/{bot_name}*/data/live_grid_runtime_state.json"),
            key=lambda path: path.stat().st_mtime, reverse=True,
        )
        runtime = self._load(runtime_candidates[0]) if runtime_candidates else {}
        grid_open_orders = sum(
            len(value.get("open_order_ids", []))
            for value in runtime.get("ledgers", {}).values()
        )
        for pair in ("BTC-FDUSD", "ETH-FDUSD"):
            values = latest.get("pairs", {}).get(pair, {})
            pnl = float(values["pnl"]) if values.get("pnl") is not None else None
            equity = None if pnl is None else 200 + pnl
            history = self.outbox.profit_history("grid", pair, days=370)
            historic_peak = max([200.0, *(row["equity"] for row in history if row["equity"] is not None)])
            peak = max(historic_peak, equity or historic_peak)
            drawdown = None if equity is None else (peak - equity) / peak * 100
            buys, sells, fees = self._grid_trade_counts(str(latest.get("database", "")), pair)
            data_age = (max(0, time.time() - float(latest.get("updated_at", 0)))
                        if latest else None)
            warnings = [] if latest else ["Grid Guard快照无可信数据"]
            if data_age is not None and data_age > MAX_DATA_AGE_SECONDS:
                warnings.append(f"{pair}: Grid快照年龄{data_age:.0f}秒")
            item = {
                "strategy": "grid", "bot": bot_name, "pair": pair,
                "quote_asset": "FDUSD", "generated_at_bjt": now.astimezone(SHANGHAI).isoformat(),
                "data_age_seconds": data_age,
                "data_sources": ["Hummingbot SQLite TradeFill",
                                 "grid-live-guard snapshot", "v22 shared contract"],
                "profit": {"all_time_mtm_quote": pnl}, "equity": equity,
                "peak_equity": peak, "drawdown_pct": drawdown,
                "owned_base": values.get("net_base"), "fees_quote": fees,
                "buys": buys, "sells": sells,
                "phase": runtime.get("pair_recovery", {}).get(pair, {}).get(
                    "phase", runtime.get("portfolio_recovery", {}).get("phase", "ACTIVE")
                ),
                "active_runtime": {"orders": grid_open_orders},
                "v22_gate": "Risk-Off" if gate.get("pairs", {}).get(pair, {}).get("risk_off_active") else "Risk-On",
                "fomc_gate": "暂停" if macro.get("pause_new_orders") else "放行",
                "warnings": warnings,
            }
            output.append(item)
        return output

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

    def cycle(self, report: Mapping[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
        now = now or datetime.now(timezone.utc)
        robots = self.update_snapshots(report, now)
        due, slot = self.outbox.slot_due(now=now)
        if self.enabled and self.profit_enabled and due:
            self._queue_profit_report(robots, slot, now)
            self.outbox.mark_slot(slot)
        sources = [self.events, self.grid_state / "telegram_events.jsonl"]
        sources.extend(self.bots_path.glob("instances/*/data/telegram_events.jsonl"))
        for source in sources:
            self.outbox.ingest(Path(source), attachment_builder=self.parameter_worker.schedule)
        worker = self.parameter_worker.poll()
        sent = self.outbox.drain(self.client) if self.enabled and self.client is not None else 0
        return {"enabled": self.enabled, "sent": sent, **self.outbox.health(),
                "slot": slot, "parameter_worker": worker}


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
        default=int(os.getenv("DCA_LIVE_REPORT_INTERVAL", "300")),
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--healthcheck", action="store_true")
    args = parser.parse_args()
    if args.healthcheck:
        return 0 if healthcheck(args.output_dir) else 1
    collector = DcaLiveReportCollector(args.bots_path, args.output_dir)
    telegram = UnifiedTelegramReporting(bots_path=args.bots_path, dca_state=args.output_dir)
    while True:
        try:
            report = collector.collect()
            telegram_status = telegram.cycle(report)
            print(
                json.dumps(
                    {
                        "generated_at": report["generated_at"],
                        "report_id": report["report_id"],
                        "bots": len(report["bots"]),
                        "warnings": report["warnings"],
                        "telegram": telegram_status,
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            print(
                json.dumps(
                    {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                ),
                flush=True,
            )
        if args.once:
            return 0
        time.sleep(max(60, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
