#!/usr/bin/env python3
"""Build a read-only V3 DCA performance snapshot and seven-day trade chart."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable

import requests
from PIL import Image, ImageDraw, ImageFont

try:
    from dca_live_common import LIVE_PAIRS
except ModuleNotFoundError:  # Repository import; container copies it to /app.
    from scripts.dca_live_common import LIVE_PAIRS


BINANCE_API = "https://api.binance.com"
SCALE = Decimal("1000000")
WINDOW_DAYS = 7
MAX_PUBLIC_FILLS = 2_000
MAX_DATA_AGE_SECONDS = 900
CHART_SIZE = (1280, 1000)


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
    start_candle = next(
        (
            item
            for item in normalized_candles
            if item["timestamp"] >= window_start_ts
        ),
        None,
    )
    start_mark = Decimal(start_candle["close"]) if start_candle else None

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
    window_metrics = row_metrics(window_rows)
    apply_emergency_adjustments(all_metrics, emergency_adjustments)
    apply_emergency_adjustments(before_metrics, before_adjustments)
    apply_emergency_adjustments(window_metrics, window_adjustments)
    ending_position = before_metrics["net_base"] + window_metrics["net_base"]

    all_time_mtm = (
        all_metrics["cashflow_quote"]
        - all_metrics["fees_quote"]
        + all_metrics["net_base"] * mark
        if market_healthy and mark is not None
        else None
    )
    seven_day_mtm = (
        window_metrics["cashflow_quote"]
        - window_metrics["fees_quote"]
        + ending_position * mark
        - before_metrics["net_base"] * start_mark
        if market_healthy and mark is not None and start_mark is not None
        else None
    )

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
            "all_time_cashflow_quote": str(all_metrics["cashflow_quote"]),
            "seven_day_cashflow_quote": str(window_metrics["cashflow_quote"]),
            "all_time_fees_quote": str(all_metrics["fees_quote"]),
            "seven_day_fees_quote": str(window_metrics["fees_quote"]),
        },
        "trades": {
            "all_time": int(all_metrics["trades"]),
            "all_time_buys": int(all_metrics["buys"]),
            "all_time_sells": int(all_metrics["sells"]),
            "seven_day": int(window_metrics["trades"]),
            "seven_day_buys": int(window_metrics["buys"]),
            "seven_day_sells": int(window_metrics["sells"]),
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
        "all_time_cashflow_quote",
        "seven_day_cashflow_quote",
        "all_time_fees_quote",
        "seven_day_fees_quote",
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
    while True:
        try:
            report = collector.collect()
            print(
                json.dumps(
                    {
                        "generated_at": report["generated_at"],
                        "report_id": report["report_id"],
                        "bots": len(report["bots"]),
                        "warnings": report["warnings"],
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
