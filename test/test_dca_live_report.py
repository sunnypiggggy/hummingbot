import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from PIL import Image

from live_guard.dca_live_report import (
    CHART_SIZE,
    DcaLiveReportCollector,
    calculate_pair_report,
    healthcheck,
    render_chart,
)
from macro_control.trading_report import (
    JsonTradingReportProvider,
    TradingReportUnavailable,
)


NOW = datetime(2026, 7, 25, 6, 0, tzinfo=timezone.utc)
SCALE = Decimal("1000000")


def raw(value):
    return int(Decimal(str(value)) * SCALE)


def row(side, price, amount, fee, at):
    return (
        side,
        raw(price),
        raw(amount),
        raw(fee),
        int(at.timestamp() * 1000),
    )


def candles():
    return [
        {
            "timestamp": int((NOW - timedelta(days=7)).timestamp() * 1000),
            "close": "105",
        },
        {
            "timestamp": int((NOW - timedelta(days=3)).timestamp() * 1000),
            "close": "115",
        },
        {"timestamp": int(NOW.timestamp() * 1000), "close": "130"},
    ]


class DcaLiveReportCalculationTests(unittest.TestCase):
    def test_all_time_and_seven_day_mtm_include_opening_position_and_fees(self):
        rows = [
            row("BUY", 100, 1, 1, NOW - timedelta(days=8)),
            row("SELL", 120, 0.5, 0.5, NOW - timedelta(days=2)),
            row("BUY", 110, 0.25, 0.25, NOW - timedelta(days=1)),
        ]
        report = calculate_pair_report(
            bot_name="dca-live-btcusdt-200",
            pair="BTC-USDT",
            rows=rows,
            candles=candles(),
            database_age_seconds=2,
            now=NOW,
        )
        self.assertEqual("0.75", report["position"]["net_base"])
        self.assertEqual(
            Decimal("28.25"),
            Decimal(report["profit"]["all_time_mtm_quote"]),
        )
        self.assertEqual(
            Decimal("24.25"),
            Decimal(report["profit"]["seven_day_mtm_quote"]),
        )
        self.assertEqual(2, report["trades"]["seven_day"])
        self.assertEqual({"BUY", "SELL"}, {
            item["side"] for item in report["trades"]["fills_7d"]
        })

    def test_negative_inventory_delta_is_preserved_and_not_account_inventory(self):
        report = calculate_pair_report(
            bot_name="dca-live-ethusdt-200",
            pair="ETH-USDT",
            rows=[row("SELL", 120, 1, 0, NOW - timedelta(hours=1))],
            candles=candles(),
            database_age_seconds=1,
            now=NOW,
        )
        self.assertEqual(
            "strategy_owned_inventory_delta", report["position"]["scope"]
        )
        self.assertEqual("-1", report["position"]["net_base"])

    def test_stale_market_data_makes_profit_unavailable(self):
        stale = candles()
        stale[-1]["timestamp"] = int(
            (NOW - timedelta(minutes=16)).timestamp() * 1000
        )
        report = calculate_pair_report(
            bot_name="dca-live-btcusdt-200",
            pair="BTC-USDT",
            rows=[],
            candles=stale,
            database_age_seconds=1,
            now=NOW,
        )
        self.assertFalse(report["market_data_healthy"])
        self.assertIsNone(report["profit"]["all_time_mtm_quote"])
        self.assertIsNone(report["profit"]["seven_day_mtm_quote"])

    def test_public_fill_limit_sets_truncated_without_changing_totals(self):
        rows = [
            row("BUY", 100 + index, 0.1, 0, NOW - timedelta(hours=index))
            for index in range(3)
        ]
        report = calculate_pair_report(
            bot_name="dca-live-btcusdt-200",
            pair="BTC-USDT",
            rows=rows,
            candles=candles(),
            database_age_seconds=1,
            now=NOW,
            max_public_fills=2,
        )
        self.assertEqual(3, report["trades"]["seven_day"])
        self.assertEqual(2, len(report["trades"]["fills_7d"]))
        self.assertTrue(report["trades"]["truncated"])

    def test_emergency_adjustment_is_included_in_position_and_profit(self):
        report = calculate_pair_report(
            bot_name="dca-live-btcusdt-200",
            pair="BTC-USDT",
            rows=[row("BUY", 100, 1, 0, NOW - timedelta(hours=2))],
            candles=candles(),
            database_age_seconds=1,
            now=NOW,
            emergency_adjustments=[
                {
                    "recorded_at": (NOW - timedelta(hours=1)).isoformat(),
                    "side": "SELL",
                    "executed_qty": "1",
                    "base_delta": "-1",
                    "quote_cashflow": "120",
                    "fee_quote": "1",
                }
            ],
        )
        self.assertEqual("0", report["position"]["net_base"])
        self.assertEqual("19", report["profit"]["all_time_mtm_quote"])
        self.assertTrue(report["trades"]["fills_7d"][-1]["emergency"])


class DcaChartAndProviderTests(unittest.TestCase):
    def _report(self):
        bot = calculate_pair_report(
            bot_name="dca-live-btcusdt-200",
            pair="BTC-USDT",
            rows=[
                row("BUY", 110, 0.1, 0, NOW - timedelta(days=2)),
                row("SELL", 120, 0.1, 0, NOW - timedelta(days=1)),
            ],
            candles=candles(),
            database_age_seconds=1,
            now=NOW,
        )
        return {
            "schema_version": 3,
            "policy_version": "dca-macro-v3",
            "generated_at": NOW.isoformat(),
            "window": {
                "days": 7,
                "start_at_epoch": (NOW - timedelta(days=7)).timestamp(),
                "end_at_epoch": NOW.timestamp(),
            },
            "bots": [bot, {**bot, "trading_pair": "ETH-USDT"}],
        }

    def test_chart_is_expected_png_size_and_contains_trade_marker_colors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chart.png"
            render_chart(self._report(), path)
            self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            with Image.open(path) as image:
                self.assertEqual(CHART_SIZE, image.size)
                colors = image.convert("RGB").getcolors(
                    maxcolors=image.width * image.height
                )
            palette = {color for _, color in colors}
            self.assertIn((22, 163, 74), palette)
            self.assertIn((220, 38, 38), palette)

    def test_provider_rejects_stale_or_mismatched_chart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chart = root / "chart.png"
            report_path = root / "report.json"
            value = b"\x89PNG\r\n\x1a\nvalid"
            chart.write_bytes(value)
            report = {
                "schema_version": 3,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "report_id": "report",
                "chart_sha256": hashlib.sha256(value).hexdigest(),
            }
            report_path.write_text(json.dumps(report), encoding="utf-8")
            provider = JsonTradingReportProvider(report_path, chart)
            self.assertEqual("report", provider.chart()[1])

            chart.write_bytes(b"\x89PNG\r\n\x1a\nchanged")
            with self.assertRaisesRegex(
                TradingReportUnavailable, "does not match"
            ):
                provider.chart()

            report["generated_at"] = (
                datetime.now(timezone.utc) - timedelta(hours=1)
            ).isoformat()
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(TradingReportUnavailable, "stale"):
                JsonTradingReportProvider(
                    report_path, chart, max_age_seconds=1
                ).report()

    def test_healthcheck_validates_fresh_matching_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chart = b"\x89PNG\r\n\x1a\nchart"
            (root / "dca_trade_chart_7d.png").write_bytes(chart)
            (root / "dca_trade_report_v3.json").write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                        "chart_sha256": hashlib.sha256(chart).hexdigest(),
                        "bots": [{}, {}],
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(healthcheck(root))

    def test_collector_atomically_writes_report_and_matching_chart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bots_root = root / "bots"
            output = root / "output"
            now = datetime.now(timezone.utc)
            for bot_name, pair in (
                ("dca-live-btcusdt-200", "BTC-USDT"),
                ("dca-live-ethusdt-200", "ETH-USDT"),
            ):
                data = bots_root / "instances" / bot_name / "data"
                data.mkdir(parents=True)
                database = data / f"{bot_name}.sqlite"
                import sqlite3

                connection = sqlite3.connect(database)
                connection.execute(
                    "CREATE TABLE TradeFill "
                    "(trade_type TEXT, price INTEGER, amount INTEGER, "
                    "trade_fee_in_quote INTEGER, timestamp INTEGER, symbol TEXT)"
                )
                connection.execute(
                    "INSERT INTO TradeFill VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        "BUY",
                        raw(100),
                        raw("0.1"),
                        raw("0.01"),
                        int((now - timedelta(hours=1)).timestamp() * 1000),
                        pair,
                    ),
                )
                connection.commit()
                connection.close()

            def candle_reader(pair, start, end):
                return [
                    {"timestamp": start.timestamp(), "close": "100"},
                    {"timestamp": end.timestamp(), "close": "110"},
                ]

            report = DcaLiveReportCollector(
                bots_root, output, candle_reader=candle_reader
            ).collect(now=now)
            self.assertEqual(2, len(report["bots"]))
            self.assertTrue(healthcheck(output))
            chart = (output / "dca_trade_chart_7d.png").read_bytes()
            self.assertEqual(
                hashlib.sha256(chart).hexdigest(), report["chart_sha256"]
            )
            self.assertFalse(
                any("_candles" in bot for bot in report["bots"])
            )

    def test_missing_v3_databases_are_explicitly_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            now = datetime.now(timezone.utc)

            def candle_reader(pair, start, end):
                return [
                    {"timestamp": start.timestamp(), "close": "100"},
                    {"timestamp": end.timestamp(), "close": "110"},
                ]

            report = DcaLiveReportCollector(
                root / "bots",
                root / "output",
                candle_reader=candle_reader,
            ).collect(now=now)
            self.assertEqual(2, len(report["bots"]))
            self.assertTrue(healthcheck(root / "output"))
            for bot in report["bots"]:
                self.assertFalse(bot["database_available"])
                self.assertIsNone(bot["position"]["net_base"])
                self.assertIsNone(bot["profit"]["all_time_mtm_quote"])
                self.assertIsNone(bot["trades"]["all_time"])


if __name__ == "__main__":
    unittest.main()
