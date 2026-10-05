"""Actual collector -> actual cloud routes, with only synthetic local inputs."""
import argparse
import json
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cardputer_monitor.collector import ROBOTS, STATUS_SCHEMA, SnapshotCollector  # noqa: E402
from cardputer_monitor.history import HistoryCollector, PublicKlinePrices  # noqa: E402
from live_guard.trading_status import evaluate_status, gate_row  # noqa: E402

NOW = 1791072000
BASE = "/api/cardputer/v1"
DEVICE = "synthetic-device-" + "a" * 40
TELEMETRY = "synthetic-telemetry-" + "b" * 40


def auth(token=DEVICE):
    return {"Authorization": "Bearer " + token}


def status(path, observed, stopped=False):
    stamp = datetime.fromtimestamp(observed, timezone.utc).isoformat()
    rows = [evaluate_status(
        generated_at=stamp, strategy=strategy, bot=bot, pair=pair,
        process_running=not (stopped and index == 0), phase="ACTIVE",
        gates=[gate_row("v22_weekly_buy_gate"), gate_row("controller_application_gate")],
    ) for index, (strategy, pair, _, bot) in enumerate(ROBOTS)]
    (path / "trading_status.json").write_text(json.dumps({"schema": STATUS_SCHEMA,
        "generated_at": stamp, "robots": rows}), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud-root", required=True, type=Path, help="serverdocker source checkout")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(args.cloud_root.resolve()))
    # These modules have no automatic config/secret or network initialization.
    from flask import Flask
    from cloud.cardputer.api import register
    from cloud.cardputer.store import Store

    with tempfile.TemporaryDirectory(prefix="cardputer-gateway-acceptance-") as directory:
        root = Path(directory)
        reports = root / "reports"
        reports.mkdir()
        writer = sqlite3.connect(reports / "telegram_outbox.sqlite")
        try:
            writer.execute("PRAGMA journal_mode=WAL")
            writer.execute("CREATE TABLE profit_snapshot(strategy TEXT,pair TEXT,observed_at REAL,"
                           "mtm_quote REAL,equity REAL,drawdown_pct REAL,payload_json TEXT,"
                           "PRIMARY KEY(strategy,pair,observed_at))")
            for strategy, pair, _, _ in ROBOTS:
                for hours, value in ((168, 100), (24, 110)):
                    writer.execute("INSERT INTO profit_snapshot VALUES(?,?,?,?,?,?,?)", (
                        strategy, pair, NOW - hours * 3600 - 60, value, 200, 0.5, "{}"))
                for index in range(73):
                    writer.execute("INSERT INTO profit_snapshot VALUES(?,?,?,?,?,?,?)", (
                        strategy, pair, NOW - 43200 + index * 600 - 60,
                        120 + index / 6, 200, 0.5, "{}"))
            writer.commit()
            status(reports, NOW - 60)

            session = Mock()
            session.headers = {}
            calls = []

            def get(url, **options):
                calls.append((url, options))
                start, end = options["params"]["startTime"], options["params"]["endTime"]
                base = 60000 if options["params"]["symbol"].startswith("BTC") else 2500
                rows = []
                # Includes an open candle to test the actual adapter's rejection.
                for opened in range((start // 300000) * 300000, end + 300000, 300000):
                    rows.append([opened, "1", "1", "1", str(base + len(rows)), "1",
                                 opened + 299999, "1", 1, "1", "1", "0"])
                response = Mock(status_code=200)
                response.iter_content.return_value = [json.dumps(rows).encode()]
                return response

            session.get.side_effect = get
            prices = PublicKlinePrices(session=session)
            collector = SnapshotCollector(reports, history=HistoryCollector(reports / "telegram_outbox.sqlite", prices=prices))
            device_file = root / "synthetic-device.json"
            token_file = root / "synthetic-telemetry"
            device_file.write_text(json.dumps({"acceptance": DEVICE}), encoding="utf-8")
            token_file.write_text(TELEMETRY, encoding="utf-8")
            settings = {"data_dir": str(root / "cloud-data"), "device_tokens_file": str(device_file),
                        "telemetry_token_file": str(token_file)}
            clock = [NOW]
            store = Store(settings["data_dir"], clock=lambda: clock[0])
            app = Flask("synthetic-cardputer-acceptance")
            app.config.update(TESTING=True)
            register(app, settings, store)
            client = app.test_client()
            largest = 0

            def collect_and_upload(at):
                clock[0] = at
                # Canonical Reader uses wall time internally. No production env
                # is loaded; all modules see this synthetic test clock.
                with patch("management_bot.clients.time.time", return_value=at):
                    sample = collector.collect(now=at)
                    response = client.post(BASE + "/telemetry/trading", json=sample, headers=auth(TELEMETRY))
                assert response.status_code == 202, response.json
                assert len(json.dumps(sample, ensure_ascii=False, separators=(",", ":")).encode()) < 65536
                return sample

            def history(identity):
                nonlocal largest
                cursor, profit, price = 0, [], []
                while cursor is not None:
                    response = client.get(BASE + "/trading/history/" + identity + "?cursor=" + str(cursor), headers=auth())
                    assert response.status_code == 200, response.json
                    assert len(response.data) <= 8192
                    largest = max(largest, len(response.data))
                    body = response.json
                    profit.extend(body["profit_points"])
                    price.extend(body["price_points"])
                    previous, cursor = cursor, body["next_cursor"]
                    assert cursor is None or cursor > previous
                body["profit_points"], body["price_points"] = profit, price
                return body

            initial = collect_and_upload(NOW)
            overview = client.get(BASE + "/trading", headers=auth()).json
            for row in overview["robots"]:
                assert row["profit"] == {"4h": 4, "24h": 22, "7d": 32, "all": 132}
                chart = history(row["id"])
                assert chart["quote_asset"] == row["quote_asset"] == row["pair"].split("-")[-1]
                assert len(chart["profit_points"]) == len(chart["price_points"]) == 73
                assert chart["profit_points"][0]["value"] == 0 and chart["profit_points"][-1]["value"] == 12
                assert chart["profit_observed_at"] == NOW - 60 and chart["price_observed_at"] == NOW - .001
                assert chart["window_complete"] and chart["profit_data_state"] == chart["price_data_state"] == "FRESH"
                assert chart["pauses"] == [], "current normal state invented a historical pause"
                assert chart["status_coverage_start_at"] == NOW - 60
                assert chart["status_coverage_end_at"] == NOW - 60
            # A missing baseline must not become zero; another unit's point gap
            # must not affect live 4h totals or the other units.
            writer.execute("DELETE FROM profit_snapshot WHERE strategy='grid' AND pair='BTC-FDUSD' AND observed_at=?", (NOW - 43200 - 60,))
            writer.execute("DELETE FROM profit_snapshot WHERE strategy='grid' AND pair='ETH-FDUSD' AND observed_at=?", (NOW - 43200 + 36 * 600 - 60,))
            writer.commit()
            status(reports, NOW, stopped=True)
            collect_and_upload(NOW + 60)
            assert all(p["value"] is None for p in history("grid:BTC-FDUSD")["profit_points"])
            curve = history("grid:ETH-FDUSD")["profit_points"]
            assert curve[36]["value"] is None and curve[37]["value"] is not None
            status(reports, NOW + 60, stopped=True)
            latest = collect_and_upload(NOW + 120)
            paused = history("grid:BTC-FDUSD")
            assert [(p["start_at"], p["end_at"]) for p in paused["pauses"]] == [(NOW, NOW + 60)]
            assert paused["pauses"][0]["scope"] == "all"
            # 300s source freshness is independent of a newer transport frame.
            clock[0] = NOW + 240
            assert history("dca:BTC-USDT")["profit_data_state"] == "FRESH"
            clock[0] += 1
            chart = history("dca:BTC-USDT")
            assert chart["profit_data_state"] == "STALE" and chart["price_data_state"] == "FRESH"
            assert chart["window_complete"] is False and chart["profit_points"][-1]["value"] == 12
            overview = client.get(BASE + "/trading", headers=auth()).json
            assert all(row["profit"]["4h"] is None for row in overview["robots"])
            clock[0] = NOW + 301
            duplicate = client.post(BASE + "/telemetry/trading", json=latest, headers=auth(TELEMETRY))
            assert duplicate.json["duplicate"] is True
            assert store.cache_get("trading")[1] == NOW + 120
            assert history("dca:BTC-USDT")["price_data_state"] == "STALE"
            # Invalid latest MTM must not make the optional history validator
            # discard this robot's otherwise fresh public price series.
            writer.execute("UPDATE profit_snapshot SET observed_at=? WHERE strategy='grid' AND pair='BTC-FDUSD' AND observed_at=?",
                           (NOW + 451, NOW - 60))
            writer.commit()
            collect_and_upload(NOW + 420)
            chart = history("grid:BTC-FDUSD")
            assert chart["profit_observed_at"] is None
            assert all(p["value"] is None for p in chart["profit_points"])
            assert chart["price_data_state"] == "FRESH" and chart["price_points"][-1]["value"] is not None
            assert client.get(BASE + "/trading/history/dca:BTC-USDT", headers=auth(TELEMETRY)).status_code == 401
            assert client.post(BASE + "/telemetry/trading", json=initial, headers=auth()).status_code == 401
            assert len(calls) == 16 and all(call[1]["allow_redirects"] is False and call[1]["verify"] is True for call in calls)
            print(json.dumps({"collector_gateway_acceptance": "passed", "robots": 4,
                              "curve_points": 73, "largest_history_response_bytes": largest,
                              "profit_windows": ["4h", "24h", "7d"], "currency_isolation": True,
                              "source_age_300_boundary": True, "real_observation_pauses": True,
                              "production_connections": 0}))
        finally:
            writer.close()


if __name__ == "__main__":
    main()
