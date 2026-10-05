from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .collector import SnapshotCollector
from .config import Config
from .history import HistoryCollector, PublicKlinePrices
from .publisher import PublishError, TelemetryPublisher


def main(argv=None) -> int:
    # Windows redirected output otherwise inherits its legacy code page.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Read-only Cardputer trading telemetry (dry-run by default)")
    parser.add_argument("--config", type=Path, help="JSON configuration containing paths, never token values")
    parser.add_argument("--once", action="store_true", help="collect one snapshot and exit")
    parser.add_argument("--publish", action="store_true", help="explicitly POST telemetry using its dedicated token file")
    args = parser.parse_args(argv)
    try:
        config = Config.load(args.config)
    except (OSError, ValueError, TypeError, UnicodeError):
        print("monitor configuration is unavailable or invalid", file=sys.stderr)
        return 2
    prices = PublicKlinePrices(timeout=config.price_timeout_seconds) if args.publish and config.price_provider == "binance-public" else None
    history = HistoryCollector(config.reports_dir / "telegram_outbox.sqlite", prices=prices)
    collector = SnapshotCollector(config.reports_dir, history=history)
    publisher = TelemetryPublisher(config.telemetry_url, config.token_file, timeout=config.request_timeout_seconds) if args.publish else None
    try:
        while True:
            started = time.monotonic()
            snapshot = collector.collect()
            failed = False
            if publisher is None:
                print(json.dumps(snapshot, ensure_ascii=False, allow_nan=False), flush=True)
            else:
                try:
                    publisher.publish(snapshot)
                except PublishError as exc:
                    failed = True
                    print(str(exc), file=sys.stderr, flush=True)
                else:
                    print(f"telemetry published sample={snapshot['sample_id']}", file=sys.stderr, flush=True)
            if args.once:
                return 1 if failed else 0
            # No durable queue: after a failure the next cycle collects a new
            # latest sample, so reconnection cannot replay an old backlog.
            time.sleep(max(0, config.interval_seconds - (time.monotonic() - started)))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
