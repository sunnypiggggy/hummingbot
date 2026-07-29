#!/usr/bin/env python3
"""Read-only verification of final order states in DCA instance databases."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


TERMINAL_MARKERS = ("cancel", "failure", "expired", "completed", "filled")


def inspect_database(path: Path) -> dict:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
    try:
        rows = connection.execute(
            'SELECT id, last_status FROM "Order" ORDER BY creation_timestamp'
        ).fetchall()
        fill_count = connection.execute("SELECT COUNT(*) FROM TradeFill").fetchone()[0]
        executor_counts = connection.execute(
            "SELECT COALESCE(SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END), 0), "
            "COALESCE(SUM(CASE WHEN is_trading = 1 THEN 1 ELSE 0 END), 0) "
            "FROM Executors"
        ).fetchone()
    finally:
        connection.close()
    active = [
        {"order_id": str(order_id), "last_status": str(status)}
        for order_id, status in rows
        if not any(marker in str(status).lower() for marker in TERMINAL_MARKERS)
    ]
    counts: dict[str, int] = {}
    for _, status in rows:
        counts[str(status)] = counts.get(str(status), 0) + 1
    return {
        "database": str(path),
        "order_status_counts": counts,
        "non_terminal_count": len(active),
        "non_terminal_orders": active,
        "fill_count": int(fill_count),
        "active_executor_count": int(executor_counts[0]),
        "trading_executor_count": int(executor_counts[1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("databases", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([inspect_database(path) for path in args.databases], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
