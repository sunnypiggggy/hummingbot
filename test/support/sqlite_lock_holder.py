"""Hold a real SQLite write lock inside the isolated scenario volume."""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("--seconds", type=float, default=8)
    args = parser.parse_args()
    deadline = time.time() + 30
    while not args.database.is_file() and time.time() < deadline:
        time.sleep(0.1)
    if not args.database.is_file():
        raise FileNotFoundError(args.database)
    connection = sqlite3.connect(args.database, timeout=15)
    try:
        connection.execute("BEGIN IMMEDIATE")
        print("SQLITE_LOCK_ACQUIRED", flush=True)
        time.sleep(max(args.seconds, 0))
        connection.rollback()
        print("SQLITE_LOCK_RELEASED", flush=True)
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
