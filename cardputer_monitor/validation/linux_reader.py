"""Synthetic acceptance helper mounted separately; not part of the image."""
import errno
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app")
from cardputer_monitor.collector import SnapshotCollector  # noqa: E402
from cardputer_monitor.history import HistoryCollector  # noqa: E402


def main():
    assert os.getuid() == 10001
    root = Path("/reports")
    assert (root / "telegram_outbox.sqlite-wal").exists()
    assert (root / "telegram_outbox.sqlite-shm").exists()
    try:
        (root / "should-never-exist").write_text("read-only probe")
    except OSError as exc:
        assert exc.errno in {errno.EROFS, errno.EACCES}
    else:
        raise AssertionError("report mount was writable")
    with sqlite3.connect((root / "telegram_outbox.sqlite").as_uri() + "?mode=ro", uri=True) as db:
        try:
            db.execute("UPDATE profit_snapshot SET mtm_quote=-999")
        except sqlite3.OperationalError as exc:
            assert "readonly" in str(exc).lower()
        else:
            raise AssertionError("read-only SQLite connection accepted a write")
    collector = SnapshotCollector(root, history=HistoryCollector(root / "telegram_outbox.sqlite"))
    good = 0
    modes = set()
    amounts = set()
    for _ in range(80):
        sample = collector.collect()
        rows = sample["robots"]
        if all(row["profit_data_state"] == "FRESH" for row in rows):
            values = [row["profit"]["all"] for row in rows]
            assert len(set(values)) == 1, "one read transaction mixed different writer commits"
            amounts.add(values[0])
            good += 1
            curves = [row["history"] for row in rows]
            assert all(curve["window_complete"] for curve in curves), "synthetic 12h baseline or points missing"
            assert len({curve["profit_points"][-1]["value"] for curve in curves}) == 1, "history mixed writer commits"
            assert all(len(curve["profit_points"]) == 73 for curve in curves)
            assert all(curve["status_coverage_start_at"] is None and not curve["pauses"] for curve in curves)
        if rows[0]["status_data_state"] == "FRESH":
            modes.add(rows[0]["status"])
        time.sleep(0.04)
    assert good >= 20, f"too few usable WAL snapshots: {good}"
    assert len(amounts) >= 2, "concurrent writer changes were never observed"
    assert {"NORMAL", "STOPPED"}.issubset(modes), "atomic JSON replacement was not reopened"
    print(json.dumps({
        "uid": os.getuid(), "wal_readonly": True, "reports_readonly": True,
        "consistent_profit_samples": good, "writer_versions_seen": len(amounts),
        "history_readonly_consistent": True,
        "atomic_json_modes": sorted(modes),
    }))


if __name__ == "__main__":
    main()
