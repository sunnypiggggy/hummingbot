"""OCI-local allowlisted history upkeep; default is a non-mutating preview."""
import argparse
import json
import sqlite3
import time
import os
from contextlib import contextmanager
from contextlib import closing
from pathlib import Path

RETENTION_DAYS = 370
# Financial flows and execution recovery state are intentionally absent.
RULES = {
    "instance": [("MarketData", "timestamp", None)],
    "inventory": [("events", "created_at", "delivered=1 AND kind IN "
                   "('inventory_unattributed_detected','inventory_dust_classified','inventory_reconciliation_recovered') "
                   "AND NOT EXISTS(SELECT 1 FROM inventory_episodes p WHERE "
                   "json_extract(events.payload,'$.episode_id')=p.episode_id)")],
    "bot": [("audit_events", "created_at", None), ("processed_updates", "processed_at", None)],
}


@contextmanager
def database_owner(path):
    """Do not create root-owned WAL/SHM beside a non-root bot database."""
    if hasattr(os,"geteuid") and os.geteuid()==0:
        stat=Path(path).stat()
        original_gid=os.getegid()
        os.setegid(stat.st_gid)
        os.seteuid(stat.st_uid)
        try:
            yield
        finally:
            os.seteuid(0)
            os.setegid(original_gid)
    else:
        yield


def maintain(path, role, *, now, apply=False):
    path = Path(path)
    with closing(sqlite3.connect(path.resolve().as_uri()+"?mode=ro", uri=True, timeout=1)) as db:
        tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        candidates = []
        for table, column, condition in RULES[role]:
            if table not in tables or column not in {r[1] for r in db.execute(f'PRAGMA table_info("{table}")')}:
                continue
            # Skip ANY incoming FK; no dependent-record guessing or cascade.
            if any(fk[2] == table for t in tables for fk in db.execute(f'PRAGMA foreign_key_list("{t}")')):
                continue
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='trigger' AND tbl_name=?",(table,)).fetchone():
                continue
            # Hummingbot MarketData.timestamp is SqliteDecimal(6), NOT ms.
            scale = 1_000_000 if role == "instance" else 1
            where = f'"{column}"<?' + (" AND " + condition if condition else "")
            cutoff = (now-RETENTION_DAYS*86400)*scale
            count = db.execute(f'SELECT count(*) FROM "{table}" WHERE {where}',(cutoff,)).fetchone()[0]
            candidates.append({"table":table,"where":where,"cutoff":cutoff,"eligible":count})
    result = {"path":str(path),"role":role,"retention_days":RETENTION_DAYS,"apply":apply,
              "candidates":candidates,"deleted":0,"protected_tables":sorted(set(tables)-{r["table"] for r in candidates})}
    if apply:
        with closing(sqlite3.connect(path, timeout=1)) as db:
            db.execute("PRAGMA foreign_keys=ON")
            for row in candidates:
                with db:
                    db.execute("BEGIN IMMEDIATE")
                    # Revalidate inside the transaction: schema can change after preview.
                    current_tables=[r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
                    if (any(fk[2]==row["table"] for t in current_tables for fk in db.execute(f'PRAGMA foreign_key_list("{t}")'))
                            or db.execute("SELECT 1 FROM sqlite_master WHERE type='trigger' AND tbl_name=?",(row["table"],)).fetchone()):
                        row["skipped"]="dependency_changed"
                        continue
                    result["deleted"] += db.execute(f'DELETE FROM "{row["table"]}" WHERE rowid IN '
                        f'(SELECT rowid FROM "{row["table"]}" WHERE {row["where"]} LIMIT 5000)',
                        (row["cutoff"],)).rowcount
            db.execute("PRAGMA wal_checkpoint(PASSIVE)")
            result["remaining"]={row["table"]:db.execute(
                f'SELECT count(*) FROM "{row["table"]}" WHERE {row["where"]}',(row["cutoff"],)).fetchone()[0]
                for row in candidates}
            result["reusable_bytes"]=db.execute("PRAGMA freelist_count").fetchone()[0]*db.execute("PRAGMA page_size").fetchone()[0]
    result["space_policy"]="freed pages reusable; no online full VACUUM"
    result["sqlite_bytes"] = sum(p.stat().st_size for p in
        (path,Path(str(path)+"-wal"),Path(str(path)+"-shm")) if p.exists())
    return result


def discover(root):
    root = Path(root).resolve()
    # Exact runtime filenames only: backups in data/ are not maintenance targets.
    instances = root/"api-files/bots/instances"
    runtime_files = {
        "grid-live-fdusd-400": "walk_forward_portfolio_grid_live_fdusd_400.sqlite",
        "dca-live-btcusdt-200": "dca-live-btcusdt-200.sqlite",
        "dca-live-ethusdt-200": "dca-live-ethusdt-200.sqlite",
    }
    for bot, filename in runtime_files.items():
        path = instances/bot/"data"/filename
        if path.exists() and not path.is_symlink() and path.resolve().is_relative_to(root):
            yield path,"instance"
    for path,role in ((root/"account-inventory-data/account_inventory.sqlite","inventory"),
                      (root/"telegram-management-data/management_bot.sqlite","bot")):
        if path.exists() and not path.is_symlink() and path.resolve().is_relative_to(root):
            yield path,role


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--apply",action="store_true")
    args=parser.parse_args()
    for path,role in discover(args.root):
        try:
            with database_owner(path):
                result=maintain(path,role,now=time.time(),apply=args.apply)
            print(json.dumps(result))
        except (sqlite3.Error,OSError) as exc:
            print(json.dumps({"path":str(path),"skipped":True,"reason":type(exc).__name__}))


if __name__ == "__main__":
    main()
