"""Consistent rollout backups; never prints credentials or database content."""
import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--destination",type=Path,required=True)
    parser.add_argument("--postgres",action="store_true")
    args=parser.parse_args()
    root=args.root.resolve()
    out=args.destination.resolve()
    if not out.is_relative_to(root/"ops-backups"):
        raise ValueError("backup must stay in the explicit ops-backups directory")
    os.umask(0o077)
    out.mkdir(parents=True,exist_ok=True)
    source=out/"source"
    source.mkdir(exist_ok=True)
    # Preserve old images in the receipt, not by writing over their tags.
    receipt={"containers":[]}
    for name in ("dca-live-report","sunnypiggy-trade-bot","binance-stocks-runtime",
                 "grid-live-fdusd-400","dca-live-btcusdt-200","dca-live-ethusdt-200","grid-live-guard","dca-live-guard"):
        record=json.loads(subprocess.check_output(["docker","inspect",name,"--format",
            '{"name":{{json .Name}},"id":{{json .Id}},"image":{{json .Image}},"started":{{json .State.StartedAt}}}'],text=True))
        receipt["containers"].append(record)
    for relative in ("live_guard","management_bot","stocks_runtime","docker-compose.yml",
                     "Dockerfile.dca-live-guard","Dockerfile.trading-management-bot","Dockerfile.binance-stocks-runtime"):
        path=root/relative
        if path.is_dir():
            shutil.copytree(path,source/relative,dirs_exist_ok=True,ignore=shutil.ignore_patterns("__pycache__"))
        elif path.exists():
            shutil.copy2(path,source/relative)
    databases=set(root.glob("api-files/bots/instances/*/data/*.sqlite"))
    for folder in ("account-inventory-data","dca-live-data/telegram","telegram-management-data"):
        databases.update((root/folder).rglob("*.sqlite"))
    receipt["sqlite"]=[]
    for path in sorted(databases):
        target=out/"sqlite"/path.relative_to(root)
        target.parent.mkdir(parents=True,exist_ok=True)
        try:
            with closing(sqlite3.connect(path.resolve().as_uri()+"?mode=ro",uri=True,timeout=3)) as db, closing(sqlite3.connect(target)) as copy:
                db.backup(copy,pages=1000,sleep=.05)
                if copy.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise RuntimeError("SQLite backup integrity failed")
        except sqlite3.Error as exc:
            raise RuntimeError(f"consistent backup failed for {path.relative_to(root)}: {exc}") from exc
        receipt["sqlite"].append({"path":str(path.relative_to(root)),"sha256":digest(target),"bytes":target.stat().st_size})
    for folder in ("grid-live-fdusd-data","dca-live-data","account-inventory-data"):
        for path in (root/folder).glob("*.json"):
            target=out/"state"/folder/path.name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
    if args.postgres:
        if shutil.disk_usage(out).free < 20*1024**3:
            raise RuntimeError("insufficient restore/backup/WAL headroom")
        target=out/"hummingbot_stocks.dump"
        with target.open("wb") as stream:
            subprocess.run(["docker","exec","hummingbot-api-postgres","nice","-n","19",
                            "pg_dump","-U","hbot","-d","hummingbot_stocks","-Fc","-Z","1"],stdout=stream,check=True)
        receipt["postgres"]={"path":target.name,"sha256":digest(target),"bytes":target.stat().st_size}
    (out/"backup_receipt.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    print(json.dumps({"backup":str(out),"sqlite_count":len(receipt["sqlite"]),"postgres":receipt.get("postgres")}))


if __name__ == "__main__":
    main()
