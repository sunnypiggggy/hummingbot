"""Read-only restore/economic checks; receipts live beside the private backup."""
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def query(database, sql):
    return json.loads(subprocess.check_output([
        "docker", "exec", "hummingbot-api-postgres", "psql", "-U", "hbot",
        "-d", database, "-At", "-v", "ON_ERROR_STOP=1", "-c", sql,
    ], text=True))


def fingerprint(database):
    schema = "binance_stocks_paper"
    result = {}
    for table, projection in {
        "paper_runs": "run_id,initial_usdc,cash_balance,status",
        "inventory_lots": "*",
        "paper_trades": "*",
        "paper_orders": "*",
    }.items():
        result[table] = query(database, f"""
          SELECT json_build_object('rows',count(*),'sha',
            md5(coalesce(string_agg(value,'' ORDER BY value),'')))
          FROM (SELECT row_to_json(t)::text AS value FROM
            (SELECT {projection} FROM {schema}.{table}) t) records
        """)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--stage", choices=("restored", "before_cutover", "after_cutover", "reclaimed"), required=True)
    args = parser.parse_args()
    root = args.destination.resolve()
    if "ops-backups" not in root.parts or not (root/"backup_receipt.json").is_file():
        raise ValueError("verified private backup directory required")
    production = fingerprint("hummingbot_stocks")
    receipt = {"stage": args.stage, "verified_at": datetime.now(timezone.utc).isoformat(),
               "production_economics": production}
    if args.stage == "restored":
        receipt["restored_economics"] = fingerprint("hummingbot_stocks_retention_restore")
        receipt["financial_restore_matches"] = receipt["restored_economics"] == production
        receipt["restored_quotes"] = query("hummingbot_stocks_retention_restore", """
          SELECT json_build_object('rows',count(*),'first',min(event_time),'last',max(event_time))
          FROM binance_stocks_paper.paper_quote_events
        """)
        if not receipt["financial_restore_matches"]:
            raise RuntimeError("restore financial evidence differs; inspect before reclaim")
    elif args.stage in {"after_cutover", "reclaimed"}:
        baseline = json.loads((root/"before_cutover.json").read_text())
        receipt["economics_unchanged"] = production == baseline["production_economics"]
        if not receipt["economics_unchanged"]:
            raise RuntimeError("economic state changed; independently reconcile real fills before acceptance")
    (root/f"{args.stage}.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    print(json.dumps({"stage": args.stage, "verified_at": receipt["verified_at"],
                      "financial_restore_matches": receipt.get("financial_restore_matches"),
                      "economics_unchanged": receipt.get("economics_unchanged")}))


if __name__ == "__main__":
    main()
