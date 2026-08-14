#!/usr/bin/env python3
"""Restore the DCA ownership ledger from a previously accepted live preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


CONFIRMATION = "RESTORE-DCA-MANAGED-INVENTORY"
PAIR_ASSETS = {"BTC-USDT": "BTC", "ETH-USDT": "ETH"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()
    if args.confirm != CONFIRMATION:
        raise ValueError(f"explicit --confirm {CONFIRMATION} is required")
    raw = args.preflight.read_bytes()
    source = json.loads(raw.decode("utf-8"))
    if source.get("account") != "binance_live_dca_200":
        raise ValueError("DCA preflight account mismatch")
    requirements = source.get("requirements", {})
    pairs = {}
    for pair, asset in PAIR_ASSETS.items():
        amount = str(requirements.get(asset, ""))
        if not amount or float(amount) <= 0:
            raise ValueError(f"preflight has no positive ownership for {asset}")
        pairs[pair] = {
            "base_asset": asset,
            "managed_base": amount,
            "verified_at": source["checked_at"],
        }
    payload = {
        "schema_version": 1,
        "account": source["account"],
        "pairs": pairs,
        "restored_at": datetime.now(timezone.utc).isoformat(),
        "source_preflight_sha256": hashlib.sha256(raw).hexdigest(),
        "restore_confirmation": CONFIRMATION,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
