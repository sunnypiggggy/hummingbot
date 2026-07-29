#!/usr/bin/env python3
"""Non-destructive live wiring check for DCA Guard instance resolution."""

from __future__ import annotations

import json

from dca_live_common import LIVE_PAIRS
from dca_live_guard import Guard


def main() -> int:
    guard = Guard()
    checks = {}
    healthy = True
    for pair, spec in LIVE_PAIRS.items():
        resolved = guard._actual_instances(spec.bot_name)
        database = guard._database(spec.bot_name)
        exact = resolved == [spec.bot_name]
        database_exists = database is not None and database.exists()
        checks[pair] = {
            "logical_name": spec.bot_name,
            "resolved_instances": resolved,
            "exact_singleton": exact,
            "database_exists": database_exists,
        }
        healthy = healthy and exact and database_exists
    checks["stop_response_validation"] = Guard._stop_succeeded(
        {"status": "success", "response": {"success": True}}
    )
    healthy = healthy and checks["stop_response_validation"]
    print(json.dumps({"healthy": healthy, "checks": checks}, indent=2))
    return 0 if healthy else 2


if __name__ == "__main__":
    raise SystemExit(main())
