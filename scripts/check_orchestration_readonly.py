#!/usr/bin/env python3
"""Read the local Hummingbot orchestration status without exposing credentials."""

import json
import os

import requests


def main() -> int:
    base = os.getenv("HUMMINGBOT_API_URL", "http://hummingbot-api:8000").rstrip("/")
    response = requests.get(
        f"{base}/bot-orchestration/status",
        auth=(os.environ["USERNAME"], os.environ["PASSWORD"]),
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    bot_name = os.getenv("GRID_STATUS_NAME", "")
    if bot_name:
        bot = payload.get("data", {}).get(bot_name)
        payload = {
            "api_status": payload.get("status"),
            "bot_name": bot_name,
            "found": bot is not None,
            "status": bot.get("status") if bot else None,
            "recently_active": bot.get("recently_active") if bot else None,
            "source": bot.get("source") if bot else None,
            "performance": bot.get("performance") if bot else None,
            "error_logs": bot.get("error_logs") if bot else None,
        }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
