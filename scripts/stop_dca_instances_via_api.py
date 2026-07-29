#!/usr/bin/env python3
"""Stop explicitly named DCA instances through Hummingbot API with cancellation."""

from __future__ import annotations

import argparse
import json
import os

import requests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("instances", nargs="*")
    parser.add_argument("--show-status", action="store_true")
    parser.add_argument("--show-status-raw", action="store_true")
    parser.add_argument("--show-status-structure", action="store_true")
    parser.add_argument("--show-api-paths", action="store_true")
    parser.add_argument("--show-api-schema", action="store_true")
    parser.add_argument("--show-live-preflight", action="store_true")
    args = parser.parse_args()
    base_url = os.getenv("HUMMINGBOT_API_URL", "http://hummingbot-api:8000").rstrip("/")
    session = requests.Session()
    session.auth = (os.environ["USERNAME"], os.environ["PASSWORD"])
    if args.show_status_structure:
        status = session.get(f"{base_url}/bot-orchestration/status", timeout=30)
        status.raise_for_status()
        payload = status.json()
        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        print(json.dumps({
            name: {
                "status": value.get("status"),
                "recently_active": value.get("recently_active"),
                "source": value.get("source"),
                "performance": value.get("performance"),
                "keys": sorted(value),
            }
            for name, value in data.items()
            if name.startswith("dca-live-")
        }, indent=2))
        if not args.instances:
            return 0
    if args.show_status_raw:
        status = session.get(f"{base_url}/bot-orchestration/status", timeout=30)
        status.raise_for_status()
        print(json.dumps(status.json(), indent=2))
        if not args.instances and not any((
            args.show_status, args.show_api_paths, args.show_api_schema,
            args.show_live_preflight,
        )):
            return 0
    if args.show_live_preflight:
        containers = session.get(
            f"{base_url}/docker/active-containers",
            params={"name_filter": "dca-live-"}, timeout=30,
        )
        containers.raise_for_status()
        orders = {}
        for pair in ("BTC-USDT", "ETH-USDT"):
            response = session.post(
                f"{base_url}/trading/orders/active",
                json={
                    "limit": 1000,
                    "account_names": ["binance_live_dca_200"],
                    "connector_names": ["binance"],
                    "trading_pairs": [pair],
                },
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            values = payload.get("data", payload) if isinstance(payload, dict) else payload
            orders[pair] = len(values) if isinstance(values, list) else None
        print(json.dumps({"containers": containers.json(), "active_order_counts": orders}, indent=2))
        if not args.instances and not any((args.show_status, args.show_api_paths, args.show_api_schema)):
            return 0
    if args.show_api_schema:
        response = session.get(f"{base_url}/openapi.json", timeout=30)
        response.raise_for_status()
        document = response.json()
        selected_paths = {
            path: document.get("paths", {}).get(path, {})
            for path in (
                "/docker/active-containers",
                "/docker/stop-container/{container_name}",
                "/trading/orders/active",
                "/trading/{account_name}/{connector_name}/orders/{client_order_id}/cancel",
            )
        }
        schemas = document.get("components", {}).get("schemas", {})
        print(json.dumps(
            {
                "paths": selected_paths,
                "schemas": {
                    name: schema
                    for name, schema in schemas.items()
                    if any(token in name.lower() for token in ("order", "cancel"))
                },
            },
            indent=2,
        ))
        if not args.instances and not args.show_status and not args.show_api_paths:
            return 0
    if args.show_api_paths:
        response = session.get(f"{base_url}/openapi.json", timeout=30)
        response.raise_for_status()
        paths = response.json().get("paths", {})
        print(json.dumps(
            {
                path: sorted(methods)
                for path, methods in paths.items()
                if any(
                    token in path.lower()
                    for token in ("docker", "stop", "bot-orchestration", "cancel", "trading/order")
                )
            },
            indent=2,
        ))
        if not args.instances and not args.show_status:
            return 0
    if args.show_status:
        status = session.get(f"{base_url}/bot-orchestration/status", timeout=30)
        status.raise_for_status()
        payload = status.json()
        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        print(json.dumps(
            {
                name: {
                    "status": value.get("status"),
                    "source": value.get("source"),
                    "recently_active": value.get("recently_active"),
                }
                for name, value in data.items()
                if "dca-live-" in name
            },
            indent=2,
        ))
        if not args.instances:
            return 0
    results = []
    for instance in args.instances:
        response = session.post(
            f"{base_url}/bot-orchestration/stop-bot",
            json={
                "bot_name": instance,
                "skip_order_cancellation": False,
                "async_backend": False,
            },
            timeout=90,
        )
        results.append(
            {
                "instance": instance,
                "status_code": response.status_code,
                "response": response.json() if response.content else {},
            }
        )
        response.raise_for_status()
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
