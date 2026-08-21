from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import requests


class ServiceError(RuntimeError):
    pass


class JsonClient:
    def __init__(self, base_url: str, username: str, password: str, timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.auth = (username, password)
        self.timeout = timeout

    def request(self, method: str, path: str, payload: Optional[dict] = None) -> Any:
        try:
            response = self.session.request(
                method.upper(), f"{self.base_url}{path}", json=payload, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise ServiceError(f"{method.upper()} {path} connection failed: {type(exc).__name__}") from exc
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text[:300])
            except Exception:
                detail = response.text[:300]
            raise ServiceError(f"{method.upper()} {path} failed ({response.status_code}): {detail}")
        if not response.content:
            return {}
        return response.json()


class HummingbotClient(JsonClient):
    def status(self) -> dict:
        value = self.request("GET", "/bot-orchestration/status")
        return value if isinstance(value, dict) else {"data": {}}

    def stop(self, bot_name: str) -> dict:
        return self.request("POST", "/bot-orchestration/stop-bot", {
            "bot_name": bot_name,
            "skip_order_cancellation": False,
            "async_backend": False,
        })

    def start(self, definition: dict[str, str]) -> dict:
        return self.request("POST", "/bot-orchestration/start-bot", {
            "bot_name": definition["bot_name"],
            "log_level": "INFO",
            "script": definition["script"],
            "conf": definition["conf"],
            "async_backend": False,
        })

    def restart(self, definition: dict[str, str]) -> dict:
        self.stop(definition["bot_name"])
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            raw = self.status().get("data", {})
            item = raw.get(definition["bot_name"], {}) if isinstance(raw, dict) else {}
            if str(item.get("status", "")).lower() not in {"running", "starting", "stopping"}:
                break
            time.sleep(1)
        else:
            raise ServiceError("bot did not stop within 30 seconds; restart was not attempted")
        return self.start(definition)


class StocksClient(JsonClient):
    def health(self) -> dict:
        return self.request("GET", "/stocks/health")

    def account(self) -> dict:
        return self.request("GET", "/stocks/account-summary")

    def positions(self) -> dict:
        return self.request("GET", "/stocks/managed-positions")

    def whitelist(self) -> list[dict]:
        value = self.request("GET", "/stocks/whitelist")
        return list(value.get("items", [])) if isinstance(value, dict) else []

    def put_whitelist(self, symbol: str, enabled: bool, max_position: str) -> dict:
        return self.request("PUT", f"/stocks/whitelist/{symbol}", {
            "symbol": symbol,
            "enabled": enabled,
            "max_position_notional": max_position,
        })

    def delete_whitelist(self, symbol: str) -> dict:
        return self.request("DELETE", f"/stocks/whitelist/{symbol}")

    def limits(self) -> dict:
        return self.request("GET", "/stocks/limits")

    def put_limits(self, order: str, symbol: str, total: str) -> dict:
        return self.request("PUT", "/stocks/limits", {
            "max_order_notional": order,
            "max_symbol_exposure": symbol,
            "max_managed_exposure": total,
        })

    def quote(self, symbol: str) -> dict:
        value = self.request("GET", f"/stocks/quotes/{symbol}")
        return value if isinstance(value, dict) else {}

    def preview(self, config: dict) -> dict:
        return self.request("POST", "/stocks/executors/preview", {
            "executor_config": config,
            "controller_id": "telegram-management-bot",
        })

    def create(self, config: dict) -> dict:
        return self.request("POST", "/stocks/executors", {
            "executor_config": config,
            "controller_id": "telegram-management-bot",
        })

    def executor(self, executor_id: str) -> dict:
        return self.request("GET", f"/stocks/executors/{executor_id}")

    def executors(self, active_only: bool = True) -> list[dict]:
        value = self.request("GET", f"/stocks/executors?active_only={'true' if active_only else 'false'}")
        return list(value.get("items", [])) if isinstance(value, dict) else []

    def pause(self, executor_id: str) -> dict:
        return self.request("POST", f"/stocks/executors/{executor_id}/cancel")

    def close(self, executor_id: str) -> dict:
        return self.request("POST", f"/stocks/executors/{executor_id}/close")

    def reduce(self, executor_id: str, amount: str, request_id: str) -> dict:
        return self.request("POST", f"/stocks/executors/{executor_id}/reduce", {
            "amount": amount, "request_id": request_id,
        })


class ContractReader:
    def __init__(self, grid_path: Path, dca_path: Path, inventory_path: Path):
        self.paths = {"grid": grid_path, "dca": dca_path, "inventory": inventory_path}

    @staticmethod
    def _read(path: Path) -> tuple[dict, Optional[str]]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return (value if isinstance(value, dict) else {}), None
        except Exception as exc:
            return {}, f"{path.name}: {type(exc).__name__}"

    def snapshot(self) -> dict:
        result: dict[str, Any] = {"generated_at": time.time(), "sources": {}, "errors": []}
        for name, path in self.paths.items():
            value, error = self._read(path)
            result["sources"][name] = value
            if error:
                result["errors"].append(error)
        return result

    @staticmethod
    def resume_allowed(strategy: str, snapshot: dict) -> tuple[bool, str]:
        source = snapshot.get("sources", {}).get(strategy, {})
        if not source:
            return False, "风控合同不可用"
        last_success = float(source.get("last_success_at", 0) or 0)
        if last_success and time.time() - last_success > 180:
            return False, "风控合同已过期"
        text = json.dumps(source, ensure_ascii=False, default=str).lower()
        for token, reason in (
            ('"latched"', "存在锁存故障"),
            ('"fail_closed"', "存在完整性 Fail-Closed"),
            ('"ownership_deficit"', "库存归属存在缺口"),
        ):
            if token in text:
                return False, reason
        return True, "当前合同未发现恢复阻塞"
