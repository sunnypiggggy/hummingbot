from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from datetime import datetime
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

    def paper_summary(self) -> dict:
        value = self.request("GET", "/stocks/paper/summary")
        return value if isinstance(value, dict) else {}

    def paper_trades(self, limit: int = 20) -> list[dict]:
        value = self.request("GET", f"/stocks/paper/trades?limit={max(1, min(limit, 100))}")
        return list(value.get("items", [])) if isinstance(value, dict) else []

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

    def market_status(self, symbol: str) -> dict:
        value = self.request("GET", f"/stocks/market-status/{symbol}")
        return value if isinstance(value, dict) else {}

    def preview(self, config: dict) -> dict:
        return self.request("POST", "/stocks/executors/preview", {
            "executor_config": config,
            "controller_id": "telegram-management-bot",
        })

    def preview_order(self, payload: dict) -> dict:
        return self.request("POST", "/stocks/order-executors/preview", payload)

    def preview_position(self, payload: dict) -> dict:
        return self.request("POST", "/stocks/position-executors/preview", payload)

    def create(self, config: dict) -> dict:
        return self.request("POST", "/stocks/executors", {
            "executor_config": config,
            "controller_id": "telegram-management-bot",
        })

    def create_order(self, payload: dict) -> dict:
        return self.request("POST", "/stocks/order-executors", payload)

    def create_position(self, payload: dict) -> dict:
        return self.request("POST", "/stocks/position-executors", payload)

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


class OperationsReportReader:
    """Read the report service's canonical trading status and owned-MTM snapshots."""

    def __init__(self, status_path: Path, profit_db_path: Path, max_age_seconds: int = 300):
        self.status_path = status_path
        self.profit_db_path = profit_db_path
        self.max_age_seconds = max_age_seconds

    @staticmethod
    def _iso_timestamp(value: Any) -> float:
        text = str(value or "").strip()
        if not text:
            return 0.0
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0

    def status(self) -> dict:
        try:
            payload = json.loads(self.status_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ServiceError(f"运行状态快照不可用：{type(exc).__name__}") from exc
        robots = payload.get("robots", []) if isinstance(payload, dict) else []
        if not isinstance(robots, list) or not robots:
            raise ServiceError("运行状态快照没有机器人数据")
        generated_at = self._iso_timestamp(payload.get("generated_at"))
        age = time.time() - generated_at if generated_at else float("inf")
        if age > self.max_age_seconds:
            raise ServiceError(f"运行状态快照已过期（{int(age)}秒）")
        return {"generated_at": generated_at, "age_seconds": max(0.0, age), "robots": robots}

    def profits(self) -> dict:
        try:
            uri = self.profit_db_path.resolve().as_uri() + "?mode=ro"
            connection = sqlite3.connect(uri, uri=True, timeout=3)
            try:
                rows = connection.execute(
                    "SELECT p.strategy,p.pair,p.observed_at,p.mtm_quote,p.equity,"
                    "p.drawdown_pct,p.payload_json FROM profit_snapshot p "
                    "JOIN (SELECT strategy,pair,MAX(observed_at) AS latest "
                    "FROM profit_snapshot GROUP BY strategy,pair) x "
                    "ON p.strategy=x.strategy AND p.pair=x.pair AND p.observed_at=x.latest "
                    "ORDER BY p.strategy,p.pair"
                ).fetchall()
            finally:
                connection.close()
        except Exception as exc:
            raise ServiceError(f"收益快照不可用：{type(exc).__name__}") from exc
        if not rows:
            raise ServiceError("收益快照没有机器人数据")
        result = []
        newest = 0.0
        for strategy, pair, observed_at, mtm, equity, drawdown, payload_json in rows:
            try:
                payload = json.loads(payload_json or "{}")
            except (TypeError, json.JSONDecodeError):
                payload = {}
            observed = float(observed_at or 0)
            newest = max(newest, observed)
            result.append({
                "strategy": str(strategy), "pair": str(pair), "observed_at": observed,
                "mtm_quote": mtm, "equity": equity, "drawdown_pct": drawdown,
                "profit": payload.get("profit", {}) if isinstance(payload, dict) else {},
            })
        age = time.time() - newest if newest else float("inf")
        if age > self.max_age_seconds:
            raise ServiceError(f"收益快照已过期（{int(age)}秒）")
        return {"observed_at": newest, "age_seconds": max(0.0, age), "robots": result}


class ParameterCatalogReader:
    """Read the report service's sanitized parameter and evidence contracts."""

    def __init__(self, catalog_path: Path, evidence_catalog_path: Path,
                 report_root: Path, max_age_seconds: int = 300):
        self.catalog_path = catalog_path
        self.evidence_catalog_path = evidence_catalog_path
        self.report_root = report_root.resolve()
        self.max_age_seconds = max_age_seconds

    @staticmethod
    def _load(path: Path) -> dict:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ServiceError(f"参数合同不可用：{type(exc).__name__}") from exc
        if not isinstance(value, dict):
            raise ServiceError("参数合同格式无效")
        return value

    @staticmethod
    def _iso_timestamp(value: Any) -> float:
        text = str(value or "").strip()
        if not text:
            return 0.0
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0

    def catalog(self) -> dict:
        value = self._load(self.catalog_path)
        generated = self._iso_timestamp(value.get("generated_at"))
        age = time.time() - generated if generated else float("inf")
        if age > self.max_age_seconds:
            raise ServiceError(f"参数合同已过期（{int(age)}秒）")
        value["age_seconds"] = max(0.0, age)
        return value

    def evidence(self) -> dict:
        return self._load(self.evidence_catalog_path)

    def evidence_set(self, set_id: str) -> Optional[dict]:
        return next((item for item in self.evidence().get("sets", [])
                     if item.get("evidence_set_id") == set_id), None)

    def attachment(self, set_id: str, strategy: str, asset: str, window: str) -> dict:
        evidence = self.evidence_set(set_id)
        if not evidence:
            raise ServiceError("证据集不存在")
        item = next((row for row in evidence.get("attachments", [])
                     if row.get("strategy") == strategy
                     and str(row.get("pair", "")).split("-", 1)[0] == asset
                     and row.get("window") == window), None)
        if not item:
            raise ServiceError("所选证据图片不存在")
        path = (self.report_root / str(item.get("relative_path", ""))).resolve()
        try:
            path.relative_to(self.report_root)
        except ValueError as exc:
            raise ServiceError("证据路径越界") from exc
        if not path.is_file():
            raise ServiceError("证据文件缺失")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item.get("sha256"):
            raise ServiceError("证据文件哈希不匹配")
        return {**item, "path": str(path), "notice": evidence.get("notice")}

    def history(self, digest_prefix: str) -> dict:
        catalog = self.catalog()
        match = next((item for item in catalog.get("history", [])
                      if str(item.get("catalog_sha256", "")).startswith(digest_prefix)), None)
        if not match:
            raise ServiceError("历史参数版本不存在")
        digest = str(match["catalog_sha256"])
        path = (self.report_root / "management" / "history" / f"{digest}.json").resolve()
        try:
            path.relative_to(self.report_root)
        except ValueError as exc:
            raise ServiceError("历史参数路径越界") from exc
        return self._load(path)
