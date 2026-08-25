from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_BOTS: dict[str, dict[str, str]] = {
    "grid": {
        "bot_name": "grid-live-fdusd-400",
        "script": "walk_forward_portfolio_grid_live",
        "conf": "walk_forward_portfolio_grid_live_fdusd_400",
    },
    "dca_btc": {
        "bot_name": "dca-live-btcusdt-200",
        "script": "v2_with_controllers",
        "conf": "dca-live-btcusdt-200",
    },
    "dca_eth": {
        "bot_name": "dca-live-ethusdt-200",
        "script": "v2_with_controllers",
        "conf": "dca-live-ethusdt-200",
    },
}


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


@dataclass(frozen=True)
class Settings:
    token_file: Path
    admin_user_id: int
    state_dir: Path
    health_path: Path
    hummingbot_api_url: str
    hummingbot_api_username: str
    hummingbot_api_password: str
    stocks_api_url: str
    stocks_api_username: str
    stocks_api_password: str
    grid_guard_path: Path
    dca_guard_path: Path
    inventory_path: Path
    approval_request_root: Path
    approval_evidence_root: Path
    approval_decision_root: Path
    mutations_enabled: bool
    bots: dict[str, dict[str, str]]
    stocks_paper_trading_enabled: bool = False
    stocks_live_trading_enabled: bool = False
    session_ttl_seconds: int = 900
    trading_status_path: Path = Path("/reports/trading_status.json")
    profit_snapshot_db_path: Path = Path("/reports/telegram_outbox.sqlite")
    current_runtime_errors_path: Path = Path("/reports/current_runtime_errors.json")
    operations_report_max_age_seconds: int = 300
    host_proc_stat_path: Path = Path("/host-metrics/proc/stat")
    host_proc_meminfo_path: Path = Path("/host-metrics/proc/meminfo")
    host_proc_loadavg_path: Path = Path("/host-metrics/proc/loadavg")
    host_proc_uptime_path: Path = Path("/host-metrics/proc/uptime")
    host_root_disk_path: Path = Path("/host-metrics/root-disk")
    host_extra_disk_path: Path = Path("/host-metrics/extra-disk")
    parameter_catalog_path: Path = Path("/reports/management_parameter_catalog.json")
    model_evidence_catalog_path: Path = Path("/reports/model_evidence_catalog.json")
    reports_root: Path = Path("/reports")

    @classmethod
    def from_env(cls) -> "Settings":
        raw_bots = os.getenv("TRADING_MANAGEMENT_BOTS_JSON", "").strip()
        bots: dict[str, dict[str, str]] = DEFAULT_BOTS
        if raw_bots:
            parsed: Any = json.loads(raw_bots)
            if not isinstance(parsed, dict):
                raise ValueError("TRADING_MANAGEMENT_BOTS_JSON must be an object")
            bots = {str(key): dict(value) for key, value in parsed.items() if isinstance(value, dict)}
        state_dir = Path(os.getenv("TRADING_MANAGEMENT_STATE_DIR", "/state"))
        return cls(
            token_file=Path(os.getenv(
                "TELEGRAM_MANAGEMENT_BOT_TOKEN_FILE",
                "/run/secrets/telegram_management_bot_token",
            )),
            admin_user_id=int(_required("ADMIN_USER_ID")),
            state_dir=state_dir,
            health_path=Path(os.getenv("TRADING_MANAGEMENT_HEALTH_PATH", str(state_dir / "health.json"))),
            hummingbot_api_url=os.getenv("HUMMINGBOT_API_URL", "http://hummingbot-api:8000").rstrip("/"),
            hummingbot_api_username=os.getenv("USERNAME", os.getenv("HUMMINGBOT_API_USERNAME", "admin")),
            hummingbot_api_password=_required("PASSWORD"),
            stocks_api_url=os.getenv("STOCKS_API_URL", "http://binance-stocks-runtime:8000").rstrip("/"),
            stocks_api_username=os.getenv("STOCKS_API_USERNAME", "stocks-admin"),
            stocks_api_password=_required("STOCKS_API_PASSWORD"),
            grid_guard_path=Path(os.getenv("GRID_GUARD_CONTRACT_PATH", "/contracts/grid_guard_state.json")),
            dca_guard_path=Path(os.getenv("DCA_GUARD_CONTRACT_PATH", "/contracts/dca_guard_state.json")),
            inventory_path=Path(os.getenv("INVENTORY_CONTRACT_PATH", "/contracts/account_inventory_status.json")),
            approval_request_root=Path(os.getenv("MODEL_APPROVAL_REQUEST_ROOT", "/approvals/weekly")),
            approval_evidence_root=Path(os.getenv("MODEL_APPROVAL_EVIDENCE_ROOT", "/approvals/evidence")),
            approval_decision_root=Path(os.getenv("MODEL_APPROVAL_DECISION_ROOT", "/approvals/decisions")),
            mutations_enabled=os.getenv("TRADING_MANAGEMENT_MUTATIONS_ENABLED", "false").lower() == "true",
            bots=bots,
            stocks_paper_trading_enabled=os.getenv(
                "STOCKS_PAPER_TELEGRAM_TRADING_ENABLED", "false"
            ).lower() == "true",
            stocks_live_trading_enabled=os.getenv(
                "STOCKS_LIVE_TELEGRAM_TRADING_ENABLED", "false"
            ).lower() == "true",
            session_ttl_seconds=int(os.getenv("TRADING_MANAGEMENT_SESSION_TTL_SECONDS", "900")),
            trading_status_path=Path(os.getenv(
                "TRADING_STATUS_PATH", "/reports/trading_status.json"
            )),
            profit_snapshot_db_path=Path(os.getenv(
                "PROFIT_SNAPSHOT_DB_PATH", "/reports/telegram_outbox.sqlite"
            )),
            current_runtime_errors_path=Path(os.getenv(
                "CURRENT_RUNTIME_ERRORS_PATH", "/reports/current_runtime_errors.json"
            )),
            operations_report_max_age_seconds=int(os.getenv(
                "OPERATIONS_REPORT_MAX_AGE_SECONDS", "300"
            )),
            host_proc_stat_path=Path(os.getenv(
                "HOST_PROC_STAT_PATH", "/host-metrics/proc/stat"
            )),
            host_proc_meminfo_path=Path(os.getenv(
                "HOST_PROC_MEMINFO_PATH", "/host-metrics/proc/meminfo"
            )),
            host_proc_loadavg_path=Path(os.getenv(
                "HOST_PROC_LOADAVG_PATH", "/host-metrics/proc/loadavg"
            )),
            host_proc_uptime_path=Path(os.getenv(
                "HOST_PROC_UPTIME_PATH", "/host-metrics/proc/uptime"
            )),
            host_root_disk_path=Path(os.getenv(
                "HOST_ROOT_DISK_PATH", "/host-metrics/root-disk"
            )),
            host_extra_disk_path=Path(os.getenv(
                "HOST_EXTRA_DISK_PATH", "/host-metrics/extra-disk"
            )),
            parameter_catalog_path=Path(os.getenv(
                "MANAGEMENT_PARAMETER_CATALOG_PATH", "/reports/management_parameter_catalog.json"
            )),
            model_evidence_catalog_path=Path(os.getenv(
                "MODEL_EVIDENCE_CATALOG_PATH", "/reports/model_evidence_catalog.json"
            )),
            reports_root=Path(os.getenv("MANAGEMENT_REPORTS_ROOT", "/reports")),
        )

    def read_token(self) -> str:
        token = self.token_file.read_text(encoding="utf-8").strip()
        if not token or ":" not in token:
            raise ValueError("Telegram management Bot token secret is invalid")
        return token
