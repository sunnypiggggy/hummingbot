from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


class TradingReportUnavailable(RuntimeError):
    pass


class JsonTradingReportProvider:
    def __init__(
        self,
        report_path: Path,
        chart_path: Path,
        *,
        max_age_seconds: float = 900,
    ) -> None:
        self.report_path = report_path
        self.chart_path = chart_path
        self.max_age_seconds = max_age_seconds
        self._lock = Lock()

    def report(self) -> dict:
        try:
            with self._lock:
                value = json.loads(self.report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise TradingReportUnavailable(
                f"trading report is unavailable: {type(exc).__name__}"
            ) from exc
        if value.get("schema_version") != 3:
            raise TradingReportUnavailable("trading report schema is not V3")
        try:
            generated = datetime.fromisoformat(str(value["generated_at"]))
        except (KeyError, ValueError) as exc:
            raise TradingReportUnavailable(
                "trading report generated_at is invalid"
            ) from exc
        if generated.tzinfo is None:
            raise TradingReportUnavailable(
                "trading report generated_at must include timezone"
            )
        age = max(0.0, (datetime.now(timezone.utc) - generated).total_seconds())
        if age > self.max_age_seconds:
            raise TradingReportUnavailable(
                f"trading report is stale ({age:.0f}s)"
            )
        return {**value, "data_age_seconds": age}

    def chart(self) -> tuple[bytes, str]:
        report = self.report()
        try:
            with self._lock:
                value = self.chart_path.read_bytes()
        except Exception as exc:
            raise TradingReportUnavailable(
                f"trading chart is unavailable: {type(exc).__name__}"
            ) from exc
        if not value.startswith(b"\x89PNG\r\n\x1a\n"):
            raise TradingReportUnavailable("trading chart is not a PNG")
        if hashlib.sha256(value).hexdigest() != report.get("chart_sha256"):
            raise TradingReportUnavailable(
                "trading chart does not match the current report"
            )
        return value, str(report["report_id"])
