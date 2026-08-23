#!/usr/bin/env python3
"""Persisted v22 weekly generation and 12-hour default-approval workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd
import requests

try:
    from ethbtc_forced_exit_contract import PACKAGE_ID, atomic_json, sha256_file
    from grid_v22_live_gate import V22LiveGateProducer
    from telegram_notifications import append_event, build_event
except ModuleNotFoundError:
    from scripts.ethbtc_forced_exit_contract import PACKAGE_ID, atomic_json, sha256_file
    from scripts.grid_v22_live_gate import V22LiveGateProducer
    from live_guard.telegram_notifications import append_event, build_event


PAIRS = ("BTC-FDUSD", "ETH-FDUSD")
AUTO_CONFIRMATION = "AUTO-PROMOTE-ETHBTC-FORCED-EXIT-AFTER-12H"
MANUAL_CONFIRMATION = "PROMOTE-ETHBTC-FORCED-EXIT"
DEFAULT_DELAY_SECONDS = 12 * 60 * 60
DEFAULT_MINIMUM_RUNWAY_SECONDS = 24 * 60 * 60
DEFAULT_GENERATION_LEAD_SECONDS = 16 * 60 * 60
DEFAULT_PREWARM_LEAD_SECONDS = 35 * 60
DEFAULT_ACTIVATION_LEAD_SECONDS = 30 * 60
DEFAULT_RETAIN_OLD_RELEASES = 3
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"


def canonical_sha256(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def approval_prompt(release_sha256: str, deadline: int) -> str:
    deadline_text = datetime.fromtimestamp(deadline, timezone.utc).astimezone().isoformat()
    return (
        f"检查 v22 周模型候选 {release_sha256}。默认审批截止时间 {deadline_text}；"
        "请先核验 BTC/ETH 周覆盖连续、模型/特征/策略/行情哈希、样本外报告、"
        "账户资金归属、交易所过滤器和当前全部风控门。若同意请选择批准；若不同意"
        "请选择拒绝。截止时间前未拒绝且所有硬门槛持续通过，系统将按默认通过策略"
        "生成哈希绑定授权；不得绕过失败门槛，也不得回退 v21、ROC/SQZMOM 或上一周模型。"
    )


@dataclass(frozen=True)
class Policy:
    enabled: bool = True
    approval_delay_seconds: int = DEFAULT_DELAY_SECONDS
    minimum_runway_seconds: int = DEFAULT_MINIMUM_RUNWAY_SECONDS
    generation_lead_seconds: int = DEFAULT_GENERATION_LEAD_SECONDS
    xgb_threads: int = 2
    retain_old_releases: int = DEFAULT_RETAIN_OLD_RELEASES

    def __post_init__(self) -> None:
        if self.approval_delay_seconds < 60:
            raise ValueError("approval delay must be at least 60 seconds")
        if self.minimum_runway_seconds < 3600:
            raise ValueError("minimum signed runway must be at least one hour")
        if not 1 <= self.xgb_threads <= 2:
            raise ValueError("xgb_threads must be 1 or 2")
        if self.generation_lead_seconds < self.approval_delay_seconds + 1800:
            raise ValueError("generation lead must leave 30m after the review window")
        if self.generation_lead_seconds > 48 * 3600:
            raise ValueError("generation lead cannot exceed 48h")
        if not 0 <= self.retain_old_releases <= 52:
            raise ValueError("old release retention must be between 0 and 52")


class WeeklyReleaseManager:
    def __init__(
        self, *, release_root: Path, work_root: Path, candle_dir: Path,
        state_path: Path, authorization_path: Path, notification_path: Path,
        grid_state_path: Path, dca_state_path: Path, policy: Policy,
        runtime_root: Path | None = None,
        now: Callable[[], int] | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self.release_root = release_root
        self.work_root = work_root
        self.candle_dir = candle_dir
        self.state_path = state_path
        self.authorization_path = authorization_path
        self.notification_path = notification_path
        self.grid_state_path = grid_state_path
        self.dca_state_path = dca_state_path
        self.evidence_receipt_root = Path(os.getenv(
            "MODEL_EVIDENCE_RECEIPT_ROOT",
            str(dca_state_path.parent / "telegram" / "evidence_receipts"),
        ))
        self.runtime_root = runtime_root or grid_state_path.parent / "v22-runtime"
        self.policy = policy
        self.now = now or (lambda: int(time.time()))
        self.runner = runner or subprocess.run
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.candle_dir.mkdir(parents=True, exist_ok=True)

    @property
    def current(self) -> Path:
        return self.release_root / "current"

    def _state(self) -> dict[str, Any]:
        return load_json(self.state_path, {}) or {}

    @staticmethod
    def _publish_public_json(path: Path, value: Mapping[str, Any]) -> None:
        """Publish a secret-free approval view readable by the unprivileged management Bot."""
        atomic_json(path, dict(value))
        os.chmod(path, 0o644)

    def _save(self, value: Mapping[str, Any]) -> None:
        atomic_json(self.state_path, dict(value))
        approval_public = self.work_root / "approval_public"
        approval_public.mkdir(parents=True, exist_ok=True)
        approval_state = {
            key: value.get(key) for key in (
                "schema", "phase", "source_release_sha256", "source_effective_end",
                "candidate_release_sha256", "review_started_at", "review_deadline",
                "activation_boundary", "approved_at", "activate_at", "approval_mode",
                "runtime_generation", "last_error",
            )
        }
        self._publish_public_json(approval_public / "automation_state.json", approval_state)
        request_path = Path(str(value.get("request_path", "")))
        release_sha = str(value.get("candidate_release_sha256", ""))
        if release_sha and request_path.is_file():
            request = load_json(request_path, {}) or {}
            if request.get("release_sha256") == release_sha:
                self._publish_public_json(
                    approval_public / f"approval-request-{release_sha}.json", request,
                )
        public = {
            "schema": "ethbtc-forced-exit-cutover-status-v1",
            "updated_at": int(self.now()),
            "phase": value.get("phase", "UNKNOWN"),
            "candidate_release_sha256": value.get("candidate_release_sha256"),
            "runtime_generation": value.get("runtime_generation"),
            "fold_boundary": value.get("activation_boundary"),
            "activate_at": value.get("activate_at"),
            "current_generation_unaffected": value.get("phase") in {
                "SCHEDULED", "AWAITING_APPROVAL", "APPROVED_PENDING_PREWARM",
                "PREWARMED_PENDING_ACTIVATION",
            },
            "last_error": value.get("last_error"),
        }
        atomic_json(self.grid_state_path.parent / "v22_cutover_status.json", public)

    def _production(self, package: Path | None = None) -> dict[str, Any]:
        value = load_json((package or self.current) / "production_lock.json", {}) or {}
        if value.get("package_id") != PACKAGE_ID:
            raise RuntimeError("current v22 production lock is missing or invalid")
        return value

    def _run(self, *arguments: str, suppress_notification: bool = False) -> dict[str, Any]:
        environment = None
        if suppress_notification:
            environment = dict(os.environ)
            environment["TELEGRAM_NOTIFICATION_EVENTS_PATH"] = ""
        try:
            result = self.runner(
                [sys.executable, *arguments], check=True, capture_output=True, text=True,
                env=environment,
            )
        except subprocess.CalledProcessError as exc:
            detail = (str(exc.stderr or exc.stdout or "").strip()[-4000:]
                      or f"exit status {exc.returncode}")
            raise RuntimeError(f"{Path(arguments[0]).name} failed: {detail}") from exc
        output = str(result.stdout).strip()
        start = output.find("{")
        return json.loads(output[start:]) if start >= 0 else {}

    def _refresh_pair(self, pair: str, cutoff: int) -> int:
        path = self.candle_dir / f"binance_{pair}_5m.csv"
        existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
        if existing.empty:
            raise RuntimeError(f"seed candle history is missing for {pair}")
        last = int(existing["timestamp"].max())
        if last > 10_000_000_000:
            last //= 1000
        cursor, end_ms = (last + 300) * 1000, cutoff * 1000
        rows: list[list[Any]] = []
        while cursor < end_ms:
            response = requests.get(
                BINANCE_KLINES,
                params={"symbol": pair.replace("-", ""), "interval": "5m",
                        "startTime": cursor, "endTime": end_ms - 1, "limit": 1000},
                timeout=30,
            )
            response.raise_for_status()
            batch = response.json()
            if not batch:
                break
            rows.extend(batch)
            next_cursor = int(batch[-1][0]) + 300_000
            if next_cursor <= cursor:
                raise RuntimeError(f"Binance candle cursor did not advance for {pair}")
            cursor = next_cursor
            time.sleep(0.05)
        if rows:
            fresh = pd.DataFrame({
                "timestamp": [int(row[0]) // 1000 for row in rows],
                "open": [row[1] for row in rows], "high": [row[2] for row in rows],
                "low": [row[3] for row in rows], "close": [row[4] for row in rows],
                "volume": [row[5] for row in rows],
            })
            combined = pd.concat([existing, fresh], ignore_index=True)
            combined = combined.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
            temporary = path.with_suffix(".csv.tmp")
            combined.to_csv(temporary, index=False)
            os.replace(temporary, path)
        timestamps = pd.to_numeric(pd.read_csv(path, usecols=["timestamp"])["timestamp"], errors="raise").astype("int64")
        if int(timestamps.max()) < cutoff - 300:
            raise RuntimeError(f"{pair} candle history does not reach cutoff")
        tail = timestamps[timestamps >= cutoff - 8 * 86400]
        if tail.duplicated().any() or not (tail.diff().dropna() == 300).all():
            raise RuntimeError(f"{pair} candle history has a gap or duplicate near cutoff")
        return len(rows)

    def _seed_candles(self) -> None:
        source = self.release_root / "inputs" / "candles" / "grid"
        for pair in PAIRS:
            target = self.candle_dir / f"binance_{pair}_5m.csv"
            packaged = source / target.name
            if not target.exists() and packaged.is_file():
                shutil.copy2(packaged, target)

    def _candidate_checks(self, release: Path, *, current_end: int) -> dict[str, bool]:
        production = self._production(release)
        verify = self._run("/app/verify_ethbtc_forced_exit_package.py", str(release))
        return {
            "immutable_package_integrity": verify.get("integrity") == "PASS",
            "release_directory_matches": release.name == production.get("release_sha256"),
            "signed_week_is_contiguous": int(production.get("effective_end", 0)) == current_end + 7 * 86400,
            "fallback_forbidden": production.get("previous_model_fallback_allowed") is False,
            "candidate_stays_closed": production.get("deployment_allowed") is False,
        }

    def _generate(self, production: Mapping[str, Any], observed: int) -> dict[str, Any]:
        cutoff = int(production["effective_end"])
        late_recovery = observed >= cutoff
        training_cutoff = min(cutoff, observed // 3600 * 3600)
        self._seed_candles()
        refreshed = {pair: self._refresh_pair(pair, training_cutoff) for pair in PAIRS}
        staged = self.work_root / f"shadow-{cutoff}"
        staged_lock = staged / "shadow_lock.json"
        if staged.exists() and not staged_lock.is_file():
            shutil.rmtree(staged)
        if not staged_lock.is_file():
            self._run(
                "/app/append_xgboost_v22_signed_week.py",
                "--source-package", str(self.current / "shadow_package"),
                "--candle-dir", str(self.candle_dir), "--cutoff", str(cutoff),
                "--training-cutoff", str(training_cutoff),
                "--output-package", str(staged), "--xgb-threads", str(self.policy.xgb_threads),
            )
        result = self._run(
            "/app/stage_ethbtc_forced_exit_release.py", "--shadow-package", str(staged),
            "--lineage-package", str(self.release_root),
            "--release-root", str(self.release_root / "releases"),
            suppress_notification=True,
        )
        release_sha = str(result["release_sha256"])
        release = self.release_root / "releases" / release_sha
        checks = self._candidate_checks(release, current_end=cutoff)
        if not all(checks.values()):
            raise RuntimeError(f"candidate hard gates failed: {checks}")
        deadline = observed + self.policy.approval_delay_seconds
        prompt = approval_prompt(release_sha, deadline)
        request = {
            "schema": "ethbtc-forced-exit-default-approval-request-v1",
            "release_sha256": release_sha, "model_sha256": self._production(release)["model_sha256"],
            "source_effective_end": cutoff, "candidate_effective_end": self._production(release)["effective_end"],
            "training_cutoff": training_cutoff, "activation_boundary": cutoff,
            "late_signed_week_recovery": late_recovery,
            "review_started_at": observed, "review_deadline": deadline,
            "default_decision": "approve", "checks": checks, "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "refreshed_candle_rows": refreshed,
        }
        request_path = self.work_root / f"approval-request-{release_sha}.json"
        atomic_json(request_path, request)
        approval_public = self.work_root / "approval_public"
        approval_public.mkdir(parents=True, exist_ok=True)
        self._publish_public_json(approval_public / request_path.name, request)
        event = build_event(
            source="v22-weekly-release-manager", strategy="grid+dca",
            bot="grid-live-fdusd-400,dca-live-btcusdt-200,dca-live-ethusdt-200",
            pair="BTC-FDUSD,ETH-FDUSD,BTC-USDT,ETH-USDT", mechanism="parameter_update",
            transition="MODEL_APPROVAL_PENDING", reason="v22 新周候选已通过硬校验，进入12小时默认审批窗口",
            severity="warning", action="review_or_reject_before_default_approval",
            release_sha256=release_sha, model_sha256=request["model_sha256"],
            correlation_id=f"review:{release_sha}", details={**request, "report_request": "v22_png_windows"},
        )
        append_event(self.notification_path, event)
        state = {
            "schema": "ethbtc-forced-exit-weekly-automation-v1", "phase": "AWAITING_APPROVAL",
            "source_release_sha256": production["release_sha256"], "source_effective_end": cutoff,
            "candidate_release_sha256": release_sha, "candidate_path": str(release),
            "request_path": str(request_path), "review_started_at": observed,
            "review_deadline": deadline, "activation_boundary": cutoff,
            "late_signed_week_recovery": late_recovery,
            "last_event_id": event["event_id"], "last_error": None,
        }
        self._save(state)
        return state

    def _runtime_checks(self, release: Path, observed: int) -> dict[str, bool]:
        production = self._production(release)
        grid, dca = load_json(self.grid_state_path, {}) or {}, load_json(self.dca_state_path, {}) or {}
        ownership = grid.get("shadow_preflight", {}).get("ownership_coverage", {})
        dca_ownership = dca.get("ownership_preflight", {})
        return {
            "candidate_not_expired": observed < int(production["effective_end"]),
            "minimum_signed_runway": int(production["effective_end"]) - observed >= self.policy.minimum_runway_seconds,
            "grid_emergency_channel_ready": bool(grid.get("emergency_ready")),
            "dca_emergency_channel_ready": bool(dca.get("emergency_ready")),
            "grid_ownership_covered": bool(ownership) and all(bool(row.get("covered")) for row in ownership.values()),
            "dca_ownership_covered": bool(dca_ownership) and all(bool(row.get("covered")) for row in dca_ownership.values()),
            "exchange_filters_verified": bool(grid.get("shadow_preflight", {}).get("test_order_no_fill")),
        }

    def _decision(self, release_sha: str) -> dict[str, Any]:
        decision_root = Path(os.getenv("MODEL_APPROVAL_DECISION_ROOT", str(self.work_root)))
        value = load_json(decision_root / "review_decision.json", {}) or {}
        if value.get("release_sha256") != release_sha:
            return {}
        if value.get("decision") not in {"approve", "reject"}:
            raise RuntimeError("invalid hash-bound review decision")
        if not str(value.get("operator", "")).strip():
            raise RuntimeError("review decision has no operator")
        return value

    def _evidence_delivered(self, state: Mapping[str, Any], release: Path) -> bool:
        """Require all twelve model-evidence PNGs to reach Telegram before approval."""
        release_sha = str(state.get("candidate_release_sha256", ""))
        receipt = load_json(self.evidence_receipt_root / f"{release_sha}.json", {}) or {}
        if receipt.get("schema") != "telegram-evidence-delivery-receipt-v1":
            return False
        production = self._production(release)
        if (
            receipt.get("identity_sha256") != release_sha
            or receipt.get("release_sha256") != release_sha
            or receipt.get("model_sha256") != production.get("model_sha256")
            or receipt.get("report_request") not in {"v22_png_windows", "v22_360d"}
            or int(receipt.get("expected_photo_count", 0)) != 12
            or len(receipt.get("photo_sha256", [])) != 12
        ):
            return False
        if state.get("last_event_id") and receipt.get("source_event_id") != state.get("last_event_id"):
            return False
        expected = str(receipt.get("delivery_receipt_sha256", ""))
        unsigned = dict(receipt)
        unsigned.pop("delivery_receipt_sha256", None)
        return expected == canonical_sha256(unsigned)

    def _switch_current(self, release_sha: str) -> None:
        if os.name == "nt":
            raise RuntimeError("automatic production switching is only supported on OCI/Linux")
        temporary = self.release_root / ".current.next"
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(Path("releases") / release_sha, target_is_directory=True)
        os.replace(temporary, self.current)

    @staticmethod
    def _is_release_sha(value: object) -> bool:
        text = str(value or "")
        return len(text) == 64 and all(character in "0123456789abcdef" for character in text)

    @staticmethod
    def _remove_directory_atomically(path: Path) -> None:
        """Remove one verified child directory without exposing a half-deleted release."""
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"retention target is not a real directory: {path}")
        parent = path.parent.resolve()
        resolved = path.resolve()
        if resolved.parent != parent or resolved.name != path.name:
            raise ValueError(f"retention target escaped its parent: {path}")
        tombstone = path.parent / f".pruning-{path.name}"
        if tombstone.exists():
            raise FileExistsError(f"retention tombstone already exists: {tombstone}")
        os.replace(path, tombstone)
        try:
            shutil.rmtree(tombstone)
        except BaseException:
            if tombstone.exists() and not path.exists():
                os.replace(tombstone, path)
            raise

    def _release_records(self) -> list[dict[str, Any]]:
        releases = self.release_root / "releases"
        if not releases.is_dir():
            return []
        records: list[dict[str, Any]] = []
        for path in releases.iterdir():
            if not path.is_dir() or path.is_symlink() or not self._is_release_sha(path.name):
                continue
            production = load_json(path / "production_lock.json", {}) or {}
            if (
                production.get("package_id") != PACKAGE_ID
                or production.get("release_sha256") != path.name
            ):
                continue
            records.append({
                "release_sha256": path.name,
                "path": path,
                "effective_start": int(production.get("effective_start", 0) or 0),
                "effective_end": int(production.get("effective_end", 0) or 0),
                "mtime_ns": path.stat().st_mtime_ns,
            })
        return records

    @staticmethod
    def _generation_release_refs(manifest: Mapping[str, Any]) -> set[str]:
        return {
            str(value) for value in (
                manifest.get("release_sha256"),
                manifest.get("predecessor_release_sha256"),
            )
            if WeeklyReleaseManager._is_release_sha(value)
        }

    def _prune_release_history(
        self, state: Mapping[str, Any], observed: int,
    ) -> dict[str, Any]:
        """Keep the active release plus N previous releases and their usable generations.

        Telegram PNGs, delivery receipts, approval records, and notification events live
        outside ``releases/`` and are deliberately not touched by this retention task.
        """
        records = self._release_records()
        active_sha = str(state.get("candidate_release_sha256") or "")
        active = load_json(self.release_root / "active_deployment.json", {}) or {}
        if self._is_release_sha(active.get("release_sha256")):
            active_sha = str(active["release_sha256"])
        if not self._is_release_sha(active_sha):
            raise RuntimeError("cannot prune releases without a valid active release hash")

        ordered_old = sorted(
            (record for record in records if record["release_sha256"] != active_sha),
            key=lambda record: (
                record["effective_end"], record["effective_start"],
                record["mtime_ns"], record["release_sha256"],
            ),
            reverse=True,
        )
        retained_old = ordered_old[:self.policy.retain_old_releases]
        retained = {active_sha, *(record["release_sha256"] for record in retained_old)}

        protected_generations: set[str] = set()
        current_pointer = load_json(self.runtime_root / "current.json", {}) or {}
        current_generation = str(current_pointer.get("runtime_generation") or "")
        if self._is_release_sha(current_pointer.get("release_sha256")):
            retained.add(str(current_pointer["release_sha256"]))
        if self._is_release_sha(current_generation):
            protected_generations.add(current_generation)
        previous_pointer = state.get("previous_runtime_pointer")
        if isinstance(previous_pointer, Mapping):
            previous_generation = str(previous_pointer.get("runtime_generation") or "")
            if self._is_release_sha(previous_generation):
                protected_generations.add(previous_generation)
            if self._is_release_sha(previous_pointer.get("release_sha256")):
                retained.add(str(previous_pointer["release_sha256"]))

        generations_root = self.runtime_root / "generations"
        generation_records: list[tuple[Path, set[str]]] = []
        if generations_root.is_dir():
            for path in generations_root.iterdir():
                if not path.is_dir() or path.is_symlink() or not self._is_release_sha(path.name):
                    continue
                manifest = load_json(path / "manifest.json", {}) or {}
                refs = self._generation_release_refs(manifest)
                generation_records.append((path, refs))
                if path.name in protected_generations:
                    retained.update(refs)

        known_releases = {record["release_sha256"] for record in records}
        candidates_for_removal = known_releases - retained
        removed_generations: list[str] = []
        generation_errors: list[str] = []
        for path, refs in generation_records:
            if path.name in protected_generations or not (refs & candidates_for_removal):
                continue
            try:
                self._remove_directory_atomically(path)
                removed_generations.append(path.name)
            except Exception as exc:
                # Never leave a surviving generation pointing at a deleted release.
                retained.update(refs)
                candidates_for_removal -= refs
                generation_errors.append(f"{path.name}: {type(exc).__name__}: {exc}")

        removed_releases: list[str] = []
        release_errors: list[str] = []
        by_sha = {record["release_sha256"]: record for record in records}
        for release_sha in sorted(
            candidates_for_removal,
            key=lambda sha: (
                by_sha[sha]["effective_end"], by_sha[sha]["effective_start"], sha,
            ),
        ):
            try:
                self._remove_directory_atomically(by_sha[release_sha]["path"])
                removed_releases.append(release_sha)
            except Exception as exc:
                release_errors.append(f"{release_sha}: {type(exc).__name__}: {exc}")

        report = {
            "schema": "ethbtc-forced-exit-release-retention-v1",
            "generated_at": int(observed),
            "active_release_sha256": active_sha,
            "retain_old_releases": self.policy.retain_old_releases,
            "retained_release_sha256": sorted(known_releases - set(removed_releases)),
            "removed_release_sha256": removed_releases,
            "removed_runtime_generation": removed_generations,
            "generation_errors": generation_errors,
            "release_errors": release_errors,
            "evidence_png_and_receipts_preserved": True,
        }
        atomic_json(self.work_root / "release_retention.json", report)
        return report

    def _apply_release_retention(
        self, state: Mapping[str, Any], observed: int,
    ) -> dict[str, Any]:
        try:
            report = self._prune_release_history(state, observed)
            if report["removed_release_sha256"] or report["removed_runtime_generation"]:
                append_event(self.notification_path, build_event(
                    source="v22-weekly-release-manager", strategy="grid+dca",
                    bot="grid-live-fdusd-400,dca-live-btcusdt-200,dca-live-ethusdt-200",
                    pair="BTC-FDUSD,ETH-FDUSD,BTC-USDT,ETH-USDT",
                    mechanism="parameter_update", transition="MODEL_RETENTION_PRUNED",
                    reason=("新模型健康激活后仅保留当前模型和最近"
                            f"{self.policy.retain_old_releases}个旧模型"),
                    severity="info", action="prune_unreferenced_old_releases",
                    release_sha256=str(report["active_release_sha256"]),
                    correlation_id=f"retention:{report['active_release_sha256']}:{observed}",
                    details=report,
                ))
            return report
        except Exception as exc:
            report = {
                "schema": "ethbtc-forced-exit-release-retention-v1",
                "generated_at": int(observed), "status": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
                "evidence_png_and_receipts_preserved": True,
            }
            atomic_json(self.work_root / "release_retention.json", report)
            append_event(self.notification_path, build_event(
                source="v22-weekly-release-manager", strategy="grid+dca",
                bot="grid-live-fdusd-400,dca-live-btcusdt-200,dca-live-ethusdt-200",
                pair="BTC-FDUSD,ETH-FDUSD,BTC-USDT,ETH-USDT",
                mechanism="parameter_update", transition="MODEL_RETENTION_FAILED",
                reason="旧模型保留清理失败；当前已激活模型和交易不回滚",
                severity="warning", action="keep_active_release_and_retry_next_update",
                release_sha256=str(state.get("candidate_release_sha256") or ""),
                correlation_id=f"retention-failed:{observed}", details=report,
            ))
            return report

    def _activate(self, state: dict[str, Any], observed: int) -> dict[str, Any]:
        release_sha = str(state["candidate_release_sha256"])
        prepared_pointer = load_json(Path(state["prepared_pointer_path"]), {}) or {}
        prepared_pointer["cutover_phase"] = "WARM_ACTIVE"
        producer = V22LiveGateProducer(
            package_dir=self.release_root, cache_dir=self.candle_dir,
            seed_cache_dir=self.candle_dir, state_dir=self.grid_state_path.parent,
            authorization_path=self.authorization_path, refresh_binance=False,
            runtime_root=self.runtime_root,
        )
        previous_pointer = load_json(self.runtime_root / "current.json", None)
        producer.commit_generation(prepared_pointer)
        state.update({
            "phase": "WARM_ACTIVE_PENDING_FOLD",
            "warm_activated_at": observed,
            "previous_runtime_pointer": previous_pointer,
            "warm_verified_cycles": 0,
            "warm_failures": 0,
            "last_error": None,
        })
        self._save(state)
        append_event(self.notification_path, build_event(
            source="v22-weekly-release-manager", strategy="grid+dca",
            bot="grid-live-fdusd-400,dca-live-btcusdt-200,dca-live-ethusdt-200",
            pair="BTC-FDUSD,ETH-FDUSD,BTC-USDT,ETH-USDT", mechanism="parameter_update",
            transition="MODEL_CUTOVER_STABLE", reason="v22 候选已在当前签名周内完成无故障热切换",
            severity="info", action="monitor_warm_generation_until_fold_boundary",
            release_sha256=release_sha,
            correlation_id=f"warm-active:{prepared_pointer.get('runtime_generation')}",
            details={"runtime_generation": prepared_pointer.get("runtime_generation"),
                     "fold_boundary": state["activation_boundary"],
                     "approval_mode": state["approval_mode"]},
        ))
        return state

    def _finalize_fold(
        self, state: dict[str, Any], observed: int, *, generation_healthy: bool,
    ) -> dict[str, Any]:
        receipt = load_json(Path(state["pending_authorization_path"]), {}) or {}
        release_sha = str(state["candidate_release_sha256"])
        pointer = {
            "schema": "ethbtc-forced-exit-active-deployment-v1",
            "package_id": PACKAGE_ID, "release_sha256": release_sha,
            "activated_at": observed, "authorization": receipt,
            "authorization_sha256": canonical_sha256(receipt),
        }
        target = self.release_root / "active_deployment.json"
        atomic_json(target, pointer)
        self._switch_current(release_sha)
        atomic_json(self.authorization_path, receipt)
        runtime_pointer = load_json(self.runtime_root / "current.json", {}) or {}
        expected_generation = str(state.get("runtime_generation") or "")
        if (
            self._is_release_sha(expected_generation)
            and runtime_pointer.get("runtime_generation") == expected_generation
        ):
            runtime_pointer["cutover_phase"] = (
                "ACTIVE" if generation_healthy else "ACTIVE_UNAVAILABLE"
            )
            producer = V22LiveGateProducer(
                package_dir=self.release_root, cache_dir=self.candle_dir,
                seed_cache_dir=self.candle_dir, state_dir=self.grid_state_path.parent,
                authorization_path=self.authorization_path, refresh_binance=False,
                runtime_root=self.runtime_root,
            )
            producer.commit_generation(runtime_pointer)
        state.update({
            "phase": "ACTIVE" if generation_healthy else "ACTIVE_UNAVAILABLE",
            "activated_at": observed,
            "active_deployment_sha256": sha256_file(target),
            "last_error": None if generation_healthy else "signed_week_unavailable",
        })
        if generation_healthy:
            state["release_retention"] = self._apply_release_retention(state, observed)
        self._save(state)
        append_event(self.notification_path, build_event(
            source="v22-weekly-release-manager", strategy="grid+dca",
            bot="grid-live-fdusd-400,dca-live-btcusdt-200,dca-live-ethusdt-200",
            pair="BTC-FDUSD,ETH-FDUSD,BTC-USDT,ETH-USDT", mechanism="parameter_update",
            transition=("MODEL_FOLD_ACTIVATED" if generation_healthy
                        else "MODEL_CUTOVER_PRECHECK_FAILED"),
            reason=("v22 已预热 release 在周边界自然进入新 fold"
                    if generation_healthy else
                    "周边界到达时当前 generation 无健康签名周，模型信号不可用并进入 Fail-Closed"),
            severity="info" if generation_healthy else "critical",
            action=("finalize_non_runtime_release_aliases" if generation_healthy
                    else "signed_week_unavailable_fail_closed"),
            release_sha256=release_sha, model_sha256=receipt.get("model_sha256", ""),
            correlation_id=f"activated:{release_sha}",
            details={"activation_boundary": state["activation_boundary"],
                     "runtime_generation": state.get("runtime_generation"),
                     "fold_boundary": state["activation_boundary"],
                     "approval_mode": state["approval_mode"],
                     "active_deployment_sha256": state["active_deployment_sha256"]},
        ))
        return state

    def _monitor_warm_generation(self, state: dict[str, Any], observed: int) -> dict[str, Any]:
        contract = load_json(
            self.grid_state_path.parent / "ethbtc_forced_exit_observation.json", {},
        ) or {}
        healthy = bool(
            contract.get("source_healthy") is True
            and contract.get("runtime_generation") == state.get("runtime_generation")
        )
        if healthy:
            state["warm_verified_cycles"] = int(state.get("warm_verified_cycles", 0)) + 1
            state["warm_failures"] = 0
        else:
            state["warm_failures"] = int(state.get("warm_failures", 0)) + 1
            state["last_error"] = (
                "warm generation not observed healthy; signed predecessor remains active"
            )
        if state.get("late_signed_week_recovery"):
            if healthy and int(state.get("warm_verified_cycles", 0)) >= 3:
                return self._finalize_fold(
                    state, observed, generation_healthy=True,
                )
            self._save(state)
            return state
        if observed >= int(state["activation_boundary"]):
            return self._finalize_fold(
                state, observed, generation_healthy=healthy,
            )
        if int(state.get("warm_failures", 0)) >= 3:
            current_pointer = self.runtime_root / "current.json"
            previous = state.get("previous_runtime_pointer")
            if isinstance(previous, dict):
                atomic_json(current_pointer, previous)
            elif current_pointer.exists():
                failed = self.runtime_root / (
                    f"failed-{state.get('runtime_generation')}-{observed}.json"
                )
                os.replace(current_pointer, failed)
            state.update({
                "phase": "APPROVED_PENDING_PREWARM",
                "prewarm_at": observed + 60,
                "retry_after": observed + 60,
            })
            self._notify_blocked(state, {
                "warm_generation_healthy": False,
                "signed_predecessor_preserved": True,
            })
        self._save(state)
        return state

    def _prewarm(self, state: dict[str, Any], observed: int) -> dict[str, Any]:
        producer = V22LiveGateProducer(
            package_dir=self.release_root, cache_dir=self.candle_dir,
            seed_cache_dir=self.candle_dir, state_dir=self.grid_state_path.parent,
            authorization_path=self.authorization_path, refresh_binance=True,
            runtime_root=self.runtime_root,
        )
        prepared = producer.prepare_generation(
            release=Path(state["candidate_path"]),
            authorization=Path(state["pending_authorization_path"]),
            predecessor_release_sha256=str(state["source_release_sha256"]),
            fold_boundary=int(state["activation_boundary"]), observed_at=observed,
            live_contract_path=(
                self.grid_state_path.parent / "ethbtc_forced_exit_observation.json"
            ),
            final_preflight_sha256=canonical_sha256(state.get("final_checks", {})),
            recovery_from_unavailable=bool(state.get("late_signed_week_recovery")),
        )
        pointer_path = self.work_root / f"runtime-pointer-{prepared['generation']}.json"
        atomic_json(pointer_path, prepared["pointer"])
        state.update({
            "phase": "PREWARMED_PENDING_ACTIVATION",
            "prewarmed_at": observed,
            "runtime_generation": prepared["generation"],
            "prepared_pointer_path": str(pointer_path),
            "semantic_parity": prepared["manifest"].get("semantic_parity"),
            "last_error": None,
        })
        self._save(state)
        append_event(self.notification_path, build_event(
            source="v22-weekly-release-manager", strategy="grid+dca",
            bot="grid-live-fdusd-400,dca-live-btcusdt-200,dca-live-ethusdt-200",
            pair="BTC-FDUSD,ETH-FDUSD,BTC-USDT,ETH-USDT", mechanism="parameter_update",
            transition="MODEL_CUTOVER_PREWARMED",
            reason="v22 候选已隔离预热并通过当前周语义一致性检查",
            severity="info", action="wait_for_early_atomic_activation",
            release_sha256=state["candidate_release_sha256"],
            correlation_id=f"prewarmed:{prepared['generation']}",
            details={"runtime_generation": prepared["generation"],
                     "activate_at": state["activate_at"],
                     "fold_boundary": state["activation_boundary"],
                     "semantic_parity": state["semantic_parity"]},
        ))
        return state

    def _notify_blocked(self, state: Mapping[str, Any], checks: Mapping[str, bool]) -> None:
        release_sha = str(state.get("candidate_release_sha256", ""))
        boundary_missed = checks.get("approved_before_fold_boundary") is False
        reason = (
            "v22 候选在周边界前仍不可用；当前签名周结束后必须 Fail-Closed"
            if boundary_missed else
            "v22 候选核验或预热未通过；当前已提交 generation 继续正常刷新，候选不会污染交易合同"
        )
        append_event(self.notification_path, build_event(
            source="v22-weekly-release-manager", strategy="grid+dca",
            bot="grid-live-fdusd-400,dca-live-btcusdt-200,dca-live-ethusdt-200",
            pair="BTC-FDUSD,ETH-FDUSD,BTC-USDT,ETH-USDT", mechanism="parameter_update",
            transition="MODEL_CUTOVER_PRECHECK_FAILED", reason=reason,
            severity="critical" if boundary_missed else "warning",
            action=("signed_week_unavailable_fail_closed" if boundary_missed
                    else "preserve_current_generation_and_retry_candidate"),
            release_sha256=release_sha,
            correlation_id=f"blocked:{release_sha}:{canonical_sha256(checks)}",
            details={"checks": dict(checks), "default_approval_suppressed": True},
        ))

    def _approve(self, state: dict[str, Any], observed: int, decision: Mapping[str, Any]) -> dict[str, Any]:
        release_sha, release = str(state["candidate_release_sha256"]), Path(state["candidate_path"])
        boundary = int(state["activation_boundary"])
        late_recovery = bool(state.get("late_signed_week_recovery"))
        if observed >= boundary and not late_recovery:
            state.update({
                "phase": "SIGNED_WEEK_UNAVAILABLE",
                "last_error": "signed_week_unavailable: candidate approval missed the signed week boundary",
            })
            self._save(state)
            self._notify_blocked(state, {"approved_before_fold_boundary": False})
            return state
        manual = decision.get("decision") == "approve"
        if not manual and observed < int(state["review_deadline"]):
            return state
        checks = {
            **self._candidate_checks(release, current_end=int(state["source_effective_end"])),
            **self._runtime_checks(release, observed),
            "telegram_model_evidence_delivered": self._evidence_delivered(state, release),
        }
        if not all(checks.values()):
            state.update({"phase": "AWAITING_APPROVAL", "retry_after": observed + 300,
                          "last_error": f"approval hard gates failed: {checks}"})
            self._save(state)
            self._notify_blocked(state, checks)
            return state
        mode = "manual_hermes" if manual else "automatic_default_after_12h"
        preferred_activation = boundary - DEFAULT_ACTIVATION_LEAD_SECONDS
        activate_at = (
            ((observed + 119) // 60) * 60
            if late_recovery else
            max(preferred_activation, ((observed + 119) // 60) * 60)
        )
        prewarm_at = max(observed, activate_at - 5 * 60)
        production = self._production(release)
        delivery_receipt = load_json(
            self.evidence_receipt_root / f"{release_sha}.json", {}
        ) or {}
        evidence = {
            "schema": "ethbtc-forced-exit-12h-review-evidence-v1", "release_sha256": release_sha,
            "request_sha256": sha256_file(Path(state["request_path"])),
            "review_started_at": state["review_started_at"], "review_deadline": state["review_deadline"],
            "approved_at": observed, "approval_mode": mode, "checks": checks,
            "telegram_evidence_delivery_sha256": delivery_receipt.get(
                "delivery_receipt_sha256"
            ),
        }
        evidence_path = self.work_root / f"approval-evidence-{release_sha}.json"
        atomic_json(evidence_path, evidence)
        receipt = {
            "schema": "ethbtc-forced-exit-authorization-v1", "package_id": PACKAGE_ID,
            "release_sha256": release_sha, "model_sha256": production["model_sha256"],
            "operator": str(decision.get("operator") or "v22-weekly-default-approval"),
            "confirmation": MANUAL_CONFIRMATION if manual else AUTO_CONFIRMATION,
            "approval_mode": mode, "approved_at": observed, "activate_at": activate_at,
            "effective_end": int(production["effective_end"]),
            "review_started_at": int(state["review_started_at"]),
            "review_deadline": int(state["review_deadline"]),
            "approval_request_sha256": sha256_file(Path(state["request_path"])),
            "observation_report_sha256": sha256_file(evidence_path),
            "telegram_evidence_delivery_sha256": delivery_receipt.get(
                "delivery_receipt_sha256"
            ),
            "preflight_sha256": canonical_sha256(checks),
            "auto_reentry_authorized": True, "consumed": False,
        }
        pending = self.work_root / f"pending-authorization-{release_sha}.json"
        atomic_json(pending, receipt)
        state.update({"phase": "APPROVED_PENDING_PREWARM", "approved_at": observed,
                      "activate_at": activate_at, "approval_mode": mode,
                      "prewarm_at": prewarm_at,
                      "final_check_at": max(observed, boundary - 60 * 60),
                      "final_check_complete": False,
                      "late_signed_week_recovery": late_recovery,
                      "pending_authorization_path": str(pending),
                      "authorization_sha256": sha256_file(pending),
                      "last_error": None})
        self._save(state)
        append_event(self.notification_path, build_event(
            source="v22-weekly-release-manager", strategy="grid+dca",
            bot="grid-live-fdusd-400,dca-live-btcusdt-200,dca-live-ethusdt-200",
            pair="BTC-FDUSD,ETH-FDUSD,BTC-USDT,ETH-USDT", mechanism="parameter_update",
            transition="MODEL_DEFAULT_APPROVED" if not manual else "PARAMETER_ACTIVATED",
            reason=("v22 候选在12小时无否决后按默认策略批准" if not manual else "v22 候选已由 Hermes 明确批准"),
            severity="info", action="wait_without_touching_current_trade_then_activate_at_boundary",
            release_sha256=release_sha, model_sha256=production["model_sha256"],
            correlation_id=f"approved:{release_sha}",
            details={"approval_mode": mode, "activate_at": activate_at,
                     "final_check_at": state["final_check_at"],
                     "prewarm_at": prewarm_at,
                     "effective_end": production["effective_end"], "checks": checks,
                     "authorization_sha256": state["authorization_sha256"]},
        ))
        return state

    def reconcile(self) -> dict[str, Any]:
        observed = self.now()
        if not self.policy.enabled:
            return {"phase": "DISABLED", "automatic_update": False}
        state = self._state()
        if state.get("phase") == "APPROVED_PENDING_PREWARM":
            boundary = int(state["activation_boundary"])
            if observed >= boundary and not state.get("late_signed_week_recovery"):
                state.update({
                    "phase": "SIGNED_WEEK_UNAVAILABLE",
                    "last_error": "signed_week_unavailable: candidate missed pre-boundary commit",
                })
                self._save(state)
                self._notify_blocked(state, {"approved_before_fold_boundary": False})
                return state
            if (
                not state.get("final_check_complete")
                and observed >= int(state.get("final_check_at", boundary - 60 * 60))
            ):
                release = Path(state["candidate_path"])
                checks = {
                    **self._candidate_checks(
                        release, current_end=int(state["source_effective_end"]),
                    ),
                    **self._runtime_checks(release, observed),
                }
                if not all(checks.values()):
                    state.update({
                        "retry_after": observed + 60,
                        "last_error": f"T-60m final checks failed: {checks}",
                    })
                    self._save(state)
                    self._notify_blocked(state, {
                        **checks, "live_generation_unchanged": True,
                    })
                    return state
                state.update({
                    "final_check_complete": True,
                    "final_checked_at": observed,
                    "final_checks": checks,
                    "last_error": None,
                })
                self._save(state)
            if observed < int(state["prewarm_at"]):
                return state
            if not state.get("final_check_complete"):
                return state
            try:
                return self._prewarm(state, observed)
            except Exception as exc:
                state.update({
                    "phase": "APPROVED_PENDING_PREWARM",
                    "retry_after": observed + 60,
                    "last_error": (
                        "prewarm failed without live impact: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                })
                self._save(state)
                self._notify_blocked(state, {
                    "candidate_prewarm": False,
                    "live_generation_unchanged": True,
                })
                return state
        if state.get("phase") == "PREWARMED_PENDING_ACTIVATION":
            if observed < int(state["activate_at"]):
                return state
            if (
                observed >= int(state["activation_boundary"])
                and not state.get("late_signed_week_recovery")
            ):
                state.update({
                    "phase": "SIGNED_WEEK_UNAVAILABLE",
                    "last_error": "signed_week_unavailable: runtime commit missed T-30m window",
                })
                self._save(state)
                self._notify_blocked(state, {"approved_before_fold_boundary": False})
                return state
            return self._activate(state, observed)
        if state.get("phase") in {"WARM_ACTIVE_PENDING_FOLD", "ACTIVE_UNAVAILABLE"}:
            return self._monitor_warm_generation(state, observed)
        if state.get("phase") == "AWAITING_APPROVAL":
            decision = self._decision(str(state["candidate_release_sha256"]))
            if decision.get("decision") == "reject":
                state.update({"phase": "REJECTED", "rejected_at": observed,
                              "rejected_by": decision["operator"], "last_error": decision.get("reason")})
                self._save(state)
                append_event(self.notification_path, build_event(
                    source="v22-weekly-release-manager", strategy="grid+dca",
                    bot="grid-live-fdusd-400,dca-live-btcusdt-200,dca-live-ethusdt-200",
                    pair="BTC-FDUSD,ETH-FDUSD,BTC-USDT,ETH-USDT", mechanism="parameter_update",
                    transition="MODEL_APPROVAL_REJECTED", reason=str(decision.get("reason") or "operator rejected"),
                    severity="warning", action="keep_fail_closed", release_sha256=state["candidate_release_sha256"],
                    correlation_id=f"rejected:{state['candidate_release_sha256']}",
                ))
                return state
            return self._approve(state, observed, decision)
        production = self._production()
        source_end = int(production["effective_end"])
        if state.get("source_effective_end") == source_end and state.get("phase") in {"ACTIVE", "REJECTED"}:
            return state
        if state.get("source_effective_end") == source_end and state.get("phase") == "BLOCKED":
            if observed < int(state.get("retry_after", 0)):
                return state
        generation_at = source_end - self.policy.generation_lead_seconds
        if observed < generation_at:
            return {"phase": "SCHEDULED", "automatic_update": True,
                    "next_generation_at": generation_at, "activation_boundary": source_end,
                    "current_release_sha256": production["release_sha256"]}
        try:
            return self._generate(production, observed)
        except Exception as exc:
            failed = {
                "schema": "ethbtc-forced-exit-weekly-automation-v1", "phase": "BLOCKED",
                "source_release_sha256": production["release_sha256"], "source_effective_end": source_end,
                "failed_at": observed, "retry_after": observed + 900,
                "last_error": f"{type(exc).__name__}: {exc}",
            }
            self._save(failed)
            self._notify_blocked(failed, {"candidate_generation": False})
            return failed


def policy_from_environment() -> Policy:
    return Policy(
        enabled=os.getenv("V22_WEEKLY_AUTO_UPDATE_ENABLED", "true").lower() == "true",
        approval_delay_seconds=int(os.getenv("V22_WEEKLY_DEFAULT_APPROVAL_DELAY_SECONDS", str(DEFAULT_DELAY_SECONDS))),
        minimum_runway_seconds=int(os.getenv("V22_WEEKLY_MINIMUM_RUNWAY_SECONDS", str(DEFAULT_MINIMUM_RUNWAY_SECONDS))),
        generation_lead_seconds=int(os.getenv("V22_WEEKLY_GENERATION_LEAD_SECONDS", str(DEFAULT_GENERATION_LEAD_SECONDS))),
        xgb_threads=int(os.getenv("V22_WEEKLY_XGB_THREADS", "2")),
        retain_old_releases=int(os.getenv(
            "V22_WEEKLY_RETAIN_OLD_RELEASES", str(DEFAULT_RETAIN_OLD_RELEASES),
        )),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=Path(os.getenv("ETHBTC_RELEASE_FAMILY_PATH", "/workspace/releases")))
    parser.add_argument("--work-root", type=Path, default=Path(os.getenv("V22_WEEKLY_WORK_PATH", "/workspace/weekly")))
    parser.add_argument("--candle-dir", type=Path, default=Path(os.getenv("V22_WEEKLY_CANDLE_PATH", "/workspace/weekly/candles")))
    parser.add_argument("--grid-state", type=Path, default=Path(os.getenv("GRID_LIVE_STATE_PATH", "/workspace/state")) / "guard_state.json")
    parser.add_argument("--dca-state", type=Path, default=Path(os.getenv("DCA_LIVE_STATE_PATH", "/workspace/dca-state")) / "guard_state.json")
    parser.add_argument("--decision", choices=("approve", "reject"))
    parser.add_argument("--release-sha256")
    parser.add_argument("--operator")
    parser.add_argument("--reason", default="")
    args = parser.parse_args()
    if args.decision:
        state = load_json(args.work_root / "automation_state.json", {}) or {}
        release_sha = str(args.release_sha256 or "")
        if state.get("phase") != "AWAITING_APPROVAL":
            raise RuntimeError("there is no v22 candidate awaiting approval")
        if release_sha != state.get("candidate_release_sha256"):
            raise RuntimeError("review decision release hash does not match the pending candidate")
        if not str(args.operator or "").strip():
            raise RuntimeError("a review decision requires an operator")
        decision = {
            "schema": "ethbtc-forced-exit-review-decision-v1",
            "release_sha256": release_sha, "decision": args.decision,
            "operator": str(args.operator).strip(), "reason": str(args.reason).strip(),
            "decided_at": int(time.time()),
        }
        decision_root = Path(os.getenv("MODEL_APPROVAL_DECISION_ROOT", str(args.work_root)))
        decision_root.mkdir(parents=True, exist_ok=True)
        atomic_json(decision_root / "review_decision.json", decision)
        print(json.dumps(decision, ensure_ascii=False, indent=2))
        return 0
    manager = WeeklyReleaseManager(
        release_root=args.release_root, work_root=args.work_root, candle_dir=args.candle_dir,
        state_path=args.work_root / "automation_state.json",
        authorization_path=Path(os.getenv("GRID_V22_AUTHORIZATION_PATH", "/workspace/state/ethbtc_forced_exit_authorization.json")),
        notification_path=Path(os.getenv("TELEGRAM_NOTIFICATION_EVENTS_PATH", "/workspace/state/telegram_events.jsonl")),
        grid_state_path=args.grid_state, dca_state_path=args.dca_state, policy=policy_from_environment(),
        runtime_root=Path(os.getenv("V22_RUNTIME_ROOT", "/workspace/state/v22-runtime")),
    )
    print(json.dumps(manager.reconcile(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
