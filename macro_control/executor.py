from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from threading import RLock

from .approval import (
    ApprovalProof,
    proposal_sha256,
    validate_approval_proof,
)
from .policy import (
    POLICY_VERSION,
    RiskWindowDecision,
    RiskWindowPolicy,
)
from .profiles import event_config
from .storage import StateStore


class DecisionRejected(ValueError):
    pass


class MacroExecutor:
    """Validate Hermes risk windows and reconcile directional DCA gates."""

    def __init__(
        self,
        api,
        telemetry,
        store: StateStore,
        bot_targets: list[dict],
        *,
        max_snapshot_age_seconds: int = 120,
        close_confirmation_timeout: float = 30,
        allowed_event_kinds: set[str] | None = None,
        max_pre_event_hours: float = 24,
        max_post_event_hours: float = 6,
        execution_enabled: bool = True,
        telegram_approver_user_id: str = "",
        telegram_approver_chat_id: str = "",
    ) -> None:
        self.api = api
        self.telemetry = telemetry
        self.store = store
        self.bot_targets = bot_targets
        self.max_snapshot_age_seconds = max_snapshot_age_seconds
        self.close_confirmation_timeout = close_confirmation_timeout
        self.execution_enabled = execution_enabled
        self.telegram_approver_user_id = str(telegram_approver_user_id)
        self.telegram_approver_chat_id = str(telegram_approver_chat_id)
        self.risk_policy = RiskWindowPolicy(
            allowed_event_kinds or {"fomc", "cpi", "nfp"},
            max_pre_event_hours=max_pre_event_hours,
            max_post_event_hours=max_post_event_hours,
        )
        self._lock = RLock()

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _fingerprint(command: dict) -> str:
        return hashlib.sha256(
            json.dumps(command, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @staticmethod
    def _hard_breaker(snapshot: dict) -> bool:
        return any(
            bool(bot.get("hard_circuit_breaker"))
            for bot in snapshot.get("bots", [])
        )

    @staticmethod
    def _buy_breaker(snapshot: dict) -> bool:
        return any(
            bool(bot.get("buy_circuit_breaker"))
            for bot in snapshot.get("bots", [])
        )

    def apply(self, command: dict, snapshot: dict | None = None) -> dict:
        with self._lock:
            snapshot = snapshot or self.telemetry.snapshot()
            try:
                decision = RiskWindowDecision.from_mapping(command)
            except (KeyError, TypeError, ValueError) as exc:
                raise DecisionRejected(f"invalid risk window decision: {exc}") from exc
            if decision.policy_version != POLICY_VERSION:
                raise DecisionRejected("unsupported policy version")
            if decision.snapshot_id != snapshot.get("snapshot_id") and hasattr(
                self.telemetry, "snapshot_by_id"
            ):
                cached = self.telemetry.snapshot_by_id(decision.snapshot_id)
                if cached is not None:
                    snapshot = cached
            self._validate_snapshot_v3(decision, snapshot)

            fingerprint = self._fingerprint(command)
            state = self.store.read()
            previous = state["decisions"].get(decision.decision_id)
            if previous:
                if previous["fingerprint"] != fingerprint:
                    raise DecisionRejected(
                        "decision_id was reused with another payload"
                    )
                return previous["result"]

            assessment = self.risk_policy.assess(decision)
            now = self._now()
            if not assessment.accepted:
                result = self._v3_result(
                    decision,
                    "rejected_no_change",
                    state,
                    assessment=assessment.to_dict(),
                )
                self._record_v3(state, decision, fingerprint, result)
                return result
            if decision.resume_at.astimezone(timezone.utc) <= now:
                result = self._v3_result(
                    decision,
                    "expired_no_change",
                    state,
                    assessment=assessment.to_dict(),
                )
                self._record_v3(state, decision, fingerprint, result)
                return result
            if decision.market_impact == "neutral":
                result = self._v3_result(
                    decision,
                    "recorded_no_change",
                    state,
                    assessment=assessment.to_dict(),
                )
                self._record_v3(state, decision, fingerprint, result)
                return result

            approval = self._validate_approval(
                decision,
                state,
                action="approve",
            )
            state["leases"][decision.decision_id] = {
                "decision_id": decision.decision_id,
                "event_id": decision.event_id,
                "event_kind": decision.event_kind,
                "market_impact": decision.market_impact,
                "effective_at": decision.effective_at.isoformat(),
                "resume_at": decision.resume_at.isoformat(),
                "decision_at": decision.decision_at.isoformat(),
                "proposal_sha256": approval.proposal_sha256,
                "approval": approval.to_dict(),
                "approval_timing": (
                    "late_approval"
                    if approval.approved_at
                    > decision.effective_at.astimezone(timezone.utc)
                    else "on_time"
                ),
                "status": (
                    "active"
                    if decision.effective_at.astimezone(timezone.utc) <= now
                    else "scheduled"
                ),
                "created_at": now.isoformat(),
            }
            self.store.write(state)
            reconcile = self._reconcile_locked(now=now, snapshot=snapshot)
            state = self.store.read()
            lease = state["leases"][decision.decision_id]
            status = (
                "scheduled"
                if lease["status"] == "scheduled"
                else reconcile["status"]
            )
            result = self._v3_result(
                decision,
                status,
                state,
                assessment=assessment.to_dict(),
                bot_results=reconcile["bot_results"],
            )
            self._record_v3(state, decision, fingerprint, result)
            return result

    def _validate_snapshot_v3(
        self, decision: RiskWindowDecision, snapshot: dict
    ) -> None:
        if decision.snapshot_id != snapshot.get("snapshot_id"):
            raise DecisionRejected("snapshot_id is stale or unknown")
        try:
            observed_at = datetime.fromisoformat(str(snapshot["observed_at"]))
        except (KeyError, ValueError) as exc:
            raise DecisionRejected("snapshot timestamp is invalid") from exc
        if observed_at.tzinfo is None:
            raise DecisionRejected("snapshot timestamp has no timezone")
        age = self._now() - observed_at.astimezone(timezone.utc)
        if age > timedelta(seconds=self.max_snapshot_age_seconds):
            raise DecisionRejected("snapshot is too old")
        if age < timedelta(seconds=-10):
            raise DecisionRejected("snapshot timestamp is in the future")
        if snapshot.get("telemetry_healthy") is False:
            raise DecisionRejected("snapshot telemetry is unhealthy")

    @staticmethod
    def _lease_phase(lease: dict, now: datetime) -> str:
        revoked_at = lease.get("revoked_at")
        if revoked_at and now >= datetime.fromisoformat(str(revoked_at)).astimezone(
            timezone.utc
        ):
            return "revoked"
        effective = datetime.fromisoformat(str(lease["effective_at"])).astimezone(
            timezone.utc
        )
        resume = datetime.fromisoformat(str(lease["resume_at"])).astimezone(
            timezone.utc
        )
        if now < effective:
            return "scheduled"
        if now < resume:
            return "active"
        return "expired"

    def _validate_approval(
        self,
        decision: RiskWindowDecision,
        state: dict,
        *,
        action: str,
    ) -> ApprovalProof:
        if decision.approval is None:
            raise DecisionRejected(
                "owner approval is required for non-neutral decisions"
            )
        try:
            approval = ApprovalProof.from_mapping(decision.approval)
            expected_hash = proposal_sha256(decision.to_dict())
            validate_approval_proof(
                approval,
                expected_proposal_sha256=expected_hash,
                expected_user_id=self.telegram_approver_user_id,
                expected_chat_id=self.telegram_approver_chat_id,
                decision_at=decision.decision_at,
                resume_at=decision.resume_at,
                action=action,
                received_at=self._now(),
            )
        except ValueError as exc:
            raise DecisionRejected(str(exc)) from exc
        replay_key = approval.replay_key
        if not replay_key:
            raise DecisionRejected("approval replay identifier is required")
        existing = state["approval_callbacks"].get(replay_key)
        claim = {
            "decision_id": decision.decision_id,
            "proposal_sha256": approval.proposal_sha256,
            "action": action,
        }
        if existing and existing != claim:
            raise DecisionRejected("approval interaction was reused")
        state["approval_callbacks"][replay_key] = claim
        return approval

    @staticmethod
    def _desired_gates(leases: list[dict]) -> dict[str, bool]:
        return {
            "buy": not any(
                lease["market_impact"] == "negative" for lease in leases
            ),
            "sell": not any(
                lease["market_impact"] == "positive" for lease in leases
            ),
        }

    def reconcile(
        self,
        *,
        now: datetime | None = None,
        snapshot: dict | None = None,
    ) -> dict:
        with self._lock:
            return self._reconcile_locked(now=now, snapshot=snapshot)

    def _reconcile_locked(
        self,
        *,
        now: datetime | None = None,
        snapshot: dict | None = None,
    ) -> dict:
        now = (now or self._now()).astimezone(timezone.utc)
        state = self.store.read()
        active: list[dict] = []
        for lease in state["leases"].values():
            lease["status"] = self._lease_phase(lease, now)
            if lease["status"] == "active":
                active.append(lease)
        desired = self._desired_gates(active)
        previous_desired = dict(state.get("desired_gates", {}))
        state["desired_gates"] = desired
        state["last_reconcile"] = now.isoformat()

        if not self.execution_enabled:
            result = {
                "status": "shadow",
                "desired_gates": desired,
                "active_lease_ids": sorted(
                    lease["decision_id"] for lease in active
                ),
                "bot_results": [],
            }
            self.store.write(state)
            return result

        snapshot = snapshot or self.telemetry.snapshot()
        hard_breaker = self._hard_breaker(snapshot)
        buy_breaker = self._buy_breaker(snapshot)
        bot_results: list[dict] = []
        lease_ids = sorted(lease["decision_id"] for lease in active)
        transition_id = self._transition_id(lease_ids, desired, now)
        desired_key = json.dumps(desired, sort_keys=True)
        for target in self.bot_targets:
            bot_name = target["bot_name"]
            retry = state["retry_state"].get(bot_name)
            if (
                retry
                and retry.get("desired_key") == desired_key
                and float(retry.get("next_retry_at", 0)) > now.timestamp()
            ):
                bot_results.append(
                    {
                        "bot_name": bot_name,
                        "trading_pair": target["trading_pair"],
                        "status": "retry_scheduled",
                        "next_retry_at": retry["next_retry_at"],
                    }
                )
                continue
            # The persisted gate state is audit/retry memory only. Always use
            # the live controller telemetry as the source of truth so a bot
            # restart or out-of-band config change cannot bypass an active
            # macro lease.
            actual = (
                None
                if snapshot.get("telemetry_healthy") is False
                else self._snapshot_gates(snapshot, bot_name)
            )
            if actual is None:
                # Do not guess a live controller's state.
                bot_results.append(
                    {
                        "bot_name": bot_name,
                        "trading_pair": target["trading_pair"],
                        "status": "failed",
                        "error": "controller gate telemetry is unavailable",
                    }
                )
                continue
            if actual == desired:
                state["bot_gate_state"][bot_name] = dict(actual)
                bot_results.append(
                    {
                        "bot_name": bot_name,
                        "trading_pair": target["trading_pair"],
                        "status": "unchanged",
                        "macro_buy_enabled": actual["buy"],
                        "macro_sell_enabled": actual["sell"],
                    }
                )
                continue
            enables_side = (
                (not actual["buy"] and desired["buy"])
                or (not actual["sell"] and desired["sell"])
            )
            target_gates = dict(desired)
            blocked_resume = False
            if enables_side and (hard_breaker or buy_breaker):
                # A hard breaker may block re-enabling a side, but must never
                # block a simultaneous risk-reducing disable on the other side.
                target_gates = {
                    "buy": (
                        desired["buy"] and actual["buy"]
                        if hard_breaker or buy_breaker
                        else desired["buy"]
                    ),
                    "sell": (
                        desired["sell"] and actual["sell"]
                        if hard_breaker
                        else desired["sell"]
                    ),
                }
                blocked_resume = target_gates != desired
                if target_gates == actual:
                    bot_results.append(
                        {
                            "bot_name": bot_name,
                            "trading_pair": target["trading_pair"],
                            "status": "blocked_by_hard_breaker",
                            "macro_buy_enabled": actual["buy"],
                            "macro_sell_enabled": actual["sell"],
                        }
                    )
                    continue
            item = self._apply_gates_to_bot(
                target,
                buy_enabled=target_gates["buy"],
                sell_enabled=target_gates["sell"],
                decision_id=transition_id,
            )
            if blocked_resume:
                item["resume_blocked_by_hard_breaker"] = True
            bot_results.append(item)
            if item["status"] in {"applied", "verified", "unchanged"}:
                state["bot_gate_state"][bot_name] = dict(target_gates)
                state["retry_state"].pop(bot_name, None)
            elif item["status"] == "failed":
                previous_delay = float(
                    state["retry_state"].get(bot_name, {}).get(
                        "delay_seconds", 2.5
                    )
                )
                delay = min(60.0, max(5.0, previous_delay * 2))
                state["retry_state"][bot_name] = {
                    "desired_key": desired_key,
                    "delay_seconds": delay,
                    "next_retry_at": now.timestamp() + delay,
                    "error": item.get("error", "gate update failed"),
                }

        statuses = [item["status"] for item in bot_results]
        successful = sum(
            value in {"applied", "verified", "unchanged"} for value in statuses
        )
        if statuses and all(value == "blocked_by_hard_breaker" for value in statuses):
            status = "blocked_by_hard_breaker"
        elif statuses and all(value == "retry_scheduled" for value in statuses):
            status = "retry_scheduled"
        elif successful == len(statuses):
            status = (
                "partially_applied"
                if any(
                    item.get("resume_blocked_by_hard_breaker")
                    for item in bot_results
                )
                else "applied"
            )
        elif successful:
            status = "partially_applied"
        else:
            status = "failed"
        self.store.write(state)
        if bot_results and (
            previous_desired != desired
            or any(value not in {"unchanged"} for value in statuses)
        ):
            self.store.append_audit(
                {
                    "type": "gate_reconcile",
                    "at": now.isoformat(),
                    "desired_gates": desired,
                    "active_lease_ids": lease_ids,
                    "status": status,
                    "bot_results": bot_results,
                }
            )
        return {
            "status": status,
            "desired_gates": desired,
            "active_lease_ids": lease_ids,
            "bot_results": bot_results,
        }

    @staticmethod
    def _snapshot_gates(snapshot: dict, bot_name: str) -> dict | None:
        bot = next(
            (
                value
                for value in snapshot.get("bots", [])
                if value.get("bot_name") == bot_name
            ),
            None,
        )
        if bot is None:
            return None
        return {
            "buy": bool(bot.get("macro_buy_enabled", True)),
            "sell": bool(bot.get("macro_sell_enabled", True)),
        }

    @staticmethod
    def _transition_id(
        lease_ids: list[str], desired: dict[str, bool], now: datetime
    ) -> str:
        value = json.dumps(
            {"leases": lease_ids, "desired": desired},
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(value.encode()).hexdigest()[:16]
        prefix = "leases" if lease_ids else "automatic-resume"
        return f"{prefix}:{digest}:{int(now.timestamp())}"

    def _apply_gates_to_bot(
        self,
        target: dict,
        *,
        buy_enabled: bool,
        sell_enabled: bool,
        decision_id: str,
    ) -> dict:
        bot_name = target["bot_name"]
        profile = event_config(
            target["trading_pair"],
            macro_buy_enabled=buy_enabled,
            macro_sell_enabled=sell_enabled,
            decision_id=decision_id,
            policy_version=POLICY_VERSION,
        )
        try:
            response = self.api.update_controller(
                bot_name, target["controller_name"], profile
            )
        except Exception as exc:
            return {
                "bot_name": bot_name,
                "trading_pair": target["trading_pair"],
                "status": "failed",
                "error": str(exc),
            }

        close_results = []
        if not buy_enabled:
            close_results.append(self._wait_for_side_close(target, "buy"))
        if not sell_enabled:
            close_results.append(self._wait_for_side_close(target, "sell"))
        confirmed = all(item["confirmed"] for item in close_results)
        return {
            "bot_name": bot_name,
            "trading_pair": target["trading_pair"],
            "status": "verified" if confirmed else "applied",
            "macro_buy_enabled": buy_enabled,
            "macro_sell_enabled": sell_enabled,
            "controller_response": response,
            "side_close": close_results,
        }

    def _wait_for_side_close(self, target: dict, side: str) -> dict:
        timeout = float(
            target.get("close_confirmation_timeout", self.close_confirmation_timeout)
        )
        deadline = time.monotonic() + timeout
        last_bot: dict | None = None
        active_key = f"active_{side}_executors"
        trading_key = f"trading_{side}_executors"
        while True:
            snapshot = self.telemetry.snapshot()
            last_bot = next(
                (
                    value
                    for value in snapshot.get("bots", [])
                    if value.get("bot_name") == target["bot_name"]
                ),
                None,
            )
            if last_bot is not None:
                active = int(last_bot.get(active_key, 0))
                trading = int(last_bot.get(trading_key, 0))
                if active == 0 and trading == 0:
                    return {
                        "side": side,
                        "confirmed": True,
                        active_key: 0,
                        trading_key: 0,
                        "note": (
                            "only the executor-owned exposure was closed; "
                            "no account-wide inventory order was submitted"
                        ),
                    }
            if time.monotonic() >= deadline:
                return {
                    "side": side,
                    "confirmed": False,
                    active_key: (
                        None if last_bot is None else int(last_bot.get(active_key, 0))
                    ),
                    trading_key: (
                        None
                        if last_bot is None
                        else int(last_bot.get(trading_key, 0))
                    ),
                    "reason": f"{side}_side_close_confirmation_timeout",
                }
            time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))

    @staticmethod
    def _v3_result(
        decision: RiskWindowDecision,
        status: str,
        state: dict,
        *,
        assessment: dict,
        bot_results: list[dict] | None = None,
    ) -> dict:
        approval_timing = None
        if decision.approval is not None:
            try:
                approval_timing = (
                    "late_approval"
                    if ApprovalProof.from_mapping(
                        decision.approval
                    ).approved_at
                    > decision.effective_at.astimezone(timezone.utc)
                    else "on_time"
                )
            except ValueError:
                approval_timing = "invalid"
        return {
            "decision_id": decision.decision_id,
            "event_id": decision.event_id,
            "market_impact": decision.market_impact,
            "status": status,
            "effective_at": decision.effective_at.isoformat(),
            "resume_at": decision.resume_at.isoformat(),
            "desired_gates": dict(
                state.get("desired_gates", {"buy": True, "sell": True})
            ),
            "assessment": assessment,
            "bot_results": bot_results or [],
            "approval": (
                dict(decision.approval)
                if decision.approval is not None
                else None
            ),
            "approval_timing": approval_timing,
        }

    def _record_v3(
        self,
        state: dict,
        decision: RiskWindowDecision,
        fingerprint: str,
        result: dict,
    ) -> None:
        state["decisions"][decision.decision_id] = {
            "fingerprint": fingerprint,
            "result": result,
        }
        state["updated_at"] = self._now().isoformat()
        if len(state["decisions"]) > 1000:
            for key in list(state["decisions"])[:-1000]:
                del state["decisions"][key]
        self.store.write(state)
        self.store.append_audit(
            {
                "type": "risk_window_decision",
                "at": self._now().isoformat(),
                "decision": decision.to_dict(),
                "result": result,
            }
        )

    def status(self) -> dict:
        state = self.store.read()
        return {
            "schema_version": state["schema_version"],
            "policy_version": POLICY_VERSION,
            "execution_enabled": self.execution_enabled,
            "approval_policy": {
                "channels": ["hermes_conversation", "telegram"],
                "preferred_channel": "hermes_conversation",
                "required_for": ["negative", "positive", "revoke"],
                "approver_configured": bool(
                    self.telegram_approver_user_id
                    and self.telegram_approver_chat_id
                ),
            },
            "desired_gates": state["desired_gates"],
            "bot_gate_state": state["bot_gate_state"],
            "leases": list(state["leases"].values()),
            "retry_state": state["retry_state"],
            "approval_callback_count": len(state["approval_callbacks"]),
            "last_reconcile": state["last_reconcile"],
        }

    def revoke(
        self,
        decision_id: str,
        approval_mapping: dict,
        *,
        snapshot: dict | None = None,
    ) -> dict:
        with self._lock:
            state = self.store.read()
            lease = state["leases"].get(decision_id)
            if lease is None:
                raise DecisionRejected("active decision lease was not found")
            existing_revocation = lease.get("revocation")
            if existing_revocation:
                return dict(existing_revocation)
            now = self._now()
            if now >= datetime.fromisoformat(str(lease["resume_at"])).astimezone(
                timezone.utc
            ):
                raise DecisionRejected("decision lease has already expired")
            try:
                approval = ApprovalProof.from_mapping(approval_mapping)
                validate_approval_proof(
                    approval,
                    expected_proposal_sha256=str(lease["proposal_sha256"]),
                    expected_user_id=self.telegram_approver_user_id,
                    expected_chat_id=self.telegram_approver_chat_id,
                    decision_at=datetime.fromisoformat(
                        str(lease["decision_at"])
                    ),
                    resume_at=datetime.fromisoformat(str(lease["resume_at"])),
                    action="revoke",
                    received_at=now,
                )
            except (KeyError, ValueError) as exc:
                raise DecisionRejected(str(exc)) from exc
            replay_key = approval.replay_key
            if not replay_key:
                raise DecisionRejected("approval replay identifier is required")
            claim = {
                "decision_id": decision_id,
                "proposal_sha256": approval.proposal_sha256,
                "action": "revoke",
            }
            used = state["approval_callbacks"].get(replay_key)
            if used and used != claim:
                raise DecisionRejected("approval interaction was reused")
            state["approval_callbacks"][replay_key] = claim
            lease["revoked_at"] = now.isoformat()
            lease["status"] = "revoked"
            lease["revocation_approval"] = approval.to_dict()
            self.store.write(state)

            snapshot = snapshot or self.telemetry.snapshot()
            reconcile = self._reconcile_locked(now=now, snapshot=snapshot)
            state = self.store.read()
            result = {
                "decision_id": decision_id,
                "status": "revoked",
                "revoked_at": now.isoformat(),
                "proposal_sha256": approval.proposal_sha256,
                "approval": approval.to_dict(),
                "desired_gates": reconcile["desired_gates"],
                "reconcile_status": reconcile["status"],
                "bot_results": reconcile["bot_results"],
            }
            state["leases"][decision_id]["revocation"] = result
            if decision_id in state["decisions"]:
                state["decisions"][decision_id]["result"]["revocation"] = result
            self.store.write(state)
            self.store.append_audit(
                {
                    "type": "risk_window_revocation",
                    "at": now.isoformat(),
                    "decision_id": decision_id,
                    "approval": approval.to_dict(),
                    "result": result,
                }
            )
            return result

    def heartbeat(self, decision_id: str = "") -> dict:
        event = {
            "type": "heartbeat",
            "at": self._now().isoformat(),
            "decision_id": decision_id,
            "strategy_changed": False,
        }
        self.store.append_audit(event)
        return {
            "status": "healthy",
            "observed_at": event["at"],
            "decision_id": decision_id,
            "strategy_changed": False,
        }

    def decision(self, decision_id: str) -> dict | None:
        record = self.store.read()["decisions"].get(decision_id)
        return None if record is None else record["result"]
