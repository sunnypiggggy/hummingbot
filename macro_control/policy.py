from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable
from urllib.parse import urlparse
from zoneinfo import ZoneInfo


POLICY_VERSION = "dca-macro-v3"
VALID_EVENT_KINDS = {"fomc", "cpi", "nfp"}
VALID_MARKET_IMPACTS = {"negative", "positive", "neutral"}
MIN_CONFIDENCE = 0.75
MIN_DECISION_EVIDENCE = 2


def _aware(value: datetime, field: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class Event:
    event_id: str
    kind: str
    starts_at: datetime
    title: str
    source_url: str

    def __post_init__(self) -> None:
        if self.kind not in VALID_EVENT_KINDS:
            raise ValueError(f"unsupported event kind: {self.kind}")
        _aware(self.starts_at, "event starts_at")

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "starts_at": self.starts_at.isoformat(),
            "title": self.title,
            "source_url": self.source_url,
        }

    @classmethod
    def from_mapping(cls, value: dict) -> "Event":
        return cls(
            event_id=str(value["event_id"]),
            kind=str(value["kind"]).lower(),
            starts_at=datetime.fromisoformat(str(value["starts_at"])),
            title=str(value.get("title", value["event_id"])),
            source_url=str(value["source_url"]),
        )


@dataclass(frozen=True)
class Evidence:
    source_id: str
    source_url: str
    observed_at: datetime
    summary: str
    kind: str = "decision"

    def __post_init__(self) -> None:
        if not self.source_id or not self.source_url or not self.summary:
            raise ValueError("evidence source_id, source_url and summary are required")
        _aware(self.observed_at, "evidence observed_at")

    def to_dict(self) -> dict:
        value = asdict(self)
        value["observed_at"] = self.observed_at.isoformat()
        return value

    @classmethod
    def from_mapping(cls, value: dict) -> "Evidence":
        return cls(
            source_id=str(value["source_id"]),
            source_url=str(value["source_url"]),
            observed_at=datetime.fromisoformat(str(value["observed_at"])),
            summary=str(value["summary"]),
            kind=str(value.get("kind", "decision")),
        )


@dataclass(frozen=True)
class RiskWindowDecision:
    """Hermes-authored decision that leases one directional DCA gate."""

    decision_id: str
    event_id: str
    event_kind: str
    event_at: datetime
    event_source_url: str
    decision_at: datetime
    market_impact: str
    confidence: float
    effective_at: datetime
    resume_at: datetime
    snapshot_id: str
    reason: str
    evidence: tuple[Evidence, ...]
    hermes: dict
    approval: dict | None = None
    policy_version: str = POLICY_VERSION

    def __post_init__(self) -> None:
        required = {
            "decision_id": self.decision_id,
            "event_id": self.event_id,
            "snapshot_id": self.snapshot_id,
            "reason": self.reason,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"required fields are empty: {', '.join(missing)}")
        if self.event_kind not in VALID_EVENT_KINDS:
            raise ValueError(f"unsupported event kind: {self.event_kind}")
        if self.market_impact not in VALID_MARKET_IMPACTS:
            raise ValueError(
                f"market_impact must be one of {sorted(VALID_MARKET_IMPACTS)}"
            )
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        for name in ("event_at", "decision_at", "effective_at", "resume_at"):
            _aware(getattr(self, name), name)
        required_hermes = {"agent_id", "model", "prompt_version", "run_id"}
        if not isinstance(self.hermes, dict) or not required_hermes.issubset(
            self.hermes
        ):
            raise ValueError(
                "hermes metadata needs agent_id, model, prompt_version and run_id"
            )
        if self.hermes.get("agent_id") != "hermes-macro":
            raise ValueError("decision must be authored by hermes-macro")

    @classmethod
    def from_mapping(cls, value: dict) -> "RiskWindowDecision":
        event = value.get("event") if isinstance(value.get("event"), dict) else {}
        return cls(
            decision_id=str(value["decision_id"]),
            event_id=str(value.get("event_id", event.get("event_id", ""))),
            event_kind=str(value.get("event_kind", event.get("kind", ""))).lower(),
            event_at=datetime.fromisoformat(
                str(value.get("event_at", event.get("starts_at", "")))
            ),
            event_source_url=str(
                value.get("event_source_url", event.get("source_url", ""))
            ),
            decision_at=datetime.fromisoformat(str(value["decision_at"])),
            market_impact=str(value["market_impact"]).lower(),
            confidence=float(value["confidence"]),
            effective_at=datetime.fromisoformat(str(value["effective_at"])),
            resume_at=datetime.fromisoformat(str(value["resume_at"])),
            snapshot_id=str(value["snapshot_id"]),
            reason=str(value["reason"]),
            evidence=tuple(
                Evidence.from_mapping(item) for item in value.get("evidence", [])
            ),
            hermes=dict(value.get("hermes", {})),
            approval=(
                dict(value["approval"])
                if isinstance(value.get("approval"), dict)
                else None
            ),
            policy_version=str(value.get("policy_version", "")),
        )

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "event_id": self.event_id,
            "event_kind": self.event_kind,
            "event_at": self.event_at.isoformat(),
            "event_source_url": self.event_source_url,
            "decision_at": self.decision_at.isoformat(),
            "market_impact": self.market_impact,
            "confidence": self.confidence,
            "effective_at": self.effective_at.isoformat(),
            "resume_at": self.resume_at.isoformat(),
            "snapshot_id": self.snapshot_id,
            "reason": self.reason,
            "evidence": [item.to_dict() for item in self.evidence],
            "hermes": dict(self.hermes),
            "policy_version": self.policy_version,
            **(
                {"approval": dict(self.approval)}
                if self.approval is not None
                else {}
            ),
        }


@dataclass(frozen=True)
class RiskWindowAssessment:
    decision: RiskWindowDecision
    accepted: bool
    creates_lease: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.to_dict(),
            "accepted": self.accepted,
            "creates_lease": self.creates_lease,
            "reasons": list(self.reasons),
        }


class RiskWindowPolicy:
    """Validate timing/evidence without changing Hermes' market judgment."""

    OFFICIAL_SOURCE_DOMAINS = {
        "fomc": "federalreserve.gov",
        "cpi": "bls.gov",
        "nfp": "bls.gov",
    }

    def __init__(
        self,
        allowed_event_kinds: Iterable[str] = VALID_EVENT_KINDS,
        *,
        max_pre_event_hours: float = 24,
        max_post_event_hours: float = 6,
    ) -> None:
        self.allowed_event_kinds = {
            str(value).strip().lower() for value in allowed_event_kinds
        }
        self.max_pre_event = timedelta(hours=max_pre_event_hours)
        self.max_post_event = timedelta(hours=max_post_event_hours)

    def assess(self, decision: RiskWindowDecision) -> RiskWindowAssessment:
        reasons: list[str] = []
        if decision.policy_version != POLICY_VERSION:
            reasons.append("policy_version_mismatch")
        if decision.event_kind not in self.allowed_event_kinds:
            reasons.append("event_kind_not_allowed")
        expected_domain = self.OFFICIAL_SOURCE_DOMAINS.get(decision.event_kind)
        source_host = (urlparse(decision.event_source_url).hostname or "").lower()
        if not expected_domain or not (
            source_host == expected_domain
            or source_host.endswith(f".{expected_domain}")
        ):
            reasons.append("official_calendar_source_missing")
        if decision.confidence < MIN_CONFIDENCE:
            reasons.append("confidence_below_0_75")

        source_ids: set[str] = set()
        source_urls: set[str] = set()
        decision_at = _aware(decision.decision_at, "decision_at")
        for item in decision.evidence:
            if _aware(item.observed_at, "evidence observed_at") > decision_at:
                reasons.append(f"evidence_after_decision:{item.source_id}")
                continue
            if item.kind == "decision":
                source_ids.add(item.source_id)
                source_urls.add(item.source_url)
        if len(source_ids) < MIN_DECISION_EVIDENCE:
            reasons.append("fewer_than_two_independent_decision_sources")
        if len(source_urls) < MIN_DECISION_EVIDENCE:
            reasons.append("decision_sources_are_not_independent")

        event_at = _aware(decision.event_at, "event_at")
        effective_at = _aware(decision.effective_at, "effective_at")
        resume_at = _aware(decision.resume_at, "resume_at")
        if decision_at > effective_at:
            reasons.append("decision_after_effective_at")
        if decision_at < event_at - self.max_pre_event:
            reasons.append("decision_too_early")
        if not event_at - self.max_pre_event <= effective_at <= event_at:
            reasons.append("effective_at_outside_event_window")
        if not event_at <= resume_at <= event_at + self.max_post_event:
            reasons.append("resume_at_outside_event_window")
        if decision.event_kind == "fomc":
            ny = ZoneInfo("America/New_York")
            if decision_at.astimezone(ny).date() != (
                event_at.astimezone(ny).date() - timedelta(days=1)
            ):
                reasons.append("fomc_decision_not_prior_new_york_day")

        return RiskWindowAssessment(
            decision=decision,
            accepted=not reasons,
            creates_lease=not reasons and decision.market_impact != "neutral",
            reasons=tuple(reasons or ("validated",)),
        )
