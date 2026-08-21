from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class ApprovalStore:
    """Read public model evidence and write only hash-bound operator decisions."""

    def __init__(self, request_root: Path, evidence_root: Path, decision_root: Path):
        self.request_root = request_root
        self.evidence_root = evidence_root
        self.decision_root = decision_root

    def _requests(self) -> list[tuple[Path, dict]]:
        values: list[tuple[Path, dict]] = []
        if not self.request_root.exists():
            return values
        patterns = ("approval-request-*.json", "*.model-approval-request.json")
        seen: set[Path] = set()
        for pattern in patterns:
            for path in self.request_root.rglob(pattern):
                if path in seen:
                    continue
                seen.add(path)
                payload = _json(path)
                release = str(payload.get("release_sha256", ""))
                if release:
                    values.append((path, payload))
        return sorted(values, key=lambda item: int(item[1].get("review_started_at", 0) or 0), reverse=True)

    def pending(self) -> list[dict]:
        state = _json(self.request_root / "automation_state.json")
        pending_release = str(state.get("candidate_release_sha256", "")) if state.get("phase") == "AWAITING_APPROVAL" else ""
        result = []
        for path, request in self._requests():
            release = str(request["release_sha256"])
            decision = _json(self.decision_root / f"{release}.json")
            is_pending = release == pending_release and decision.get("decision") not in {"approve", "reject"}
            result.append({
                "candidate_id": release[:16],
                "model_type": str(request.get("model_type", "v22_weekly_buy_gate")),
                "release_sha256": release,
                "model_sha256": str(request.get("model_sha256", "")),
                "review_started_at": request.get("review_started_at"),
                "review_deadline": request.get("review_deadline"),
                "effective_start": request.get("activation_boundary"),
                "effective_end": request.get("candidate_effective_end"),
                "checks": request.get("checks", {}),
                "default_decision": request.get("default_decision", "approve"),
                "request_sha256": _sha(path),
                "request_path": str(path),
                "status": "PENDING" if is_pending else decision.get("decision", "HISTORY").upper(),
            })
        return result

    def find(self, candidate_id: str) -> Optional[dict]:
        for candidate in self.pending():
            if candidate["candidate_id"] == candidate_id or candidate["release_sha256"] == candidate_id:
                return candidate
        return None

    def evidence(self, candidate: dict) -> dict:
        release = candidate["release_sha256"]
        paths: set[Path] = set()
        if self.evidence_root.exists():
            for path in self.evidence_root.rglob("*"):
                relative = str(path.relative_to(self.evidence_root)) if path.is_file() else ""
                if path.is_file() and release in relative:
                    paths.add(path)
                if path.is_file() and path.suffix.lower() == ".json":
                    payload = _json(path)
                    if str(payload.get("release_sha256", "")) != release:
                        continue
                    paths.add(path)
                    for attachment in payload.get("attachments", []):
                        if not isinstance(attachment, dict):
                            continue
                        name = Path(str(attachment.get("path", ""))).name
                        if name:
                            paths.update(candidate_path for candidate_path in self.evidence_root.rglob(name) if candidate_path.is_file())
        attachments = [{
            "name": path.name,
            "path": str(path),
            "sha256": _sha(path),
            "size": path.stat().st_size,
        } for path in sorted(paths)]
        return {"candidate": candidate, "attachments": attachments}

    def decide(self, candidate: dict, decision: str, *, operator: str, reason: str,
               telegram_user_id: int, telegram_chat_id: int, telegram_update_id: int,
               telegram_callback_query_id: str) -> dict:
        if decision not in {"approve", "reject"}:
            raise ValueError("decision must be approve or reject")
        if decision == "reject" and not reason.strip():
            raise ValueError("reject decision requires a reason")
        current = self.find(candidate["candidate_id"])
        if not current or current.get("status") != "PENDING":
            raise RuntimeError("candidate is no longer pending")
        payload = {
            "schema": "ethbtc-forced-exit-review-decision-v1",
            "candidate_id": candidate["candidate_id"],
            "model_type": candidate["model_type"],
            "release_sha256": candidate["release_sha256"],
            "model_sha256": candidate["model_sha256"],
            "approval_request_sha256": candidate["request_sha256"],
            "decision": decision,
            "operator": operator,
            "reason": reason.strip(),
            "decided_at": int(time.time()),
            "telegram_user_id": str(telegram_user_id),
            "telegram_chat_id": str(telegram_chat_id),
            "telegram_update_id": str(telegram_update_id),
            "telegram_callback_query_id": telegram_callback_query_id,
        }
        history = self.decision_root / f"{candidate['release_sha256']}.json"
        generic = self.decision_root / "review_decision.json"
        if history.exists():
            existing = _json(history)
            if existing != payload:
                raise RuntimeError("candidate already has a different decision")
            return existing
        _atomic_json(history, payload)
        _atomic_json(generic, payload)
        return payload
