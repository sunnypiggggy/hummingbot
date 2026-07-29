from __future__ import annotations

import argparse
import json
import os
import ssl
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .approval import proposal_payload, proposal_sha256
from .calendar import load_calendar, sync_official_calendar
from .ledger import (
    append_record,
    load_records,
    new_record,
    proposal_record,
    replay_records,
    validate_records,
)
from .hermes_conversation import (
    create_approval_request,
    dumps_request,
    resolve_approval_request,
)
from .policy import (
    RiskWindowDecision,
    RiskWindowPolicy,
)
from .security import sign_request
from .telegram_bot import DEFAULT_APPROVAL_TIMEOUT_SECONDS, TelegramApprovalBot


def signed_request(
    base_url: str,
    secret: str,
    method: str,
    path: str,
    payload: dict | None = None,
    timeout: float = 15,
    key_id: str = "",
    client_cert: str = "",
    client_key: str = "",
    ca_file: str = "",
) -> dict:
    body = b"" if payload is None else json.dumps(
        payload, separators=(",", ":")
    ).encode()
    timestamp = str(time.time())
    nonce = uuid.uuid4().hex
    signature = sign_request(secret, method, path, timestamp, nonce, body)
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=body if method != "GET" else None,
        method=method,
        headers={
            "Content-Type": "application/json",
            "X-Hermes-Timestamp": timestamp,
            "X-Hermes-Nonce": nonce,
            "X-Hermes-Signature": signature,
            "X-Hermes-Key-Id": key_id,
        },
    )
    context = None
    if base_url.lower().startswith("https://"):
        context = ssl.create_default_context(cafile=ca_file or None)
        if client_cert:
            context.load_cert_chain(client_cert, client_key or None)
    try:
        with urllib.request.urlopen(
            request, timeout=timeout, context=context
        ) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"gateway returned HTTP {exc.code}: {detail}") from exc
    return {} if not raw else json.loads(raw.decode())


def signed_download(
    base_url: str,
    secret: str,
    path: str,
    *,
    timeout: float = 15,
    key_id: str = "",
    client_cert: str = "",
    client_key: str = "",
    ca_file: str = "",
) -> bytes:
    body = b""
    timestamp = str(time.time())
    nonce = uuid.uuid4().hex
    signature = sign_request(secret, "GET", path, timestamp, nonce, body)
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        method="GET",
        headers={
            "X-Hermes-Timestamp": timestamp,
            "X-Hermes-Nonce": nonce,
            "X-Hermes-Signature": signature,
            "X-Hermes-Key-Id": key_id,
        },
    )
    context = None
    if base_url.lower().startswith("https://"):
        context = ssl.create_default_context(cafile=ca_file or None)
        if client_cert:
            context.load_cert_chain(client_cert, client_key or None)
    try:
        with urllib.request.urlopen(
            request, timeout=timeout, context=context
        ) as response:
            value = response.read()
            if response.headers.get_content_type() != "image/png":
                raise RuntimeError("gateway trading chart is not image/png")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"gateway returned HTTP {exc.code}: {detail}") from exc
    if not value.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("gateway trading chart has an invalid PNG signature")
    return value


def default_chart_path() -> Path:
    directory = Path(
        os.environ.get(
            "HERMES_DCA_CHART_CACHE",
            "~/.hermes/cache/dca-macro-control",
        )
    ).expanduser()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return directory / f"dca-trades-7d-{stamp}.png"


def clean_chart_cache(directory: Path, *, now: float | None = None) -> None:
    now = time.time() if now is None else now
    if not directory.exists():
        return
    for path in directory.glob("dca-trades-7d-*.png"):
        try:
            if path.is_file() and path.stat().st_mtime < now - 86_400:
                path.unlink()
        except OSError:
            continue


def load_dossier(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    value.setdefault("snapshot_id", "pending-live-telemetry")
    return RiskWindowDecision.from_mapping(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hermes DCA macro event control V3")
    parser.add_argument(
        "--gateway-url", default=os.environ.get("DCA_MACRO_GATEWAY_URL", "")
    )
    parser.add_argument(
        "--hmac-secret", default=os.environ.get("HERMES_HMAC_SECRET", "")
    )
    parser.add_argument(
        "--key-id", default=os.environ.get("HERMES_HMAC_KEY_ID", "primary")
    )
    parser.add_argument(
        "--client-cert", default=os.environ.get("HERMES_CLIENT_CERT", "")
    )
    parser.add_argument(
        "--client-key", default=os.environ.get("HERMES_CLIENT_KEY", "")
    )
    parser.add_argument(
        "--ca-file", default=os.environ.get("HERMES_GATEWAY_CA_FILE", "")
    )
    parser.add_argument(
        "--calendar", type=Path, default=Path("data/hermes/official_events.json")
    )
    parser.add_argument(
        "--telegram-token", default=os.environ.get("TELEGRAM_TOKEN", "")
    )
    parser.add_argument(
        "--telegram-user-id",
        default=os.environ.get("HERMES_APPROVER_TELEGRAM_USER_ID", ""),
    )
    parser.add_argument(
        "--telegram-chat-id",
        default=os.environ.get("HERMES_APPROVER_TELEGRAM_CHAT_ID", ""),
    )
    parser.add_argument(
        "--conversation-surface",
        choices=("telegram", "cli", "dashboard"),
        default=os.environ.get("HERMES_CONVERSATION_SURFACE", "telegram"),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sync = sub.add_parser("sync-calendar")
    sync.add_argument("--years", default=str(datetime.now().year))

    due = sub.add_parser("due-events")
    due.add_argument("--window-minutes", type=int, default=60)
    due.add_argument("--at")

    for name in ("validate-dossier", "submit-dossier", "submit-approved"):
        command = sub.add_parser(name)
        command.add_argument("--dossier", type=Path, required=True)
        if name == "submit-approved":
            command.add_argument(
                "--ledger",
                type=Path,
                default=Path("data/hermes/dca_macro_v3/event_ledger.jsonl"),
            )

    approve = sub.add_parser("approve")
    approve.add_argument("--dossier", type=Path, required=True)
    approve.add_argument("--output", type=Path, required=True)
    approve.add_argument(
        "--ledger",
        type=Path,
        default=Path("data/hermes/dca_macro_v3/event_ledger.jsonl"),
    )
    approve.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_APPROVAL_TIMEOUT_SECONDS,
    )

    conversation_request = sub.add_parser("conversation-request")
    conversation_request.add_argument("--dossier", type=Path, required=True)
    conversation_request.add_argument("--output", type=Path, required=True)
    conversation_request.add_argument(
        "--action", choices=("approve", "revoke"), default="approve"
    )

    conversation_resolve = sub.add_parser("conversation-resolve")
    conversation_resolve.add_argument("--request", type=Path, required=True)
    conversation_resolve.add_argument("--response", required=True)
    conversation_resolve.add_argument("--output", type=Path, required=True)
    conversation_resolve.add_argument(
        "--ledger",
        type=Path,
        default=Path("data/hermes/dca_macro_v3/event_ledger.jsonl"),
    )

    revoke = sub.add_parser("revoke")
    revoke.add_argument("decision_id")
    revoke.add_argument("--dossier", type=Path, required=True)
    revoke.add_argument(
        "--ledger",
        type=Path,
        default=Path("data/hermes/dca_macro_v3/event_ledger.jsonl"),
    )
    revoke.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_APPROVAL_TIMEOUT_SECONDS,
    )
    revoke_approved = sub.add_parser("revoke-approved")
    revoke_approved.add_argument("decision_id")
    revoke_approved.add_argument("--approval", type=Path, required=True)
    revoke_approved.add_argument(
        "--ledger",
        type=Path,
        default=Path("data/hermes/dca_macro_v3/event_ledger.jsonl"),
    )

    ledger_validate = sub.add_parser("ledger-validate")
    ledger_validate.add_argument(
        "--ledger",
        type=Path,
        default=Path("data/hermes/dca_macro_v3/event_ledger.jsonl"),
    )
    ledger_replay = sub.add_parser("ledger-replay")
    ledger_replay.add_argument(
        "--ledger",
        type=Path,
        default=Path("data/hermes/dca_macro_v3/event_ledger.jsonl"),
    )
    ledger_replay.add_argument("--output", type=Path)

    heartbeat = sub.add_parser("heartbeat")
    heartbeat.add_argument("--decision-id", default="")
    sub.add_parser("status")
    report = sub.add_parser("report")
    report.add_argument("--chart-output", type=Path)
    sub.add_parser("telemetry")
    resolve = sub.add_parser("resolve")
    resolve.add_argument("decision_id")
    return parser


def _require_gateway(args) -> None:
    if not args.gateway_url or not args.hmac_secret:
        raise RuntimeError("--gateway-url and --hmac-secret are required")


def _telegram_bot(args) -> TelegramApprovalBot:
    return TelegramApprovalBot(
        args.telegram_token,
        args.telegram_user_id,
        args.telegram_chat_id,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "sync-calendar":
        years = {int(value.strip()) for value in args.years.split(",")}
        events = sync_official_calendar(args.calendar, years)
        print(json.dumps({"events": len(events), "output": str(args.calendar)}))
        return 0

    if args.command == "due-events":
        now = (
            datetime.fromisoformat(args.at)
            if args.at
            else datetime.now(timezone.utc)
        )
        if now.tzinfo is None:
            raise ValueError("--at must include a timezone")
        window = timedelta(minutes=args.window_minutes)
        due = [
            event.to_dict()
            for event in load_calendar(args.calendar)
            if abs((event.starts_at - timedelta(hours=24) - now).total_seconds())
            <= window.total_seconds()
        ]
        print(json.dumps({"at": now.isoformat(), "events": due}, indent=2))
        return 0

    if args.command in {"validate-dossier", "submit-dossier", "submit-approved"}:
        dossier = load_dossier(args.dossier)
        assessment = RiskWindowPolicy().assess(dossier)
        if args.command == "validate-dossier":
            print(json.dumps(assessment.to_dict(), indent=2))
            return 0
        _require_gateway(args)
        payload = dossier.to_dict()
        if args.command == "submit-approved":
            telemetry = signed_request(
                args.gateway_url,
                args.hmac_secret,
                "GET",
                "/v1/telemetry",
                key_id=args.key_id,
                client_cert=args.client_cert,
                client_key=args.client_key,
                ca_file=args.ca_file,
            )
            payload["snapshot_id"] = telemetry["snapshot_id"]
        result = signed_request(
            args.gateway_url,
            args.hmac_secret,
            "POST",
            "/v1/event-decisions",
            payload,
            key_id=args.key_id,
            client_cert=args.client_cert,
            client_key=args.client_key,
            ca_file=args.ca_file,
        )
        if args.command == "submit-approved":
            append_record(
                args.ledger,
                new_record(
                    "execution",
                    proposal_id=payload["decision_id"],
                    proposal_sha256=proposal_sha256(payload),
                    submitted_at=datetime.now(timezone.utc).isoformat(),
                    snapshot_id=payload["snapshot_id"],
                    gateway_result=result,
                ),
            )
    elif args.command == "conversation-request":
        value = json.loads(args.dossier.read_text(encoding="utf-8"))
        request = create_approval_request(
            value,
            action=args.action,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(dumps_request(request), encoding="utf-8")
        result = {
            "interaction_id": request["interaction_id"],
            "proposal_sha256": request["proposal_sha256"],
            "prompt": request["prompt"],
            "choices": request["choices"],
            "request": str(args.output),
        }
    elif args.command == "conversation-resolve":
        request = json.loads(args.request.read_text(encoding="utf-8"))
        approval = resolve_approval_request(
            request,
            args.response,
            approver_user_id=args.telegram_user_id,
            chat_id=args.telegram_chat_id,
            surface=args.conversation_surface,
        )
        proposal = request["proposal"]
        existing = load_records(args.ledger)
        if not any(
            record.get("record_type") == "proposal"
            and record.get("proposal_sha256") == request["proposal_sha256"]
            for record in existing
        ):
            append_record(args.ledger, proposal_record(proposal))
        append_record(
            args.ledger,
            new_record(
                "approval",
                proposal_id=proposal["decision_id"],
                proposal_sha256=request["proposal_sha256"],
                approval=approval,
            ),
        )
        output = (
            {**proposal, "approval": approval}
            if request["action"] == "approve"
            else {"approval": approval}
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        result = {
            "status": approval["status"],
            "action": approval["action"],
            "proposal_sha256": approval["proposal_sha256"],
            "output": str(args.output),
            "ledger": str(args.ledger),
        }
    elif args.command == "approve":
        value = json.loads(args.dossier.read_text(encoding="utf-8"))
        proposal = proposal_payload(value)
        bot = _telegram_bot(args)
        request_record = bot.request_approval(proposal)
        approval = bot.wait_for_approval(
            request_record,
            timeout_seconds=args.timeout_seconds,
        )
        existing = load_records(args.ledger)
        if not any(
            record.get("record_type") == "proposal"
            and record.get("proposal_sha256") == request_record["proposal_sha256"]
            for record in existing
        ):
            append_record(args.ledger, proposal_record(proposal))
        append_record(
            args.ledger,
            new_record(
                "approval",
                proposal_id=proposal["decision_id"],
                proposal_sha256=request_record["proposal_sha256"],
                approval=approval,
            ),
        )
        output = {**value, "approval": approval}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        result = {
            "status": approval["status"],
            "proposal_sha256": request_record["proposal_sha256"],
            "output": str(args.output),
            "ledger": str(args.ledger),
        }
    elif args.command == "revoke-approved":
        _require_gateway(args)
        value = json.loads(args.approval.read_text(encoding="utf-8"))
        approval = value.get("approval", value)
        if approval.get("status") != "approved":
            result = {"status": "rejected_no_change"}
        else:
            result = signed_request(
                args.gateway_url,
                args.hmac_secret,
                "POST",
                f"/v1/decisions/{args.decision_id}/revoke",
                {"approval": approval},
                key_id=args.key_id,
                client_cert=args.client_cert,
                client_key=args.client_key,
                ca_file=args.ca_file,
            )
            append_record(
                args.ledger,
                new_record(
                    "revocation",
                    proposal_id=args.decision_id,
                    proposal_sha256=approval["proposal_sha256"],
                    revoked_at=result["revoked_at"],
                    approval=approval,
                    gateway_result=result,
                ),
            )
    elif args.command == "revoke":
        _require_gateway(args)
        value = json.loads(args.dossier.read_text(encoding="utf-8"))
        proposal = proposal_payload(value)
        if proposal["decision_id"] != args.decision_id:
            raise ValueError("revoke decision_id does not match dossier")
        bot = _telegram_bot(args)
        request_record = bot.request_approval(proposal, action="revoke")
        approval = bot.wait_for_approval(
            request_record,
            timeout_seconds=args.timeout_seconds,
        )
        if approval["status"] != "approved":
            result = {"status": "rejected_no_change"}
        else:
            result = signed_request(
                args.gateway_url,
                args.hmac_secret,
                "POST",
                f"/v1/decisions/{args.decision_id}/revoke",
                {"approval": approval},
                key_id=args.key_id,
                client_cert=args.client_cert,
                client_key=args.client_key,
                ca_file=args.ca_file,
            )
            append_record(
                args.ledger,
                new_record(
                    "revocation",
                    proposal_id=args.decision_id,
                    proposal_sha256=proposal_sha256(proposal),
                    revoked_at=result["revoked_at"],
                    approval=approval,
                    gateway_result=result,
                ),
            )
    elif args.command == "ledger-validate":
        result = validate_records(load_records(args.ledger))
    elif args.command == "ledger-replay":
        result = replay_records(load_records(args.ledger))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    elif args.command == "heartbeat":
        _require_gateway(args)
        result = signed_request(
            args.gateway_url,
            args.hmac_secret,
            "POST",
            "/v1/heartbeat",
            {"decision_id": args.decision_id},
            key_id=args.key_id,
            client_cert=args.client_cert,
            client_key=args.client_key,
            ca_file=args.ca_file,
        )
    elif args.command == "resolve":
        _require_gateway(args)
        result = signed_request(
            args.gateway_url,
            args.hmac_secret,
            "GET",
            f"/v1/decisions/{args.decision_id}",
            key_id=args.key_id,
            client_cert=args.client_cert,
            client_key=args.client_key,
            ca_file=args.ca_file,
        )
    elif args.command == "report":
        _require_gateway(args)
        request_options = {
            "key_id": args.key_id,
            "client_cert": args.client_cert,
            "client_key": args.client_key,
            "ca_file": args.ca_file,
        }
        components: dict[str, dict | None] = {}
        warnings: list[str] = []
        for name, path in (
            ("status", "/v1/status"),
            ("telemetry", "/v1/telemetry"),
            ("trading_report", "/v1/trading-report"),
        ):
            try:
                components[name] = signed_request(
                    args.gateway_url,
                    args.hmac_secret,
                    "GET",
                    path,
                    **request_options,
                )
            except Exception as exc:
                components[name] = None
                warnings.append(f"{name}: {exc}")

        chart_path: Path | None = None
        try:
            chart = signed_download(
                args.gateway_url,
                args.hmac_secret,
                "/v1/trading-chart",
                **request_options,
            )
            chart_path = (
                args.chart_output.expanduser()
                if args.chart_output is not None
                else default_chart_path()
            ).resolve()
            chart_path.parent.mkdir(parents=True, exist_ok=True)
            clean_chart_cache(chart_path.parent)
            temporary = chart_path.with_suffix(chart_path.suffix + ".tmp")
            temporary.write_bytes(chart)
            temporary.replace(chart_path)
        except Exception as exc:
            warnings.append(f"trading_chart: {exc}")

        if not any(value is not None for value in components.values()):
            raise RuntimeError(
                "DCA report unavailable: " + "; ".join(warnings)
            )
        result = {
            "schema_version": 3,
            "policy_version": "dca-macro-v3",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **components,
            "chart_path": str(chart_path) if chart_path else None,
            "warnings": warnings,
            "read_only": True,
        }
    elif args.command == "status":
        _require_gateway(args)
        result = signed_request(
            args.gateway_url,
            args.hmac_secret,
            "GET",
            "/v1/status",
            key_id=args.key_id,
            client_cert=args.client_cert,
            client_key=args.client_key,
            ca_file=args.ca_file,
        )
    else:
        _require_gateway(args)
        result = signed_request(
            args.gateway_url,
            args.hmac_secret,
            "GET",
            "/v1/telemetry",
            key_id=args.key_id,
            client_cert=args.client_cert,
            client_key=args.client_key,
            ca_file=args.ca_file,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
