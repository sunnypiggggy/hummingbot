#!/usr/bin/env python3
"""Run the isolated Grid/DCA container scenario and produce audit artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker-compose.risk-scenarios.yml"
PROJECT = "hummingbot-risk-scenarios"


def run(arguments: list[str], *, check: bool = True) -> str:
    result = subprocess.run(
        arguments, cwd=ROOT, text=True, capture_output=True, check=False,
    )
    if check and result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(arguments)}\n"
            f"stdout={result.stdout[-4000:]}\nstderr={result.stderr[-4000:]}"
        )
    return result.stdout.strip()


def compose(*arguments: str, check: bool = True) -> str:
    return run(["docker", "compose", "-f", str(COMPOSE), *arguments], check=check)


def start_only(service: str) -> None:
    """Start exactly one existing container without Compose dependency traversal."""
    container_id = compose("ps", "-q", "--all", service)
    if not container_id:
        raise RuntimeError(f"scenario container does not exist for {service}")
    run(["docker", "start", container_id])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def running_production_containers() -> list[str]:
    values = run(["docker", "ps", "--format", "{{.Names}}"], check=False).splitlines()
    return sorted(name for name in values if not name.startswith(f"{PROJECT}-"))


def wait_exit(service: str, timeout: float = 120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        container_id = compose("ps", "-q", "--all", service)
        if container_id:
            state = run(["docker", "inspect", "-f", "{{.State.Status}} {{.State.ExitCode}}", container_id])
            status, code = state.split()
            if status == "exited":
                if code != "0":
                    raise RuntimeError(f"{service} exited {code}:\n{compose('logs', service)}")
                return
        time.sleep(1)
    raise TimeoutError(f"timed out waiting for {service}")


def wait_exit_code(service: str, timeout: float = 120) -> int:
    deadline = time.time() + timeout
    while time.time() < deadline:
        container_id = compose("ps", "-q", "--all", service)
        if container_id:
            state = run(
                ["docker", "inspect", "-f", "{{.State.Status}} {{.State.ExitCode}}", container_id]
            )
            status, code = state.split()
            if status == "exited":
                return int(code)
        time.sleep(1)
    raise TimeoutError(f"timed out waiting for {service}")


def validate_isolation() -> None:
    document = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    network = document["networks"]["risk-scenario"]
    if network.get("internal") is not True:
        raise RuntimeError("risk scenario network must remain internal")
    serialized = json.dumps(document, sort_keys=True)
    forbidden = [
        "/var/run/docker.sock", "api.binance.com", "api.telegram.org",
        "dca_binance_emergency_credentials.json", "grid_binance_emergency_credentials.json",
    ]
    hits = [value for value in forbidden if value in serialized]
    if hits:
        raise RuntimeError(f"scenario compose references forbidden production resource: {hits}")
    if "GUARD_SCENARIO_MODE" not in serialized:
        raise RuntimeError("scenario interlock is absent")


def simulator_state() -> dict:
    command = (
        "import json,urllib.request;"
        "print(json.dumps(json.load(urllib.request.urlopen('http://localhost:8080/scenario/state'))))"
    )
    return json.loads(compose("exec", "-T", "binance-sim", "python", "-c", command))


def assertions(state: dict, extracted: Path) -> list[str]:
    checks = []
    btc_orders = [row for row in state["orders"] if row["symbol"] == "BTCUSDT"]
    eth_orders = [row for row in state["orders"] if row["symbol"] == "ETHUSDT"]
    assert len(btc_orders) == 1, btc_orders
    checks.append("exactly one BTC economic fill")
    assert not eth_orders
    checks.append("ETH remained dust with zero orders")
    assert btc_orders[0]["executedQty"] == "0.00155"
    assert btc_orders[0]["cummulativeQuoteQty"] == "101.1330050"
    checks.append("BTC quantity and USDT proceeds match the redacted incident")
    assert state["open_orders"] == []
    checks.append("no exchange orders remain active")
    dca_state = json.loads((extracted / "dca" / "guard_state.json").read_text(encoding="utf-8"))
    for bot in dca_state["bots"].values():
        assert bot.get("manual_exit_required") is True
        assert bot.get("exit_status") == "pending_manual_existing_dca_inventory"
    checks.append("pre-existing DCA LATCHED inventory was not retroactively sold")
    inventory = json.loads(
        (extracted / "account-inventory" / "account_inventory_status.json").read_text(encoding="utf-8")
    )
    assert inventory["open_order_counts"] == {
        "BTC-FDUSD": 0, "BTC-USDT": 0, "ETH-FDUSD": 0, "ETH-USDT": 0,
    }
    checks.append("final shared inventory contract has zero open orders")
    assert state["telegram"]
    assert len(state["telegram"]) == 7, state["telegram"]
    outbox_path = extracted / "dca" / "telegram" / "telegram_outbox.sqlite"
    if not outbox_path.is_file():
        raise FileNotFoundError(f"Telegram outbox was not preserved: {outbox_path}")
    with sqlite3.connect(f"file:{outbox_path}?mode=ro", uri=True) as connection:
        pending = connection.execute(
            "SELECT COUNT(*) FROM outbox WHERE status='pending'"
        ).fetchone()[0]
        sent, distinct_ids = connection.execute(
            "SELECT COUNT(*),COUNT(DISTINCT id) FROM outbox WHERE status='sent'"
        ).fetchone()
    assert pending == 0
    assert sent == distinct_ids == 7
    checks.append("Telegram 429 survived report restart: 7 delivered, zero pending/duplicates")
    return checks


def write_timeline(state: dict, extracted: Path, destination: Path) -> None:
    """Write an auditable UTC sequence from exchange, ledger, and notification evidence."""
    entries: list[tuple[float, str, str]] = []
    for row in state.get("orders", []):
        entries.append((
            float(row.get("transactTime", 0)) / 1000,
            "Binance order",
            f"{row.get('symbol')} {row.get('status')} client={row.get('clientOrderId')} "
            f"executed={row.get('executedQty')} quote={row.get('cummulativeQuoteQty')}",
        ))
    for row in state.get("trades", []):
        entries.append((
            float(row.get("time", 0)) / 1000,
            "Binance trade",
            f"{row.get('symbol')} order={row.get('orderId')} qty={row.get('qty')} "
            f"fee={row.get('commission')} {row.get('commissionAsset')}",
        ))
    for row in state.get("telegram", []):
        entries.append((
            float(row.get("received_at", 0)), "Telegram sink",
            f"{row.get('method')} message_id={row.get('message_id')} "
            f"body_sha256={row.get('body_sha256')}",
        ))
    database = extracted / "account-inventory" / "account_inventory.sqlite"
    with sqlite3.connect(database) as connection:
        for job_id, asset, scope, status, created_at, updated_at in connection.execute(
            "SELECT job_id,asset,scope,status,created_at,updated_at "
            "FROM liquidation_jobs ORDER BY created_at,job_id"
        ):
            entries.append((
                float(created_at), "Inventory job created",
                f"{asset} scope={scope} status={status} job={job_id}",
            ))
            entries.append((
                float(updated_at), "Inventory job final",
                f"{asset} scope={scope} status={status} job={job_id}",
            ))
    for event_path in sorted(extracted.rglob("telegram_events.jsonl")):
        for raw in event_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(raw)
                occurred = datetime.fromisoformat(
                    str(event["occurred_at"]).replace("Z", "+00:00")
                ).timestamp()
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            entries.append((
                occurred, "Risk event",
                f"{event.get('transition')} {event.get('pair')} id={event.get('event_id')}",
            ))
    lines = [
        "# Grid/DCA real scenario timeline", "",
        "All timestamps are UTC. Sensitive account and credential identifiers are absent.", "",
        "| UTC time | Source | Evidence |", "|---|---|---|",
    ]
    for timestamp, source, evidence in sorted(entries, key=lambda item: (item[0], item[1], item[2])):
        shown = datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
        lines.append(f"| `{shown}` | {source} | {evidence} |")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "risk_scenarios" / "aug10-container-replay")
    parser.add_argument("--skip-kill", action="store_true")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--soak-seconds", type=int, default=0)
    args = parser.parse_args()
    validate_isolation()
    allowed_output_root = (ROOT / "results" / "risk_scenarios").resolve()
    resolved_output = args.output.resolve()
    if resolved_output != allowed_output_root and allowed_output_root not in resolved_output.parents:
        raise RuntimeError(
            f"scenario output must stay under {allowed_output_root}: {resolved_output}"
        )
    args.output = resolved_output
    before = running_production_containers()
    args.output.mkdir(parents=True, exist_ok=True)
    extracted = args.output / "state"
    if extracted.exists():
        shutil.rmtree(extracted)
    started = datetime.now(timezone.utc)
    try:
        compose("down", "--volumes", "--remove-orphans", check=False)
        compose("build", "grid-guard-scenario", "dca-guard-scenario")
        compose("up", "-d", "binance-sim", "scenario-init")
        wait_exit("scenario-init", timeout=30)
        compose("up", "-d", "grid-guard-scenario")
        wait_exit("grid-guard-scenario", timeout=60)
        compose("up", "-d", "sqlite-lock-scenario")
        lock_deadline = time.time() + 15
        while "SQLITE_LOCK_ACQUIRED" not in compose("logs", "sqlite-lock-scenario"):
            if time.time() >= lock_deadline:
                raise TimeoutError("SQLite scenario lock was not acquired")
            time.sleep(0.2)
        compose("up", "-d", "dca-guard-scenario")
        if not args.skip_kill:
            time.sleep(10)
            compose("kill", "-s", "KILL", "dca-guard-scenario")
            start_only("dca-guard-scenario")
            # Disconnect only the scenario container from its internal network.
            # The process must fail closed, then recover from the same volume
            # after reconnection without creating another economic fill.
            time.sleep(3)
            container_id = compose("ps", "-q", "--all", "dca-guard-scenario")
            network = run([
                "docker", "inspect", "-f",
                "{{range $key, $value := .NetworkSettings.Networks}}{{$key}}{{end}}",
                container_id,
            ])
            run(["docker", "network", "disconnect", network, container_id])
            interrupted_code = wait_exit_code("dca-guard-scenario", timeout=45)
            if interrupted_code == 0:
                raise AssertionError("network-isolated Guard unexpectedly exited successfully")
            run(["docker", "network", "connect", network, container_id])
            start_only("dca-guard-scenario")
        wait_exit("dca-guard-scenario", timeout=120)
        soak_runs = 0
        soak_deadline = time.time() + max(args.soak_seconds, 0)
        while time.time() < soak_deadline:
            start_only("dca-guard-scenario")
            wait_exit("dca-guard-scenario", timeout=120)
            current = simulator_state()
            btc_orders = [row for row in current["orders"] if row["symbol"] == "BTCUSDT"]
            if len(btc_orders) != 1:
                raise AssertionError(f"soak restart created duplicate BTC fills: {btc_orders}")
            soak_runs += 1
        compose("up", "-d", "report-scenario")
        wait_exit("report-scenario", timeout=60)
        # The fixture makes the first Telegram request return HTTP 429.  The
        # first report process persists that row with a five-second backoff;
        # a real container restart must drain it without re-ingesting or
        # duplicating the six rows already sent.
        time.sleep(6)
        start_only("report-scenario")
        wait_exit("report-scenario", timeout=60)
        state = simulator_state()
        container_id = compose("ps", "-q", "--all", "dca-guard-scenario")
        run(["docker", "cp", f"{container_id}:/scenario", str(extracted)])
        checks = assertions(state, extracted)
        after = running_production_containers()
        assert before == after, {"before": before, "after": after}
        checks.append("production container set unchanged")
        files = {
            str(path.relative_to(args.output)): sha256(path)
            for path in sorted(extracted.rglob("*")) if path.is_file()
        }
        timeline_path = args.output / "timeline.md"
        write_timeline(state, extracted, timeline_path)
        report = {
            "schema": "grid-dca-real-scenario-report-v1",
            "scenario_id": "aug10-container-replay",
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "verdict": "PASS", "checks": checks,
            "soak_seconds_requested": args.soak_seconds,
            "soak_restart_runs": soak_runs,
            "production_containers_before": before,
            "production_containers_after": after,
            "state_file_sha256": files,
            "timeline_sha256": sha256(timeline_path),
            "simulator_state": state,
        }
        report_path = args.output / "report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        markdown = [
            "# Grid/DCA real scenario acceptance", "",
            f"- Verdict: **{report['verdict']}**",
            f"- Scenario: `{report['scenario_id']}`",
            f"- Report SHA-256: `{sha256(report_path)}`",
            f"- Timeline SHA-256: `{report['timeline_sha256']}`", "", "## Checks", "",
            *[f"- [x] {item}" for item in checks],
        ]
        (args.output / "report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
        print(json.dumps({
            "verdict": "PASS", "report": str(report_path),
            "sha256": sha256(report_path), "checks": len(checks),
        }))
        return 0
    finally:
        if not args.keep:
            compose("down", "--volumes", "--remove-orphans", check=False)


if __name__ == "__main__":
    raise SystemExit(main())
