"""Run only a named isolated synthetic reader, never a production service."""
import json
import contextlib
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
from cardputer_monitor.collector import ROBOTS, STATUS_SCHEMA  # noqa: E402
from live_guard.trading_status import evaluate_status, gate_row  # noqa: E402

NAME = "cardputer-monitor-acceptance"
IMAGE = "hummingbot/cardputer-monitor:acceptance"


def docker(*args, check=True, timeout=60):
    return subprocess.run(["docker", *args], text=True, capture_output=True, check=check, timeout=timeout)


def document(observed, stopped=False):
    rows = [evaluate_status(
        generated_at=datetime.fromtimestamp(observed, timezone.utc).isoformat(),
        strategy=strategy, bot=bot, pair=pair, process_running=not (stopped and index == 0),
        phase="ACTIVE", gates=[gate_row("v22_weekly_buy_gate"), gate_row("controller_application_gate")],
    ) for index, (strategy, pair, _, bot) in enumerate(ROBOTS)]
    return {"schema": STATUS_SCHEMA, "generated_at": rows[0]["generated_at"], "robots": rows}


def main():
    if docker("container", "inspect", NAME, check=False).returncode == 0:
        raise RuntimeError("acceptance container name already exists; refusing to touch it")
    root = Path(tempfile.mkdtemp(prefix="cardputer-monitor-acceptance-")).resolve()
    nonce = uuid.uuid4().hex
    database = root / "telegram_outbox.sqlite"
    observed = time.time()
    keeper = sqlite3.connect(database)
    stop = threading.Event()
    errors = []
    created = False
    writer = None
    try:
        keeper.execute("PRAGMA journal_mode=WAL")
        keeper.execute("CREATE TABLE profit_snapshot(strategy TEXT,pair TEXT,observed_at REAL,mtm_quote REAL,equity REAL,drawdown_pct REAL,payload_json TEXT,PRIMARY KEY(strategy,pair,observed_at))")
        for strategy, pair, _, _ in ROBOTS:
            for hours, value in ((168, 0), (24, 1), (4, 2), (0, 10)):
                keeper.execute("INSERT INTO profit_snapshot VALUES(?,?,?,?,?,?,?)", (strategy, pair, observed - hours * 3600, value, 200 + value, 0.5, "{}"))
            for index in range(73):
                keeper.execute("INSERT INTO profit_snapshot VALUES(?,?,?,?,?,?,?)", (
                    strategy, pair, int(observed) - 43200 + index * 600 - 10,
                    index / 72 * 10, 200, 0.5, "{}",
                ))
        keeper.commit()
        (root / "trading_status.json").write_text(json.dumps(document(observed)), encoding="utf-8")

        def write_synthetic():
            try:
                with contextlib.closing(sqlite3.connect(database, timeout=3)) as db:
                    sequence = 0
                    while not stop.is_set():
                        sequence += 1
                        db.execute("BEGIN IMMEDIATE")
                        db.execute("UPDATE profit_snapshot SET mtm_quote=? WHERE observed_at=?", (10 + sequence, observed))
                        db.commit()
                        temporary = root / "status.tmp"
                        temporary.write_text(json.dumps(document(observed, stopped=bool(sequence % 2))), encoding="utf-8")
                        try:
                            temporary.replace(root / "trading_status.json")
                        except PermissionError:
                            # Windows may briefly hold an atomic-read handle.
                            pass
                        stop.wait(0.025)
            except Exception as exc:
                errors.append(type(exc).__name__)

        writer = threading.Thread(target=write_synthetic, daemon=True)
        writer.start()
        helper = Path(__file__).with_name("linux_reader.py").resolve()
        creation = docker(
            "create", "--name", NAME, "--label", f"cardputer.acceptance={nonce}",
            "--network", "none", "--read-only", "--user", "10001:10001", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges:true", "--tmpfs", "/tmp",
            "--mount", f"type=bind,source={root},target=/reports,readonly",
            "--mount", f"type=bind,source={helper},target=/acceptance.py,readonly",
            "--entrypoint", "python", IMAGE, "/acceptance.py",
        )
        created = True
        result = docker("start", "--attach", NAME)
        state = docker("inspect", "--format", "{{.State.ExitCode}}", NAME)
        if state.stdout.strip() != "0":
            raise RuntimeError("synthetic Linux reader failed: " + result.stderr[-3000:])
        assert not errors, f"synthetic writer failed: {errors}"
    finally:
        stop.set()
        if writer:
            writer.join(timeout=5)
        keeper.close()
        if created:
            label = docker("inspect", "--format", '{{index .Config.Labels "cardputer.acceptance"}}', NAME).stdout.strip()
            if label != nonce:
                raise RuntimeError("container ownership changed; refusing cleanup")
            docker("rm", "-f", NAME)  # No volume deletion.
        # Verify the exact absolute cleanup target is our newly created temp child.
        assert root.parent == Path(tempfile.gettempdir()).resolve()
        assert root.name.startswith("cardputer-monitor-acceptance-")
        shutil.rmtree(root)
    print(result.stdout.strip())
    print("Synthetic acceptance succeeded; temporary container and files were cleaned without deleting volumes.")


if __name__ == "__main__":
    main()
