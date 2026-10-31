"""Exercise the real report-to-audit clock boundary without external requests."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from live_guard import simple_audit
from test.test_cardputer_report_integration import reporting  # noqa: F401
from test.test_simple_audit import inputs


@pytest.mark.parametrize("delay", [6, 12])
def test_delayed_report_audits_new_balance_at_actual_sampling_time(reporting, monkeypatch, delay):
    instance, report_now = reporting
    sampled_at = report_now.timestamp() + delay
    _, grid, dca, inventory, quotes = inputs(sampled_at)
    (instance.grid_state / "guard_state.json").write_text(json.dumps(grid), encoding="utf-8")
    (instance.dca_state / "guard_state.json").write_text(json.dumps(dca), encoding="utf-8")
    instance.inventory_status_path.parent.mkdir(parents=True, exist_ok=True)
    instance.inventory_status_path.write_text(json.dumps(inventory), encoding="utf-8")
    monkeypatch.setattr(simple_audit, "time", SimpleNamespace(time=lambda: sampled_at))
    monkeypatch.setattr(simple_audit, "quote_snapshot", lambda _: quotes)
    monkeypatch.setattr(simple_audit, "render", lambda path, *_args, **_kwargs: path.write_bytes(b"test-image"))
    instance.simple_audit = simple_audit.SimpleAudit(instance.output)
    try:
        instance.cycle({}, now=report_now)
        summary = json.loads((instance.output / "simple_audit/summary.json").read_text(encoding="utf-8"))
        assert summary["account_total"] is not None
        assert summary["issues"] == []
        assert summary["observed_at"] == sampled_at
        assert instance.simple_audit.db.execute("SELECT observed_at FROM hours").fetchall() == [(sampled_at,)]
        status = json.loads((instance.output / "trading_status.json").read_text(encoding="utf-8"))
        assert status["generated_at"] == report_now.isoformat()
        assert instance.management_parameters.publish.call_args.kwargs["now"] == report_now
    finally:
        instance.simple_audit.db.close()


def test_audit_hour_boundary_uses_sampling_hour_without_backfilling(reporting, monkeypatch):
    instance, original_now = reporting
    hour_start = int(original_now.timestamp()) // 3600 * 3600
    report_now = datetime.fromtimestamp(hour_start + 3594, timezone.utc)
    sampled_at = hour_start + 3606
    _, grid, dca, inventory, quotes = inputs(sampled_at)
    (instance.grid_state / "guard_state.json").write_text(json.dumps(grid), encoding="utf-8")
    (instance.dca_state / "guard_state.json").write_text(json.dumps(dca), encoding="utf-8")
    instance.inventory_status_path.parent.mkdir(parents=True, exist_ok=True)
    instance.inventory_status_path.write_text(json.dumps(inventory), encoding="utf-8")
    monkeypatch.setattr(simple_audit, "time", SimpleNamespace(time=lambda: sampled_at))
    monkeypatch.setattr(simple_audit, "quote_snapshot", lambda _: quotes)
    monkeypatch.setattr(simple_audit, "render", lambda path, *_args, **_kwargs: path.write_bytes(b"test-image"))
    instance.simple_audit = simple_audit.SimpleAudit(instance.output)
    try:
        instance.cycle({}, now=report_now)
        rows = instance.simple_audit.db.execute("SELECT slot,observed_at FROM hours").fetchall()
        assert rows == [(int(sampled_at) // 3600, sampled_at)]
        instance.cycle({}, now=report_now)
        assert instance.simple_audit.db.execute("SELECT slot,observed_at FROM hours").fetchall() == rows
    finally:
        instance.simple_audit.db.close()


def test_actual_sampling_time_still_rejects_expired_balance(reporting, monkeypatch):
    instance, report_now = reporting
    sampled_at = report_now.timestamp() + 12
    _, grid, dca, inventory, quotes = inputs(sampled_at)
    dca["simple_audit_balance"]["observed_at"] = sampled_at - 61
    (instance.grid_state / "guard_state.json").write_text(json.dumps(grid), encoding="utf-8")
    (instance.dca_state / "guard_state.json").write_text(json.dumps(dca), encoding="utf-8")
    instance.inventory_status_path.parent.mkdir(parents=True, exist_ok=True)
    instance.inventory_status_path.write_text(json.dumps(inventory), encoding="utf-8")
    monkeypatch.setattr(simple_audit, "time", SimpleNamespace(time=lambda: sampled_at))
    monkeypatch.setattr(simple_audit, "quote_snapshot", lambda _: quotes)
    monkeypatch.setattr(simple_audit, "render", lambda path, *_args, **_kwargs: path.write_bytes(b"test-image"))
    instance.simple_audit = simple_audit.SimpleAudit(instance.output)
    try:
        instance.cycle({}, now=report_now)
        summary = json.loads((instance.output / "simple_audit/summary.json").read_text(encoding="utf-8"))
        assert summary["account_total"] is None
        assert summary["difference"] is None
        assert summary["issues"] == ["账户余额缺失或过期"]
        assert instance.simple_audit.db.execute("SELECT count(*) FROM hours").fetchone()[0] == 0
    finally:
        instance.simple_audit.db.close()
