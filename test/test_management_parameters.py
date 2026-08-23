import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from live_guard.management_parameters import HISTORY_LIMIT, ManagementParameterPublisher
from management_bot.clients import ParameterCatalogReader, ServiceError


def write(path: Path, value: str | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")


def test_publisher_builds_sanitized_catalog_history_and_verified_evidence():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        bots = root / "bots"
        grid = root / "grid"
        dca = root / "dca"
        releases = root / "releases"
        approvals = root / "approvals"
        output = dca / "telegram"
        write(bots / "instances/grid-live-fdusd-400/conf/scripts/"
              "walk_forward_portfolio_grid_live_fdusd_400.yml", """
pair_budget_quote: 200
side_budget_quote: 100
grid_range: 0.06
grid_levels: 10
take_profit: 0.006
connector_secret: must-not-leak
""")
        write(bots / "instances/grid-live-fdusd-400/data/live_grid_runtime_state.json", {
            "active_parameter_version": "fixed-v1", "active_parameter_sha256": "a" * 64,
            "active_pair_parameters": {"BTC-FDUSD": {"grid_range": .06, "grid_levels": 10}},
        })
        for pair, bot, name in (
            ("BTC-USDT", "dca-live-btcusdt-200", "dca_btcusdt_live_200.yml"),
            ("ETH-USDT", "dca-live-ethusdt-200", "dca_ethusdt_live_200.yml"),
        ):
            write(bots / f"instances/{bot}/conf/controllers/{name}",
                  f"trading_pair: {pair}\ntotal_amount_quote: 190\nstop_loss: 0.05\napi_secret: no\n")
        write(grid / "guard_state.json", {"mechanisms": {"v22_weekly_buy_gate": True},
              "mechanism_parameters": {"strategy_loss_breaker": {"loss_limit_quote": "6"}}})
        write(dca / "guard_state.json", {"mechanisms": {"v22_weekly_buy_gate": True},
              "mechanism_parameters": {"strategy_loss_breaker": {"loss_limit_quote": "16"}}})
        write(grid / "xgboost_risk_gate.json", {
            "release_sha256": "r" * 64, "model_sha256": "m" * 64,
            "runtime_generation": "g" * 64, "source_healthy": True, "cutover_phase": "ACTIVE",
            "pairs": {"BTC-FDUSD": {"probability": .2, "entry_threshold": .5}},
        })
        release = releases / "releases" / ("r" * 64)
        write(release / "release.json", {"release_sha256": "r" * 64, "effective_end": 10})
        write(release / "production_lock.json", {"model_sha256": "m" * 64})
        image = b"verified-png"
        audit = releases / "audits" / "evidence"
        image_path = audit / "grid_btcfdusd_360d.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image)
        image_sha = hashlib.sha256(image).hexdigest()
        write(audit / "manifest.json", {
            "generated_at": "2026-08-22T00:00:00Z", "evidence_model_sha256": "x" * 64,
            "production_model_sha256": "m" * 64,
            "release_sha256": "r" * 64,
            "images": [{"strategy": "grid", "pair": "BTC-FDUSD", "window": "360d",
                        "path": "release_packages\\ethbtc-forced-exit\\audits\\evidence\\grid_btcfdusd_360d.png",
                        "sha256": image_sha, "width": 1440, "height": 2400}],
        })
        write(approvals / "automation_state.json", {"phase": "IDLE"})
        publisher = ManagementParameterPublisher(
            bots_path=bots, dca_state=dca, grid_state=grid, release_root=releases,
            approval_root=approvals, output=output,
        )
        robot = {"strategy": "grid", "pair": "BTC-FDUSD", "trading_normal": True,
                 "final_permissions": {"buy": True, "sell": True}, "gate_statuses": []}
        result = publisher.publish([robot], now=datetime.now(timezone.utc))
        raw_catalog = (output / "management_parameter_catalog.json").read_text(encoding="utf-8")
        assert "must-not-leak" not in raw_catalog
        assert "api_secret" not in raw_catalog
        assert result["history_limit"] == HISTORY_LIMIT == 3
        evidence = json.loads((output / "model_evidence_catalog.json").read_text(encoding="utf-8"))
        assert evidence["sets"][0]["relation_to_active"] == "EXACT"
        assert evidence["sets"][0]["attachments"][0]["sha256"] == image_sha

        reader = ParameterCatalogReader(
            output / "management_parameter_catalog.json",
            output / "model_evidence_catalog.json", output, 300,
        )
        attachment = reader.attachment(evidence["sets"][0]["evidence_set_id"], "grid", "BTC", "360d")
        assert Path(attachment["path"]).read_bytes() == image
        exact = reader.model_attachment("r" * 64, "m" * 64, "grid", "BTC")
        assert Path(exact["path"]).read_bytes() == image
        with __import__("pytest").raises(ServiceError, match="精确360天回测缺失"):
            reader.model_attachment("x" * 64, "m" * 64, "grid", "BTC")

        first_sha = result["catalog_sha256"]
        write(grid / "xgboost_risk_gate.json", {
            "release_sha256": "r" * 64, "model_sha256": "m" * 64,
            "runtime_generation": "g" * 64, "source_healthy": True, "cutover_phase": "ACTIVE",
            "generated_at": "2026-08-22T00:01:00Z", "valid_until": "2026-08-22T00:03:30Z",
            "pairs": {"BTC-FDUSD": {"probability": .8, "entry_threshold": .5,
                       "model_signal": "RISK_OFF"}},
        })
        dynamic_robot = {**robot, "trading_normal": False,
                         "final_permissions": {"buy_enabled": False, "sell_enabled": True}}
        dynamic = publisher.publish([dynamic_robot], now=datetime.now(timezone.utc))
        assert dynamic["catalog_sha256"] == first_sha
        assert len(dynamic["history"]) == 1

        runtime_path = bots / "instances/grid-live-fdusd-400/data/live_grid_runtime_state.json"
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        runtime["active_parameter_sha256"] = "b" * 64
        runtime["active_pair_parameters"]["BTC-FDUSD"]["grid_range"] = .08
        write(runtime_path, runtime)
        changed = publisher.publish([dynamic_robot], now=datetime.now(timezone.utc))
        assert changed["catalog_sha256"] != first_sha
        assert len(changed["history"]) == 2


def test_unbound_legacy_evidence_is_not_current_or_navigable():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        output = root / "state" / "telegram"
        image = root / "releases" / "audits" / "legacy" / "grid_btcfdusd_360d.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"legacy")
        write(image.parent / "manifest.json", {
            "evidence_model_sha256": "m" * 64,
            "production_model_sha256": "m" * 64,
            "images": [{"strategy": "grid", "pair": "BTC-FDUSD", "window": "360d",
                        "path": image.name, "sha256": hashlib.sha256(b"legacy").hexdigest()}],
        })
        write(root / "grid" / "xgboost_risk_gate.json", {
            "release_sha256": "r" * 64, "model_sha256": "m" * 64,
            "runtime_generation": "g" * 64, "source_healthy": True, "cutover_phase": "ACTIVE",
            "pairs": {},
        })
        publisher = ManagementParameterPublisher(
            bots_path=root / "bots", dca_state=root / "dca", grid_state=root / "grid",
            release_root=root / "releases", approval_root=root / "approvals", output=output,
        )
        result = publisher.publish([], now=datetime.now(timezone.utc))
        assert result["models"]["active"]["exact_evidence_count"] == 0
        evidence = json.loads((output / "model_evidence_catalog.json").read_text(encoding="utf-8"))
        assert evidence["sets"][0]["relation_to_active"] == "UNBOUND_LEGACY"
        reader = ParameterCatalogReader(output / "management_parameter_catalog.json",
                                        output / "model_evidence_catalog.json", output, 300)
        with __import__("pytest").raises(ServiceError, match="精确360天回测缺失"):
            reader.model_attachment("r" * 64, "m" * 64, "grid", "BTC")


def test_model_history_requires_trusted_cutover_and_excludes_current():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        grid = root / "grid"
        output = root / "state" / "telegram"
        publisher = ManagementParameterPublisher(
            bots_path=root / "bots", dca_state=root / "dca", grid_state=grid,
            release_root=root / "releases", approval_root=root / "approvals", output=output,
        )
        write(grid / "xgboost_risk_gate.json", {
            "release_sha256": "a" * 64, "model_sha256": "b" * 64,
            "runtime_generation": "c" * 64, "source_healthy": True, "cutover_phase": "ACTIVE",
            "pairs": {"BTC-FDUSD": {"model_week": 39, "week_start": 100, "week_end": 200}},
        })
        first = publisher.publish([], now=datetime.now(timezone.utc))
        assert first["models"]["history"] == []
        first_event = json.dumps({
            "transition": "MODEL_CUTOVER_STABLE", "occurred_at": "2026-08-17T05:44:43Z",
            "release_sha256": "a" * 64, "details": {"runtime_generation": "c" * 64},
        })
        write(grid / "telegram_events.jsonl", first_event + "\n")
        publisher.publish([], now=datetime.now(timezone.utc))
        write(grid / "xgboost_risk_gate.json", {
            "release_sha256": "d" * 64, "model_sha256": "e" * 64,
            "runtime_generation": "f" * 64, "source_healthy": True, "cutover_phase": "ACTIVE",
            "pairs": {"BTC-FDUSD": {"model_week": 40, "week_start": 200, "week_end": 300}},
        })
        second_event = json.dumps({
            "transition": "MODEL_CUTOVER_STABLE", "occurred_at": "2026-08-24T05:44:43Z",
            "release_sha256": "d" * 64, "details": {"runtime_generation": "f" * 64},
        })
        write(grid / "telegram_events.jsonl", first_event + "\n" + second_event + "\n")
        second = publisher.publish([], now=datetime.now(timezone.utc))
        assert len(second["models"]["history"]) == 1
        assert second["models"]["history"][0]["release_sha256"] == "a" * 64
        assert second["models"]["history"][0]["activated_at"] == "2026-08-17T05:44:43Z"
        assert second["models"]["active"]["release_sha256"] == "d" * 64


def test_evidence_reader_rejects_path_escape_and_hash_mismatch():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        write(root / "catalog.json", {"generated_at": datetime.now(timezone.utc).isoformat()})
        outside = root.parent / "outside-model-evidence.png"
        outside.write_bytes(b"x")
        write(root / "evidence.json", {"sets": [{"evidence_set_id": "bad", "attachments": [{
            "strategy": "grid", "pair": "BTC-FDUSD", "window": "360d",
            "relative_path": "../outside-model-evidence.png", "sha256": hashlib.sha256(b"x").hexdigest(),
        }]}]})
        reader = ParameterCatalogReader(root / "catalog.json", root / "evidence.json", root, 300)
        try:
            with __import__("pytest").raises(ServiceError, match="越界"):
                reader.attachment("bad", "grid", "BTC", "360d")
        finally:
            outside.unlink(missing_ok=True)


def test_grid_approved_runtime_override_is_applied_not_mismatch():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        bots = root / "bots"
        write(bots / "instances/grid-live-fdusd-400/conf/scripts/"
              "walk_forward_portfolio_grid_live_fdusd_400.yml", """
grid_range: 0.06
grid_levels: 10
take_profit: 0.006
""")
        write(bots / "instances/grid-live-fdusd-400/data/live_grid_runtime_state.json", {
            "active_parameter_version": "approved-runtime-v2",
            "active_parameter_sha256": "b" * 64,
            "active_parameters": {"grid_range": .06, "grid_levels": 10, "take_profit": .006},
            "active_pair_parameters": {
                "BTC-FDUSD": {"grid_range": .126984, "grid_levels": 18, "take_profit": .004},
                "ETH-FDUSD": {"grid_range": .524651, "grid_levels": 18, "take_profit": .01418},
            },
        })
        publisher = ManagementParameterPublisher(
            bots_path=bots, dca_state=root / "dca", grid_state=root / "grid",
            release_root=root / "releases", approval_root=root / "approvals",
            output=root / "output",
        )
        grid = publisher._grid()
        assert grid["application_state"] == "APPLIED"
        assert grid["runtime_override_active"] is True
        assert grid["difference_reason"] == "approved_runtime_parameter_override"
        assert grid["pairs"]["BTC-FDUSD"]["effective"]["grid_range"] == .126984
