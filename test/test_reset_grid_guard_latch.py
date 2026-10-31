"""Offline reset adapter checks. All writes are to temporary synthetic files."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from ops import reset_grid_guard_latch as reset

NOW = 2_000_000_000.0


def iso(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def encoded(value):
    return (json.dumps(value, indent=2) + "\n").encode()


def fixture():
    episode = {"triggered_at": NOW - 1000, "mechanism": "portfolio_drawdown_breaker", "scope": "portfolio"}
    stop = {"logical_bot_name": reset.BOT, "verified_no_active_orders": True, "verified_no_live_instances": True,
            "resolved_instances": [], "exchange_cancellations": {pair: [] for pair in reset.PAIRS}}
    flatten = {pair: "dust" for pair in reset.PAIRS}
    bot = {"tripped": True, "action_complete": True, "stop_complete": True, "tripped_at": NOW - 100.5,
           "reason": "recoverable PORTFOLIO exit remained incomplete for 899.5s", "stop": stop, "flatten": flatten,
           "emergency_adjustments": [{"base_delta": "-0.001", "source": "offline-fill"}], "peak_equity": "501",
           "latest": {"observed_at": NOW - 1, "pairs": {pair: {"owned_base": "0.000001", "quote_balance": "99",
                       "fees_quote": "1", "net_base": "-0.001", "mark": "50000"} for pair in reset.PAIRS}}}
    state = {"version": 3, "armed": True, "shadow": False, "last_success_at": NOW - 1,
             "bots": {reset.BOT: bot, "untouched-bot": {"accounting": [1, 2, 3]}}, "other_log": ["preserve"]}
    raw = {"schema_version": 13, "portfolio_recovery": {**episode, "phase": "EXITING", "healthy_cycles": 0,
           "exit_completed_at": None, "cooldown_until": None}, "ledgers": {"offline": "immutable"}}
    event = {"schema": "ethbtc-telegram-event-v1", "source": "grid-live-guard", "strategy": "grid",
             "bot": reset.BOT, "pair": ",".join(reset.PAIRS), "mechanism": "infrastructure_integrity_breaker",
             "transition": "LATCHED", "phase_from": "", "phase_to": "LATCHED", "requires_manual_action": True,
             "action": "manual_recovery_required", "occurred_at": iso(NOW - 100.3),
             "correlation_id": "grid-breaker:offline-episode:2", "reason": "recoverable PORTFOLIO exit remained incomplete",
             "details": {"recovery_episode": {**episode, "triggered_at": str(episode["triggered_at"]),
                          "scope": "PORTFOLIO", "bot": reset.BOT}, "failure_type": "complete"}}
    identity = {key: event.get(key, "") for key in ("source", "strategy", "bot", "pair", "mechanism", "transition",
                "phase_from", "phase_to", "correlation_id", "reason")}
    event["event_id"] = reset.canonical_sha256(identity)
    proof = {"schema": "grid-guard-latch-preflight-v1", "scope": reset.SCOPE, "bot": reset.BOT, "observed_at": NOW - 1,
        "authorization": {"type": "direct_user_request", "scope": reset.SCOPE},
        "guard_state_sha256": reset.bytes_sha256(encoded(state)), "runtime_sha256": reset.canonical_sha256(raw),
        "event_id": event["event_id"], "event_sha256": reset.canonical_sha256(event),
        "checks": {key: {"ok": True, "observed_at": NOW - 1} for key in reset.CHECKS},
        "snapshot": {"guard_container_id": "offline-guard-id", "grid_container_id": "offline-grid-id",
                     "running_guard_writers": 0, "guard_running": False, "grid_running": False,
                     "grid_open_orders": {pair: 0 for pair in reset.PAIRS}, "portfolio_phase": "EXITING",
                     "ordinary_permissions": {"buy_enabled": False, "sell_enabled": False}},
        "completion_audit": {"event": "grid_circuit_breaker_complete", "bot": reset.BOT,
            "timestamp": iso(NOW - 100.4), "reason": bot["reason"], "stop": deepcopy(stop), "flatten": deepcopy(flatten),
            "snapshot": {"recovery_episodes": {"PORTFOLIO": deepcopy(episode)}}}}
    return state, raw, event, proof


class ResetTests(unittest.TestCase):
    def plan(self, documents, **kwargs):
        state, raw, event, proof = documents
        return reset.plan(state, raw, event, proof, reset_id=kwargs.get("reset_id", "offline-reset-1"),
                          guard_state_sha256=proof["guard_state_sha256"], now=kwargs.get("now", NOW))

    def assert_rejected(self, documents):
        before = deepcopy(documents)
        with self.assertRaises((reset.ResetRejected, ValueError, TypeError)):
            self.plan(documents)
        self.assertEqual(documents, before)

    def files(self, directory, documents):
        paths = [Path(directory) / name for name in ("guard.json", "runtime.json", "event.json", "proof.json")]
        for path, value in zip(paths, documents):
            path.write_bytes(encoded(value))
        paths[0].chmod(0o640)
        return paths

    def apply(self, paths, directory, documents, **kwargs):
        return reset.apply(*paths, reset_id=kwargs.get("reset_id", "offline-reset-1"),
                           expected_state_sha256=documents[3]["guard_state_sha256"],
                           private_dir=Path(directory) / "private", now=NOW)

    def test_plan_changes_only_latch_fields_and_archives_exact_prior(self):
        documents = fixture();before = deepcopy(documents);planned = self.plan(documents)
        result = planned["new_state"];target = result["bots"][reset.BOT];old = documents[0]["bots"][reset.BOT]
        receipt = target.pop("manual_guard_reset")
        self.assertEqual(receipt["prior"], {key: old[key] for key in (*reset.FLAGS, "tripped_at", "reason", "stop", "flatten")})
        expected = deepcopy(documents[0]);bot = expected["bots"][reset.BOT]
        bot.update({key: False for key in reset.FLAGS});bot.pop("reason");bot.pop("tripped_at")
        self.assertEqual(result, expected);self.assertEqual(documents, before)
        self.assertEqual(receipt["runtime_episode"], {key: documents[1]["portfolio_recovery"][key]
                         for key in ("triggered_at", "mechanism", "scope")})
        for key in ("event_sha256", "runtime_sha256", "preflight_sha256"):
            self.assertTrue(reset.valid_sha(receipt[key]))
        self.assertFalse(planned["summary"]["starts_grid"])

    def test_every_check_must_be_true_present_and_fresh(self):
        for key in reset.CHECKS:
            for change in ("false", "missing", "old", "future", "boolean_time"):
                with self.subTest(check=key, change=change):
                    docs = fixture();check = docs[3]["checks"][key]
                    if change == "false":check["ok"] = False
                    elif change == "missing":docs[3]["checks"].pop(key)
                    elif change == "old":check["observed_at"] = NOW - 60.001
                    elif change == "future":check["observed_at"] = NOW + 5.001
                    else:check["observed_at"] = True
                    self.assert_rejected(docs)

    def test_model_or_writer_proof_cannot_be_omitted_or_forged_truthy(self):
        for key in ("guard_running", "grid_running", "running_guard_writers", "ordinary_permissions", "grid_open_orders"):
            docs = fixture()
            if key in ("guard_running", "grid_running"):docs[3]["snapshot"][key] = 0
            elif key == "running_guard_writers":docs[3]["snapshot"][key] = False
            elif key == "ordinary_permissions":docs[3]["snapshot"][key]["buy_enabled"] = True
            else:docs[3]["snapshot"][key]["BTC-FDUSD"] = True
            with self.subTest(key=key):self.assert_rejected(docs)

    def test_event_schema_scope_body_and_fresh_binding_are_required(self):
        for key, value in (("schema", "wrong"), ("bot", "other-bot"), ("source", "report"), ("phase_to", "ACTIVE"),
                           ("transition", "ACTION_FAILED"), ("requires_manual_action", False), ("event_id", "a" * 64)):
            docs = fixture();docs[2][key] = value
            with self.subTest(key=key):self.assert_rejected(docs)
        for key in ("guard_state_sha256", "runtime_sha256", "event_sha256", "event_id"):
            docs = fixture();docs[3][key] = "f" * 64
            if key == "guard_state_sha256":
                with self.assertRaises(reset.ResetRejected):
                    reset.validate(*docs, guard_state_sha256="a" * 64, now=NOW)
            else:self.assert_rejected(docs)

    def test_episode_and_completion_stop_must_match_current_state(self):
        for change in ("event_trigger", "event_scope", "audit_trigger", "audit_stop", "audit_flatten", "audit_reason", "audit_time"):
            docs = fixture()
            if change == "event_trigger":docs[2]["details"]["recovery_episode"]["triggered_at"] = "1"
            elif change == "event_scope":docs[2]["details"]["recovery_episode"]["scope"] = "BTC-FDUSD"
            elif change == "audit_trigger":docs[3]["completion_audit"]["snapshot"]["recovery_episodes"]["PORTFOLIO"]["triggered_at"] = 1
            elif change == "audit_stop":docs[3]["completion_audit"]["stop"]["verified_no_live_instances"] = False
            elif change == "audit_flatten":docs[3]["completion_audit"]["flatten"]["BTC-FDUSD"] = "filled"
            elif change == "audit_reason":docs[3]["completion_audit"]["reason"] = "another episode"
            else:docs[3]["completion_audit"]["timestamp"] = iso(NOW - 80)
            with self.subTest(change=change):self.assert_rejected(docs)

    def test_incomplete_current_exit_or_new_error_is_rejected(self):
        for change in (*reset.FLAGS, "error", "stop", "flatten", "tripped_at"):
            docs = fixture();bot = docs[0]["bots"][reset.BOT]
            if change in reset.FLAGS:bot[change] = False
            elif change == "error":bot["last_action_error"] = "new independent error"
            elif change == "stop":bot["stop"]["verified_no_active_orders"] = False
            elif change == "flatten":bot["flatten"]["BTC-FDUSD"] = "ownership_unavailable_no_action"
            else:bot["tripped_at"] = NOW + 10
            with self.subTest(change=change):self.assert_rejected(docs)

    def test_runtime_schema_and_phase_cannot_be_generalized(self):
        for schema in (True, "13", 12, 14):
            docs = fixture();docs[1]["schema_version"] = schema;docs[3]["runtime_sha256"] = reset.canonical_sha256(docs[1])
            with self.subTest(schema=schema):self.assert_rejected(docs)
        docs = fixture();docs[1]["portfolio_recovery"]["phase"] = "ACTIVE"
        docs[3]["runtime_sha256"] = reset.canonical_sha256(docs[1]);self.assert_rejected(docs)

    def test_scope_authorization_and_reset_id_are_exact(self):
        docs = fixture();docs[3]["authorization"]["scope"] = "resume_trading";self.assert_rejected(docs)
        for value in ("", "../escape", "a" * 97, "恢复"):
            with self.subTest(value=value), self.assertRaises(reset.ResetRejected):self.plan(fixture(), reset_id=value)

    def test_apply_preserves_all_inputs_accounting_permissions_and_private_backup(self):
        docs = fixture()
        with tempfile.TemporaryDirectory() as directory:
            paths = self.files(directory, docs);before = [path.read_bytes() for path in paths];mode = stat.S_IMODE(paths[0].stat().st_mode)
            result = self.apply(paths, directory, docs)
            self.assertTrue(result["applied"])
            self.assertEqual([path.read_bytes() for path in paths[1:]], before[1:])
            self.assertEqual(stat.S_IMODE(paths[0].stat().st_mode), mode)
            saved = Path(result["audit_directory"])
            self.assertEqual((saved / "state.before.json").read_bytes(), before[0])
            self.assertEqual((saved / "state.after.json").read_bytes(), paths[0].read_bytes())
            updated = json.loads(paths[0].read_bytes());bot = updated["bots"][reset.BOT]
            self.assertEqual(bot["emergency_adjustments"], docs[0]["bots"][reset.BOT]["emergency_adjustments"])
            self.assertEqual(updated["bots"]["untouched-bot"], docs[0]["bots"]["untouched-bot"])
            with self.assertRaises(reset.ResetRejected):self.apply(paths, directory, docs)

    def test_no_write_on_cas_mismatch_or_changed_source(self):
        docs = fixture()
        with tempfile.TemporaryDirectory() as directory:
            paths = self.files(directory, docs);before = paths[0].read_bytes()
            paths[1].write_text(json.dumps({**docs[1], "updated_at": NOW}), encoding="utf-8")
            with self.assertRaises(reset.ResetRejected):self.apply(paths, directory, docs)
            self.assertEqual(paths[0].read_bytes(), before)
            self.assertFalse((Path(directory) / "private").exists())

    def test_final_cas_detects_another_writer_without_overwriting_it(self):
        docs = fixture()
        with tempfile.TemporaryDirectory() as directory:
            paths = self.files(directory, docs);verify = reset.verify_inputs
            concurrent = {**docs[0], "new_log": "another writer"}
            def changed(selected, expected):
                paths[0].write_bytes(encoded(concurrent));verify(selected, expected)
            with patch.object(reset, "verify_inputs", side_effect=changed), self.assertRaises(reset.ResetRejected):
                self.apply(paths, directory, docs)
            self.assertEqual(json.loads(paths[0].read_bytes()), concurrent)
            self.assertFalse(paths[0].with_name(paths[0].name + ".reset.lock").exists())

    def test_existing_backup_is_never_overwritten(self):
        docs = fixture()
        with tempfile.TemporaryDirectory() as directory:
            paths = self.files(directory, docs);before = paths[0].read_bytes()
            folder = Path(directory) / "private/offline-reset-1";folder.mkdir(parents=True)
            old = folder / "previous-audit.json";old.write_bytes(b"immutable history")
            with self.assertRaises(FileExistsError):self.apply(paths, directory, docs)
            self.assertEqual(old.read_bytes(), b"immutable history");self.assertEqual(paths[0].read_bytes(), before)

    def test_duplicate_id_cannot_be_planned_again(self):
        original = fixture();docs = (self.plan(original)["new_state"], *original[1:])
        with self.assertRaisesRegex(reset.ResetRejected, "already_consumed"):self.plan(docs)

    def test_post_commit_audit_failure_reports_applied_true(self):
        docs = fixture()
        with tempfile.TemporaryDirectory() as directory:
            paths = self.files(directory, docs);write = reset.private_write
            def fail_final(path, content):
                if Path(path).name == "applied.json":raise OSError("offline audit disk failure")
                return write(path, content)
            with patch.object(reset, "private_write", side_effect=fail_final):
                result = self.apply(paths, directory, docs)
            self.assertTrue(result["applied"])
            self.assertEqual(result["audit_finalize_error_type"], "OSError")
            self.assertFalse(json.loads(paths[0].read_bytes())["bots"][reset.BOT]["tripped"])
            self.assertTrue((Path(result["audit_directory"]) / "prepared-audit.json").exists())
            self.assertTrue((Path(result["audit_directory"]) / "state.before.json").exists())

    def rollback_fixture(self):
        docs = fixture();saved = self.plan(docs)["new_state"];current = deepcopy(saved)
        current["last_success_at"] = NOW
        current["other_log"].append("new health log")
        current["bots"][reset.BOT]["latest"]["observed_at"] = NOW
        current["bots"][reset.BOT]["latest"]["pairs"]["BTC-FDUSD"]["mark"] = "60000"
        proof = {"schema": "grid-guard-latch-rollback-preflight-v1", "scope": "grid_guard_latch_rollback_only",
                 "bot": reset.BOT, "reset_id": "offline-reset-1", "observed_at": NOW,
                 "guard_state_sha256": reset.bytes_sha256(encoded(current)),
                 "post_reset_accounting_sha256": reset.accounting_sha256(saved),
                 "snapshot": deepcopy(docs[3]["snapshot"]), "checks": {key: {"ok": True, "observed_at": NOW}
                     for key in ("unique_guard_writer_stopped", "grid_physically_stopped", "no_grid_orders", "no_new_economic_progress")}}
        return current, docs[1], saved, proof

    def rollback(self, docs):
        return reset.plan_rollback(*docs, reset_id="offline-reset-1",
                                   guard_state_sha256=docs[3]["guard_state_sha256"], now=NOW)

    def test_selective_rollback_preserves_new_logs_marks_and_all_accounting(self):
        docs = self.rollback_fixture();before = deepcopy(docs);planned = self.rollback(docs)
        bot = planned["new_state"]["bots"][reset.BOT];prior = docs[0]["bots"][reset.BOT]["manual_guard_reset"]["prior"]
        for key in (*reset.FLAGS, "reason", "tripped_at"):self.assertEqual(bot[key], prior[key])
        self.assertEqual(bot["latest"], docs[0]["bots"][reset.BOT]["latest"])
        self.assertEqual(bot["emergency_adjustments"], docs[0]["bots"][reset.BOT]["emergency_adjustments"])
        self.assertEqual(planned["new_state"]["other_log"], docs[0]["other_log"])
        self.assertIn("rollback", bot["manual_guard_reset"])
        self.assertTrue(planned["summary"]["planned_only"]);self.assertEqual(docs, before)

    def test_rollback_rejects_new_trip_accounting_runtime_or_weak_proof(self):
        for change in ("trip", "receipt", "adjustment", "owned", "runtime", "orders", "writer", "stale", "check", "repeated"):
            docs = self.rollback_fixture();bot = docs[0]["bots"][reset.BOT]
            if change == "trip":bot["tripped"] = True
            elif change == "receipt":bot["manual_guard_reset"]["reset_id"] = "other-reset"
            elif change == "adjustment":bot["emergency_adjustments"].append({"new": "economic fill"})
            elif change == "owned":bot["latest"]["pairs"]["BTC-FDUSD"]["owned_base"] = "0.1"
            elif change == "runtime":docs[1]["new_progress"] = NOW
            elif change == "orders":docs[3]["snapshot"]["grid_open_orders"]["BTC-FDUSD"] = 1
            elif change == "writer":docs[3]["snapshot"]["running_guard_writers"] = 1
            elif change == "stale":docs[3]["observed_at"] = NOW - 61
            elif change == "check":docs[3]["checks"]["no_new_economic_progress"]["ok"] = False
            else:
                docs = (self.rollback(docs)["new_state"], *docs[1:])
            with self.subTest(change=change), self.assertRaises(reset.ResetRejected):self.rollback(docs)


if __name__ == "__main__":
    unittest.main()
