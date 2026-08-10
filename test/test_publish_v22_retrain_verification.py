import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.publish_v22_retrain_verification import build_verified_event


class PublishV22RetrainVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.release = root / "release"
        self.retrain = root / "retrain"
        current = self.release / "current"
        active_model = current / "shadow_package/models/xgboost_long_risk_gate_v22_weekly.joblib"
        retrain_model = self.retrain / "models/xgboost_long_risk_gate_v22_weekly.joblib"
        active_model.parent.mkdir(parents=True)
        retrain_model.parent.mkdir(parents=True)
        active_model.write_bytes(b"identical-real-model")
        retrain_model.write_bytes(b"identical-real-model")
        model_hash = hashlib.sha256(b"identical-real-model").hexdigest()
        release_sha = "a" * 64
        production = {
            "release_sha256": release_sha,
            "model_sha256": model_hash,
            "effective_end": 200,
        }
        lock = {"model_sha256": model_hash, "effective_end": 200, "deployment_allowed": False}
        (current / "production_lock.json").write_text(json.dumps(production), encoding="utf-8")
        (current / "shadow_package/shadow_lock.json").write_text(json.dumps(lock), encoding="utf-8")
        (self.retrain / "shadow_lock.json").write_text(json.dumps(lock), encoding="utf-8")
        audit = self.release / "audits" / f"retrain-verification-{release_sha}"
        audit.mkdir(parents=True)
        (audit / "retrain_report.json").write_text(json.dumps({
            "release_sha256": release_sha,
            "byte_for_byte_match": True,
            "production_switch_performed": False,
            "mode": "real_market_data_retrain_reproducibility_check",
            "fold": 38,
            "training_cutoff": 100,
            "effective_start": 100,
            "effective_end": 200,
            "feature_schema_sha256": "b" * 64,
            "strategy_schema_sha256": "c" * 64,
            "training_data_sha256": "d" * 64,
            "pairs": {},
            "next_scheduled_candidate_training_bjt": "2026-08-16T10:00:00+08:00",
        }), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def test_builds_event_only_after_every_hash_and_policy_check_passes(self):
        event = build_verified_event(self.release, self.retrain)
        self.assertEqual("PARAMETER_RETAINED", event["transition"])
        self.assertTrue(all(event["details"]["checks"].values()))
        self.assertEqual([], event["attachments"])
        self.assertEqual("v22_png_windows", event["details"]["report_request"])
        self.assertFalse(event["details"]["model_attachment_included"])
        self.assertFalse(event["details"]["production_switch_performed"])

    def test_rejects_non_reproducible_model(self):
        model = self.retrain / "models/xgboost_long_risk_gate_v22_weekly.joblib"
        model.write_bytes(b"different-model")
        with self.assertRaisesRegex(RuntimeError, "retrain verification failed"):
            build_verified_event(self.release, self.retrain)


if __name__ == "__main__":
    unittest.main()
