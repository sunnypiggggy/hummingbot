import io
import json
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from macro_control.hermes_cli import clean_chart_cache, main


PNG = b"\x89PNG\r\n\x1a\nchart"


class HermesReportCliTests(unittest.TestCase):
    @staticmethod
    def gateway_args():
        return [
            "--gateway-url",
            "https://example.test",
            "--hmac-secret",
            "secret",
            "--key-id",
            "v3",
        ]

    def test_report_collects_components_and_writes_absolute_chart(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.png"

            def response(*args, **kwargs):
                path = args[3]
                return {
                    "/v1/status": {"execution_enabled": False},
                    "/v1/telemetry": {"telemetry_healthy": True},
                    "/v1/trading-report": {"report_id": "report"},
                }[path]

            stdout = io.StringIO()
            with (
                patch("macro_control.hermes_cli.signed_request", side_effect=response),
                patch("macro_control.hermes_cli.signed_download", return_value=PNG),
                redirect_stdout(stdout),
            ):
                self.assertEqual(
                    0,
                    main(
                        self.gateway_args()
                        + ["report", "--chart-output", str(output)]
                    ),
                )
            value = json.loads(stdout.getvalue())
            self.assertFalse(value["status"]["execution_enabled"])
            self.assertTrue(value["telemetry"]["telemetry_healthy"])
            self.assertEqual("report", value["trading_report"]["report_id"])
            self.assertEqual(str(output.resolve()), value["chart_path"])
            self.assertEqual(PNG, output.read_bytes())
            self.assertTrue(value["read_only"])
            self.assertEqual([], value["warnings"])

    def test_report_returns_status_when_report_and_chart_are_unavailable(self):
        def response(*args, **kwargs):
            path = args[3]
            if path == "/v1/status":
                return {"execution_enabled": False}
            raise RuntimeError("unavailable")

        stdout = io.StringIO()
        with (
            patch("macro_control.hermes_cli.signed_request", side_effect=response),
            patch(
                "macro_control.hermes_cli.signed_download",
                side_effect=RuntimeError("no chart"),
            ),
            redirect_stdout(stdout),
        ):
            self.assertEqual(0, main(self.gateway_args() + ["report"]))
        value = json.loads(stdout.getvalue())
        self.assertIsNotNone(value["status"])
        self.assertIsNone(value["trading_report"])
        self.assertIsNone(value["chart_path"])
        self.assertEqual(3, len(value["warnings"]))

    def test_chart_cache_removes_only_expired_matching_png(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "dca-trades-7d-old.png"
            fresh = root / "dca-trades-7d-fresh.png"
            other = root / "other.png"
            for path in (old, fresh, other):
                path.write_bytes(PNG)
            now = time.time()
            old_time = now - 90_000
            old.touch()
            fresh.touch()
            other.touch()
            import os

            os.utime(old, (old_time, old_time))
            clean_chart_cache(root, now=now)
            self.assertFalse(old.exists())
            self.assertTrue(fresh.exists())
            self.assertTrue(other.exists())


if __name__ == "__main__":
    unittest.main()
