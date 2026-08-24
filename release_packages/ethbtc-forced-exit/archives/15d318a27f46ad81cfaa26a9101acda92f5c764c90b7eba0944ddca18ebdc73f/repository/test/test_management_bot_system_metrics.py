from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from management_bot.system_metrics import HostSystemMetrics


class DeterministicMetrics(HostSystemMetrics):
    def __init__(self, samples, **kwargs):
        super().__init__(**kwargs)
        self.samples = list(samples)

    def _cpu_sample(self, _path):
        return self.samples.pop(0)


class HostSystemMetricsTests(TestCase):
    def test_snapshot_reads_host_proc_and_both_filesystem_anchors(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "stat").write_text("cpu 1 2 3 4\n", encoding="utf-8")
            (root / "meminfo").write_text(
                "MemTotal: 1048576 kB\nMemAvailable: 262144 kB\n", encoding="utf-8"
            )
            (root / "loadavg").write_text("0.10 0.20 0.30 1/100 1\n", encoding="utf-8")
            (root / "uptime").write_text("90000.00 1.00\n", encoding="utf-8")
            metrics = DeterministicMetrics(
                [(100, 50, 2), (200, 60, 2)],
                proc_stat=root / "stat",
                proc_meminfo=root / "meminfo",
                proc_loadavg=root / "loadavg",
                proc_uptime=root / "uptime",
                root_disk=root,
                extra_disk=root,
                sample_seconds=0,
            ).snapshot()
            self.assertAlmostEqual(90.0, metrics["cpu"]["used_pct"])
            self.assertEqual(2, metrics["cpu"]["cores"])
            self.assertEqual(0.2, metrics["load"]["five"])
            self.assertEqual(1024**3, metrics["memory"]["total_bytes"])
            self.assertEqual(90000.0, metrics["uptime_seconds"])
            self.assertIn("root", metrics["disks"])
            self.assertIn("extra", metrics["disks"])
            self.assertEqual([], metrics["errors"])
