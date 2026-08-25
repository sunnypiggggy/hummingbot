from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any


class HostSystemMetrics:
    """Read a minimal, read-only view of the OCI host.

    The container receives only selected proc files and two empty filesystem
    anchors. It never needs the Docker socket or a mount of the host root.
    """

    def __init__(
        self,
        *,
        proc_stat: Path,
        proc_meminfo: Path,
        proc_loadavg: Path,
        proc_uptime: Path,
        root_disk: Path,
        extra_disk: Path,
        sample_seconds: float = 0.15,
    ):
        self.proc_stat = proc_stat
        self.proc_meminfo = proc_meminfo
        self.proc_loadavg = proc_loadavg
        self.proc_uptime = proc_uptime
        self.root_disk = root_disk
        self.extra_disk = extra_disk
        self.sample_seconds = sample_seconds

    @staticmethod
    def _cpu_sample(path: Path) -> tuple[int, int, int]:
        lines = path.read_text(encoding="utf-8").splitlines()
        values = [int(value) for value in lines[0].split()[1:]]
        if len(values) < 4:
            raise ValueError("host /proc/stat CPU row is incomplete")
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        cores = sum(1 for line in lines[1:] if line.startswith("cpu") and line[3:4].isdigit())
        return sum(values), idle, cores

    @staticmethod
    def _disk(path: Path) -> dict[str, Any]:
        usage = shutil.disk_usage(path)
        total = usage.total
        available = usage.free
        used = usage.used
        return {
            "total_bytes": total,
            "used_bytes": used,
            "available_bytes": available,
            "used_pct": used / total * 100 if total else 0.0,
        }

    def snapshot(self) -> dict[str, Any]:
        result: dict[str, Any] = {"errors": []}
        try:
            total_1, idle_1, cores = self._cpu_sample(self.proc_stat)
            time.sleep(self.sample_seconds)
            total_2, idle_2, cores_2 = self._cpu_sample(self.proc_stat)
            delta_total = total_2 - total_1
            delta_idle = idle_2 - idle_1
            if delta_total <= 0:
                raise ValueError("host CPU counters did not advance")
            result["cpu"] = {
                "used_pct": max(0.0, min(100.0, (1 - delta_idle / delta_total) * 100)),
                "cores": max(cores, cores_2),
            }
        except Exception as exc:
            result["errors"].append(f"cpu:{type(exc).__name__}")

        try:
            load = [float(value) for value in self.proc_loadavg.read_text(encoding="utf-8").split()[:3]]
            result["load"] = {"one": load[0], "five": load[1], "fifteen": load[2]}
        except Exception as exc:
            result["errors"].append(f"load:{type(exc).__name__}")

        try:
            memory = {}
            for line in self.proc_meminfo.read_text(encoding="utf-8").splitlines():
                key, raw = line.split(":", 1)
                memory[key] = int(raw.strip().split()[0]) * 1024
            total = memory["MemTotal"]
            available = memory["MemAvailable"]
            used = max(0, total - available)
            result["memory"] = {
                "total_bytes": total,
                "used_bytes": used,
                "available_bytes": available,
                "used_pct": used / total * 100 if total else 0.0,
            }
        except Exception as exc:
            result["errors"].append(f"memory:{type(exc).__name__}")

        try:
            result["uptime_seconds"] = float(
                self.proc_uptime.read_text(encoding="utf-8").split()[0]
            )
        except Exception as exc:
            result["errors"].append(f"uptime:{type(exc).__name__}")

        for name, path in (("root", self.root_disk), ("extra", self.extra_disk)):
            try:
                result.setdefault("disks", {})[name] = self._disk(path)
            except Exception as exc:
                result["errors"].append(f"disk_{name}:{type(exc).__name__}")
        return result
