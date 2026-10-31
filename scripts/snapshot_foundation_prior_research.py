"""Read-only hash contract for studies preceding the foundation-model tests."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/backtests/crypto_foundation_models_20261024"
EXCLUDED = {"node_modules", "__pycache__", ".git", ".pytest_cache", ".vite", ".venv", "venv"}
STUDY_NAMES = ("crypto_rl_top10_20261024", "crypto_rl_a2c_20261024",
               "crypto_rl_ex_ada_20261024", "crypto_rl_elegantrl_20261024",
               "rl_framework_comparison_20261024", "spot_strategy_suite_365d_20261023",
               "spot_strategy_suite_round2_365d_20261024")


def sha(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def linked(path: Path):
    return path.is_symlink() or path.is_junction()


def walk_files(path: Path):
    for current, directories, files in os.walk(path, followlinks=False):
        directory = Path(current)
        directories[:] = [name for name in directories
                          if name not in EXCLUDED and not linked(directory / name)]
        for name in files:
            file = directory / name
            if not linked(file):
                yield file


def main():
    target = OUTPUT / "original_research_preservation.json"
    if target.exists():
        raise FileExistsError("Preserve the original hash contract; never backdate a fresh snapshot")
    sources = []
    for path in (ROOT / "scripts").iterdir():
        if path.is_file() and any(part in path.name for part in
                                 ("crypto_rl", "rl_elegantrl", "rl_ex_ada", "rl_framework",
                                  "backtest_rl_", "build_rl_", "audit_rl_", "spot_strategy_suite",
                                  "spot_strategy_round2")):
            sources.append(path)
    for path in (ROOT / "test").iterdir():
        if path.is_file() and any(part in path.name for part in ("crypto_rl", "spot_strategy_suite")):
            sources.append(path)
    directories = [ROOT / "results/backtests" / name for name in STUDY_NAMES]
    directories += [ROOT / "data" / name for name in ("crypto_rl_top10", "crypto_rl_top10_ex_ada")]
    all_files = set(sources)
    for directory in directories:
        if not directory.is_dir() or linked(directory):
            raise FileNotFoundError("Expected an actual prior study/cache directory: " + str(directory))
        all_files.update(walk_files(directory))
    records = {}
    for path in sorted(all_files):
        before = path.stat()
        digest = sha(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("A prior file changed while hashing: " + str(path))
        records[path.relative_to(ROOT).as_posix()] = {
            "sha256": digest, "size_bytes": after.st_size, "mtime_ns": after.st_mtime_ns}
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "All preceding studies in this user-requested spot/RL chain, including ElegantRL native models and complete delivered report, plus related source/tests and both frozen RL datasets. No fixed expected file count.",
              "study_roots": [str(path.relative_to(ROOT)) for path in directories],
              "excluded_rebuildable_directory_names": sorted(EXCLUDED),
              "symlinks_and_junctions_followed": False,
              "files": records, "originally_absent": [],
              "file_count": len(records), "total_bytes": sum(row["size_bytes"] for row in records.values())}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(target), "sha256": sha(target),
                      "file_count": result["file_count"], "total_bytes": result["total_bytes"]}))


if __name__ == "__main__":
    main()
