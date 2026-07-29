#!/usr/bin/env python3
import os
from pathlib import Path
import sys


def find_project_root() -> Path:
    candidates = []
    configured = os.environ.get("HUMMINGBOT_PROJECT_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend((Path.cwd(), *Path.cwd().parents))
    candidates.extend(Path(__file__).resolve().parents)
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "macro_control").is_dir():
            return candidate
    raise RuntimeError(
        "Cannot find macro_control. Start Hermes in the Hummingbot checkout "
        "or set HUMMINGBOT_PROJECT_ROOT to that checkout."
    )


PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from macro_control.hermes_cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
