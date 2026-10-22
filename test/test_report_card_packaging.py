"""Exercise report imports using only the Python files declared by each image."""

import ast
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CARD_SOURCE = "live_guard/report_cards.py"


def _python_copies(dockerfile):
    for line in dockerfile.read_text(encoding="utf-8").splitlines():
        if not line.startswith("COPY "):
            continue
        parts = shlex.split(line)[1:]
        sources, destination = parts[:-1], parts[-1]
        for source in sources:
            if not source.endswith(".py"):
                continue
            target = Path(destination.removeprefix("/app/"))
            if destination.endswith("/") or len(sources) > 1:
                target /= Path(source).name
            yield source, target


def _copy_modules(dockerfile, app, *, exclude=()):
    for source, relative in _python_copies(dockerfile):
        if source in exclude:
            continue
        target = app / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / source).read_bytes())


def test_mqtt_rollout_allowlist_contains_all_overlay_python_sources():
    tree = ast.parse((ROOT / "ops/deploy_cardputer_report.py").read_text(encoding="utf-8"))
    assignment = next(
        node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "SOURCE_FILES" for target in node.targets)
    )
    allowed = set(ast.literal_eval(assignment.value))
    copies = dict(_python_copies(ROOT / "Dockerfile.dca-live-report-mqtt"))
    assert CARD_SOURCE in copies
    assert set(copies) <= allowed


@pytest.mark.parametrize("overlay", [False, True], ids=["base-report", "mqtt-over-old-base"])
def test_report_card_dependency_is_available_in_flat_image_layout(tmp_path, overlay):
    app = tmp_path / "app"
    # The pinned previous report image cannot be assumed to contain a new module.
    _copy_modules(ROOT / "Dockerfile.dca-live-guard", app, exclude=(CARD_SOURCE,) if overlay else ())
    if overlay:
        assert not (app / "report_cards.py").exists()
        _copy_modules(ROOT / "Dockerfile.dca-live-report-mqtt", app)
    code = (
        "import sys; from pathlib import Path; "
        f"root=Path({str(ROOT)!r}).resolve(); app=Path({str(app)!r}).resolve(); "
        "sys.path[:]=[p for p in sys.path if not Path(p).resolve().is_relative_to(root)]; "
        "sys.path.insert(0,str(app)); import dca_live_report, report_cards; "
        "assert Path(dca_live_report.__file__).resolve().is_relative_to(app); "
        "assert Path(report_cards.__file__).resolve().is_relative_to(app)"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-X", "utf8", "-c", code],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
