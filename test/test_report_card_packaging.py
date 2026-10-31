"""Exercise report imports using only the Python files declared by each image."""

import ast
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
CARD_SOURCE = "live_guard/report_cards.py"
PROBABILITY_SOURCE = "live_guard/model_probability_history.py"


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


def _assert_flat_imports(app, code):
    setup = (
        "import sys; from pathlib import Path; "
        f"root=Path({str(ROOT)!r}).resolve(); app=Path({str(app)!r}).resolve(); "
        "sys.path[:]=[p for p in sys.path if not Path(p).resolve().is_relative_to(root)]; "
        "sys.path.insert(0,str(app)); "
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-X", "utf8", "-c", setup + code],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_mqtt_rollout_allowlist_contains_all_overlay_python_sources():
    tree = ast.parse((ROOT / "ops/deploy_cardputer_report.py").read_text(encoding="utf-8"))
    assignment = next(
        node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "SOURCE_FILES" for target in node.targets)
    )
    allowed = set(ast.literal_eval(assignment.value))
    copies = dict(_python_copies(ROOT / "Dockerfile.dca-live-report-mqtt"))
    assert CARD_SOURCE in copies
    assert copies[PROBABILITY_SOURCE] == Path("model_probability_history.py")
    assert set(copies) <= allowed


@pytest.mark.parametrize("overlay", [False, True], ids=["base-report", "mqtt-over-old-base"])
def test_report_card_dependency_is_available_in_flat_image_layout(tmp_path, overlay):
    app = tmp_path / "app"
    # The pinned previous report image cannot be assumed to contain a new module.
    _copy_modules(ROOT / "Dockerfile.dca-live-guard", app, exclude=(CARD_SOURCE,) if overlay else ())
    if overlay:
        assert not (app / "report_cards.py").exists()
        # An importable old module must be replaced, rather than silently inherited.
        (app / "model_probability_history.py").write_text(
            "def collect(*args, **kwargs): return {'legacy': True}\n"
            "def read_market_prices(*args, **kwargs): return []\n",
            encoding="utf-8",
        )
        _assert_flat_imports(
            app,
            "import model_probability_history as probability; "
            "assert probability.collect() == {'legacy': True}",
        )
        _copy_modules(ROOT / "Dockerfile.dca-live-report-mqtt", app)
    expected_sha = hashlib.sha256((ROOT / PROBABILITY_SOURCE).read_bytes()).hexdigest()
    code = (
        "import hashlib; import dca_live_report, report_cards, management_parameters; "
        "import model_probability_history as probability; "
        "assert Path(dca_live_report.__file__).resolve().is_relative_to(app); "
        "assert Path(report_cards.__file__).resolve().is_relative_to(app); "
        "assert Path(probability.__file__).resolve() == app / 'model_probability_history.py'; "
        f"assert hashlib.sha256(Path(probability.__file__).read_bytes()).hexdigest() == {expected_sha!r}; "
        "assert management_parameters.collect_probability_history is probability.collect; "
        "assert management_parameters.read_market_prices is probability.read_market_prices"
    )
    _assert_flat_imports(app, code)


def test_mqtt_runtime_verifies_updated_probability_hash_in_flat_layout(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    new_probability = (ROOT / PROBABILITY_SOURCE).read_bytes()
    (app / "model_probability_history.py").write_bytes(new_probability)
    preserved = {
        "live_guard/telegram_notifications.py": b"unchanged notifications",
        "live_guard/risk_history.py": b"unchanged risk history",
    }
    for name, content in preserved.items():
        (app / Path(name).name).write_bytes(content)
    baseline = {
        "containers": [],
        "target_runtime": {"mounts": [], "network_names": []},
        "source_files": {
            PROBABILITY_SOURCE: {"sha256": hashlib.sha256(b"old probability module").hexdigest()},
            **{name: {"sha256": hashlib.sha256(content).hexdigest()}
               for name, content in preserved.items()},
        },
    }
    manifest = {
        "image_id": "new-report-image",
        "source_sha256": {PROBABILITY_SOURCE: hashlib.sha256(new_probability).hexdigest()},
    }
    after = [{"name": "dca-live-report", "image_id": manifest["image_id"], "health": "healthy"}]

    def remote_read(_client, command):
        if command == "docker inspect dca-live-report":
            return json.dumps([{
                "Mounts": [{"Destination": path, "RW": False} for path in (
                    "/run/secrets/cardputer_report_mqtt", "/etc/cardputer/isrg-root-x1.pem")],
                "HostConfig": {"PortBindings": None},
                "NetworkSettings": {"Networks": {}},
            }])
        if command == "docker logs --tail 30 dca-live-report":
            return json.dumps({"generated_at": 1, "cardputer_mqtt": {}})
        parts = shlex.split(command)
        assert parts[:5] == ["docker", "exec", "dca-live-report", "python", "-c"]
        program = ast.parse(parts[5])
        assignment = next(node for node in program.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "paths"
                                  for target in node.targets))
        paths = ast.literal_eval(assignment.value)
        if isinstance(paths, dict):
            actual = {name: app / path.removeprefix("/app/") for name, path in paths.items()}
        else:
            actual = {name: app / Path(name).name for name in paths}
        return json.dumps({name: hashlib.sha256(path.read_bytes()).hexdigest()
                           for name, path in actual.items()})

    # Exercise the real verifier without importing deployment-only dependencies or connecting.
    tree = ast.parse((ROOT / "ops/deploy_cardputer_report.py").read_text(encoding="utf-8"))
    verifier = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "verify_runtime")
    namespace = {
        "json": json, "Path": Path, "shlex": shlex, "time": time, "run": remote_read,
        "current": lambda *_: after, "unchanged": lambda *_: None,
    }
    exec(compile(ast.Module(body=[verifier], type_ignores=[]), "verify_runtime", "exec"), namespace)
    namespace["verify_runtime"](None, baseline, manifest, tmp_path)
    verified = json.loads((tmp_path / "runtime-verification.json").read_text())
    assert verified["loaded_source_sha256"] == manifest["source_sha256"]
    assert set(verified["preserved_source_sha256"]) == set(preserved)
