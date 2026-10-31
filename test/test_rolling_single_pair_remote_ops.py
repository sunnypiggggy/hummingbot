"""Control-only transport guards; no SSH or model phase is run by these tests."""
import ast
import base64
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tarfile

import pytest

from scripts import rolling_single_pair_remote_ops as ops


def fixture_root(tmp_path, monkeypatch):
    monkeypatch.setattr(ops, "ROOT", tmp_path)
    study = tmp_path / "results/backtests" / ops.STUDY
    study.mkdir(parents=True)
    for name in ops.SOURCE_NAMES:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# " + name, encoding="utf-8")
    for name in ops.INPUT_NAMES:
        p = study / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}", encoding="utf-8")
    return study


def remote_functions(tmp_path):
    root = tmp_path / "remote"
    root.mkdir()
    proc = tmp_path / "proc"
    proc.mkdir()
    tree = ast.parse(ops._REMOTE)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ("safe", "sha", "read", "write_new", "guard_stage")]
    environment = dict(root=root, hashlib=hashlib, json=json, active=lambda: [], proc=lambda pid: None,
        pathlib=SimpleNamespace(Path=lambda name: proc if name == "/proc" else Path(name)),
        REPLACEABLE_SOURCES={"source/" + p for p in ops.SOURCE_NAMES})
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<remote-control-guards>", "exec"), environment)
    return environment, root


def test_remote_code_compiles_and_exact_source_whitelist_matches_runner():
    compile(ops._REMOTE, "<remote-control>", "exec")
    from scripts import backtest_rolling_single_pair as runner
    assert ops.SOURCE_NAMES == runner.SOURCES
    assert len(ops.SOURCE_NAMES) == 14


def test_remote_receipts_are_valid_round_trip_json(tmp_path):
    namespace, root = remote_functions(tmp_path)
    target = root / "control/test.json"
    namespace["write_new"](target, {"passed": True, "seed": 0})
    assert json.loads(target.read_text()) == {"passed": True, "seed": 0}


def test_local_study_scope_and_full_plan(tmp_path, monkeypatch):
    study = fixture_root(tmp_path, monkeypatch)
    plan = ops.stage_plan(study)
    assert plan["staged_not_frozen"] is True
    assert not plan["missing_formal_inputs"]
    assert len(plan["file_sha256"]) == 14 + 6 + 1
    for name, path in plan["local_files"].items():
        assert plan["file_sha256"][name] == ops.sha(path)
    with pytest.raises(ValueError):
        ops.stage_plan(tmp_path / "old_study")
    (study / "selection_inputs/native_validation_summary.csv").unlink()
    with pytest.raises(FileNotFoundError):
        ops.stage_plan(study)
    partial = ops.stage_plan(study, sources_only=True)
    assert len(partial["file_sha256"]) == 15
    assert set(partial["missing_formal_inputs"]) == set(ops.INPUT_NAMES)


@pytest.mark.parametrize("name", ["../old/file", "/absolute/file", "source/../old", "source\\old.py", "C:/secret", "source//x", "source/./x", ""])
def test_remote_scope_rejects_unsafe_members(tmp_path, name):
    namespace, unused = remote_functions(tmp_path)
    with pytest.raises(AssertionError):
        namespace["safe"](name)


def test_remote_guard_allows_only_reversible_unfrozen_source_replacement(tmp_path):
    namespace, root = remote_functions(tmp_path)
    name = "source/" + ops.SOURCE_NAMES[0]
    path = root / name
    path.parent.mkdir(parents=True)
    path.write_text("old", encoding="utf-8")
    pins = {name: hashlib.sha256(b"new").hexdigest()}
    assert namespace["guard_stage"](pins) == "staged_not_frozen"
    (root / "protocol.json").write_text("{}", encoding="utf-8")
    with pytest.raises(AssertionError, match="Frozen"):
        namespace["guard_stage"](pins)
    assert namespace["guard_stage"]({name: ops.sha(path)}) == "already_frozen_identical"


def test_remote_guard_never_replaces_existing_selection_input(tmp_path):
    namespace, root = remote_functions(tmp_path)
    path = root / "inventory.json"
    path.write_text("old", encoding="utf-8")
    with pytest.raises(AssertionError, match="whitelist"):
        namespace["guard_stage"]({"inventory.json": hashlib.sha256(b"new").hexdigest()})


def test_staging_archive_and_receipt_are_byte_bound_without_launch(tmp_path, monkeypatch):
    study = fixture_root(tmp_path, monkeypatch)
    calls = []
    def fake_remote(action, payload, **kwargs):
        calls.append(action)
        assert action in ("check_stage", "stage")
        if action == "check_stage":
            return {"state": "staged_not_frozen"}
        raw = base64.b64decode(payload["archive_base64"])
        assert hashlib.sha256(raw).hexdigest() == payload["archive_sha256"]
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
            assert {m.name for m in archive.getmembers()} == set(payload["file_sha256"])
            for member in archive.getmembers():
                assert member.isfile()
                assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == payload["file_sha256"][member.name]
        return dict(file_sha256=payload["file_sha256"], staged_not_frozen=True, operation_id=payload["operation_id"])
    monkeypatch.setattr(ops, "remote_request", fake_remote)
    receipt = ops.stage(study)
    assert calls == ["check_stage", "stage"]
    for name in ops.SOURCE_NAMES:
        assert ops.sha(study / "source" / name) == ops.sha(tmp_path / name)
    saved = study / "control/stage_receipts" / (receipt["operation_id"] + ".json")
    assert json.loads(saved.read_text()) == receipt
    for path in study.rglob("*"):
        assert path.name not in ("protocol.json", "fit_complete.json", "preflight_complete.json")


def test_launch_rejects_unplanned_phase_before_any_remote_request(tmp_path, monkeypatch):
    monkeypatch.setattr(ops, "remote_request", lambda *args, **kwargs: pytest.fail("No SSH expected"))
    with pytest.raises(ValueError):
        ops.launch(tmp_path, "unplanned")
