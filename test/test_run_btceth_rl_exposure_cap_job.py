"""Subprocess lifecycle checks; no market data, policies or GPU are loaded."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "scripts/run_btceth_rl_exposure_cap_job.py"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def job(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    wrapper = scripts / SOURCE.name
    shutil.copyfile(SOURCE, wrapper)
    runner = scripts / "btceth_rl_exposure_cap.py"
    runner.write_text("import sys\nprint('SYNTHETIC_LIFECYCLE_FIXTURE_ONLY')\nsys.exit(0)\n", encoding="utf-8")
    main, output = tmp_path / "main", tmp_path / "overlay"
    main.mkdir()
    output.mkdir()
    spec = main / "frozen_spec.json"
    spec.write_text('{"fixture_only":true}', encoding="utf-8")
    dataset = tmp_path / "dataset.fixture"
    dataset.write_bytes(b"FIXTURE_NO_MARKET_DATA")

    def seal():
        protocol = {"source_sha256": {"scripts/btceth_rl_exposure_cap.py": digest(runner)},
                    "dataset_sha256": digest(dataset), "original_frozen_spec_sha256": digest(spec)}
        (output / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    seal()

    def execute(phase="validate"):
        return subprocess.run([sys.executable, str(wrapper), "--phase", phase,
                               "--main-output", str(main), "--dataset", str(dataset),
                               "--output-dir", str(output)], capture_output=True, text=True, timeout=20)
    return main, output, runner, dataset, seal, execute


@pytest.mark.parametrize("phase", ("validate", "test"))
def test_actual_terminal_exit_and_log_sha_are_recorded(job, phase):
    main, output, runner, dataset, seal, execute = job
    result = execute(phase)
    assert result.returncode == 0
    proof = json.loads((output / (phase + "_process.json")).read_text(encoding="utf-8"))
    log = output / (phase + ".log")
    assert proof["status"] == "completed" and proof["exit_code"] == 0
    assert proof["child_pid"] > 0 and proof["supervisor_pid"] > 0
    assert proof["child_pid"] != proof["supervisor_pid"]
    assert proof["started_at"] <= proof["completed_at"]
    assert proof["log_sha256"] == digest(log)
    assert proof["runner_sha256"] == digest(runner)
    assert proof["protocol_sha256"] == digest(output / "protocol.json")
    assert "SYNTHETIC_LIFECYCLE_FIXTURE_ONLY" in log.read_text(encoding="utf-8")
    assert proof["command"][proof["command"].index("--main-output") + 1] == str(main.resolve())


def test_nonzero_child_exit_is_failed_and_preserved(job):
    main, output, runner, dataset, seal, execute = job
    runner.write_text("raise SystemExit(7)\n", encoding="utf-8")
    seal()
    result = execute()
    proof = json.loads((output / "validate_process.json").read_text(encoding="utf-8"))
    assert result.returncode == 7 and proof["status"] == "failed" and proof["exit_code"] == 7
    original = (output / "validate_process.json").read_bytes()
    assert execute().returncode != 0
    assert (output / "validate_process.json").read_bytes() == original


@pytest.mark.parametrize("changed", ("runner", "dataset", "spec"))
def test_changed_inputs_fail_before_creating_process_or_log(job, changed):
    main, output, runner, dataset, seal, execute = job
    target = {"runner": runner, "dataset": dataset, "spec": main / "frozen_spec.json"}[changed]
    target.write_bytes(target.read_bytes() + b"\nCHANGED")
    assert execute().returncode != 0
    assert not (output / "validate_process.json").exists()
    assert not (output / "validate.log").exists()


def test_existing_log_prevents_another_child(job):
    main, output, runner, dataset, seal, execute = job
    log = output / "validate.log"
    log.write_bytes(b"PRESERVE_PRIOR_ATTEMPT")
    assert execute().returncode != 0
    assert log.read_bytes() == b"PRESERVE_PRIOR_ATTEMPT"
    assert not (output / "validate_process.json").exists()
