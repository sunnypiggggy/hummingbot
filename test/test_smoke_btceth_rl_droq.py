"""Pure schema/source fixtures: these tests never construct an NN or use CUDA."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/smoke_btceth_rl_droq.py"
spec = importlib.util.spec_from_file_location("droq_smoke_schema_only", SOURCE)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def authorization(tmp_path):
    hashes = {name: "a" * 64 for name in smoke.DEPENDENCIES}
    auth = dict(scope=smoke.SCOPE, authorized=True, market_training_authorized=False,
        validation_authorized=False, annual_evaluation_authorized=False,
        synthetic_steps=5120, seed=1, gpu_uuid="GPU-actual", output_relative_name="native_smoke_fixture",
        preflight_root=str(tmp_path.resolve()), source_sha256=hashes,
        resource_allowance_sha256="b" * 64, upstream_manifest_sha256=smoke.MANIFEST_SHA,
        effective_config_contract_sha256=smoke.CONFIG_SHA, hyperparameters=smoke.recipe())
    values = dict(hashes=hashes, gpu_uuid="GPU-actual", allowance_sha="b" * 64,
        output_name="native_smoke_fixture", preflight_root=tmp_path)
    return auth, values


def test_exact_native_counts_are_not_td3_or_finite_driver_off_by_one():
    expected = smoke.expected_counts()
    assert expected == dict(environment_steps=5120, replay_adds=5120,
        random_exploration_actions=5001, learning_train_calls=120, q_inner_rounds=2400,
        q_optimizer_steps=[2400, 2400], total_q_optimizer_steps=4800,
        target_updates_per_network=[2400, 2400], policy_optimizer_steps=120, alpha_optimizer_steps=120)
    smoke.validate_counts(expected)
    bad = {**expected, "policy_optimizer_steps": 2400}
    with pytest.raises(ValueError, match="Actual Adam"):
        smoke.validate_counts(bad)
    assert smoke.expected_counts(5000)["q_inner_rounds"] == 0
    assert smoke.expected_counts(5001)["random_exploration_actions"] == 5001
    assert smoke.expected_counts(5001)["q_inner_rounds"] == 20


def test_actual_architecture_and_alpha_not_only_kwargs_are_required():
    actual = {key: value for key, value in smoke.recipe().items() if key not in
              ("alpha", "target_entropy", "target_drop_rate", "layer_norm")}
    actual.update(delay_update_steps=5000, target_entropy=-2., alpha_current=.2,
        log_alpha_current=0., dropout_per_q=[[.01, .01], [.01, .01]],
        layer_norm_per_q=[[[256], [256]], [[256], [256]]], target_training=[True, True])
    smoke.validate_effective_parameters(actual)
    for key, value in (("alpha_current", 1.), ("dropout_per_q", [[], []]),
                       ("layer_norm_per_q", [[], []]), ("target_training", [False, False])):
        with pytest.raises(ValueError):
            smoke.validate_effective_parameters({**actual, key: value})


@pytest.mark.parametrize("value", [0, -1, True, 5.1])
def test_noninteger_or_nonpositive_budget_rejected(value):
    with pytest.raises(ValueError):
        smoke.expected_counts(value)


def test_authorization_pins_scope_math_sources_output_and_resource(tmp_path):
    auth, values = authorization(tmp_path)
    smoke.validate_authorization(auth, **values)
    mutations = {
        "authorized": False, "market_training_authorized": True,
        "validation_authorized": True, "annual_evaluation_authorized": True,
        "scope": "old_td3_market_training", "synthetic_steps": 32768,
        "seed": 0, "gpu_uuid": "GPU-other", "output_relative_name": "formal_study",
        "preflight_root": str(tmp_path / "other"), "source_sha256": {},
        "resource_allowance_sha256": "c" * 64, "upstream_manifest_sha256": "d" * 64,
        "effective_config_contract_sha256": "e" * 64,
        "hyperparameters": {**smoke.recipe(), "target_drop_rate": .02},
    }
    for key, value in mutations.items():
        bad = copy.deepcopy(auth); bad[key] = value
        with pytest.raises(ValueError, match="synthetic-only"):
            smoke.validate_authorization(bad, **values)
    missing = copy.deepcopy(auth); missing.pop("annual_evaluation_authorized")
    with pytest.raises(ValueError):
        smoke.validate_authorization(missing, **values)


def test_new_output_cannot_reuse_or_escape_preflight(tmp_path):
    target = tmp_path / "native_smoke_a"
    assert smoke.new_output(tmp_path, target) == target
    for bad in (tmp_path, tmp_path / "formal_study", tmp_path / "nested/native_smoke_a", tmp_path.parent / "native_smoke_a"):
        with pytest.raises(ValueError):
            smoke.new_output(tmp_path, bad)
    target.mkdir()
    with pytest.raises(ValueError):
        smoke.new_output(tmp_path, target)


def test_resource_guard_does_not_filter_unknown_tasks_or_pid_reuse():
    instance = dict(pid=17, uid=1000, username="sunny", exe="/usr/bin/ptyxis",
                    starttime_ticks="100", boot_id="boot-one")
    allowed = dict(gpu_uuid="GPU-original", pid=17)
    contract = dict(scope=smoke.SCOPE, allowed_instance=instance, allowed_gpu_rows=[allowed])
    inspect = lambda pid: instance
    smoke.verify_resource([allowed], contract, inspect)
    with pytest.raises(ValueError, match="Unknown GPU task"):
        smoke.verify_resource([allowed, dict(gpu_uuid="GPU-original", pid=18)], contract, inspect)
    with pytest.raises(ValueError, match="identity changed"):
        smoke.verify_resource([allowed], contract, lambda pid: {**instance, "starttime_ticks": "101"})
    with pytest.raises(ValueError):
        smoke.verify_resource([], contract, inspect)
    child = dict(gpu_uuid="GPU-smoke", pid=123)
    smoke.verify_resource([allowed, child], contract, inspect, child=child)
    with pytest.raises(ValueError):
        smoke.verify_resource([allowed, dict(gpu_uuid="GPU-wrong", pid=123)], contract, inspect, child=child)
    with pytest.raises(ValueError):
        smoke.verify_resource([allowed], {**contract, "scope": "old_scope"}, inspect)


def test_unfiltered_gpu_parser_rejects_duplicate_or_wrong_schema():
    assert smoke.resource_rows("GPU-one, 17\n") == [dict(gpu_uuid="GPU-one", pid=17)]
    for value in ("GPU-one,17\nGPU-one,17", "GPU-one,17,ignored", "garbage,17", "GPU-one,not_a_pid"):
        with pytest.raises(ValueError):
            smoke.resource_rows(value)


def timeout_fixture(steps=5120):
    rows = []
    for ordinal, transition in enumerate(range(32, steps + 1, 32), 1):
        before, terminal = [0.] * 27, [0.] * 27
        before[0], terminal[0] = 31 / 32, 1.
        rows.append(dict(transition_number=transition, replay_position=transition - 1,
            done=1., timeout=1., effective_done=0., stored_done=0., time_limit_truncated=True,
            terminated=False, truncated=True, info_date=f"synthetic_episode_{ordinal}_step_32",
            terminal_observation_values=terminal, terminal_observation_sha256=smoke.f32_sha(terminal),
            stored_next_observation_values=terminal.copy(), stored_next_observation_sha256=smoke.f32_sha(terminal),
            stored_observation_values=before, stored_observation_sha256=smoke.f32_sha(before)))
    return rows


def test_all_actual_terminal_next_values_required_instead_of_reset():
    rows = timeout_fixture(); smoke.validate_timeouts(rows)
    for field, value in (("effective_done", 1.), ("stored_done", 1.),
                         ("info_date", "2025-10-23"), ("terminal_observation_sha256", "a" * 64)):
        bad = copy.deepcopy(rows); bad[0][field] = value
        with pytest.raises(ValueError):
            smoke.validate_timeouts(bad)
    bad = copy.deepcopy(rows); bad[0]["stored_next_observation_values"] = [0.] * 27
    with pytest.raises(ValueError):
        smoke.validate_timeouts(bad)
    with pytest.raises(ValueError):
        smoke.validate_timeouts(rows[:-1])


def test_source_contract_archive_and_recipe_only_no_nn():
    upstream = ROOT / "results/backtests/btceth_rl_droq_20261030_preflight/upstream/DroQ"
    files = smoke.verify_official(upstream)
    assert len(files) == 27
    contract = json.loads((upstream.parents[1] / "effective_config_contract.json").read_text(encoding="utf-8"))
    assert contract["effective_constructor_kwargs"] == smoke.recipe()
    assert smoke.sha(upstream.parents[1] / "effective_config_contract.json") == smoke.CONFIG_SHA
    assert files["OriginalREDQCodebase/redq/algos/redq_sac.py"] == "1417451e076f59584e1d5f3cdea1c47366c6625dff5591e3d864dd7c88152cf7"


def test_public_import_and_cli_contract_never_load_frameworks_or_an_environment():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = [alias.name for node in top_imports if isinstance(node, ast.Import) for alias in node.names]
    names += [node.module for node in top_imports if isinstance(node, ast.ImportFrom)]
    assert not any((name or "").split(".")[0] in ("torch", "numpy", "gymnasium", "pandas", "scripts") for name in names)
    code = ("import runpy,sys,json;sys.argv=['smoke','--phase','contract'];"
            f"runpy.run_path({str(SOURCE)!r},run_name='__main__');"
            "assert not any(n in sys.modules for n in ['torch','numpy','gymnasium','scripts.btceth_rl_env','scripts.crypto_rl_portfolio_env'])")
    result = subprocess.run([sys.executable, "-B", "-c", code], text=True, capture_output=True, check=True)
    contract = json.loads(result.stdout)
    assert contract["prepared_only"] and contract["synthetic_steps"] == 5120
    assert contract["expected_counters"]["q_inner_rounds"] == 2400
    assert contract["training_executed"] is False and contract["model_construction_executed"] is False
    help_result = subprocess.run([sys.executable, "-B", str(SOURCE), "--help"], capture_output=True, text=True)
    assert help_result.returncode == 0 and "authorization-sha256" in help_result.stdout


def test_first_alpha_probe_uses_the_original_cpu_exp_route():
    source = SOURCE.read_text(encoding="utf-8")
    assert "exp_log_alpha=float(agent.log_alpha.detach().cpu().exp().item())" in source
    assert "agent.log_alpha.detach().exp().cpu().item()" not in source


def test_execution_without_authorization_fails_before_output_or_nn(tmp_path):
    result = subprocess.run([sys.executable, "-B", str(SOURCE), "--phase", "supervise", "--output-dir", str(tmp_path / "native_smoke_missing")], capture_output=True, text=True)
    assert result.returncode != 0
    assert "required before execution" in result.stderr
    assert not (tmp_path / "native_smoke_missing").exists()


def test_child_claim_waits_only_for_its_actual_parent_atomic_publication():
    launching = dict(phase=smoke.SCOPE, status="launching", supervisor_pid=100)
    running = {**launching, "status": "running", "child_pid": 200}
    versions = iter([launching, running]); sleeps = []
    proof = smoke.wait_for_child_claim("unused", child_pid=200, supervisor_pid=100,
        reader=lambda path: next(versions), clock=lambda: 0., sleep=sleeps.append)
    assert proof == running and sleeps == [.02]
    for wrong in ({**running, "child_pid": 201}, {**running, "supervisor_pid": 101},
                  {**running, "phase": "market"}, {**running, "status": "failed"}):
        with pytest.raises(ValueError):
            smoke.wait_for_child_claim("unused", child_pid=200, supervisor_pid=100,
                reader=lambda path: wrong, sleep=lambda seconds: pytest.fail("Wrong identity cannot wait"))


def test_unpublished_child_times_out_without_framework_or_nn_work():
    elapsed = [0.]
    def sleep(seconds):
        elapsed[0] += seconds
    with pytest.raises(ValueError, match="no NN work started"):
        smoke.wait_for_child_claim("unused", child_pid=200, supervisor_pid=100,
            reader=lambda path: dict(phase=smoke.SCOPE, status="launching", supervisor_pid=100),
            clock=lambda: elapsed[0], sleep=sleep, timeout=.05)
    assert .05 <= elapsed[0] < .08
