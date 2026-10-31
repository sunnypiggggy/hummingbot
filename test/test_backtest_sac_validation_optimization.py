"""Protocol/control-flow and continuous-window arithmetic, never study results."""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import backtest_sac_validation_optimization as job


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")


def test_exact_search_population_and_separate_seed_sets():
    assert len(job.CANDIDATES) == 4 and job.TUNING_SEEDS == tuple(range(100, 105))
    assert job.FINAL_SEEDS == tuple(range(30)) and not set(job.FINAL_SEEDS) & set(job.TUNING_SEEDS)
    assert 4 * 2 * 5 == 40 and 30 * 2 == 60
    assert 5 * 2 * 2 + 4 * 2 * 2 == 36
    assert 30 * 2 * 2 + 4 * 2 * 2 == 136


@pytest.mark.parametrize("name", ["../x", "/x", "C:/x", "a\\b", "a/../b", "a//b", ""])
def test_canonical_evidence_paths(tmp_path, name):
    with pytest.raises(ValueError):
        job.safe(tmp_path, name)


def test_copy_tree_preserves_exact_bytes_and_refuses_overwrite(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "copy"
    write(source / "record.json", {"fixture": True})
    (source / "checkpoint").mkdir()
    (source / "checkpoint/model.zip").write_bytes(b"not-a-native-model; copy fixture only")
    result = job.copy_tree(source, destination)
    assert job.tree(source) == job.tree(destination) == result
    with pytest.raises(FileExistsError):
        job.copy_tree(source, destination)
    assert job.tree(source) == job.tree(destination)


def test_continuous_subwindows_use_previous_hour_not_new_cash(tmp_path):
    start, cut1, cut2, end = map(pd.Timestamp, ["2025-01-01T00:00:00Z", "2025-04-01T00:00:00Z", "2025-07-01T00:00:00Z", "2025-10-23T00:00:00Z"])
    windows = [[str(start), str(cut1)], [str(cut1), str(cut2)], [str(cut2), str(end)]]
    dates = pd.date_range(start, end, freq="h", inclusive="left")
    # Synthetic equity math only: there is no claimed cash/coin accounting here.
    values = np.where(dates < cut1, 10000., np.where(dates < cut2, 5000., 6000.))
    rows = []
    for seed in job.TUNING_SEEDS:
        for cost in ("base", "stress"):
            directory = tmp_path / job._case_directory("validation", "BTCUSDT", "sac", seed, cost)
            directory.mkdir(parents=True)
            pd.DataFrame({"timestamp": dates, "equity": values}).to_csv(directory / "equity_1h.csv.gz", index=False)
            rows.append({"symbol": "BTCUSDT", "algorithm": "sac", "seed": seed, "scenario": cost,
                "net_return": -.4, "max_drawdown": -.5})
    pd.DataFrame(rows).to_csv(tmp_path / "validation_summary.csv", index=False)
    result = job.candidate_metrics(tmp_path, "BTCUSDT", windows)
    expected = [0., math.log(.5) - .25 - .6, math.log(1.2)]
    np.testing.assert_allclose(result["subwindow_median_scores"], expected, atol=1e-14, rtol=0)
    assert result["score_iqr"] == 0 and result["base_positive_seeds"] == 0
    assert result["stress_p90_drawdown"] == .5


def metrics(score=0., seeds=None, *, stress=0., dd=.1, windows=None):
    return {"seed_scores": seeds or [score]*5, "score_median": score, "score_iqr": 0., "robust_score": score,
        "base_positive_seeds": 4, "stress_median_return": stress, "stress_p90_drawdown": dd,
        "subwindow_median_scores": windows or [score]*3}


@pytest.mark.parametrize("all_fallback", [False, True])
def test_validation_selects_only_all_gate_eligible_candidate_or_default(tmp_path, monkeypatch, all_fallback):
    write(tmp_path / "protocol.json", {"fixture": True})
    recipes = {c: {"fixture_id": c} for c in job.CANDIDATES}
    spec = {"candidate_recipes": recipes}
    write(tmp_path / "search_plan.json", {"subwindows": [["a", "b"], ["b", "c"], ["c", "d"]]})
    for c in job.CANDIDATES:
        root = job.candidate_root(tmp_path, c)
        (root / "validation_summary.csv").parent.mkdir(parents=True)
        (root / "validation_summary.csv").write_bytes(b"score fixture only\n")
        write(root / "validation_audit.json", {"passed": True, "accounts_checked": 36, "repeats_checked": 36})
        write(root / "validation_optimization_audit.json", {"passed": True, "failure_count": 0,
            "inputs_sha256": {"protocol.json": job.sha(tmp_path / "protocol.json")}})
    monkeypatch.setattr(job, "check_top", lambda *args: (spec, None))
    table = {"c00_baseline": metrics(), "c01_lr1e4": metrics(.02),
        "c02_gamma099": metrics(.05, stress=-.01), "c03_lr1e4_gamma099": metrics(.02)}
    if all_fallback:
        table = {key: metrics(.005) for key in table}
    monkeypatch.setattr(job, "candidate_metrics", lambda root, symbol, subwindows: deepcopy(table[root.name]))
    selected = job.choose_parameters(tmp_path, tmp_path / "never_loaded")
    assert set(selected.values()) == {"c00_baseline" if all_fallback else "c01_lr1e4"}
    frozen = job.read(tmp_path / "selection.json")
    assert frozen["uses_test"] is False and frozen["uses_final_validation"] is False and frozen["seed_selection"] is False
    assert job.read(tmp_path / "selection_receipt.json")["sha256"] == job.sha(tmp_path / "selection.json")
    if not all_fallback:
        assert frozen["selection"]["BTCUSDT"]["gates_by_candidate"]["c02_gamma099"]["stress_return_median_at_least"] is False
    with pytest.raises(FileExistsError):
        job.choose_parameters(tmp_path, tmp_path / "never_loaded")


def test_tuning_fit_uses_one_joint_forty_task_gpu_queue(tmp_path, monkeypatch):
    calls = []
    core = SimpleNamespace(ROOT=tmp_path / "core", run_parallel=lambda output, tasks, fn, phase, workers: calls.append((output,tasks,fn,phase,workers)))
    monkeypatch.setattr(job, "check_top", lambda *args: ({}, core))
    monkeypatch.setattr(job, "check_child", lambda *args: {"seeds": list(job.TUNING_SEEDS)})
    def finish(top, child):
        write(child / "training_counter_audit.json", {"passed": True, "fixture": True})
        return {"models_checked": 10}
    monkeypatch.setattr(job, "finish_fit", finish)
    write(tmp_path / "protocol.json", {"fixture": True})
    result = job.fit(tmp_path, tmp_path / "never_loaded", stage="tuning", max_workers=28)
    assert result == {"fresh_models": 40, "completed_models": 40}
    assert len(calls) == 1
    output,tasks,function,phase,workers = calls[0]
    assert output == tmp_path and function is job.train_one and phase == "tuning_fit" and workers == 28
    assert len(tasks) == 40 and {t["seed"] for t in tasks} == set(job.TUNING_SEEDS)
    assert {Path(t["child"]).name for t in tasks} == set(job.CANDIDATES)


def test_default_final_is_imported_not_retrained(tmp_path, monkeypatch):
    write(tmp_path / "protocol.json", {"fixture": True})
    write(tmp_path / "selection.json", {"selection": {s: {"candidate_id": "c00_baseline"} for s in job.SYMBOLS}})
    for symbol in job.SYMBOLS:
        for seed in job.FINAL_SEEDS:
            directory = tmp_path / f"references/{symbol}/training/{symbol}/sac/seed{seed}"
            write(directory / "record.json", {"fixture": [symbol,seed]})
            (directory / "model_bytes").write_bytes(b"original")
    monkeypatch.setattr(job, "check_top", lambda *args: ({}, None))
    monkeypatch.setattr(job, "_selection_gate", lambda top: job.read(top / "selection.json"))
    def prepare_child(top, child, stage, ids):
        write(child / "protocol.json", {"fixture": True, "recipe_ids_by_symbol": ids})
    monkeypatch.setattr(job, "_prepare_child", prepare_child)
    result = job.prepare_final(tmp_path, tmp_path / "never_loaded")
    assert result == {"retained_models": 60, "new_models": 0}
    manifest = job.read(tmp_path / "final/import_manifest.json")
    assert len(manifest["entries"]) == 60 and all(e["origin"] == "retained_baseline_fallback" for e in manifest["entries"])
    for entry in manifest["entries"]:
        assert job.tree(tmp_path / entry["source_directory"]) == job.tree(tmp_path / entry["destination_directory"]) == entry["file_sha256"]


def test_tuning_has_no_annual_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(job, "check_top", lambda *args: ({}, None))
    with pytest.raises(ValueError, match="no annual"):
        job.evaluate(tmp_path, tmp_path / "never_loaded", stage="tuning", split="test")


@pytest.mark.parametrize("audit", [{"passed": "true", "failure_count": 0, "inputs_sha256": {}},
    {"passed": True, "failure_count": 1, "inputs_sha256": {"x": "y"}},
    {"passed": True, "failure_count": 0, "inputs_sha256": {}}])
def test_audit_requires_real_pass_and_current_input_hashes(tmp_path, audit):
    with pytest.raises(ValueError):
        job._verify_audit_inputs(tmp_path, audit)
