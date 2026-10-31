"""Read-only audit of the explicit early-seed4 training schedule.

This checks JSON, original bytes and chronology only. It never imports a
financial environment, a tensor library, a policy, or a training entry point.
The returned cohort envelope is a derived TIME BOUND, never a replacement for
the original failed fit_process.json or a claim that financial audits passed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

PROTOCOL_SHA256 = "e29c93d37571dd917f95b7f517a49ff8dc52cece712f14720ce1ab496ed226da"
SCHEDULER_SHA256 = "52ba244d55abf36fc49cdbaa8374c179c939dc35b07f3d27bbd8c7a07b233453"
SOURCES = {
    "scripts/backtest_btceth_rl.py": "dbc7ffae69ddc1c7b043302d770b2c36066cd0b9c8577fa231d6d46d8e9e16a1",
    "scripts/btceth_rl_env.py": "9bf67454009ba03bf53720b3fe1f3587430c76599d610acdfce4f03ac8befe02",
    "scripts/crypto_rl_portfolio_env.py": "56091e7d29a79316b0a3f1d3defb2531856de9fd90e9cfb3f4e63a3df48a6179",
    "scripts/crypto_rl_reporting.py": "b77e0c55d23a2adf65dd87aab8df322458754e6c908d4b2152305543af6ff6d5",
    "scripts/audit_btceth_rl.py": "3c5e6840538cadd45f53026423946358de45da35dc8d30f8984571d78ba3728c",
    "scripts/audit_btceth_rl_exposure_cap.py": "d8e941874dd05f96910785bec36d9243bea36dcc2f1ee1cc4bb9142c7483e762",
    "scripts/audit_btceth_rl_execution_band.py": "b4d699bc57b22a1b436491bd501e4ed7af784834e8156f6a0042956f5eb6285f",
    "scripts/btceth_rl_droq.py": "f1463c5e206aabd6a77e934f7972a55c84c15901b4a380e470c7dcbbcf08443a",
    "scripts/audit_btceth_rl_droq.py": "bf33c879ff5eeaa13ee288c561fb5e6f18efb3fddc0da5179e6d646339b8c2bc",
}
COUNTS = dict(environment_steps=32768, replay_adds=32768, random_exploration_actions=5001,
    learning_train_calls=27768, q_inner_rounds=555360, q_optimizer_steps=[555360, 555360],
    total_q_optimizer_steps=1110720, target_updates_per_network=[555360, 555360],
    policy_optimizer_steps=27768, alpha_optimizer_steps=27768)
PERFORMANCE_FILES = {
    "droq_execution_adapter.py": "21dcb3bf1afbd5b4285929589f1c2b0096f7ceb048017f944cefac347207f53c",
    "replay_cache_adapter.py": "4f2f2c9ef2a992e8937940adb6794d3c65de7db7a6fe580d1e03acdfa929441b",
    "polyak_cuda_graph.py": "d97700f67b49b4e626e03fc53a99764b6589590b3aa4bbfd7a7aae8210127093",
    "benchmark_gpu_replay_cache.py": "16e176531ac416e3dc25b60c752e1bef037a003a37edafcc51c5ba2887019009",
    "benchmark_gpu_replay_polyak_graph.py": "943235eb1d752a8f5a8b497220e0affd9aceb0b8b2648326521906871e093190",
}
LEAVES = ("record.json", "model_tensors.pt", "model.json", "training_timeout_transitions.jsonl")
IDENTITY_KEYS = ("phase", "started_at", "supervisor_pid", "child_pid", "command", "source_sha256", "protocol_sha256")
PROCESS_KEYS = ("pid", "exe", "starttime_ticks", "boot_id", "uid", "username")


def require(value, message):
    if not value:
        raise ValueError(message)


def regular(path):
    path = Path(path)
    for part in (path, *path.parents):
        require(not part.is_symlink() and not (hasattr(part, "is_junction") and part.is_junction()), "Links/junctions are not audit evidence")
    return path


def safe(root, relative):
    text = str(relative).replace("\\", "/"); key = PurePosixPath(text)
    require(bool(text) and not key.is_absolute() and ".." not in key.parts and ":" not in text, "Safe relative evidence path required")
    root = regular(root).resolve(); path = regular(root.joinpath(*key.parts))
    path.resolve().relative_to(root)
    return path


def sha(path):
    with regular(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.tzinfo is not None, "Explicit UTC timestamps required")
    return result.astimezone(timezone.utc)


def integer(value, *, zero=False):
    return type(value) is int and value >= (0 if zero else 1)


def argument(command, flag):
    require(isinstance(command, list) and all(isinstance(v, str) for v in command) and command.count(flag) == 1, "One actual command argument required: " + flag)
    index = command.index(flag)
    require(index + 1 < len(command), "Missing actual command value")
    return command[index + 1]


class Inputs:
    def __init__(self, study, main, source):
        self.roots = dict(study=regular(study), main=regular(main), source=regular(source))
        self.maps = {key: {} for key in self.roots}; self.external = {}; self.checks = 0

    def capture(self, root, name):
        path = safe(self.roots[root], name); digest = sha(path)
        prior = self.maps[root].setdefault(name, digest)
        require(prior == digest, "Evidence changed during audit: " + name)
        self.checks += 1
        return path

    def json(self, root, name):
        path = self.capture(root, name); value = json.loads(path.read_text(encoding="utf-8"))
        require(sha(path) == self.maps[root][name], "JSON changed during read")
        return value

    def outside(self, path):
        path = regular(path).resolve(); key = path.as_posix(); digest = sha(path)
        require(self.external.setdefault(key, digest) == digest, "External source changed")
        self.checks += 1
        return digest

    def bound(self, root, name, digest):
        require(sha(self.capture(root, name)) == digest, "SHA mismatch: " + root + "/" + name)

    def stable(self):
        for root, values in self.maps.items():
            for name, digest in values.items():
                require(sha(safe(self.roots[root], name)) == digest, "Evidence changed before audit completion")
        for name, digest in self.external.items():
            require(sha(name) == digest, "External evidence changed before audit completion")


def verify_protocol(inputs, dataset, scheduler_source):
    p = inputs.json("study", "protocol.json"); receipt = inputs.json("study", "protocol_receipt.json")
    require(inputs.maps["study"]["protocol.json"] == PROTOCOL_SHA256 == receipt.get("protocol_sha256"), "Exact e29 protocol and receipt required")
    require(p.get("source_sha256") == SOURCES and len(SOURCES) == 9, "Exactly the frozen nine financial sources required")
    for name, digest in SOURCES.items():
        inputs.bound("source", name, digest); inputs.bound("study", "sources/" + name, digest)
    require(inputs.outside(scheduler_source) == SCHEDULER_SHA256, "Actually used 52ba scheduler source required")
    require(inputs.outside(dataset) == p.get("dataset_sha256") and inputs.outside(Path(dataset).parent / "manifest.json") == p.get("dataset_manifest_sha256"), "Original data bytes required; no pickle loading")
    inputs.bound("main", "frozen_spec.json", p["original_frozen_spec_sha256"])
    require(p.get("model_count") == 5 and p.get("timesteps_per_model") == 32768 and p.get("expected_counters") == COUNTS
            and p.get("seeds") == list(range(5)) and p.get("algorithm") == "droq"
            and p.get("profiles") == {"balanced": {"drawdown_penalty": .25, "rebalance_band": .025}}, "Unchanged finite five-seed recipe required")
    return p, receipt


def verify_resources(annex, p):
    snap = annex["resource_snapshot"]
    require(snap.get("command") == ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], "Unfiltered compute query required")
    rows = []
    for line in snap["unfiltered_stdout"].splitlines():
        if line.strip():
            uuid, pid = [part.strip() for part in line.split(",")]; rows.append(dict(gpu_uuid=uuid, pid=int(pid)))
    require(rows == snap.get("rows"), "Parsed resource rows must match original bytes")
    active = annex["original_running_2_3"]
    require(set(active) == {"2", "3"}, "Only original seed2 and seed3 were active")
    expected = {(p["seed_gpu_assignments"][seed], value["process"]["pid"]) for seed, value in active.items()}
    allowed = expected | {(row["gpu_uuid"], row["pid"]) for row in p.get("allowed_gpu_rows", [])}
    actual = {(row["gpu_uuid"], row["pid"]) for row in rows}
    require(len(rows) == len(actual) and expected <= actual <= allowed, "Unknown GPU process cannot be suppressed")
    for _, pid in actual:
        observed = snap["instances"][str(pid)]
        declared = next((item["process"] for item in active.values() if item["process"]["pid"] == pid), p.get("allowed_compute_instance"))
        require(declared and all(observed.get(k) == declared.get(k) for k in PROCESS_KEYS), "Exact original PID instance required")
    require(snap.get("physical_command") == ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], "Original physical-device query required")
    devices = []
    for line in snap["physical_stdout"].splitlines():
        if line.strip():
            index, uuid, name, total, used, util = [v.strip() for v in line.split(",")]
            devices.append(dict(index=int(index), uuid=uuid, name=name, memory_total_mib=int(total), memory_used_mib=int(used), utilization_percent=int(util)))
    require(devices == snap.get("physical_gpus") and len(devices) == 2, "Two actual device rows required")
    keys = ("index", "uuid", "name", "memory_total_mib")
    require([{k: d[k] for k in keys} for d in sorted(devices, key=lambda d: d["index"])] == [{k: d[k] for k in keys} for d in p["physical_gpus"]], "Original physical V100 identity required")
    require(all("V100" in d["name"] and 0 <= d["memory_used_mib"] <= d["memory_total_mib"] and 0 <= d["utilization_percent"] <= 100 for d in devices), "Finite real resource observations required")


def verify_record(inputs, p, seed):
    folder = f"training/droq__balanced__seed{seed}"; record = inputs.json("study", folder + "/record.json")
    require((record.get("algorithm"), record.get("profile"), record.get("seed"), record.get("actual_timesteps")) == ("droq", "balanced", seed, 32768), "Actual completed seed identity required")
    require(record.get("actual_counters") == COUNTS and record.get("hyperparameters") == p["hyperparameters"] and record.get("protocol_sha256") == PROTOCOL_SHA256, "Actual optimizer counts and original recipe required")
    require(not safe(inputs.roots["study"], folder + "/failure.json").exists(), "Failed seed cannot be adopted")
    device = record["device"]
    require(device.get("device") == "cuda:0" and device.get("capability") == [7, 0] and "V100" in device.get("name", "") and device.get("visible_gpu_uuid") == p["seed_gpu_assignments"][str(seed)], "Actual fixed-seed device required")
    for leaf, field, digest in (("model_tensors.pt", "model_tensor_file", "model_tensor_sha256"), ("model.json", "model_manifest_file", "model_manifest_sha256"), ("training_timeout_transitions.jsonl", "timeout_trace_relative_path", "timeout_trace_sha256")):
        require(record.get(field) == folder + "/" + leaf, "Original record-relative artifact path required")
        inputs.bound("study", folder + "/" + leaf, record[digest])
    manifest = inputs.json("study", folder + "/model.json")
    require(manifest.get("tensor_sha256") == record["model_tensor_sha256"] and manifest.get("parameter_sha256") == record["parameter_sha256"] and manifest.get("counters") == COUNTS, "Tensor/JSON identities and counts required")
    history = record.get("training_episode_history", []); first, last = map(utc, p["train"])
    require(len(history) == 1024, "All 1024 completed training fragments required")
    for item in history:
        start, end, until = (utc(item[key]) for key in ("start", "end_exclusive", "completed_until"))
        require(first <= start < end <= last and (end - start).total_seconds() == 32 * 86400 and until == end and item.get("completed_steps") == 32 and item.get("terminated") is False and item.get("truncated") is True, "Completed causal TRAIN-only calendar required")
    started = inputs.json("study", folder + "/started.json")
    require(started.get("protocol_sha256") == PROTOCOL_SHA256 and started.get("task", {}).get("seed") == seed and started.get("task", {}).get("profile") == "balanced" and integer(started.get("pid")), "Actual source-bound seed start required")
    require(started.get("device", {}).get("visible_gpu_uuid") == device["visible_gpu_uuid"], "Start and record physical devices agree")
    return record, started


def verify_performance(inputs, annex, worker, performance_root):
    performance = annex.get("performance_adapter")
    require(performance is not None, "This schedule requires the explicitly approved adapter")
    require(performance.get("sha256") == PERFORMANCE_FILES["droq_execution_adapter.py"] and Path(performance["path"]).name == "droq_execution_adapter.py", "Approved worker-local adapter identity required")
    parent = Path(performance["path"]).parent
    require(performance.get("files") == {str(parent / name): digest for name, digest in PERFORMANCE_FILES.items()}, "All five exact execution-source identities required")
    actual = regular(performance_root if performance_root is not None else parent)
    for name, digest in PERFORMANCE_FILES.items():
        require(inputs.outside(actual / name) == digest, "Performance source SHA changed")
    require(worker.get("performance_adapter") == performance and worker.get("performance_install_count") == 1, "One explicit new-worker installation required")
    provenance = worker.get("performance_provenance", {})
    require(provenance.get("scope") == "explicit_new_worker_execution_adapter" and provenance.get("sources") == PERFORMANCE_FILES
            and provenance.get("live_attach") is False and provenance.get("install_before_first_store") is True
            and provenance.get("pinned_core_sha256") == "16fec92b1de1ce5942599f57079bc3a002b2869429515b4cd68606bda8fb52db"
            and provenance.get("pinned_algorithm_sha256") == "1417451e076f59584e1d5f3cdea1c47366c6625dff5591e3d864dd7c88152cf7", "Original learning-source and pre-store execution-adapter provenance required")


def verify_original_terminal(inputs, original, command):
    require(original.get("status") == "failed" and integer(original.get("exit_code")), "Original fit must retain actual collision failure")
    inputs.bound("study", "fit.log", original["log_sha256"])
    collision_path = argument(command, "--output-dir").replace("\\", "/").rstrip("/") + "/training/droq__balanced__seed4"
    text = safe(inputs.roots["study"], "fit.log").read_text(encoding="utf-8", errors="replace")
    errors = [line for line in re.findall(r"(?m)^[A-Za-z_][\w.]*: [^\n]*", text) if not line.startswith("concurrent.futures.process._RemoteTraceback:")]
    require(errors and all(line.startswith("FileExistsError:") and collision_path in line.replace("\\", "/") for line in errors), "Only actual planned seed4 collision may explain original failure")
    return dict(file="fit_process.json", sha256=inputs.maps["study"]["fit_process.json"], status="failed", exit_code=original["exit_code"],
        log_sha256=original["log_sha256"], collision_verified=True, collision_path=collision_path)


def verify_accelerated_fit(output, main, dataset, source_root, *, scheduler_source=None, performance_root=None):
    """Return pending/failed/passed without dispatch, tensor loading, or mutation."""
    inputs = Inputs(output, main, source_root)
    result = dict(schema_version=1, audited_at=datetime.now(timezone.utc).isoformat(), status="failed", passed=False,
        audit_scope="training_schedule_and_artifact_identity_only", financial_audit_passed=False, tensor_payload_loaded=False,
        training_model_count=0, failures=[], pending_reasons=[], derived_cohort_envelope=None)
    try:
        scheduler_source = scheduler_source or Path(__file__).with_name("btceth_rl_droq_scheduling.py")
        inputs.outside(Path(__file__))
        p, receipt = verify_protocol(inputs, dataset, scheduler_source)
        annex = inputs.json("study", "scheduling_annex.json"); annex_sha = inputs.maps["study"]["scheduling_annex.json"]
        require(annex.get("protocol_sha256") == PROTOCOL_SHA256 and annex.get("scheduler_source_sha256") == SCHEDULER_SHA256 and annex.get("source_sha256") == SOURCES, "Annex binds unchanged core and actual scheduling source")
        require(annex.get("mathematics_changed") is False and annex.get("no_signals") is True and annex.get("new_seed_count") == 0 and annex.get("only_queued_seed4_started_earlier") is True, "Scheduling-only scope must be explicit")
        initial_name = annex["original_fit_initial_snapshot_file"]; initial = inputs.json("study", initial_name)
        require(inputs.maps["study"][initial_name] == annex.get("original_fit_initial_sha256") and initial.get("status") == "running", "Original running fit snapshot required")
        initial_identity = {key: initial[key] for key in IDENTITY_KEYS}
        require(initial_identity == annex.get("original_fit_identity") and initial.get("phase") == "fit" and initial.get("source_sha256") == SOURCES["scripts/btceth_rl_droq.py"] and initial.get("protocol_sha256") == PROTOCOL_SHA256, "Original real fit identity required")
        original = inputs.json("study", "fit_process.json")
        require({key: original[key] for key in IDENTITY_KEYS} == initial_identity, "Original fit identity cannot change")
        for role, expected_pid, expected_parent in (("original_supervisor", initial["supervisor_pid"], None), ("original_child", initial["child_pid"], initial["supervisor_pid"])):
            instance = annex[role]
            require(instance.get("pid") == expected_pid and integer(expected_pid) and (expected_parent is None or instance.get("ppid") == expected_parent), "Original supervisor/child ancestry required")
        verify_resources(annex, p)
        expected_initial = {f"training/droq__balanced__seed{seed}/{leaf}" for seed in (0, 1) for leaf in LEAVES}
        require(set(annex.get("original_0_1_file_sha256", {})) == expected_initial, "Both already completed models must be immutable")
        for name, digest in annex["original_0_1_file_sha256"].items():
            inputs.bound("study", name, digest)
        for seed in (2, 3):
            item = annex["original_running_2_3"][str(seed)]
            name = f"training/droq__balanced__seed{seed}/started.json"; inputs.bound("study", name, item["started_sha256"])
            started = inputs.json("study", name)
            require(started.get("pid") == item["process"].get("pid") and item["process"].get("ppid") == initial["child_pid"] and utc(started["at"]) <= utc(annex["prepared_at"]), "Original seed2/3 start instance preserved")
        process = inputs.json("study", "accelerated_fit_process.json"); owner = inputs.json("study", "accelerated_fit.owner.json")
        require(process.get("phase") == "accelerated_fit" and process.get("source_sha256") == SCHEDULER_SHA256 and process.get("protocol_sha256") == PROTOCOL_SHA256 and process.get("annex_sha256") == annex_sha, "New actual supervisor source/protocol/annex chain required")
        require(integer(process.get("supervisor_pid")) and owner.get("supervisor_pid") == process["supervisor_pid"] and owner.get("source_sha256") == SCHEDULER_SHA256 and isinstance(owner.get("token"), str) and len(owner["token"]) == 32, "Exclusive supervisor ownership required")
        command = process["command"]
        require(len(command) > 3 and Path(command[2]).name == "btceth_rl_droq_scheduling.py" and "--worker" in command and argument(command, "--owner-token") == owner["token"], "Actual independent worker invocation required")
        require(utc(receipt["sealed_at"]) <= utc(initial["started_at"]) <= utc(annex["prepared_at"]) <= utc(owner["at"]) <= utc(process["started_at"]), "Original freeze and real scheduling chronology required")
        require(utc(initial["started_at"]) <= utc(annex["resource_snapshot"]["at"]) <= utc(annex["prepared_at"]), "Resource snapshot precedes the real scheduling annex")
        records, starts = [], []
        for seed in range(5):
            if safe(output, f"training/droq__balanced__seed{seed}/record.json").exists():
                record, started = verify_record(inputs, p, seed); records.append(record); starts.append(started)
                require(utc(initial["started_at"]) <= utc(started["at"]) <= utc(record["completed_at"]), "Original cohort starts before actual model completion")
                if seed in (0, 1):
                    require(utc(record["completed_at"]) <= utc(annex["prepared_at"]), "Completed seed0/1 precede schedule change")
        result["training_model_count"] = len(records)
        require(process.get("status") in ("launching", "running", "completed", "failed"), "Known true new process status required")
        require(original.get("status") in ("running", "failed"), "Original fit cannot be relabelled as successful")
        require(process.get("status") != "failed", "Actual accelerated supervisor failed; retain its terminal")
        if original.get("status") == "failed":
            verify_original_terminal(inputs, original, command)
        if process.get("status") != "launching":
            require(integer(process.get("child_pid")) and process["child_pid"] != process["supervisor_pid"], "Actual distinct independent worker PID required")
        if safe(output, "accelerated_seed4_worker.json").exists():
            observed = inputs.json("study", "accelerated_seed4_worker.json")
            require(observed.get("pid") == process.get("child_pid") and observed.get("ppid") == process["supervisor_pid"]
                    and observed.get("seed") == 4 and observed.get("protocol_sha256") == PROTOCOL_SHA256
                    and observed.get("annex_sha256") == annex_sha and observed.get("status") in ("running", "completed"), "Actual nonfailed independent worker identity required")
            if observed.get("performance_install_count") == 1:
                verify_performance(inputs, annex, observed, performance_root)
            else:
                require(observed.get("performance_install_count") == 0 and observed.get("status") == "running", "Pending install cannot masquerade as completed execution")
        if process.get("status") != "completed":
            result.update(status="pending", pending_reasons=["Accelerated supervisor has not reached its real terminal completion"])
        else:
            require(integer(process.get("child_pid")) and process["child_pid"] != process["supervisor_pid"] and type(process.get("exit_code")) is int and process["exit_code"] == 0 and type(process.get("worker_exit_code")) is int and process["worker_exit_code"] == 0, "Actual new supervisor and worker terminal0 required")
            require(process.get("log_relative_path") == "accelerated_seed4.log", "Original new-worker log identity required")
            inputs.bound("study", "accelerated_seed4.log", process["log_sha256"])
            completion = inputs.json("study", "accelerated_fit_completion.json"); inputs.bound("study", "accelerated_fit_completion.json", process["completion_sha256"])
            worker = inputs.json("study", "accelerated_seed4_worker.json"); inputs.bound("study", "accelerated_seed4_worker.json", completion["worker_proof_sha256"])
            canonical = inputs.json("study", "training_completion.json"); inputs.bound("study", "training_completion.json", completion["training_completion_sha256"])
            require(len(records) == 5 and [r["seed"] for r in records] == list(range(5)) and completion.get("model_count") == canonical.get("model_count") == 5 and completion.get("records") == canonical.get("records") == records, "All five real records required; partial cannot pass")
            require(completion.get("protocol_sha256") == canonical.get("protocol_sha256") == PROTOCOL_SHA256 and completion.get("annex_sha256") == annex_sha and completion.get("source_sha256") == SOURCES and completion.get("scheduler_source_sha256") == SCHEDULER_SHA256 and completion.get("no_signals") is True and completion.get("original_0_1_hashes_preserved") is True, "Explicit completion scope and fixed sources required")
            expected_files = {f"training/droq__balanced__seed{seed}/{leaf}" for seed in range(5) for leaf in (*LEAVES, "started.json")}
            require(set(completion.get("record_file_sha256", {})) == expected_files, "Complete 25-file training inventory required")
            for name, digest in completion["record_file_sha256"].items():
                inputs.bound("study", name, digest)
            terminal = completion["original_fit_terminal"]
            require(terminal == verify_original_terminal(inputs, original, command), "Original failed terminal and log must remain bound")
            require(worker.get("status") == "completed" and worker.get("pid") == process["child_pid"] == starts[4]["pid"] and worker.get("ppid") == process["supervisor_pid"] and worker.get("seed") == 4 and worker.get("protocol_sha256") == PROTOCOL_SHA256 and worker.get("annex_sha256") == annex_sha and worker.get("record_sha256") == inputs.maps["study"]["training/droq__balanced__seed4/record.json"] and worker.get("parameter_sha256") == records[4]["parameter_sha256"], "Actual independent worker4 identity and completion required")
            verify_performance(inputs, annex, worker, performance_root)
            require(utc(process["started_at"]) <= utc(worker["started_at"]) <= utc(starts[4]["at"]) <= utc(records[4]["completed_at"]) <= utc(worker["completed_at"]) <= utc(process["worker_completed_at"]), "New worker execution chronology required")
            require(utc(initial["started_at"]) <= utc(original["completed_at"]) <= utc(canonical["at"]) <= utc(completion["at"]) <= utc(completion["cohort_completed_at"]) <= utc(process["completed_at"]), "Original/new distinct cohort and adoption chronology required")
            require(utc(process["worker_completed_at"]) <= utc(canonical["at"]) and completion.get("cohort_started_at") == initial["started_at"] and all(utc(r["completed_at"]) <= utc(canonical["at"]) for r in records), "Five actual model completions precede canonical adoption")
            envelope = dict(kind="derived_training_time_bounds_only", started_at=initial["started_at"], completed_at=process["completed_at"],
                model_count=5, original_terminal_status="failed", original_terminal_exit_code=original["exit_code"], accelerated_terminal_status="completed",
                original_terminal_relabelled=False, source="accelerated_fit_completion.json + both actual process terminals",
                original_fit_process_sha256=inputs.maps["study"]["fit_process.json"], accelerated_process_sha256=inputs.maps["study"]["accelerated_fit_process.json"],
                accelerated_completion_sha256=inputs.maps["study"]["accelerated_fit_completion.json"], training_completion_sha256=inputs.maps["study"]["training_completion.json"])
            result.update(status="passed", passed=True, original_terminal_status="failed", original_collision_verified=True,
                original_0_1_preserved=True, original_2_3_start_instances_preserved=True, derived_cohort_envelope=envelope)
        inputs.stable()
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        result.update(status="failed", passed=False, failures=[type(error).__name__ + ": " + str(error)], derived_cohort_envelope=None)
    result.update(checks_total=inputs.checks, input_sha256=inputs.maps["study"], main_input_sha256=inputs.maps["main"], source_sha256=inputs.maps["source"], external_input_sha256=inputs.external)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output-dir", "reference-main", "dataset", "source-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--scheduler-source", type=Path); parser.add_argument("--performance-root", type=Path); parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    value = verify_accelerated_fit(args.output_dir, args.reference_main, args.dataset, args.source_root, scheduler_source=args.scheduler_source, performance_root=args.performance_root)
    if args.report is not None:
        with regular(args.report).open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({key: value[key] for key in ("status", "passed", "training_model_count", "failures", "pending_reasons")}, ensure_ascii=False))
    return 0 if value["status"] in ("pending", "passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
