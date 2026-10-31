"""Scheduling-only synthetic fixtures: no SSH/GPU/NN/market or real PID access."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts import btceth_rl_droq_scheduling as s


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value),encoding="utf-8")


@pytest.fixture
def state(tmp_path,monkeypatch):
    output,main,source,data=(tmp_path/name for name in ("study","main","source","data"))
    for folder in (output,main,source,data): folder.mkdir()
    dataset=data/"not_a_pickle"; dataset.write_bytes(b"synthetic byte identity"); put(data/"manifest.json",{})
    put(main/"frozen_spec.json",{"synthetic":True})
    names=["scripts/btceth_rl_droq.py","scripts/audit_btceth_rl_droq.py",*(f"scripts/fixture{i}.py" for i in range(7))]
    for name in names:
        path=source/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text("synthetic:"+name)
        copy=output/"sources"/name; copy.parent.mkdir(parents=True,exist_ok=True); copy.write_bytes(path.read_bytes())
    code=s.hashes(source,names); monkeypatch.setattr(s,"RUNNER_SHA256",code[names[0]]); monkeypatch.setattr(s,"AUDITOR_SHA256",code[names[1]])
    gpus=[dict(index=i,uuid=f"GPU-{i}",name="Tesla V100",memory_total_mib=32768,memory_used_mib=19,utilization_percent=0) for i in range(2)]
    p=dict(source_sha256=code,algorithm="droq",model_count=5,timesteps_per_model=32768,expected_counters=s.COUNTS,seeds=list(range(5)),
        profiles={"balanced":{"drawdown_penalty":.25,"rebalance_band":.025}},hyperparameters={"synthetic_recipe":True},
        dataset_sha256=s.sha(dataset),dataset_manifest_sha256=s.sha(data/"manifest.json"),original_frozen_spec_sha256=s.sha(main/"frozen_spec.json"),
        seed_gpu_assignments={str(i):f"GPU-{i%2}" for i in range(5)},train=["2020-01-01T00:00:00Z","2025-01-01T00:00:00Z"],physical_gpus=gpus,allowed_gpu_rows=[])
    put(output/"protocol.json",p); monkeypatch.setattr(s,"PROTOCOL_SHA256",s.sha(output/"protocol.json")); put(output/"protocol_receipt.json",dict(protocol_sha256=s.PROTOCOL_SHA256))
    fit=dict(phase="fit",started_at="2026-10-30T01:00:00Z",supervisor_pid=101,child_pid=102,command=["synthetic"],source_sha256=s.RUNNER_SHA256,protocol_sha256=s.PROTOCOL_SHA256,status="running")
    put(output/"fit_process.json",fit)
    for seed in (2,3):
        put(output/f"training/droq__balanced__seed{seed}/started.json",dict(at="2026-10-30T01:01:00Z",pid=200+seed,protocol_sha256=s.PROTOCOL_SHA256,
            task=dict(seed=seed),device=dict(visible_gpu_uuid=p["seed_gpu_assignments"][str(seed)])))
    def proc(pid):
        return dict(pid=pid,ppid=101 if pid==102 else 102,state="S",exe="/synthetic/python",starttime_ticks=str(pid),boot_id="synthetic",uid=1000,username="fixture")
    monkeypatch.setattr(s,"proc_identity",proc)
    def snapshot(argv,**kw):
        return "GPU-0, 202\nGPU-1, 203\n" if "--query-compute-apps=gpu_uuid,pid" in argv else "0, GPU-0, Tesla V100, 32768, 500, 13\n1, GPU-1, Tesla V100, 32768, 501, 12\n"
    monkeypatch.setattr(s.subprocess,"check_output",snapshot)
    def model(seed):
        folder=output/f"training/droq__balanced__seed{seed}"; folder.mkdir(parents=True,exist_ok=True)
        (folder/"model_tensors.pt").write_bytes(b"No Tensor payload: synthetic scheduler only")
        (folder/"training_timeout_transitions.jsonl").write_bytes(b"synthetic trace byte identity")
        param=str(seed)*64; put(folder/"model.json",dict(counters=s.COUNTS,tensor_sha256=s.sha(folder/"model_tensors.pt"),parameter_sha256=param))
        start=datetime(2020,1,1,tzinfo=timezone.utc); end=start+timedelta(days=32)
        episode=dict(start=start.isoformat(),end_exclusive=end.isoformat(),completed_until=end.isoformat(),completed_steps=32,terminated=False,truncated=True)
        record=dict(algorithm="droq",profile="balanced",seed=seed,actual_timesteps=32768,actual_counters=s.COUNTS,hyperparameters=p["hyperparameters"],protocol_sha256=s.PROTOCOL_SHA256,
            device=dict(device="cuda:0",capability=[7,0],name="Tesla V100",visible_gpu_uuid=p["seed_gpu_assignments"][str(seed)]),parameter_sha256=param,
            training_episode_history=[episode]*1024,completed_at="2026-10-30T02:00:00Z")
        for leaf,field,digest in (("model_tensors.pt","model_tensor_file","model_tensor_sha256"),("model.json","model_manifest_file","model_manifest_sha256"),("training_timeout_transitions.jsonl","timeout_trace_relative_path","timeout_trace_sha256")):
            record[field]=f"training/droq__balanced__seed{seed}/"+leaf; record[digest]=s.sha(folder/leaf)
        put(folder/"record.json",record)
        if seed not in (2,3): put(folder/"started.json",dict(at="2026-10-30T01:01:00Z",pid=900 if seed==4 else 200+seed,protocol_sha256=s.PROTOCOL_SHA256,task=dict(seed=seed)))
        return record
    model(0); model(1)
    return SimpleNamespace(output=output,main=main,dataset=dataset,source=source,p=p,fit=fit,model=model,proc=proc)


def test_readiness_preserves0_1_and_accepts_only_original2_3_and_physical_cards(state):
    p,annex=s.readiness(state.output,state.main,state.dataset,state.source)
    assert p==state.p and len(annex["original_0_1_file_sha256"])==8 and annex["no_signals"] is True
    assert annex["resource_snapshot"]["rows"]==[dict(gpu_uuid="GPU-0",pid=202),dict(gpu_uuid="GPU-1",pid=203)]
    (state.output/"training/droq__balanced__seed4").mkdir()
    with pytest.raises(FileExistsError): s.readiness(state.output,state.main,state.dataset,state.source)


@pytest.mark.parametrize("damage",["count","device","date","model"])
def test_completed_record_requires_real_counts_dates_device_and_byte_identity(state,damage):
    folder=state.output/"training/droq__balanced__seed0"; record=s.read(folder/"record.json")
    if damage=="count": record["actual_counters"]={**record["actual_counters"],"q_inner_rounds":1}
    elif damage=="device": record["device"]["visible_gpu_uuid"]="OTHER"
    elif damage=="date": record["training_episode_history"][0]["end_exclusive"]="2026-01-01T00:00:00Z"
    else: (folder/"model_tensors.pt").write_bytes(b"changed")
    put(folder/"record.json",record)
    with pytest.raises(ValueError): s.completed_record(state.output,0,state.p)


def terminal_fixture(state):
    _,annex=s.readiness(state.output,state.main,state.dataset,state.source)
    # Synthetic fixed times keep the provenance assertions deterministic.
    annex["prepared_at"]="2026-10-30T01:00:30Z"; put(state.output/"scheduling_annex.json",annex)
    for seed in (2,3,4): state.model(seed)
    log=state.output/"fit.log"; log.write_text("FileExistsError: [Errno 17] File exists: '"+str(state.output/"training/droq__balanced__seed4")+"'\n")
    put(state.output/"fit_process.json",{**state.fit,"status":"failed","exit_code":1,"completed_at":"2026-10-30T02:00:30Z","log_sha256":s.sha(log)})
    put(state.output/"accelerated_fit_process.json",dict(child_pid=900,supervisor_pid=800,worker_exit_code=0,worker_completed_at="2026-10-30T02:00:20Z"))
    put(state.output/"accelerated_seed4_worker.json",dict(status="completed",pid=900,ppid=800,protocol_sha256=s.PROTOCOL_SHA256,annex_sha256=s.sha(state.output/"scheduling_annex.json"),
        record_sha256=s.sha(state.output/"training/droq__balanced__seed4/record.json"),completed_at="2026-10-30T02:00:10Z"))
    return annex


def test_finalize_adopts5_records_without_relabelling_original_collision(state):
    annex=terminal_fixture(state); before=(state.output/"fit_process.json").read_bytes(),(state.output/"fit.log").read_bytes()
    result=s.finalize(state.output,state.main,state.dataset,state.source,annex)
    assert result["model_count"]==5 and result["original_fit_terminal"]["collision_verified"] is True
    assert s.read(state.output/"fit_process.json")["status"]=="failed" and s.read(state.output/"training_completion.json")["model_count"]==5
    assert before==((state.output/"fit_process.json").read_bytes(),(state.output/"fit.log").read_bytes())


@pytest.mark.parametrize("damage",["other_error","worker_exit","changed0","restart2"])
def test_bad_handoff_never_generates_canonical_success(state,damage):
    annex=terminal_fixture(state)
    if damage=="other_error":
        log=state.output/"fit.log"; log.write_text(log.read_text()+"RuntimeError: unrelated failure\n")
        fit=s.read(state.output/"fit_process.json"); fit["log_sha256"]=s.sha(log); put(state.output/"fit_process.json",fit)
    elif damage=="worker_exit":
        path=state.output/"accelerated_fit_process.json"; value=s.read(path); value["worker_exit_code"]=7; put(path,value)
    elif damage=="changed0": (state.output/"training/droq__balanced__seed0/model_tensors.pt").write_bytes(b"changed original")
    else:
        path=state.output/"training/droq__balanced__seed2/started.json"; value=s.read(path); value["pid"]=999; put(path,value)
    with pytest.raises(ValueError): s.finalize(state.output,state.main,state.dataset,state.source,annex)
    assert not (state.output/"training_completion.json").exists() and not (state.output/"accelerated_fit_completion.json").exists()


def test_unknown_gpu_process_is_preserved_as_rejected_evidence(state,monkeypatch):
    old=s.subprocess.check_output
    monkeypatch.setattr(s.subprocess,"check_output",lambda argv,**kw: old(argv,**kw)+"GPU-0, 99999\n" if "--query-compute-apps=gpu_uuid,pid" in argv else old(argv,**kw))
    with pytest.raises(s.ResourceConflict): s.supervise(state.output,state.main,state.dataset,state.source)
    files=list((state.output/"preflight/scheduling").glob("resource_rejected_*.json"))
    assert len(files)==1 and "99999" in s.read(files[0])["snapshot"]["unfiltered_stdout"]
    assert not (state.output/"accelerated_fit.owner.json").exists()


def test_performance_install_is_worker_local_and_import_path_is_restored(tmp_path):
    directory=tmp_path/"perf"; directory.mkdir(); (directory/"synthetic_neighbor.py").write_text("VALUE=17\n")
    adapter=directory/"adapter.py"; adapter.write_text("from types import SimpleNamespace\ndef install(agent):\n import synthetic_neighbor\n return SimpleNamespace(provenance={'neighbor':synthetic_neighbor.VALUE})\n")
    output=tmp_path/"out"; output.mkdir(); s.write_new(output/"accelerated_seed4_worker.json",{})
    original=lambda: {"synthetic_agent":True}; core=SimpleNamespace(observe_agent=original)
    core.train_one=lambda task: dict(actual=core.observe_agent())
    before=list(sys.path); proof={}
    result=s.run_original(core,{"output":str(output)},dict(path=str(adapter)),proof)
    assert result==dict(actual={"synthetic_agent":True}) and proof["performance_install_count"]==1
    assert proof["performance_provenance"]=={"neighbor":17} and core.observe_agent is original and sys.path==before
    sys.modules.pop("synthetic_neighbor",None)
