"""Explicit seed4 scheduling annex; never rewrites the original fit terminal.

The current seed2/3 processes receive no signals or code changes. A separate
worker calls the frozen train_one for seed4 in the same output. Its atomic
mkdir wins that slot; the old queued invocation must fail honestly on exactly
that directory. This coordinator adopts only five complete verified records.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time
import types
import uuid

PROTOCOL_SHA256 = "e29c93d37571dd917f95b7f517a49ff8dc52cece712f14720ce1ab496ed226da"
RUNNER_SHA256 = "f1463c5e206aabd6a77e934f7972a55c84c15901b4a380e470c7dcbbcf08443a"
AUDITOR_SHA256 = "bf33c879ff5eeaa13ee288c561fb5e6f18efb3fddc0da5179e6d646339b8c2bc"
LEAVES = ("record.json", "model_tensors.pt", "model.json", "training_timeout_transitions.jsonl")
COUNTS = dict(environment_steps=32768, replay_adds=32768, random_exploration_actions=5001, learning_train_calls=27768,
    q_inner_rounds=555360, q_optimizer_steps=[555360,555360], total_q_optimizer_steps=1110720,
    target_updates_per_network=[555360,555360], policy_optimizer_steps=27768, alpha_optimizer_steps=27768)
PERFORMANCE_FILES = {
    "droq_execution_adapter.py":"21dcb3bf1afbd5b4285929589f1c2b0096f7ceb048017f944cefac347207f53c",
    "replay_cache_adapter.py":"4f2f2c9ef2a992e8937940adb6794d3c65de7db7a6fe580d1e03acdfa929441b",
    "polyak_cuda_graph.py":"d97700f67b49b4e626e03fc53a99764b6589590b3aa4bbfd7a7aae8210127093",
    "benchmark_gpu_replay_cache.py":"16e176531ac416e3dc25b60c752e1bef037a003a37edafcc51c5ba2887019009",
    "benchmark_gpu_replay_polyak_graph.py":"943235eb1d752a8f5a8b497220e0affd9aceb0b8b2648326521906871e093190",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def utc(value):
    value = datetime.fromisoformat(value.replace("Z","+00:00"))
    if value.tzinfo is None: raise ValueError("Explicit UTC evidence required")
    return value.astimezone(timezone.utc)


def regular(path):
    path = Path(path)
    for item in (path,*path.parents):
        if item.is_symlink() or hasattr(item,"is_junction") and item.is_junction():
            raise ValueError("Scheduling inputs cannot cross links/junctions")
    return path


def safe(root, name):
    text=str(name).replace("\\","/"); key=PurePosixPath(text)
    if not text or key.is_absolute() or ".." in key.parts or ":" in text: raise ValueError("Relative owned evidence path required")
    root=regular(root).resolve(); path=regular(root.joinpath(*key.parts)); path.resolve().relative_to(root)
    return path


def sha(path):
    with regular(path).open("rb") as stream: return hashlib.file_digest(stream,"sha256").hexdigest()


def read(path):
    return json.loads(regular(path).read_text(encoding="utf-8"))


def write_new(path,value):
    path=regular(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False); stream.write("\n")


def publish(path,value):
    path=regular(path); temporary=path.with_suffix(path.suffix+".pending")
    write_new(temporary,value); os.replace(temporary,path)


def hashes(root,names):
    return {name:sha(safe(root,name)) for name in sorted(names)}


def verify_hashes(root,values):
    if not isinstance(values,dict) or not values: raise ValueError("Immutable SHA map required")
    for name,expected in values.items():
        if sha(safe(root,name))!=expected: raise ValueError("Original evidence changed:"+name)


def core_guard(output,main,dataset,source):
    output,main,dataset,source=map(regular,(output,main,dataset,source))
    p=read(output/"protocol.json"); receipt=read(output/"protocol_receipt.json")
    code=p.get("source_sha256",{})
    if (sha(output/"protocol.json")!=PROTOCOL_SHA256 or receipt.get("protocol_sha256")!=PROTOCOL_SHA256
            or len(code)!=9 or code.get("scripts/btceth_rl_droq.py")!=RUNNER_SHA256
            or code.get("scripts/audit_btceth_rl_droq.py")!=AUDITOR_SHA256
            or p.get("algorithm")!="droq" or p.get("model_count")!=5 or p.get("timesteps_per_model")!=32768
            or p.get("expected_counters")!=COUNTS or p.get("seeds")!=list(range(5))
            or p.get("profiles")!={"balanced":{"drawdown_penalty":.25,"rebalance_band":.025}}
            or sha(dataset)!=p.get("dataset_sha256") or sha(dataset.parent/"manifest.json")!=p.get("dataset_manifest_sha256")
            or sha(main/"frozen_spec.json")!=p.get("original_frozen_spec_sha256")):
        raise ValueError("Original e29/f146/bf33 protocol, data and finite recipe must remain unchanged")
    verify_hashes(source,code); verify_hashes(output/"sources",code)
    return p


def completed_record(output,seed,p):
    folder=f"training/droq__balanced__seed{seed}"; record=read(safe(output,folder+"/record.json"))
    if (record.get("algorithm"),record.get("profile"),record.get("seed"),record.get("actual_timesteps"))!=("droq","balanced",seed,32768):
        raise ValueError("Exact completed seed identity required")
    if record.get("actual_counters")!=COUNTS or record.get("hyperparameters")!=p["hyperparameters"] or record.get("protocol_sha256")!=PROTOCOL_SHA256:
        raise ValueError("Actual optimizer/environment counts or recipe differ")
    device=record.get("device",{})
    if (device.get("device")!="cuda:0" or device.get("capability")!=[7,0] or "V100" not in device.get("name","")
            or device.get("visible_gpu_uuid")!=p["seed_gpu_assignments"][str(seed)]):
        raise ValueError("Original seed's physical V100 identity required")
    for leaf,field,digest in (("model_tensors.pt","model_tensor_file","model_tensor_sha256"),("model.json","model_manifest_file","model_manifest_sha256"),
            ("training_timeout_transitions.jsonl","timeout_trace_relative_path","timeout_trace_sha256")):
        name=folder+"/"+leaf
        if record.get(field)!=name or sha(safe(output,name))!=record.get(digest): raise ValueError("Completed model/manifest/timeout bytes differ")
    manifest=read(safe(output,folder+"/model.json"))
    if manifest.get("counters")!=COUNTS or manifest.get("tensor_sha256")!=record["model_tensor_sha256"] or manifest.get("parameter_sha256")!=record.get("parameter_sha256"):
        raise ValueError("Actual Tensor/JSON identities and counts differ")
    history=record.get("training_episode_history",[]); first,last=map(utc,p["train"])
    if len(history)!=1024: raise ValueError("All1024 real32-day training fragments required")
    for item in history:
        start,end,until=(utc(item[key]) for key in ("start","end_exclusive","completed_until"))
        if (not first<=start<end<=last or (end-start).total_seconds()!=32*86400 or until!=end or item.get("completed_steps")!=32
                or item.get("terminated") is not False or item.get("truncated") is not True):
            raise ValueError("Completed causal TRAIN fragment outside its declared boundary")
    if safe(output,folder+"/failure.json").exists(): raise ValueError("A failed seed cannot be adopted as successful")
    return record,hashes(output,{folder+"/"+leaf for leaf in LEAVES})


def proc_identity(pid):
    import pwd
    base=Path("/proc")/str(pid); fields=(base/"stat").read_text().rsplit(")",1)[1].split()
    if fields[0]=="Z": raise ValueError("Original process is a zombie")
    uid=int(next(line for line in (base/"status").read_text().splitlines() if line.startswith("Uid:")).split()[1])
    return dict(pid=int(pid),ppid=int(fields[1]),state=fields[0],exe=os.readlink(base/"exe"),starttime_ticks=fields[19],
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),uid=uid,username=pwd.getpwuid(uid).pw_name)


class ResourceConflict(RuntimeError):
    def __init__(self,message,snapshot):
        super().__init__(message); self.snapshot=snapshot


def resource_snapshot(p,running):
    query=["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"]
    physical_query=["nvidia-smi","--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu","--format=csv,noheader,nounits"]
    raw=subprocess.check_output(query,text=True); hardware=subprocess.check_output(physical_query,text=True)
    rows=[dict(gpu_uuid=parts[0].strip(),pid=int(parts[1])) for line in raw.splitlines() if line.strip() for parts in [line.split(",")]]
    devices=[]
    for line in hardware.splitlines():
        if not line.strip(): continue
        index,identifier,name,total,used,util=[part.strip() for part in line.split(",")]
        devices.append(dict(index=int(index),uuid=identifier,name=name,memory_total_mib=int(total),memory_used_mib=int(used),utilization_percent=int(util)))
    snapshot=dict(at=now(),command=query,unfiltered_stdout=raw,physical_command=physical_query,physical_stdout=hardware,physical_gpus=devices,rows=rows,instances={})
    keys=("index","uuid","name","memory_total_mib")
    if (len(devices)!=2 or [{k:d[k] for k in keys} for d in sorted(devices,key=lambda d:d["index"])]!=[{k:d[k] for k in keys} for d in p["physical_gpus"]]
            or any(not 0<=d["memory_used_mib"]<=d["memory_total_mib"] or not 0<=d["utilization_percent"]<=100 for d in devices)):
        raise ResourceConflict("Original physical V100 identity or valid observed usage differs",snapshot)
    expected={(p["seed_gpu_assignments"][seed],item["process"]["pid"]) for seed,item in running.items()}
    allowed=expected|{(item["gpu_uuid"],item["pid"]) for item in p.get("allowed_gpu_rows",[])}
    actual={(item["gpu_uuid"],item["pid"]) for item in rows}
    if len(actual)!=len(rows) or not expected.issubset(actual) or actual-allowed:
        raise ResourceConflict("Unknown/missing GPU process; retain evidence and never stop another task",snapshot)
    for _,pid in actual:
        found=proc_identity(pid); snapshot["instances"][str(pid)]=found
        declared=next((item["process"] for item in running.values() if item["process"]["pid"]==pid),p.get("allowed_compute_instance"))
        if declared is None or any(found.get(k)!=declared.get(k) for k in ("pid","exe","starttime_ticks","boot_id","uid","username")):
            raise ResourceConflict("GPU PID belongs to another process instance",snapshot)
    return snapshot


def original_identity(fit):
    return {key:fit[key] for key in ("phase","started_at","supervisor_pid","child_pid","command","source_sha256","protocol_sha256")}


def original_guard(output,*,running=False,collision=False):
    fit=read(Path(output)/"fit_process.json")
    if fit.get("phase")!="fit" or fit.get("source_sha256")!=RUNNER_SHA256 or fit.get("protocol_sha256")!=PROTOCOL_SHA256:
        raise ValueError("Original fit identity changed")
    if running and fit.get("status")!="running": raise ValueError("Original fit must still be running before scheduling")
    if collision:
        log=Path(output)/"fit.log"; text=regular(log).read_text(encoding="utf-8",errors="replace")
        expected=str(Path(output)/"training/droq__balanced__seed4").replace("\\","/")
        errors=[line for line in re.findall(r"(?m)^[A-Za-z_][\w.]*: [^\n]*",text) if not line.startswith("concurrent.futures.process._RemoteTraceback:")]
        if (fit.get("status")!="failed" or type(fit.get("exit_code")) is not int or fit["exit_code"]<=0
                or fit.get("log_sha256")!=sha(log) or not errors
                or any(not line.startswith("FileExistsError:") or expected not in line.replace("\\","/") for line in errors)):
            raise ValueError("Original terminal must honestly be only the planned seed4 directory collision")
    return fit


def performance_guard(path,digest):
    if path is None and digest is None: return None
    if path is None or digest!=PERFORMANCE_FILES["droq_execution_adapter.py"] or Path(path).name!="droq_execution_adapter.py":
        raise ValueError("Only the explicitly approved SHA-bound performance adapter is allowed")
    path=regular(path).resolve(); verify_hashes(path.parent,PERFORMANCE_FILES)
    return dict(path=str(path),sha256=digest,files={str(path.parent/name):value for name,value in PERFORMANCE_FILES.items()})


def readiness(output,main,dataset,source,performance=None):
    p=core_guard(output,main,dataset,source); fit=original_guard(output,running=True)
    if any((Path(output)/name).exists() for name in ("training_completion.json","fit_completion.json","validation","test","test_started.json")):
        raise ValueError("Only the still-incomplete original training phase may be accelerated")
    if (Path(output)/"training/droq__balanced__seed4").exists(): raise FileExistsError("Seed4 is already claimed; never start a duplicate")
    completed={}; initial={}
    for seed in (0,1):
        _,values=completed_record(output,seed,p); completed.update(values)
    parent=proc_identity(fit["child_pid"]); supervisor=proc_identity(fit["supervisor_pid"])
    if parent["ppid"]!=fit["supervisor_pid"]: raise ValueError("Original supervisor/child relationship differs")
    for seed in (2,3):
        folder=f"training/droq__balanced__seed{seed}"; started=read(safe(output,folder+"/started.json"))
        if safe(output,folder+"/record.json").exists() or safe(output,folder+"/failure.json").exists(): raise ValueError("Original seed2/3 must remain active, not restarted")
        identity=proc_identity(started["pid"])
        if identity["ppid"]!=fit["child_pid"] or started.get("device",{}).get("visible_gpu_uuid")!=p["seed_gpu_assignments"][str(seed)]:
            raise ValueError("Original seed2/3 process ownership/device changed")
        initial[str(seed)]=dict(started_sha256=sha(safe(output,folder+"/started.json")),process=identity)
    resources=resource_snapshot(p,initial)
    return p,dict(prepared_at=now(),protocol_sha256=PROTOCOL_SHA256,source_sha256=p["source_sha256"],scheduler_source_sha256=sha(Path(__file__)),
        original_fit_identity=original_identity(fit),original_fit_initial_sha256=sha(Path(output)/"fit_process.json"),original_0_1_file_sha256=completed,
        original_running_2_3=initial,original_supervisor=supervisor,original_child=parent,seed4_gpu_uuid=p["seed_gpu_assignments"]["4"],
        performance_adapter=performance,resource_snapshot=resources,mathematics_changed=False,no_signals=True,new_seed_count=0,only_queued_seed4_started_earlier=True,
        note="Original fit will retain a real collision failure; this annex never relabels that terminal as success")


def worker_gate(output,token):
    output=Path(output); proof=read(output/"accelerated_fit_process.json"); claim=read(output/"accelerated_fit.owner.json")
    for _ in range(30):
        if proof.get("status")!="launching": break
        time.sleep(.1); proof=read(output/"accelerated_fit_process.json")
    if (proof.get("status")!="running" or proof.get("child_pid")!=os.getpid() or proof.get("supervisor_pid")!=os.getppid()
            or claim.get("token")!=token or claim.get("supervisor_pid")!=os.getppid() or proof.get("source_sha256")!=sha(Path(__file__))
            or proof.get("protocol_sha256")!=PROTOCOL_SHA256 or proof.get("annex_sha256")!=sha(output/"scheduling_annex.json")):
        raise ValueError("Only the actual unique source-bound accelerated worker may run")


def run_original(core,task,performance,worker_proof):
    original=core.observe_agent; installed=[]; old_path=list(sys.path)
    try:
        if performance is not None:
            sys.path.insert(0,str(Path(performance["path"]).parent))
            module=types.ModuleType("_droq_seed4_performance"); module.__file__=performance["path"]
            exec(compile(Path(module.__file__).read_bytes(),module.__file__,"exec"),module.__dict__)
            def observe(*args,**kwargs):
                agent=original(*args,**kwargs); result=module.install(agent); installed.append(result)
                worker_proof.update(performance_install_count=len(installed),performance_provenance=result.provenance)
                publish(Path(task["output"])/"accelerated_seed4_worker.json",worker_proof)
                return agent
            core.observe_agent=observe
        result=core.train_one(task)
        if performance is not None and len(installed)!=1: raise ValueError("Exactly one explicit performance installation required")
        return result
    finally:
        core.observe_agent=original
        sys.path[:]=old_path


def worker(output,main,dataset,source,token):
    output=Path(output); worker_gate(output,token); p=core_guard(output,main,dataset,source); annex=read(output/"scheduling_annex.json")
    performance=annex.get("performance_adapter")
    if performance: performance=performance_guard(performance["path"],performance["sha256"])
    os.environ.update(CUDA_VISIBLE_DEVICES=p["seed_gpu_assignments"]["4"],CUBLAS_WORKSPACE_CONFIG=":4096:8",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    sys.path.insert(0,str(regular(source).resolve()))
    core=importlib.import_module("scripts.btceth_rl_droq")
    if Path(core.__file__).resolve()!=(Path(source)/"scripts/btceth_rl_droq.py").resolve() or sha(core.__file__)!=RUNNER_SHA256:
        raise ValueError("Actually imported original producer differs")
    proof=dict(status="running",pid=os.getpid(),ppid=os.getppid(),started_at=now(),seed=4,protocol_sha256=PROTOCOL_SHA256,
        annex_sha256=sha(output/"scheduling_annex.json"),performance_adapter=performance,performance_install_count=0)
    write_new(output/"accelerated_seed4_worker.json",proof)
    try:
        result=run_original(core,dict(main=str(main),dataset=str(dataset),output=str(output),profile="balanced",seed=4),performance,proof)
        saved,_=completed_record(output,4,p)
        if result!=saved: raise ValueError("Worker returned something other than the actual saved record")
        proof.update(status="completed",completed_at=now(),record_sha256=sha(output/"training/droq__balanced__seed4/record.json"),parameter_sha256=saved["parameter_sha256"])
        publish(output/"accelerated_seed4_worker.json",proof)
    except BaseException as error:
        proof.update(status="failed",completed_at=now(),error_type=type(error).__name__,error=str(error)); publish(output/"accelerated_seed4_worker.json",proof); raise


def finalize(output,main,dataset,source,annex):
    output=Path(output); p=core_guard(output,main,dataset,source); original=original_guard(output,collision=True)
    if original_identity(original)!=annex["original_fit_identity"]: raise ValueError("Original fit identity changed during handoff")
    verify_hashes(output,annex["original_0_1_file_sha256"])
    records=[]; values={}
    for seed in range(5):
        record,files=completed_record(output,seed,p); records.append(record); values.update(files)
        started=read(output/f"training/droq__balanced__seed{seed}/started.json")
        if (started.get("protocol_sha256")!=PROTOCOL_SHA256 or started.get("task",{}).get("seed")!=seed
                or not utc(original["started_at"])<=utc(started["at"])<=utc(record["completed_at"])):
            raise ValueError("Actual training start/completion ancestry differs")
        values[f"training/droq__balanced__seed{seed}/started.json"]=sha(output/f"training/droq__balanced__seed{seed}/started.json")
        if seed in (2,3) and values[f"training/droq__balanced__seed{seed}/started.json"]!=annex["original_running_2_3"][str(seed)]["started_sha256"]:
            raise ValueError("Original seed2/3 was restarted or changed")
    worker_proof=read(output/"accelerated_seed4_worker.json")
    process=read(output/"accelerated_fit_process.json"); started4=read(output/"training/droq__balanced__seed4/started.json")
    if (worker_proof.get("status")!="completed" or worker_proof.get("record_sha256")!=sha(output/"training/droq__balanced__seed4/record.json")
            or process.get("worker_exit_code")!=0 or type(process.get("worker_exit_code")) is not int
            or worker_proof.get("pid")!=process.get("child_pid") or started4.get("pid")!=process.get("child_pid")
            or worker_proof.get("ppid")!=process.get("supervisor_pid") or worker_proof.get("protocol_sha256")!=PROTOCOL_SHA256
            or worker_proof.get("annex_sha256")!=sha(output/"scheduling_annex.json")
            or not utc(annex["prepared_at"])<=utc(started4["at"])<=utc(worker_proof["completed_at"])<=utc(process["worker_completed_at"])):
        raise ValueError("Actual independent seed4 worker completion required")
    if annex.get("performance_adapter") is not None and (worker_proof.get("performance_adapter")!=annex["performance_adapter"] or worker_proof.get("performance_install_count")!=1):
        raise ValueError("Exactly the approved worker-local performance adaptation required")
    canonical=dict(at=now(),model_count=5,records=records,protocol_sha256=PROTOCOL_SHA256)
    destination=output/"training_completion.json"
    if destination.exists():
        old=read(destination)
        if old.get("model_count")!=5 or old.get("records")!=records or old.get("protocol_sha256")!=PROTOCOL_SHA256: raise ValueError("Existing canonical completion cannot be overwritten")
    else:
        write_new(destination,canonical)
    evidence=dict(at=now(),protocol_sha256=PROTOCOL_SHA256,annex_sha256=sha(output/"scheduling_annex.json"),model_count=5,records=records,
        training_completion_sha256=sha(destination),record_file_sha256=values,original_0_1_hashes_preserved=True,no_signals=True,
        original_fit_terminal=dict(file="fit_process.json",sha256=sha(output/"fit_process.json"),status=original["status"],exit_code=original["exit_code"],
            log_sha256=sha(output/"fit.log"),collision_verified=True,collision_path=str(output/"training/droq__balanced__seed4")),
        cohort_started_at=original["started_at"],cohort_completed_at=now(),worker_proof_sha256=sha(output/"accelerated_seed4_worker.json"),
        source_sha256=p["source_sha256"],scheduler_source_sha256=sha(Path(__file__)))
    write_new(output/"accelerated_fit_completion.json",evidence)
    return evidence


def supervise(output,main,dataset,source,*,performance_adapter=None,performance_sha=None,poll_seconds=30):
    if not 1<=poll_seconds<=60: raise ValueError("Bounded polling interval required")
    output=regular(output).resolve(); performance=performance_guard(performance_adapter,performance_sha)
    try:
        p,annex=readiness(output,main,dataset,source,performance)
    except ResourceConflict as error:
        write_new(output/"preflight/scheduling"/("resource_rejected_"+uuid.uuid4().hex+".json"),dict(at=now(),source_sha256=sha(Path(__file__)),error=str(error),snapshot=error.snapshot))
        raise
    for name in ("scheduling_annex.json","accelerated_fit_process.json","accelerated_fit_completion.json","accelerated_seed4_worker.json","accelerated_seed4.log"):
        if (output/name).exists(): raise FileExistsError("Preserve every previous scheduling attempt:"+name)
    token=uuid.uuid4().hex
    write_new(output/"accelerated_fit.owner.json",dict(token=token,supervisor_pid=os.getpid(),at=now(),source_sha256=sha(Path(__file__))))
    snapshot="preflight/scheduling/original_fit_process.initial.json"
    content=(output/"fit_process.json").read_bytes()
    if hashlib.sha256(content).hexdigest()!=annex["original_fit_initial_sha256"]: raise ValueError("Original fit changed during scheduling readiness")
    target=safe(output,snapshot); target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("xb") as stream: stream.write(content)
    annex["original_fit_initial_snapshot_file"]=snapshot
    write_new(output/"scheduling_annex.json",annex)
    command=[sys.executable,"-u",str(Path(__file__).resolve()),"--worker","--output-dir",str(output),"--reference-main",str(Path(main).resolve()),
        "--dataset",str(Path(dataset).resolve()),"--source-root",str(Path(source).resolve()),"--owner-token",token]
    proof=dict(phase="accelerated_fit",status="launching",started_at=now(),supervisor_pid=os.getpid(),command=command,
        source_sha256=sha(Path(__file__)),protocol_sha256=PROTOCOL_SHA256,annex_sha256=sha(output/"scheduling_annex.json"),log_relative_path="accelerated_seed4.log")
    path=output/"accelerated_fit_process.json"; log=output/"accelerated_seed4.log"; write_new(path,proof)
    try:
        with log.open("x",encoding="utf-8") as stream:
            child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT); proof.update(status="running",child_pid=child.pid); publish(path,proof)
            while child.poll() is None:
                print(json.dumps(dict(at=now(),status="seed4_running",child_pid=child.pid,original_fit_status=read(output/"fit_process.json").get("status"))),flush=True)
                time.sleep(poll_seconds)
            code=child.wait()
        proof.update(worker_exit_code=code,worker_completed_at=now(),log_sha256=sha(log)); publish(path,proof)
        if code: raise RuntimeError("Independent seed4 worker failed; preserve its partial evidence")
        while read(output/"fit_process.json").get("status") in ("launching","running"):
            print(json.dumps(dict(at=now(),status="waiting_original_2_3_and_collision_terminal",no_signals=True)),flush=True); time.sleep(poll_seconds)
        completed=finalize(output,main,dataset,source,annex)
        proof.update(status="completed",exit_code=0,completed_at=now(),completion_sha256=sha(output/"accelerated_fit_completion.json")); publish(path,proof)
        return completed
    except BaseException as error:
        proof.update(status="failed",exit_code=1,completed_at=now(),error_type=type(error).__name__,error=str(error),log_sha256=sha(log) if log.exists() else None)
        publish(path,proof); raise


def main():
    parser=argparse.ArgumentParser(description=__doc__); mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--supervise",action="store_true"); mode.add_argument("--worker",action="store_true")
    for name in ("output-dir","reference-main","dataset","source-root"): parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--owner-token"); parser.add_argument("--performance-adapter",type=Path); parser.add_argument("--performance-sha")
    parser.add_argument("--poll-seconds",type=int,default=30); args=parser.parse_args()
    if args.worker:
        worker(args.output_dir,args.reference_main,args.dataset,args.source_root,args.owner_token)
    else:
        result=supervise(args.output_dir,args.reference_main,args.dataset,args.source_root,performance_adapter=args.performance_adapter,
            performance_sha=args.performance_sha,poll_seconds=args.poll_seconds); print(json.dumps({key:result[key] for key in ("model_count","cohort_completed_at","original_0_1_hashes_preserved")}))


if __name__=="__main__": main()
