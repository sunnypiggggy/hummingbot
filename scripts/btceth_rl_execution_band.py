"""One post-annual, fixed execution-band comparison; validation only.

TQC balanced, its five existing frozen models and the .75 target mapping stay
unchanged. Only the environment constructor's band changes from .025 to .05.
This is designed after the first annual result, never a fresh holdout.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts import backtest_btceth_rl as formal
from scripts import btceth_rl_exposure_cap as cap
from scripts import btceth_rl_goal_candidate as goal
from scripts.crypto_rl_reporting import canonical_csv, result_fingerprints, save_result, summarize

BAND, REFERENCE_BAND, MULTIPLIER = .05, .025, .75
FAMILY, PROFILE = "cap075_band05", "balanced"
CASE_FILES, GROUP_FILES = cap.CASE_FILES, cap.GROUP_FILES
SOURCE_FILES = (*formal.SOURCE_FILES, "scripts/btceth_rl_goal_candidate.py", "scripts/btceth_rl_exposure_cap.py",
    "scripts/audit_btceth_rl.py", "scripts/audit_btceth_rl_exposure_cap.py",
    "scripts/btceth_rl_execution_band.py", "scripts/audit_btceth_rl_execution_band.py")
now, sha, read_json, relative = formal.now, formal.sha, formal.read_json, goal.relative_file


def write_new(path, value):
    goal.exclusive_json(Path(path), value)


def pin(root, names):
    return {name:sha(relative(root, name)) for name in sorted(names)}


def verify_pins(root, values):
    if not isinstance(values, dict) or not values:
        raise ValueError("Complete frozen evidence pins required")
    for name, expected in values.items():
        if sha(relative(root, name)) != expected:
            raise ValueError("Frozen source/evidence changed: " + name)


def separate(main, reference, output):
    main, reference, output = (Path(p).resolve() for p in (main, reference, output))
    for a, b in ((main,reference),(main,output),(reference,output)):
        if a == b or a.is_relative_to(b) or b.is_relative_to(a):
            raise ValueError("Original, reference and new experiment require separate trees")
    return main, reference, output


def reference_files():
    folder = "validation/balanced"
    return {*(f"{folder}/{f}" for f in GROUP_FILES),
        *(f"{folder}/{c}/{k}/{f}" for c in formal.COSTS for k in (*map(str,formal.SEEDS),"ensemble") for f in CASE_FILES),
        "protocol.json","protocol_receipt.json","selection.json","selection_receipt.json",
        "validation_completion.json","independent_audit.json","goal_candidate_evidence.json"}


def models_manifest(main, reference):
    existing = read_json(reference / "protocol.json")
    timesteps = read_json(main / "frozen_spec.json")["timesteps_per_model"]
    models = [r for r in existing["models"] if r["profile"] == PROFILE]
    if len(models) != 5 or [r["seed"] for r in models] != list(formal.SEEDS):
        raise ValueError("Use exactly the same five balanced TQC checkpoints")
    for model in models:
        record = read_json(relative(main, model["record_file"]))
        if ((record.get("algorithm"),record.get("profile"),record.get("seed")) != ("tqc",PROFILE,model["seed"])
                or sha(relative(main,model["record_file"])) != model["record_sha256"]
                or sha(relative(main,model["model_file"])) != model["model_sha256"]
                or record.get("model_sha256") != model["model_sha256"]
                or record.get("parameter_sha256") != model["parameter_sha256"]
                or record.get("actual_timesteps") != timesteps):
            raise ValueError("Original balanced checkpoint/parameter identity changed")
    return models


def prepare(main, reference, dataset, output):
    main, reference, output = separate(main,reference,output)
    dataset = Path(dataset).resolve()
    if output.exists():
        raise FileExistsError("Preserve any existing execution-band experiment")
    spec = formal.check_frozen(main,dataset)
    cap.verify_protocol(main,dataset,reference)
    old = read_json(reference/"independent_audit.json")
    evidence = read_json(reference/"goal_candidate_evidence.json")
    if (old.get("passed") is not True or old.get("status") != "passed" or old.get("run_count") != 36
            or old.get("test_complete") is not True or old.get("failure_count") != 0
            or evidence.get("audit_sha256") != sha(reference/"independent_audit.json")
            or evidence.get("candidate") != {"family":"cap075","algorithm":"tqc","profile":PROFILE}):
        raise ValueError("Require the completed original annual experiment and preserved evidence")
    models = models_manifest(main,reference)
    names = {"frozen_spec.json","training_completion.json","runtime.json","fit_process.json","fit.log",
             *(r[k] for r in models for k in ("record_file","model_file"))}
    sources = pin(ROOT,SOURCE_FILES)
    runtime = read_json(main/"runtime.json")
    gpus = runtime["physical_gpus"]
    if len(gpus)!=2 or len({g["uuid"] for g in gpus})!=2 or any("V100" not in g["name"] for g in gpus):
        raise ValueError("Freeze both original distinct V100 identities")
    protocol = dict(schema_version=1,prepared_at=now(),algorithm="tqc",profile=PROFILE,
        source_sha256=sources,original_inputs={"main":pin(main,names),"reference":pin(reference,reference_files())},
        original_frozen_spec_sha256=sha(main/"frozen_spec.json"),models=models,
        dataset_sha256=spec["dataset_sha256"],dataset_manifest_sha256=spec["dataset_manifest_sha256"],
        validation_start=formal.VALIDATION[0],validation_end_exclusive=formal.VALIDATION[1],end_exclusive=formal.VALIDATION[1],
        new_rebalance_band=BAND,reference_rebalance_band=REFERENCE_BAND,exposure_multiplier=MULTIPLIER,
        capital=10000.,costs=formal.COSTS,seeds=list(formal.SEEDS),cash_floor=.05,max_asset_weight=.95,
        drawdown_penalty=.25,new_training=0,seed_selection=False,selection_uses_test=False,
        annual_evaluation_enabled=False,designed_after_first_annual=True,test_year_already_seen=True,not_fresh_holdout=True,
        physical_gpus=gpus,runtime_packages=spec["runtime_packages"],
        gpu_cost_assignments={cost:g["uuid"] for cost,g in zip(formal.COSTS,gpus)},
        validation_cases=12,repeat_count=12,comparison_population=["reference_band025","band05"],
        selection_score=spec["validation_selection"]["score"],tie_break="prefer_reference",
        mapping="Original action mapping; BTC and ETH targets times .75; USDT residual; five targets mean in one actual funded account",
        execution="Daily prior completed data; 00UTC sell, 01UTC buy; original cash exit and hourly close accounting",
        feature_input_cutoff="decision day minus one hour, no new annual input or future features",
        motivation="Fixed single 5pp band after first annual exposed fees and 2/5 positive seeds; no band search; old failed goal preserved")
    output.mkdir(parents=True)
    for name in sources:
        target=output/"sources"/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    write_new(output/"protocol.json",protocol)
    write_new(output/"protocol_receipt.json",dict(protocol_file="protocol.json",protocol_sha256=sha(output/"protocol.json"),sealed_at=now()))
    return protocol


def verify_protocol(main,reference,dataset,output):
    main,reference,output=separate(main,reference,output)
    if any((output/name).exists() for name in ("test","test_started.json","test_completion.json","test_process.json")):
        raise ValueError("This frozen round has no annual phase or authorization")
    p,r=read_json(output/"protocol.json"),read_json(output/"protocol_receipt.json")
    spec=formal.check_frozen(main,dataset)
    expected=dict(algorithm="tqc",profile=PROFILE,new_rebalance_band=BAND,reference_rebalance_band=REFERENCE_BAND,
        exposure_multiplier=MULTIPLIER,capital=10000.,costs=formal.COSTS,seeds=list(formal.SEEDS),cash_floor=.05,
        max_asset_weight=.95,drawdown_penalty=.25,new_training=0,seed_selection=False,selection_uses_test=False,
        annual_evaluation_enabled=False,designed_after_first_annual=True,test_year_already_seen=True,not_fresh_holdout=True,
        validation_start=formal.VALIDATION[0],validation_end_exclusive=formal.VALIDATION[1],end_exclusive=formal.VALIDATION[1],validation_cases=12,repeat_count=12,
        comparison_population=["reference_band025","band05"],tie_break="prefer_reference")
    if any(p.get(k)!=v for k,v in expected.items()) or set(p.get("source_sha256",{}))!=set(SOURCE_FILES):
        raise ValueError("Only the declared single post-annual 5pp comparison is allowed")
    if (r.get("protocol_file")!="protocol.json" or r.get("protocol_sha256")!=sha(output/"protocol.json")
            or goal.stamp(r["sealed_at"])<goal.stamp(p["prepared_at"])
            or p["dataset_sha256"]!=spec["dataset_sha256"] or p["dataset_manifest_sha256"]!=spec["dataset_manifest_sha256"]
            or p["selection_score"]!=spec["validation_selection"]["score"]
            or p["original_frozen_spec_sha256"]!=sha(main/"frozen_spec.json")
            or p["models"]!=models_manifest(main,reference)):
        raise ValueError("Protocol receipt, model population, dataset or source changed")
    verify_pins(ROOT,p["source_sha256"]);verify_pins(output/"sources",p["source_sha256"])
    verify_pins(main,p["original_inputs"]["main"]);verify_pins(reference,p["original_inputs"]["reference"])
    return p


def evaluate_band(data,models,seeds,costs,*,start=None,end=None):
    from scripts.btceth_rl_env import BTCETHPortfolioEnv
    if (len(models) not in (1,5) or len(models)!=len(seeds) or len(set(seeds))!=len(seeds)
            or not set(seeds).issubset(formal.SEEDS) or (len(models)==5 and set(seeds)!=set(formal.SEEDS))):
        raise ValueError("One fixed seed or all five original models required")
    start,end=(start or formal.VALIDATION[0]),(end or formal.VALIDATION[1])
    before=[formal.parameter_digest(m.policy) for m in models]
    # Explicit constructor argument; the frozen PROFILES dict is never changed.
    env=BTCETHPortfolioEnv(data,start,end,costs=costs,capital=10000.,record_history=True,
        rebalance_band=BAND,drawdown_penalty=formal.PROFILES[PROFILE]["drawdown_penalty"])
    obs,_=env.reset(seed=0);trace=[];done=False
    while not done:
        day=env.current_date;observation=np.asarray(obs,dtype="<f4")
        members=[];targets=[]
        for seed,model in zip(seeds,models):
            action,_=model.predict(obs,deterministic=True);raw=env.action_to_weights(action);target=cap.cap_target(raw)
            targets.append(target)
            members.append(dict(day=day.isoformat(),seed=seed,raw_action0=float(action[0]),raw_action1=float(action[1]),
                raw_btc=float(raw[0]),raw_eth=float(raw[1]),raw_cash=float(raw[2]),
                capped_btc=float(target[0]),capped_eth=float(target[1]),capped_cash=float(target[2]),
                observation_sha256=hashlib.sha256(observation.tobytes()).hexdigest(),
                observation_asof=(day-pd.Timedelta(hours=1)).isoformat(),feature_input_cutoff=(day-pd.Timedelta(hours=1)).isoformat(),
                observation_values_json=json.dumps(observation.astype(float).tolist(),separators=(",",":")),
                observation_btc_weight=float(obs[-4]),observation_eth_weight=float(obs[-3]),
                observation_cash_weight=float(obs[-2]),observation_drawdown=float(obs[-1])))
        mean=np.mean(targets,axis=0)
        for member in members:
            member.update(target_btc=float(mean[0]),target_eth=float(mean[1]),target_cash=float(mean[2]));trace.append(member)
        obs,_,terminated,truncated,_=env.step_target(mean);done=bool(terminated or truncated)
    result=env.get_result();env.close();after=[formal.parameter_digest(m.policy) for m in models]
    if before!=after:raise AssertionError("Inference changed original parameters")
    result["metadata"].update(algorithm="tqc",profile=PROFILE,family=FAMILY,exposure_multiplier=MULTIPLIER,
        model_count=len(models),parameter_sha256_before=before,parameter_sha256_after=after,
        old_actions_or_equity_scaled=False,band_change_only=True,designed_after_first_annual=True,
        test_year_already_seen=True,not_fresh_holdout=True)
    return result,pd.DataFrame(trace)


def evaluate_scenario(task):
    p=verify_protocol(task["main"],task["reference"],task["dataset"],task["output"])
    device=formal.torch_setup(123456)
    if (device["visible_gpu_uuid"]!=task["uuid"] or device["capability"]!=[7,0]
            or "V100" not in device["name"] or formal.runtime_packages()!=p["runtime_packages"]):
        raise ValueError("Original package versions and UUID-isolated V100 required")
    data=formal.load_dataset(task["dataset"])
    for name in list(data):
        if name.startswith("hourly_"):data[name]=data[name].loc[data[name].index<pd.Timestamp(formal.VALIDATION[1])].copy()
    models=[]
    for record in p["models"]:
        model=formal.model_class("tqc").load(relative(task["main"],record["model_file"]),device="cuda:0")
        if formal.parameter_digest(model.policy)!=record["parameter_sha256"]:raise ValueError("Loaded model digest changed")
        models.append(model)
    scenario=task["scenario"];costs=formal.COSTS[scenario];output=Path(task["output"])
    directory=output/"validation"/PROFILE/scenario
    directory.mkdir(parents=True,exist_ok=False)
    rows=[];history=[];monthly=[];repeats=[]
    plans=[(str(seed),[model],[seed],seed) for seed,model in zip(formal.SEEDS,models)]+[("ensemble",models,list(formal.SEEDS),None)]
    for key,members,seeds,seed in plans:
        run_id=f"validation__{scenario}__{FAMILY}__tqc__{PROFILE}__{key}"
        result,trace=evaluate_band(data,members,seeds,costs);repeat,repeated_trace=evaluate_band(data,members,seeds,costs)
        fingerprints,again=result_fingerprints(result),result_fingerprints(repeat)
        trace_bytes,again_trace=canonical_csv(trace),canonical_csv(repeated_trace)
        if fingerprints!=again or trace_bytes!=again_trace:raise AssertionError("Exact repeated account/trace differs")
        case=directory/key;save_result(case,result);(case/"member_targets.csv").write_bytes(trace_bytes)
        proof=dict(run_id=run_id,passed=True,repeat_exact=True,fingerprints=fingerprints,repeated_fingerprints=again,
            member_targets_sha256=hashlib.sha256(trace_bytes).hexdigest(),repeated_member_targets_sha256=hashlib.sha256(again_trace).hexdigest(),
            parameter_sha256_before=result["metadata"]["parameter_sha256_before"],parameter_sha256_after=result["metadata"]["parameter_sha256_after"],
            model_records=[r for r in p["models"] if r["seed"] in seeds],device=device,worker_pid=os.getpid())
        write_new(case/"replay.json",proof)
        row,daily,months=summarize(result,run_id=run_id,strategy="tqc_cap075_band05",seed=seed,scenario=scenario,
            split="validation",start=formal.VALIDATION[0],end=formal.VALIDATION[1],capital=10000.,**costs)
        row.update(algorithm="tqc",profile=PROFILE,family=FAMILY,primary_policy=seed is None,exposure_multiplier=MULTIPLIER,rebalance_band=BAND)
        daily["profile"],months["profile"]=PROFILE,PROFILE
        rows.append(row);history.append(daily);monthly.append(months);repeats.append(proof)
        print(f"BAND05 {run_id} return={row['net_return']:.6%} DD={row['max_drawdown']:.6%}",flush=True)
    for name,frame in (("summary.csv",pd.DataFrame(rows)),("daily_equity.csv",pd.concat(history,ignore_index=True)),("monthly_returns.csv",pd.concat(monthly,ignore_index=True))):
        frame.to_csv(directory/name,index=False,float_format="%.17g")
    write_new(directory/"completion.json",dict(at=now(),protocol_sha256=sha(output/"protocol.json"),case_count=6,repeats=repeats,worker_pid=os.getpid(),device=device))
    return dict(scenario=scenario,case_count=6,worker_pid=os.getpid())


def run_validation(main,reference,dataset,output):
    p=verify_protocol(main,reference,dataset,output);output=Path(output)
    if (output/"validation").exists() or (output/"validation_completion.json").exists():raise FileExistsError("Preserve prior validation")
    if sys.platform!="linux":raise RuntimeError("Run the original two UUID-isolated V100s on Linux")
    cap.terminal_process(main,"fit");cap.terminal_process(main,"test")
    status=subprocess.run(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"],capture_output=True,text=True,check=True)
    if any(g["uuid"] in status.stdout for g in p["physical_gpus"]):raise RuntimeError("Original GPUs are busy; do not stop another task")
    ctx=mp.get_context("spawn");pools=[];futures=[]
    try:
        for scenario,uuid in p["gpu_cost_assignments"].items():
            pool=ProcessPoolExecutor(max_workers=1,mp_context=ctx,initializer=formal.gpu_initializer,initargs=(uuid,));pools.append(pool)
            futures.append(pool.submit(evaluate_scenario,dict(main=str(main),reference=str(reference),dataset=str(dataset),output=str(output),scenario=scenario,uuid=uuid)))
        workers=[future.result() for future in futures]
    finally:
        for pool in pools:pool.shutdown(wait=True)
    directory=output/"validation"/PROFILE
    for name in ("summary.csv","daily_equity.csv","monthly_returns.csv"):
        frame=pd.concat([pd.read_csv(directory/c/name,float_precision="round_trip") for c in formal.COSTS],ignore_index=True)
        frame.to_csv(directory/name,index=False,float_format="%.17g");frame.to_csv(output/name,index=False,float_format="%.17g")
    repeats=[item for c in formal.COSTS for item in read_json(directory/c/"completion.json")["repeats"]]
    write_new(directory/"completion.json",dict(at=now(),protocol_sha256=sha(output/"protocol.json"),case_count=12,repeats=repeats))
    verify_protocol(main,reference,dataset,output)
    write_new(output/"validation_completion.json",dict(at=now(),protocol_sha256=sha(output/"protocol.json"),case_count=12,repeat_count=12,split="validation",workers=workers))
    return dict(case_count=12,repeat_count=12,workers=workers)


def supervise(main,reference,dataset,output):
    verify_protocol(main,reference,dataset,output);output=Path(output)
    proof,log=output/"runner_process.json",output/"validation.log"
    if proof.exists() or log.exists():raise FileExistsError("Preserve existing process attempt")
    command=[sys.executable,"-u",str(Path(__file__).resolve()),"--phase","_validate","--main-output",str(main),"--reference-output",str(reference),"--dataset",str(dataset),"--output-dir",str(output)]
    record=dict(phase="validate",started_at=now(),supervisor_pid=os.getpid(),command=command,status="launching",
        protocol_sha256=sha(output/"protocol.json"),source_sha256=sha(Path(__file__)),log_relative_path=log.name)
    def publish():
        temporary=proof.with_suffix(".tmp");temporary.write_text(json.dumps(record,indent=2)+"\n");temporary.replace(proof)
    publish()
    with log.open("x",encoding="utf-8") as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        record.update(status="running",child_pid=child.pid);publish();code=child.wait()
    record.update(status="completed" if code==0 else "failed",exit_code=code,completed_at=now(),log_sha256=sha(log));publish()
    if code:raise RuntimeError("Validation failed; preserve process/log/partial outputs")
    return record


def candidate(frame,name,band):
    values={}
    for scenario in formal.COSTS:
        row=frame.loc[frame.scenario.eq(scenario)&frame.primary_policy]
        if len(row)!=1:raise ValueError("Exactly one actual five-model ensemble per cost required")
        r=row.iloc[0];values[scenario]={"run_id":str(r.run_id),**{k:float(r[k]) for k in ("net_return","max_drawdown","final_equity","fees")}}
    def score(r):
        drawdown=abs(r["max_drawdown"])
        return math.log1p(r["net_return"])-.5*drawdown-2*max(drawdown-.20,0.)
    return dict(name=name,family="cap075" if band==REFERENCE_BAND else FAMILY,algorithm="tqc",profile=PROFILE,
        rebalance_band=band,score=(score(values["base"])+score(values["stress"]))/2,**values)


def select(main,reference,dataset,output):
    p=verify_protocol(main,reference,dataset,output);output=Path(output)
    for name in ("comparison.json","selection.json","selection_receipt.json"):
        if (output/name).exists():raise FileExistsError("Preserve the sealed two-candidate validation decision")
    audit=read_json(output/"independent_audit.validation.json")
    if (audit.get("passed") is not True or audit.get("status")!="passed" or audit.get("run_count")!=12
            or audit.get("comparator_run_count")!=12 or audit.get("repeat_count")!=12 or audit.get("test_complete") is not False
            or audit.get("failures")!=[]):raise ValueError("Complete independent twelve-new/twelve-reference validation audit required")
    roots={"extension":output,"main":Path(main),"reference":Path(reference),"source":ROOT}
    for kind,mapping in audit["input_sha256"].items():verify_pins(roots[kind],mapping)
    old=pd.read_csv(Path(reference)/"validation/balanced/summary.csv",float_precision="round_trip")
    new=pd.read_csv(output/"summary.csv",float_precision="round_trip")
    pool=[candidate(old,"reference_band025",REFERENCE_BAND),candidate(new,"band05",BAND)]
    pool.sort(key=lambda r:(-r["score"],0 if r["name"]=="reference_band025" else 1))
    result=dict(selected_at=now(),protocol_sha256=sha(output/"protocol.json"),audit_sha256=sha(output/"independent_audit.validation.json"),
        candidates=pool,candidate=pool[0],selection_uses_test=False,seed_selection=False,annual_evaluation_enabled=False,
        designed_after_first_annual=True,test_year_already_seen=True,not_fresh_holdout=True,
        validation_input_sha256=audit["input_sha256"],old_results_and_failed_goal_preserved=True)
    write_new(output/"comparison.json",dict(at=now(),candidates=pool,chosen=pool[0]["name"],tie_break=p["tie_break"]))
    write_new(output/"selection.json",result)
    write_new(output/"selection_receipt.json",dict(sealed_at=now(),selection_file="selection.json",selection_sha256=sha(output/"selection.json"),protocol_sha256=sha(output/"protocol.json")))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",required=True,choices=("prepare","validate","_validate","select"))
    for name in ("main-output","reference-output","dataset","output-dir"):parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();main,reference,dataset,output=(getattr(args,n).resolve() for n in ("main_output","reference_output","dataset","output_dir"))
    if args.phase=="prepare":result=prepare(main,reference,dataset,output)
    elif args.phase=="validate":result=supervise(main,reference,dataset,output)
    elif args.phase=="select":result=select(main,reference,dataset,output)
    else:
        for _ in range(20):
            record=read_json(output/"runner_process.json")
            if record.get("child_pid")==os.getpid():break
            time.sleep(.05)
        if record.get("status")!="running" or record.get("child_pid")!=os.getpid() or record.get("supervisor_pid")!=os.getppid():raise ValueError("Private worker must belong to the unique current supervisor")
        result=run_validation(main,reference,dataset,output)
    print(json.dumps(dict(phase=args.phase,result=result),ensure_ascii=False),flush=True)


if __name__=="__main__":main()
