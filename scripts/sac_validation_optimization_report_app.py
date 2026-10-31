"""Read-only, audit-gated comparison of original and validation-selected SAC.

Native model identity remains SAC. Separate presentation identities prevent
before/after account collisions without changing any saved source run or ledger.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import importlib
import json
import math
from pathlib import Path,PurePosixPath
import shutil
import sys
import tarfile

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import btc_rl_seed_stability_report_app as old_btc
from scripts import eth_rl_seed_stability_report_app as old_eth
from scripts import backtest_single_pair_rl as core
from scripts.btceth_rl_report_app import DataRuntime,host_path
from scripts.crypto_rl_reporting import clean_json
from scripts.single_pair_rl_six_report_app import csv,verify_case,verify_native_training_counters

SYMBOLS=("BTCUSDT","ETHUSDT")
SEEDS=tuple(range(30))
TUNING_SEEDS=tuple(range(100,105))
VARIANTS=("before","after")
CANDIDATES=("c00_baseline","c01_lr1e4","c02_gamma099","c03_lr1e4_gamma099")
SCOPE="sac_train_validation_optimization_comparison"
PLAN_SHA256="8ab80ceb21720691af055c7ed3505b69997087437f87315484ba2c64af625c36"
EXTENSION_FILES=("scripts/backtest_sac_validation_optimization.py","scripts/sac_validation_optimization_models.py","scripts/audit_sac_validation_optimization.py")
ORIGINAL_AUDITOR_SHA256="f84e18ea2d587cbce962f3c9ec9f7385fb56ce5f91284f7dd8c102f6d92fe3ad"
REVISION_FILES=("scripts/audit_sac_validation_optimization_v2.py","scripts/continue_sac_validation_optimization.py")
REVISION_NORMALIZATION="net_arch unchanged; use_sde absent or boolean False only; reject unknown policy kwargs"
PARENT_AUDITOR_SHA256="a53f412c748899bec6cc961316eedad52c354f522248a5f8b9d6d99b279ab468"
IMPORT_REVISION_FILES=("scripts/audit_sac_validation_optimization_v3.py","scripts/finish_sac_fallback_audit_v3.py")
IMPORT_REVISION_CHANGE="rename inner imported-file expected digest; preserve exact population and byte hashes"
FINAL_AUDIT_PATHS={split:"final/"+split+"_optimization_audit_v3.json" for split in ("validation","test")}
ETH_READER_SHA256="cf4f24e5393209c35395f8aea084075e943b3ca5910d1081c62b67ae9e7e5171"
RESULT_FILES=("equity_1h.csv.gz","fills.csv","weights.csv","boundary_equity.csv","metadata.json","fingerprints.json","actions.json","summary.json","daily_equity.csv","monthly_returns.csv")
QUERY_COUNTS={"account_summary":256,"equity_history":256*366,"monthly_returns":256*13,"seed_statistics":8,"matched_seed_results":120,"matched_statistics":4,
    "selection":2,"validation_candidates":8,"validation_seed_scores":40,"validation_gates":48,"tuning_validation_accounts":144,"final_validation_accounts":136,
    "training_records":160,"reference_provenance":2,"audit_evidence":14,"protocol":1}


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean_json(value),ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def unlinked_root(value):
    path=Path(value).absolute()
    require(not any(p.is_symlink() or hasattr(p,"is_junction") and p.is_junction() for p in (path,*path.parents)),"Research roots cannot cross a link")
    return path.resolve()


def relative(root,name):
    name=str(name);parsed=PurePosixPath(name)
    require(name and parsed.as_posix()==name and not parsed.is_absolute() and ".." not in parsed.parts and ":" not in name and "\\" not in name,"Canonical root-relative evidence path required")
    root=unlinked_root(root);path=root.joinpath(*parsed.parts)
    require(not any(p.is_symlink() or hasattr(p,"is_junction") and p.is_junction() for p in (path,*path.parents)),"Evidence cannot cross a link")
    path.resolve().relative_to(root)
    require(path.is_file(),"Missing evidence: "+name)
    return path


def account_identity(row,variant):
    """Keep raw run_id unchanged; add a unique presentation/account identity."""
    require(variant in (*VARIANTS,"benchmark"),"Unknown SAC comparison policy variant")
    require(row.get("symbol") in SYMBOLS and row.get("scenario") in core.COSTS and row.get("split")=="test","Annual single-coin account required")
    require(row.get("seed") is None or type(row["seed"]) is int and row["seed"] in SEEDS,"Predeclared final seed required")
    if variant in VARIANTS:require(row.get("algorithm")=="sac" and row.get("seed") is not None,"Before/after native model identity must remain SAC")
    else:require(row.get("algorithm") in core.BASELINES and row.get("seed") is None,"Benchmark account identity changed")
    account_id=f"{variant}::{row['run_id']}"
    return {**deepcopy(row),"policy_variant":variant,"source_run_id":row["run_id"],"account_id":account_id}


def seed_statistics(rows):
    output=[]
    for symbol in SYMBOLS:
        for variant in VARIANTS:
            for scenario in core.COSTS:
                group=[r for r in rows if r["symbol"]==symbol and r["policy_variant"]==variant and r["scenario"]==scenario]
                require(len(group)==30 and {r["seed"] for r in group}==set(SEEDS),"Complete matched30-seed outcome population required")
                returns=np.asarray([r["net_return"] for r in group],dtype=float);dds=np.abs([r["max_drawdown"] for r in group])
                require(np.isfinite(returns).all() and np.isfinite(dds).all(),"Finite actual account outcomes required")
                q=np.quantile(returns,[.1,.25,.5,.75,.9],method="linear")
                output.append({"symbol":symbol,"policy_variant":variant,"scenario":scenario,"seeds":30,
                    "mean_return":float(returns.mean()),"p10_return":float(q[0]),"q1_return":float(q[1]),"median_return":float(q[2]),"q3_return":float(q[3]),"p90_return":float(q[4]),
                    "min_return":float(returns.min()),"max_return":float(returns.max()),"positive_seeds":int((returns>0).sum()),"loss_seeds":int((returns<0).sum()),"zero_seeds":int((returns==0).sum()),
                    "loss_rate":float((returns<0).mean()),"median_drawdown":float(np.quantile(dds,.5,method="linear")),"p90_drawdown":float(np.quantile(dds,.9,method="linear")),
                    "worst_drawdown":float(dds.max()),"drawdown_above_20_seeds":int((dds>.2).sum()),"quantile_method":"linear"})
    return output


def matched_seed_results(rows):
    output=[]
    for symbol in SYMBOLS:
        for seed in SEEDS:
            for scenario in core.COSTS:
                group={r["policy_variant"]:r for r in rows if r["symbol"]==symbol and r["seed"]==seed and r["scenario"]==scenario and r["policy_variant"] in VARIANTS}
                require(set(group)==set(VARIANTS),"Each final seed requires both before and after actual accounts")
                before,after=group["before"],group["after"]
                output.append({"symbol":symbol,"seed":seed,"scenario":scenario,"before_account_id":before["account_id"],"after_account_id":after["account_id"],
                    "before_return":before["net_return"],"after_return":after["net_return"],"return_difference":after["net_return"]-before["net_return"],
                    "before_drawdown":abs(before["max_drawdown"]),"after_drawdown":abs(after["max_drawdown"]),"drawdown_difference":abs(after["max_drawdown"])-abs(before["max_drawdown"]),
                    "higher_return":after["net_return"]>before["net_return"],"lower_drawdown":abs(after["max_drawdown"])<abs(before["max_drawdown"])})
    return output


def matched_statistics(pairs):
    output=[]
    for symbol in SYMBOLS:
        for scenario in core.COSTS:
            group=[r for r in pairs if r["symbol"]==symbol and r["scenario"]==scenario]
            require(len(group)==30 and {r["seed"] for r in group}==set(SEEDS),"Complete matched differences required")
            differences=np.asarray([r["return_difference"] for r in group]);dd_differences=np.asarray([r["drawdown_difference"] for r in group])
            output.append({"symbol":symbol,"scenario":scenario,"seeds":30,"median_return_difference":float(np.median(differences)),
                "q1_return_difference":float(np.quantile(differences,.25,method="linear")),"q3_return_difference":float(np.quantile(differences,.75,method="linear")),
                "min_return_difference":float(differences.min()),"max_return_difference":float(differences.max()),"median_drawdown_difference":float(np.median(dd_differences)),
                "higher_return_seeds":sum(r["higher_return"] for r in group),"lower_drawdown_seeds":sum(r["lower_drawdown"] for r in group),
                "both_better_seeds":sum(r["higher_return"] and r["lower_drawdown"] for r in group),"unchanged_seeds":int(((differences==0)&(dd_differences==0)).sum())})
    return output


def load_original_snapshots(btc_root,eth_root,retained_root):
    require(sha(ROOT/"scripts/eth_rl_seed_stability_report_app.py")==ETH_READER_SHA256,"Preserve original ETH reader")
    snapshots={};evidence={}
    for symbol,root,reader in (("BTCUSDT",btc_root,old_btc),("ETHUSDT",eth_root,old_eth)):
        snapshots[symbol],evidence[symbol]=reader.make_reviewed_snapshot(root,retained_root)
    return snapshots,evidence


def verify_inputs(root,evidence):
    for name,expected in evidence["input_sha256"].items():require(sha(relative(root,name))==expected,"Optimization input changed: "+name)
    for name,expected in evidence["external_input_sha256"].items():
        path=Path(name);unlinked_root(path.parent)
        require(path.is_file() and not path.is_symlink() and sha(path)==expected,"Original study/readback dependency changed: "+name)
    for name,expected in evidence["source_sha256"].items():require(sha(relative(ROOT,name))==expected,"Report/helper source changed: "+name)


def verify_audit_revision(top,extension,pins):
    """Keep the original execution sources and explicitly identify the active audit."""
    path=relative(top,"audit_revision.json");revision=read(path);pins["audit_revision.json"]=sha(path)
    receipt_path=relative(top,"audit_revision_receipt.json");receipt=read(receipt_path);pins["audit_revision_receipt.json"]=sha(receipt_path)
    require(receipt.get("sha256")==pins["audit_revision.json"],"Versioned audit revision receipt no longer matches its explicit contract")
    expected={"schema_version":1,"revision_id":"native_sac_policy_kwargs_default_false_v2","reason_code":"sb3_sac_saved_policy_kwargs_use_sde_false",
        "original_protocol_sha256":sha(relative(top,"protocol.json")),"search_plan_sha256":PLAN_SHA256,
        "original_extension_manifest_sha256":sha(relative(top,"extension_manifest.json")),"original_audit_source_sha256":ORIGINAL_AUDITOR_SHA256,
        "revised_audit_source_path":"audit_revision_source/"+REVISION_FILES[0],"controller_source_path":"audit_revision_source/"+REVISION_FILES[1],
        "original_pipeline_launch_sha256":sha(relative(top,"pipeline_launch.json")),"permitted_metadata_normalization":REVISION_NORMALIZATION,
        "financial_math_changed":False,"model_recipe_changed":False,"training_restarted":False}
    require(extension["extension_sources"].get(EXTENSION_FILES[2])==ORIGINAL_AUDITOR_SHA256,"The original f84 audit source must remain frozen")
    require(sha(relative(top,"search_plan.json"))==PLAN_SHA256,"Revision cannot replace the predeclared search plan")
    for key,value in expected.items():
        require(revision.get(key)==value and (not isinstance(value,bool) or type(revision.get(key)) is bool),"Invalid audit-only revision: "+key)
    require(revision.get("financial_oracle_source_sha256")==read(relative(top,"protocol.json"))["sources"]["scripts/audit_single_pair_rl.py"],"Original financial oracle changed")
    created=pd.Timestamp(revision["created_at_utc"]);frozen=pd.Timestamp(read(relative(top,"protocol.json"))["frozen_at"])
    require(created.tzinfo is not None and created>=frozen,"Audit correction must follow the immutable protocol freeze")
    received=pd.Timestamp(receipt["at_utc"]);require(received.tzinfo is not None and received>=created,"Versioned audit revision receipt predates its contract")
    for name,key in zip(REVISION_FILES,("revised_audit_source_sha256","controller_source_sha256")):
        copied="audit_revision_source/"+name;digest=sha(relative(top,copied))
        require(digest==sha(relative(ROOT,name))==revision.get(key),"Active audit/controller source differs from the explicit revision: "+name)
        pins[copied]=digest
    pins["pipeline_launch.json"]=expected["original_pipeline_launch_sha256"]
    return revision


def verify_revision_launch(top,revision,pins):
    """Bind the recorded continuation identity; the collector verifies process exit."""
    name="pipeline_launch_revision.json";path=relative(top,name);launch=read(path)
    expected={"audit_revision_sha256":pins["audit_revision.json"],"original_pipeline_launch_sha256":revision["original_pipeline_launch_sha256"],
        "controller_source_sha256":revision["controller_source_sha256"],"revised_audit_source_sha256":revision["revised_audit_source_sha256"],"training_restarted":False}
    for key,value in expected.items():
        require(launch.get(key)==value and (not isinstance(value,bool) or type(launch.get(key)) is bool),"Continuation launch does not bind the explicit metadata correction: "+key)
    pins[name]=sha(path)
    return launch


def verify_import_audit_revision(top,parent_revision,pins):
    """Preserve passed v2 evidence and the failed wrapper; accept final v3 only."""
    def take(name):
        path=relative(top,name);pins[name]=sha(path);return read(path)
    revision=take("audit_import_revision.json");receipt=take("audit_import_revision_receipt.json")
    failed=take("final/validation_optimization_audit.json");selection=take("selection.json");selected_audit=take("selection_audit.json")
    expected={"schema_version":1,"revision_id":"retained_import_population_shadowing_v3","reason_code":"import_manifest_expected_digest_shadowing",
        "original_protocol_sha256":sha(relative(top,"protocol.json")),"search_plan_sha256":PLAN_SHA256,
        "parent_audit_revision_sha256":pins["audit_revision.json"],"parent_auditor_source_sha256":PARENT_AUDITOR_SHA256,
        "revised_audit_source_path":"audit_import_revision_source/"+IMPORT_REVISION_FILES[0],"finisher_source_path":"audit_import_revision_source/"+IMPORT_REVISION_FILES[1],
        "failed_final_validation_audit_sha256":pins["final/validation_optimization_audit.json"],"selection_sha256":pins["selection.json"],"selection_audit_sha256":pins["selection_audit.json"],
        "permitted_change":IMPORT_REVISION_CHANGE,"financial_math_changed":False,"model_recipe_changed":False,"selection_changed":False,"training_restarted":False}
    require(parent_revision["revised_audit_source_sha256"]==PARENT_AUDITOR_SHA256,"The passed parent v2 auditor must remain immutable")
    for key,value in expected.items():
        require(revision.get(key)==value and (not isinstance(value,bool) or type(revision.get(key)) is bool),"Invalid final import-audit revision: "+key)
    require(receipt.get("sha256")==pins["audit_import_revision.json"] and pd.Timestamp(receipt["at_utc"])>=pd.Timestamp(revision["created_at_utc"]),"Final import-audit revision receipt changed or predates its contract")
    require(failed.get("passed") is False and failed.get("failure_count",0)>0 and failed.get("audit_source_sha256")==PARENT_AUDITOR_SHA256
            and failed.get("core_financial_audit_passed") is False and failed.get("phase")=="final-validation"
            and failed.get("protocol_sha256")==expected["original_protocol_sha256"],"Preserve the exact failed v2 wrapper as failed evidence")
    require(pd.Timestamp(revision["created_at_utc"])>=pd.Timestamp(failed["audited_at"])>=pd.Timestamp(selection["at"]),"Import-audit correction must follow the preserved failure and frozen selection")
    require(selected_audit.get("passed") is True and selected_audit.get("failure_count")==0 and selected_audit.get("audit_source_sha256")==PARENT_AUDITOR_SHA256
            and selected_audit.get("selection_sha256")==pins["selection.json"],"The original passed v2 selection cannot be replaced")
    require(set(selection.get("selection",{}))==set(SYMBOLS) and all(c.get("candidate_id")=="c00_baseline" and c.get("fallback_to_baseline") is True for c in selection["selection"].values()),"This continuation only permits the already-selected unchanged baseline fallback")
    for name,key in zip(IMPORT_REVISION_FILES,("revised_audit_source_sha256","finisher_source_sha256")):
        copied="audit_import_revision_source/"+name;digest=sha(relative(top,copied))
        require(digest==sha(relative(ROOT,name))==revision.get(key),"Final auditor/finisher differs from its versioned contract: "+name);pins[copied]=digest
    return revision


def verify_import_revision_launch(top,revision,pins):
    name="pipeline_launch_import_revision.json";path=relative(top,name);launch=read(path)
    expected={"audit_import_revision_sha256":pins["audit_import_revision.json"],"parent_pipeline_launch_revision_sha256":pins["pipeline_launch_revision.json"],
        "finisher_source_sha256":revision["finisher_source_sha256"],"revised_audit_source_sha256":revision["revised_audit_source_sha256"],"training_restarted":False}
    for key,value in expected.items():
        require(launch.get(key)==value and (not isinstance(value,bool) or type(launch.get(key)) is bool),"Final fallback continuation launch changed: "+key)
    pins[name]=sha(path);return launch


def verify_transport_external(top,completion,active_auditor_sha256,pins,*,audit_revision_sha256=None,pipeline_launch_revision_sha256=None,audit_import_revision_sha256=None,pipeline_launch_import_revision_sha256=None):
    """Use the exact audited export receipt instead of reopening Linux data paths."""
    external=completion.get("external_inputs_sha256",{})
    spec=read(relative(top,"protocol.json"))
    require(all(k in spec for k in ("dataset_sha256","dataset_manifest_sha256")),"Frozen original dataset identities are missing")
    require(set(spec.get("reference_sources",{}))==set(SYMBOLS),"Explicit two-coin original external source mapping required")
    references=read(relative(top,"reference_manifest.json"))["files_sha256"]
    annual=read(relative(top,"reference_annual_import.json"))["files_sha256"]
    expected_references={}
    for name,digest in {**references,**annual}.items():
        parsed=PurePosixPath(name)
        require(len(parsed.parts)>2 and parsed.parts[0]=="references" and parsed.parts[1] in SYMBOLS
                and parsed.as_posix()==name and ".." not in parsed.parts and "\\" not in name,"Canonical external original-reference identity required")
        source_root=PurePosixPath(spec["reference_sources"][parsed.parts[1]]["source_root"])
        require(source_root.is_absolute() and ".." not in source_root.parts,"Absolute immutable original Linux source root required")
        source=(source_root/PurePosixPath(*parsed.parts[2:])).as_posix()
        require(source not in expected_references or expected_references[source]==digest,"Conflicting original reference hashes")
        expected_references[source]=digest
    require(references and annual and all(external.get(p)==h for p,h in expected_references.items()),"External original-reference evidence is incomplete or changed")
    remaining={p:h for p,h in external.items() if p not in expected_references}
    expected={"dataset.pkl.gz":spec["dataset_sha256"],"manifest.json":spec["dataset_manifest_sha256"]}
    require(len(remaining)==2 and {PurePosixPath(p).name for p in remaining}==set(expected)
            and len({PurePosixPath(p).parent for p in remaining})==1
            and all(PurePosixPath(p).is_absolute() and PurePosixPath(p).as_posix()==p and ".." not in PurePosixPath(p).parts
                    and "\\" not in p and h==expected[PurePosixPath(p).name] for p,h in remaining.items()),"External dataset/manifest hashes differ from the frozen original source")
    matching=[]
    for path in sorted((top/"transport").glob("*.tar.gz.receipt.json")):
        receipt=read(path)
        if receipt.get("completion_contract_audit_sha256")==sha(top/"completion_contract_audit.json"):matching.append((path,receipt))
    require(matching,"Collect the audited study with an explicit external-verification receipt before building")
    receipt_path,receipt=matching[-1]
    if audit_revision_sha256 is not None:require(receipt.get("audit_revision_sha256")==audit_revision_sha256,"Audited export does not identify the explicit active audit revision")
    if audit_import_revision_sha256 is not None:require(receipt.get("audit_import_revision_sha256")==audit_import_revision_sha256,"Audited export does not bind the final import-audit revision")
    require(receipt.get("external_inputs_sha256")==external and receipt.get("external_verified_sha256")==external
            and receipt.get("external_rechecked_after_archive") is True
            and receipt.get("verifier_source_sha256")==active_auditor_sha256,"Audited export external verification changed")
    archive=receipt_path.parent/PurePosixPath(receipt["archive"]).name
    require(sha(archive)==receipt.get("archive_sha256"),"Collected archive differs from the external-verification receipt")
    files=receipt.get("files",{})
    if pipeline_launch_revision_sha256 is not None:require(files.get("pipeline_launch_revision.json")==pipeline_launch_revision_sha256,"Audited export does not bind the recorded continuation launch")
    if pipeline_launch_import_revision_sha256 is not None:require(files.get("pipeline_launch_import_revision.json")==pipeline_launch_import_revision_sha256,"Audited export does not bind the final fallback continuation launch")
    require(files.get("completion_contract_audit.json")==sha(top/"completion_contract_audit.json")
            and files.get("protocol.json")==sha(top/"protocol.json"),"Export does not bind this exact complete protocol")
    with tarfile.open(archive,"r:gz") as stream:
        members=stream.getmembers()
        require(len(members)==len(files) and {m.name for m in members}==set(files),"Export archive membership differs from receipt")
        require(all(m.isfile() and PurePosixPath(m.name).as_posix()==m.name and not PurePosixPath(m.name).is_absolute()
                and ".." not in PurePosixPath(m.name).parts and ":" not in m.name and "\\" not in m.name for m in members),"Export must contain only canonical relative regular files")
    for name,expected_sha in files.items():require(sha(relative(top,name))==expected_sha,"Collected source file changed: "+name)
    pins[receipt_path.relative_to(top).as_posix()]=sha(receipt_path);pins[archive.relative_to(top).as_posix()]=receipt["archive_sha256"]
    return {"receipt":receipt_path.relative_to(top).as_posix(),"receipt_sha256":sha(receipt_path),"archive_sha256":receipt["archive_sha256"],
        "external_inputs_sha256":external,"external_rechecked_after_archive":True,"verifier_source_sha256":receipt["verifier_source_sha256"]}


def bind_audit(top,name,pins,*,phase=None,active_auditor_sha256=None,audit_revision_sha256=None,audit_revision_receipt_sha256=None,audit_import_revision_sha256=None,audit_import_revision_receipt_sha256=None,protocol_sha256=None):
    path=relative(top,name);result=read(path);pins[name]=sha(path)
    require(result.get("passed") is True and result.get("failure_count")==0 and not result.get("failures")
            and result.get("checks_total",0)>0,"A complete passed audit is required: "+name)
    if phase:require(result.get("phase")==phase,"Wrong independent audit phase: "+name)
    if active_auditor_sha256 is not None:require(result.get("audit_source_sha256")==active_auditor_sha256,"Passed receipt does not identify the active versioned auditor: "+name)
    if protocol_sha256 is not None:require(result.get("protocol_sha256")==protocol_sha256,"Passed receipt does not bind the original protocol: "+name)
    if audit_revision_sha256 is not None:require(result.get("inputs_sha256",{}).get("audit_revision.json")==audit_revision_sha256,"Passed receipt omits the explicit audit-only revision: "+name)
    if audit_revision_receipt_sha256 is not None:require(result.get("inputs_sha256",{}).get("audit_revision_receipt.json")==audit_revision_receipt_sha256,"Passed receipt omits the versioned audit sidecar: "+name)
    if audit_import_revision_sha256 is not None:require(result.get("inputs_sha256",{}).get("audit_import_revision.json")==audit_import_revision_sha256,"Final receipt omits the import-audit revision: "+name)
    if audit_import_revision_receipt_sha256 is not None:require(result.get("inputs_sha256",{}).get("audit_import_revision_receipt.json")==audit_import_revision_receipt_sha256,"Final receipt omits the import-audit sidecar: "+name)
    for filename,expected in result.get("inputs_sha256",{}).items():
        require(sha(relative(top,filename))==expected,"Changed completed audit input: "+filename)
        require(filename not in pins or pins[filename]==expected,"Conflicting audit evidence binding")
        pins[filename]=expected
    return result


def expected_phase_cases(split,seeds):
    cases=[(s,"sac",n,c) for s in SYMBOLS for n in seeds for c in core.COSTS]
    cases.extend((s,a,None,c) for s in SYMBOLS for a in core.BASELINES for c in core.COSTS)
    return {f"{split}__{c}__{s}__{a}__{'baseline' if n is None else n}":
        (s,a,n,c,f"evaluation/{split}/{s}/{a}/{'baseline' if n is None else 'seed'+str(n)}/{c}") for s,a,n,c in cases}


def read_phase(top,child_name,split,seeds,pins):
    child=top/child_name;expected=expected_phase_cases(split,seeds);count=len(expected)
    audit=read(relative(child,split+"_audit.json"));audit_name=child_name+"/"+split+"_audit.json";pins[audit_name]=sha(top/audit_name)
    require(audit.get("passed") is True and audit.get("failure_count")==0 and not audit.get("failures") and audit.get("phase")==split
            and all(audit.get(k)==count for k in ("expected_accounts","accounts_checked","repeats_checked","accounts_count"))
            and audit.get("protocol_sha256")==sha(relative(child,"protocol.json"))
            and audit.get("summary_sha256")==sha(relative(child,split+"_summary.csv")),"Full bound stage financial audit required")
    require(set(audit.get("inputs_sha256",{}))=={"protocol.json",split+"_summary.csv",split+"_complete.json"},"Complete financial phase input binding required")
    for name,expected_sha in audit["inputs_sha256"].items():
        require(sha(relative(child,name))==expected_sha,"Changed financial phase bytes");pins[child_name+"/"+name]=expected_sha
    frame=csv(relative(child,split+"_summary.csv"));cases={r["relative_path"]:r for r in audit["cases"]}
    require(len(frame)==count and not frame.run_id.duplicated().any() and set(frame.run_id)==set(expected)
            and len(audit["cases"])==count and set(cases)=={r[-1] for r in expected.values()},"Complete exact stage account population required")
    complete=read(relative(child,split+"_complete.json"))
    require(complete.get("accounts")==complete.get("repeats")==count and complete.get("repeat_equal") is True
            and len(complete.get("results",[]))==count,"Actual repeated execution completion required")
    completed={r["summary"]["run_id"]:r for r in complete["results"]};require(set(completed)==set(expected),"Execution receipt identities changed")
    for kind in ("evaluation","repeat"):
        found={p.parent.relative_to(child).as_posix() for p in (child/kind/split).rglob("summary.json")}
        require(found=={i[-1].replace("evaluation/",kind+"/",1) for i in expected.values()},"Unexpected stage account folders")
    rows=[];days=[];months=[]
    for aggregate in frame.to_dict("records"):
        identity=expected[aggregate["run_id"]];case=cases[identity[-1]]
        for name in case["inputs_sha256"]:relative(child,name)
        row,daily,monthly=verify_case(child,split,identity,aggregate,case)
        require(row["seed"] is None or type(row["seed"]) is int,"Integer native account seed required")
        item=completed[row["run_id"]]
        require(item["summary"]==row and item["path"]==identity[-1] and item["repeat_path"]==identity[-1].replace("evaluation/","repeat/",1) and item["repeat_equal"] is True,"Changed canonical/repeated execution receipt")
        pins.update({child_name+"/"+name:digest for name,digest in case["inputs_sha256"].items()})
        rows.append(row);days.extend(daily);months.extend(monthly)
    return rows,days,months,audit


def gate_rows(selection,plan):
    labels={"score_improvement_at_least":"稳健评分提高","paired_better_seed_count_at_least":"同seed效用更好数","positive_base_seed_count_at_least":"主成本盈利数",
        "stress_return_median_at_least":"压力收益中位数","stress_p90_drawdown_above_baseline_at_most":"压力回撤P90相对原候选增量","nonworse_subwindows_at_least":"不差的连续验证子窗口数"}
    rows=[]
    for symbol,choice in selection["selection"].items():
        base=choice["metrics_by_candidate"]["c00_baseline"]
        for cid in CANDIDATES:
            stat=choice["metrics_by_candidate"][cid]
            actual={"score_improvement_at_least":stat["robust_score"]-base["robust_score"],"paired_better_seed_count_at_least":sum(a>b for a,b in zip(stat["seed_scores"],base["seed_scores"])),
                "positive_base_seed_count_at_least":stat["base_positive_seeds"],"stress_return_median_at_least":stat["stress_median_return"],
                "stress_p90_drawdown_above_baseline_at_most":stat["stress_p90_drawdown"]-base["stress_p90_drawdown"],"nonworse_subwindows_at_least":sum(a>=b for a,b in zip(stat["subwindow_median_scores"],base["subwindow_median_scores"]))}
            for key,passed in choice["gates_by_candidate"][cid].items():
                threshold=plan["acceptance"][key];operator="≤" if key=="stress_p90_drawdown_above_baseline_at_most" else "≥"
                rows.append({"gate_key":f"{symbol}:{cid}:{key}","symbol":symbol,"candidate_id":cid,"gate":key,"gate_label":labels[key],"actual":actual[key],"threshold":threshold,
                    "actual_text":f"{actual[key]:.8g}","threshold_text":f"{operator}{threshold:.8g}","passed":passed,"applicable":cid!="c00_baseline"})
    return rows


def make_reviewed_snapshot(study_root,*,baseline_btc_root,baseline_eth_root,retained_study_root):
    top=unlinked_root(study_root);roots={"BTCUSDT":unlinked_root(baseline_btc_root),"ETHUSDT":unlinked_root(baseline_eth_root)}
    require(all(not top.is_relative_to(r) and not r.is_relative_to(top) for r in roots.values()),"Optimization and original studies must remain separate")
    pins={};external={}
    def take(name):
        path=relative(top,name);pins[name]=sha(path);return read(path)
    extension=take("extension_manifest.json")
    require(set(extension.get("extension_sources",{}))==set(EXTENSION_FILES),"Complete fixed three-file optimization source inventory required")
    for name,expected in extension["extension_sources"].items():
        require(sha(relative(top,"extension_source/"+name))==sha(relative(ROOT,name))==expected,"Optimization executed/frozen source changed: "+name)
        pins["extension_source/"+name]=expected
    revision=verify_audit_revision(top,extension,pins);verify_revision_launch(top,revision,pins)
    import_revision=verify_import_audit_revision(top,revision,pins);verify_import_revision_launch(top,import_revision,pins)
    active_auditor_sha256=import_revision["revised_audit_source_sha256"]
    completion=take("completion_contract_audit.json")
    require(completion.get("passed") is True and completion.get("failure_count")==0 and not completion.get("failures")
            and completion.get("checks_total",0)>0 and completion.get("expected_tuning_models")==40 and completion.get("final_models")==60
            and completion.get("audit_source_sha256")==active_auditor_sha256
            and completion.get("inputs_sha256",{}).get("audit_revision.json")==pins["audit_revision.json"]
            and completion.get("inputs_sha256",{}).get("audit_import_revision.json")==pins["audit_import_revision.json"]
            and completion.get("financial_arithmetic_rerun") is False,"The full independent optimization completion contract must pass before report preparation")
    for name,expected in completion.get("inputs_sha256",{}).items():
        require(sha(relative(top,name))==expected,"Completed optimization input bytes changed: "+name);pins[name]=expected
    verifier=importlib.import_module("scripts.audit_sac_validation_optimization_v3")
    require(Path(verifier.__file__).resolve()==(ROOT/IMPORT_REVISION_FILES[0]).resolve()
            and sha(verifier.__file__)==active_auditor_sha256,"Wrong active versioned metadata verifier module")
    checks=verifier.Checks();verified_pins={}
    plan,spec,_=verifier._base_contract(top,checks,verified_pins)
    require(sha(relative(top,"search_plan.json"))==PLAN_SHA256 and plan["tuning_seeds"]==list(TUNING_SEEDS)
            and plan["comparison_seeds"]==list(SEEDS),"Original fixed search plan and disjoint seed populations required")
    selection=verifier._selection_contract(top,plan,spec,checks,verified_pins,recompute=True)
    require(completion.get("protocol_sha256")==pins.get("protocol.json")==sha(top/"protocol.json")
            and completion.get("selection_sha256")==sha(top/"selection.json"),"Completion does not bind these exact parameters and selected candidates")
    transport=verify_transport_external(top,completion,active_auditor_sha256,pins,audit_revision_sha256=pins["audit_revision.json"],
        pipeline_launch_revision_sha256=pins["pipeline_launch_revision.json"],audit_import_revision_sha256=pins["audit_import_revision.json"],
        pipeline_launch_import_revision_sha256=pins["pipeline_launch_import_revision.json"])
    def passed_audit(name,phase,*,final=False):
        return bind_audit(top,name,pins,phase=phase,active_auditor_sha256=active_auditor_sha256 if final else PARENT_AUDITOR_SHA256,
            audit_revision_sha256=pins["audit_revision.json"],audit_revision_receipt_sha256=pins["audit_revision_receipt.json"],protocol_sha256=sha(top/"protocol.json"),
            audit_import_revision_sha256=pins["audit_import_revision.json"] if final else None,audit_import_revision_receipt_sha256=pins["audit_import_revision_receipt.json"] if final else None)
    audit_rows=[];candidate_frames=[];final_validation=[];after_raw=[];after_days=[];after_months=[]
    for cid in CANDIDATES:
        child_name="tuning/"+cid;child=top/child_name
        _,recipes,seeds=verifier._child_contract(top,child,plan,spec,checks,verified_pins,cid)
        verifier._training_checks(top,child,plan,recipes,checks,verified_pins,candidate_id=cid)
        rows,_,_,financial=read_phase(top,child_name,"validation",TUNING_SEEDS,pins)
        candidate_frames.extend({**r,"candidate_id":cid,"candidate_account_id":cid+"::"+r["run_id"]} for r in rows)
        wrapper=passed_audit(child_name+"/validation_optimization_audit.json","tuning-validation")
        require(wrapper.get("core_financial_audit_passed") is True and wrapper.get("core_financial_audit_sha256")==pins[child_name+"/validation_audit.json"],"Candidate contract/financial binding changed")
        for kind,value in (("financial",financial),("contract",wrapper)):
            audit_rows.append({"audit_id":cid+":"+kind,"stage":"tuning-validation","candidate_id":cid,"audit_kind":kind,"passed":value["passed"],"accounts_checked":value.get("accounts_checked",36),"failure_count":value["failure_count"]})
    selection_audit=passed_audit("selection_audit.json","selection")
    require(selection_audit.get("protocol_sha256")==pins["protocol.json"] and selection_audit.get("selection_sha256")==sha(top/"selection.json"),"Selection audit binds different search results")
    audit_rows.append({"audit_id":"selection","stage":"selection","candidate_id":None,"audit_kind":"selection","passed":True,"accounts_checked":None,"failure_count":0})
    final=top/"final"
    _,recipes,_=verifier._child_contract(top,final,plan,spec,checks,verified_pins,None)
    verifier._training_checks(top,final,plan,recipes,checks,verified_pins,candidate_id=None)
    for split in ("validation","test"):
        rows,days,months,financial=read_phase(top,"final",split,SEEDS,pins)
        wrapper=passed_audit(FINAL_AUDIT_PATHS[split],"final-test" if split=="test" else "final-validation",final=True)
        require(wrapper.get("core_financial_audit_passed") is True and wrapper.get("core_financial_audit_sha256")==pins["final/"+split+"_audit.json"],"Final contract/financial binding changed")
        for kind,value in (("financial",financial),("contract",wrapper)):
            audit_rows.append({"audit_id":"final:"+split+":"+kind,"stage":"final-"+split,"candidate_id":None,"audit_kind":kind,"passed":value["passed"],"accounts_checked":value.get("accounts_checked",136),"failure_count":value["failure_count"]})
        if split=="test":after_raw,after_days,after_months=rows,days,months
        else:final_validation=rows
    require(checks.result()["passed"] and checks.result()["failure_count"]==0,"Read-only provenance/recipe/selection metadata verification failed: "+str(checks.result()["failures"][:5]))
    for name,expected in verified_pins.items():
        require(name not in pins or pins[name]==expected,"Contradictory independent metadata/source binding");pins[name]=expected
    require(all(name in completion["inputs_sha256"] and completion["inputs_sha256"][name]==digest for name,digest in verified_pins.items()),"Completion omitted independently verified source/model inputs")
    before_snapshots,before_evidence=load_original_snapshots(roots["BTCUSDT"],roots["ETHUSDT"],unlinked_root(retained_study_root))
    references=take("reference_manifest.json");annual=take("reference_annual_import.json");model_import=take("final/import_manifest.json")
    require(annual.get("selection_sha256")==sha(top/"selection.json") and annual.get("selection_audit_sha256")==pins["selection_audit.json"]
            and pd.Timestamp(annual["at"])>=pd.Timestamp(selection["at"]),"Annual reference import must follow exact immutable selection")
    for mapping in (references["files_sha256"],annual["files_sha256"]):
        for name,expected in mapping.items():
            parsed=PurePosixPath(name);require(len(parsed.parts)>2 and parsed.parts[0]=="references" and parsed.parts[1] in SYMBOLS,"Only explicitly mapped original reference roots allowed")
            symbol=parsed.parts[1];source_name=PurePosixPath(*parsed.parts[2:]).as_posix();source_path=relative(roots[symbol],source_name)
            require(sha(source_path)==expected==sha(relative(top,name)),"Original/reference byte identities changed");external[str(source_path)]=expected;pins[name]=expected
    require(not any("test" in PurePosixPath(n).parts or PurePosixPath(n).name=="test_audit.json" for n in references["files_sha256"]),"Original prepare manifest contains annual results")
    accounts=[];daily_rows=[];monthly_rows=[];training_rows=[];provenance=[]
    for symbol in SYMBOLS:
        base_snapshot=before_snapshots[symbol];base_root=roots[symbol];base_spec=base_snapshot["report"]["studyConfig"]
        original=spec["reference_sources"][symbol]
        require(original["protocol_sha256"]==sha(base_root/"protocol.json")
                and original["validation_audit_sha256"]==sha(base_root/"validation_audit.json")
                and original["test_audit_sha256"]==sha(base_root/"test_audit.json"),"Explicit original baseline root mapping changed")
        require(all(base_spec.get(k)==spec.get(k) for k in ("train","validation","test","capital_per_account","ensemble","costs","environment","dataset_sha256","dataset_manifest_sha256","sources","runtime_packages")),"Original and selected-account financial semantics differ")
        chosen=selection["selection"][symbol];cid=chosen["candidate_id"];reused=sum(e["symbol"]==symbol for e in model_import["entries"])
        require(reused==(30 if chosen["fallback_to_baseline"] else 0),"Selected c00 must reuse all original30 models; nondefault selection cannot claim fallback")
        provenance.append({"symbol":symbol,"selected_candidate":cid,"fallback_to_baseline":chosen["fallback_to_baseline"],"reused_final_models":reused,"fresh_final_models":30-reused,
            "annual_imported_at":annual["at"],"original_protocol_sha256":original["protocol_sha256"],"reference_manifest_sha256":pins["reference_manifest.json"]})
        before_rows=[r for r in base_snapshot["queries"]["account_summary"]["rows"] if r["algorithm"]=="sac"]
        require(len(before_rows)==60,"Original thirty SAC models at two costs required")
        before_ids={r["run_id"] for r in before_rows}
        accounts.extend({**account_identity(r,"before"),"candidate_id":"c00_baseline","model_origin":"original_baseline"} for r in before_rows)
        for query_id,target in (("equity_history",daily_rows),("monthly_returns",monthly_rows)):
            target.extend({**deepcopy(r),"policy_variant":"before","account_id":"before::"+r["run_id"],"source_run_id":r["run_id"],"candidate_id":"c00_baseline"} for r in base_snapshot["queries"][query_id]["rows"] if r["run_id"] in before_ids)
        for r in base_snapshot["queries"]["training_records"]["rows"]:
            if r["algorithm"]=="sac":training_rows.append({**r,"symbol":symbol,"model_id":f"before:{symbol}:{r['seed']}","policy_variant":"before","stage":"baseline","candidate_id":"c00_baseline","model_origin":"original_baseline"})
        original_map={r["run_id"]:r for r in base_snapshot["queries"]["account_summary"]["rows"]}
        for row in [r for r in after_raw if r["symbol"]==symbol]:
            variant="after" if row["algorithm"]=="sac" else "benchmark"
            normalized=account_identity(row,variant);normalized.update(candidate_id=cid if variant=="after" else None,model_origin="retained_baseline_fallback" if variant=="after" and chosen["fallback_to_baseline"] else "trained_final" if variant=="after" else "retained_fixed_baseline")
            accounts.append(normalized)
            if variant=="benchmark" or chosen["fallback_to_baseline"]:
                require(row==original_map[row["run_id"]],"Reused unchanged model/baseline account differs from its original summary")
        for key,target in (("equity_history",daily_rows),("monthly_returns",monthly_rows)):
            source=after_days if key=="equity_history" else after_months
            for r in source:
                if r["symbol"]==symbol:
                    variant="after" if r["algorithm"]=="sac" else "benchmark"
                    target.append({**deepcopy(r),"policy_variant":variant,"account_id":variant+"::"+r["run_id"],"source_run_id":r["run_id"],"candidate_id":cid if variant=="after" else None})
    for child_name,stage in [("tuning/"+cid,"tuning") for cid in CANDIDATES]+[("final","final")]:
        child_spec=take(child_name+"/protocol.json");fit=take(child_name+"/fit_complete.json");counter=take(child_name+"/training_counter_audit.json")
        counters={(r["symbol"],r["seed"]):r for r in counter["models"]}
        for record in fit["records"]:
            symbol,seed=record["symbol"],record["seed"];origin=counters[(symbol,seed)]["training_origin"]
            training_rows.append({"model_id":f"{child_name}:{symbol}:{seed}","symbol":symbol,"algorithm":"sac","seed":seed,"candidate_id":child_spec["recipe_ids_by_symbol"][symbol],
                "stage":stage,"policy_variant":"after" if stage=="final" else "tuning","model_origin":origin,"actual_steps":record["training"]["actual_steps"],
                "maximum_input_hour":record["maximum_input_hour"],"record_sha256":counters[(symbol,seed)]["record_sha256"],"native_counters_json":json.dumps(record["training"]["counters"],sort_keys=True)})
    require(len(accounts)==256 and len({r["account_id"] for r in accounts})==256 and len(training_rows)==160
            and len({r["model_id"] for r in training_rows})==160,"Complete original/candidate/final account/model namespaces required")
    stats=seed_statistics(accounts);matched=matched_seed_results(accounts);matched_stats=matched_statistics(matched)
    choices=[];candidate_rows=[];seed_scores=[]
    for symbol,choice in selection["selection"].items():
        choices.append({"symbol":symbol,"candidate_id":choice["candidate_id"],"fallback_to_baseline":choice["fallback_to_baseline"],"learning_rate":choice["selected_recipe"]["learning_rate"],"gamma":choice["selected_recipe"]["gamma"],"frozen_at":selection["at"],"selected_recipe_json":json.dumps(choice["selected_recipe"],sort_keys=True)})
        for cid in CANDIDATES:
            value=choice["metrics_by_candidate"][cid];recipe=spec["candidate_recipes"][cid]
            candidate_rows.append({**value,"candidate_key":symbol+":"+cid,"symbol":symbol,"candidate_id":cid,"learning_rate":recipe["learning_rate"],"gamma":recipe["gamma"],"selected":cid==choice["candidate_id"],"eligible":cid in choice["eligible_candidates"]})
            seed_scores.extend({"symbol":symbol,"candidate_id":cid,"seed":seed,"utility":utility} for seed,utility in zip(TUNING_SEEDS,value["seed_scores"]))
    audit_rows.append({"audit_id":"completion","stage":"completion","candidate_id":None,"audit_kind":"completion-contract","passed":True,"accounts_checked":None,"failure_count":0})
    for symbol,value in before_evidence.items():
        for name,digest in value["input_sha256"].items():external[str(roots[symbol]/name)]=digest
        for name,digest in value["external_input_sha256"].items():require(name not in external or external[name]==digest,"Original source hash conflict");external[name]=digest
    source_names=["scripts/sac_validation_optimization_report_app.py","scripts/sac_validation_optimization_report_app.jsx","scripts/sac_validation_optimization_report_app.css",
        "scripts/btc_rl_seed_stability_report_app.py","scripts/eth_rl_seed_stability_report_app.py","scripts/eth_rl_seed_stability_report_app.jsx","scripts/eth_rl_seed_stability_report_app.css",
        "scripts/single_pair_rl_six_report_app.py","scripts/btceth_rl_report_app.py",*EXTENSION_FILES,*REVISION_FILES,*IMPORT_REVISION_FILES,*core.SOURCE_FILES,"scripts/extend_single_pair_rl.py",
        "scripts/backtest_btc_rl_seed_stability.py","scripts/audit_btc_rl_seed_stability.py","scripts/backtest_eth_rl_seed_stability.py","scripts/audit_eth_rl_seed_stability.py"]
    sources={name:sha(relative(ROOT,name)) for name in source_names}
    common={"label":"SAC原参数与只用训练／验证选参后的独立账户","classification":"modeled","timezone":"UTC","period":{"start":core.TEST[0],"end":core.TEST[1]},
        "caveats":["年度此前已被研究观察；本轮选择仅用训练／验证，但不属于新的盲测。","最终比较固定seed0..29；候选选择用不重叠seed100..104。","不同seed不是独立市场样本；分布和差值只作描述。","c00回退会复用原模型和相同口径结果，不计为新训练或新增收益。","每账户独立10000USDT，只持本币和自身USDT，不合成跨币或模型平均权益。"],
        "metricDefinitions":[{"label":"扣费收益","definition":"各账户期末现金加本币市值/10000-1，已扣模拟手续费和不利滑点；期末真实持仓不强平。"},
            {"label":"最大回撤","definition":"完整小时收盘权益相对含期初10000的历史峰值最大跌幅；表中为绝对跌幅。"},
            {"label":"账户身份","definition":"native algorithm仍为sac，另加before/after版本与account_id；原source_run_id及记录文件保持不变。"}],
        "auditRevision":{"contract":"audit_revision.json","contractSha256":pins["audit_revision.json"],"receipt":"audit_revision_receipt.json","receiptSha256":pins["audit_revision_receipt.json"],"revisionId":revision["revision_id"],
            "originalAuditSourceSha256":ORIGINAL_AUDITOR_SHA256,"activeAuditSourceSha256":PARENT_AUDITOR_SHA256,"role":"passed tuning and immutable selection",
            "activeAuditSourcePath":revision["revised_audit_source_path"],"controllerSourcePath":revision["controller_source_path"],
            "controllerSourceSha256":revision["controller_source_sha256"],"permittedMetadataNormalization":REVISION_NORMALIZATION,
            "continuationLaunch":"pipeline_launch_revision.json","continuationLaunchSha256":pins["pipeline_launch_revision.json"],"originalLaunchSha256":revision["original_pipeline_launch_sha256"],
            "financialMathChanged":False,"modelRecipeChanged":False,"trainingRestarted":False},
        "auditImportRevision":{"contract":"audit_import_revision.json","contractSha256":pins["audit_import_revision.json"],"receipt":"audit_import_revision_receipt.json","receiptSha256":pins["audit_import_revision_receipt.json"],
            "revisionId":import_revision["revision_id"],"parentAuditSourceSha256":PARENT_AUDITOR_SHA256,"activeFinalAuditSourceSha256":active_auditor_sha256,
            "activeFinalAuditSourcePath":import_revision["revised_audit_source_path"],"finisherSourcePath":import_revision["finisher_source_path"],"finisherSourceSha256":import_revision["finisher_source_sha256"],
            "failedWrapper":"final/validation_optimization_audit.json","failedWrapperSha256":pins["final/validation_optimization_audit.json"],"failedWrapperAcceptedAsPassed":False,
            "continuationLaunch":"pipeline_launch_import_revision.json","continuationLaunchSha256":pins["pipeline_launch_import_revision.json"],"permittedChange":IMPORT_REVISION_CHANGE,
            "financialMathChanged":False,"modelRecipeChanged":False,"selectionChanged":False,"trainingRestarted":False},
        "evidenceFlow":[{"title":"先冻结有限搜索计划","detail":"四候选只改学习率和gamma，seed100..104训练与验证；选参在年度导入和最终训练前冻结。"},
            {"title":"核对全部候选与接受门","detail":"完整36账户候选验证与独立选择算术审计，未通过时回退原参数。"},
            {"title":"实际账本读回比较","detail":"最终136账户及重复执行、原60SAC模型和年度来源逐SHA绑定，逐账户读回权益、回撤与日/月图表。"}]}
    queries={};anchors=["search_plan.json","search_plan_receipt.json","protocol.json","protocol_receipt.json","extension_manifest.json","audit_revision.json","audit_revision_receipt.json","pipeline_launch.json","pipeline_launch_revision.json",*["audit_revision_source/"+n for n in REVISION_FILES],
        "audit_import_revision.json","audit_import_revision_receipt.json","pipeline_launch_import_revision.json",*["audit_import_revision_source/"+n for n in IMPORT_REVISION_FILES],"final/validation_optimization_audit.json",*FINAL_AUDIT_PATHS.values(),
        "selection.json","selection_receipt.json","selection_audit.json","reference_manifest.json","reference_annual_import.json","completion_contract_audit.json"]
    def query(key,rows,files,method,period=core.TEST):
        source=deepcopy(common);names=list(dict.fromkeys([*files,*anchors]))
        source.update(files=names,fileHashes={name:pins[name] for name in names},period={"start":period[0],"end":period[1]},executedAt=read(top/FINAL_AUDIT_PATHS["test"])["audited_at"],transportExternalVerification=transport)
        queries[key]={"rows":clean_json(rows),"source":source,"methods":[{"language":"text","code":method}]}
    annual_summaries=["final/test_summary.csv",*[n for n in pins if n.startswith("references/") and "/evaluation/test/" in n and n.endswith("/summary.json")]]
    query("account_summary",accounts,annual_summaries,"256 actual accounts: original60 SAC and final60 SAC at two costs, plus16 same-coin fixed-rule accounts; presentation account_id namespaces only, no pooled funding.")
    def annual_plot_sources(filename):
        return [n for n in pins if (n.startswith("final/evaluation/test/") or n.startswith("references/") and "/evaluation/test/" in n) and n.endswith("/"+filename)]
    query("equity_history",daily_rows,annual_plot_sources("daily_equity.csv"),"Actual366 boundaries including initial10000 per account, two actual policies per selected coin/seed/cost; never average curves.")
    query("monthly_returns",monthly_rows,annual_plot_sources("monthly_returns.csv"),"Actual13 partial/calendar month fragments per account, preserving exact original and selected account identities.")
    query("seed_statistics",stats,annual_summaries,"Each coin/version/cost contains exact30 predeclared final seeds; linear quantiles over actual terminal outcomes and hourly max drawdowns, not traded median portfolios.")
    query("matched_seed_results",matched,annual_summaries,"Exact same-label seed0..29 after-minus-before differences, each policy separately funded; descriptive only.")
    query("matched_statistics",matched_stats,annual_summaries,"Median/IQR of seed differences and counts of higher return/lower drawdown/both/unchanged; denominator30 per coin and cost.")
    query("selection",choices,["selection.json"],"Immutable per-coin selected candidate, recipe and fallback status, independently recomputed from tuning validation only; final validation/year never selects parameters.",core.VALIDATION)
    query("validation_candidates",candidate_rows,["tuning/"+c+"/validation_summary.csv" for c in CANDIDATES],"All4x2 candidate metrics with exact five-seed utilities, robust median-.25IQR score and fixed eligibility, retained frozen display values.",core.VALIDATION)
    query("validation_seed_scores",seed_scores,["tuning/"+c+"/validation_summary.csv" for c in CANDIDATES],"seed100..104: mean over actual base/stress accounts of log1p(return)-.5abs(DD)-2max(abs(DD)-.20,0).",core.VALIDATION)
    query("validation_gates",gate_rows(selection,plan),["selection.json"],"Six predeclared replacement gates per candidate/coin; c00 raw gate booleans retained but replacement applicability false because it is the original-reference fallback.",core.VALIDATION)
    query("tuning_validation_accounts",candidate_frames,["tuning/"+c+"/validation_summary.csv" for c in CANDIDATES],"4x36 complete tuning validation accounts with candidate-qualified identities; annual files are not used.",core.VALIDATION)
    query("final_validation_accounts",final_validation,["final/validation_summary.csv"],"Full136 final validation accounts and repeat, run only after parameter choice; results cannot revise the frozen choice.",core.VALIDATION)
    query("training_records",training_rows,["final/fit_complete.json",*["tuning/"+c+"/fit_complete.json" for c in CANDIDATES]],"160 logical source records =60 original+40 tuning+60 final. Reused c00 final records are not new training; actual fresh model count is40 plus nonfallback final count.",core.TRAIN)
    queries["training_records"]["payloadColumns"]=["native_counters_json"]
    query("reference_provenance",provenance,["reference_manifest.json","reference_annual_import.json","final/import_manifest.json"],"Explicit original BTC/ETH root mappings, immutable complete model/validation copies, annual imports after selection and mandatory fallback reuse.")
    query("audit_evidence",audit_rows,["completion_contract_audit.json"],"Every tuning financial/contract audit, independent validation-only choice, final validation/test financial/contract and completion must pass; all current input bytes rehashed.")
    fresh_final=sum(r["fresh_final_models"] for r in provenance)
    query("protocol",[{"capital_per_account":10000,"original_models":60,"before_after_model_records":120,"tuning_models":40,"final_models":60,"reused_final_models":60-fresh_final,"fresh_final_models":fresh_final,"new_training_models":40+fresh_final,"annual_accounts":256,
        "tuning_seeds":list(TUNING_SEEDS),"final_seeds":list(SEEDS),"search_plan_sha256":PLAN_SHA256,"known_history_not_fresh_holdout":True,"no_pooled_capital":True}], ["protocol.json","search_plan.json"],"Frozen finite train/validation-only search and account-level comparison contract.")
    require(set(queries)==set(QUERY_COUNTS) and all(len(queries[name]["rows"])==count for name,count in QUERY_COUNTS.items()),"Complete exact optimization-report query populations required")
    snapshot={"surface":"report","status":"reviewed","buildStatus":"complete","title":"只用训练与验证选参：SAC在BTC和ETH上的前后比较","generatedAt":datetime.now(timezone.utc).isoformat(),"filters":[],"queries":queries,
        "report":{"asOf":"2026-10-22","scope":SCOPE,"studyConfig":spec,"searchPlan":plan,"selection":selection,"noEnsemble":True,"noPooledCapital":True,"auditStatus":"passed"}}
    evidence={"input_sha256":pins,"external_input_sha256":external,"source_sha256":sources,"transport_external_verification":transport,"audit_revision":common["auditRevision"],"audit_import_revision":common["auditImportRevision"],
        "annual_accounts":256,"original_models":60,"tuning_models":40,"final_models":60,"fresh_final_models":fresh_final,"new_training_models":40+fresh_final,
        "native_algorithm":"sac","numeric_results_changed":False,"pickle_loaded":False,"policy_constructed":False,"original_source_roots":{s:str(r) for s,r in roots.items()}}
    verify_inputs(top,evidence)
    return snapshot,evidence


def build_report(study_root,*,baseline_btc_root,baseline_eth_root,retained_study_root,output_dir=None,build_mode="native",wsl_node=None,wsl_distro="Ubuntu-26.04",node_path=None,plugin_root=None):
    top=unlinked_root(study_root);snapshot,evidence=make_reviewed_snapshot(top,baseline_btc_root=baseline_btc_root,baseline_eth_root=baseline_eth_root,retained_study_root=retained_study_root)
    destination=unlinked_root(output_dir) if output_dir else top/"report";project=destination/"report_app";ownership=project/".sac-optimization-authoring.json"
    prior=read(ownership) if ownership.is_file() else None
    if destination.exists() and any(destination.iterdir()):
        require(prior is not None and prior.get("scope")==SCOPE,"Preserve an unowned or different report")
        for name,expected in prior.get("authored_sha256",{}).items():require(sha(relative(project,name))==expected,"Preserve edited presentation")
        for name,expected in prior.get("delivery_sha256",{}).items():require(sha(relative(destination,name))==expected,"Preserve changed delivered evidence")
    elif project.exists():raise FileExistsError("Unowned app project")
    destination.mkdir(parents=True,exist_ok=True);reviewed=destination/"reviewed.json";write(reviewed,snapshot);write(destination/"report_input_manifest.json",evidence)
    runtime=DataRuntime(node_path=node_path,plugin_root=plugin_root,build_mode=build_mode,wsl_node=wsl_node,wsl_distro=wsl_distro)
    if project.exists():preparation=runtime.run("data-app.mjs","prepare","--project-dir",project,cwd=destination)
    else:preparation=runtime.run("prepare-data-app.mjs","--surface","report","--output",project,"--snapshot",reviewed,cwd=destination)
    require("src/content/" in (project/"AGENTS.md").read_text(encoding="utf-8"),"Unexpected official authoring boundary")
    docs=preparation["documentation"];host_path(docs["entryPoint"]).read_text(encoding="utf-8");refs=host_path(docs.get("referenceRoot",str(runtime.plugin/"docs/components")))
    for name in ("narrative/reports.md","charts/charts.md","tables/tables.md","controls/controls.md","sources/sources.md"):(refs/name).read_text(encoding="utf-8")
    content=project/"src/content/report";authoring={Path(__file__).with_suffix(".jsx"):content/"ReportContent.jsx",Path(__file__).with_suffix(".css"):content/"sac-validation-optimization.css"}
    for source,target in authoring.items():shutil.copyfile(source,target)
    prepared=read(project/"src/data.json");snapshot["id"]=prepared["id"]
    if "legacyPresentationTitle" in prepared:snapshot["legacyPresentationTitle"]=prepared["legacyPresentationTitle"]
    write(project/"src/data.json",snapshot);authored={path.relative_to(project).as_posix():sha(path) for path in authoring.values()};write(ownership,{"scope":SCOPE,"authored_sha256":authored})
    build=runtime.run("data-app.mjs","build","--project-dir",project,"--separate-data",cwd=destination);exported=project/".data-app-offline/exports/sac-validation-optimization.html"
    export=runtime.run("data-app.mjs","export-html","--project-dir",project,"--output",exported,cwd=destination);verify_inputs(top,evidence)
    delivered=destination/"report.html";shutil.copyfile(exported,delivered)
    result={"scope":SCOPE,"html_path":str(delivered),"preview_path":str(project/"dist/index.html"),"project_dir":str(project),"reviewed_sha256":sha(reviewed),"html_sha256":sha(delivered),"runtime":build,"export":export,"build_mode":build_mode,"published":False,"numeric_results_changed":False,"authoring_source_sha256":{p.name:sha(p) for p in (Path(__file__),*authoring)}}
    write(destination/"report_build.json",result);write(ownership,{"scope":SCOPE,"authored_sha256":authored,"delivery_sha256":{p.name:sha(p) for p in (reviewed,destination/"report_input_manifest.json",delivered,destination/"report_build.json")}})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--study-root",required=True,type=Path);p.add_argument("--baseline-btc-root",required=True,type=Path);p.add_argument("--baseline-eth-root",required=True,type=Path);p.add_argument("--retained-study-root",required=True,type=Path)
    p.add_argument("--output-dir",type=Path);p.add_argument("--build-mode",choices=("native","wsl"),default="native");p.add_argument("--wsl-node");p.add_argument("--wsl-distro",default="Ubuntu-26.04");p.add_argument("--node-path");p.add_argument("--plugin-root")
    a=p.parse_args();print(json.dumps(build_report(a.study_root,baseline_btc_root=a.baseline_btc_root,baseline_eth_root=a.baseline_eth_root,retained_study_root=a.retained_study_root,output_dir=a.output_dir,build_mode=a.build_mode,wsl_node=a.wsl_node,wsl_distro=a.wsl_distro,node_path=a.node_path,plugin_root=a.plugin_root),ensure_ascii=False))


if __name__=="__main__":main()
