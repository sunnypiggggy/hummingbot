"""Join two independently audited single-coin seed studies for comparison only.

No pooled account, cross-coin equity, ensemble, model inference or training is
created. The completed ETH reader is pinned and remains byte-identical.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import btc_rl_seed_stability_report_app as btc
from scripts import eth_rl_seed_stability_report_app as eth
from scripts.btceth_rl_report_app import DataRuntime,host_path
from scripts.crypto_rl_reporting import clean_json

require,sha,read,write,relative=btc.require,btc.sha,btc.read,btc.write,btc.relative
SCOPE="btceth_independent_seed_stability"
ETH_READER_SHA256="cf4f24e5393209c35395f8aea084075e943b3ca5910d1081c62b67ae9e7e5171"
QUERY_COUNTS={"account_summary":256,"equity_history":93696,"monthly_returns":3328,"cohort_statistics":16,
    "paired_seed_results":120,"paired_statistics":8,"validation_summary":256,"training_records":120,
    "training_import":20,"audit_evidence":4,"protocol":2}
COMMON_CONTRACT=("train","validation","test","primary_seed","primary_seed_fixed_before_training","capital_per_account",
    "capital_shared_between_pairs","ensemble","costs","environment","sources","runtime_packages","baselines",
    "research_gates_per_pair","python","no_production_change","fresh_holdout","test_year_already_observed",
    "dataset_sha256","dataset_manifest_sha256","seeds","timesteps_per_model","recipes","seed_extension","validation_selection")


def unlinked_root(value):
    path=Path(value).absolute()
    require(not any(p.is_symlink() or hasattr(p,"is_junction") and p.is_junction() for p in (path,*path.parents)),"Study/report root cannot cross links")
    return path.resolve()


def merge_verified_snapshots(parts):
    """Preserve each source row and add its explicit single-coin identity."""
    require(set(parts)=={"BTCUSDT","ETHUSDT"},"Exactly two separately verified studies are required")
    queries={}
    for query_id,count in QUERY_COUNTS.items():
        rows=[]
        for symbol,(snapshot,_) in parts.items():
            require(query_id in snapshot["queries"],"Missing source query: "+query_id)
            for row in snapshot["queries"][query_id]["rows"]:
                require(row.get("symbol",symbol)==symbol,"Foreign coin row in verified single-coin query")
                rows.append({**deepcopy(row),"symbol":symbol})
        require(len(rows)==count,"Incomplete joined query population: "+query_id)
        if query_id in ("account_summary","validation_summary"):
            require(len({r["run_id"] for r in rows})==count,"Cross-coin or within-study account identity collision")
        queries[query_id]={"rows":rows}
    return queries


def verify_inputs(root,evidence):
    for name,expected in evidence["input_sha256"].items():require(sha(relative(root,name))==expected,"BTC evidence changed during report build: "+name)
    for name,expected in evidence["external_input_sha256"].items():
        path=Path(name);unlinked_root(path.parent)
        require(path.is_file() and not path.is_symlink() and sha(path)==expected,"ETH/original study evidence changed: "+name)
    for name,expected in evidence["source_sha256"].items():require(sha(relative(ROOT,name))==expected,"Comparison reader/presentation changed: "+name)


def make_reviewed_snapshot(study_root,eth_study_root,retained_study_root):
    roots={"BTCUSDT":unlinked_root(study_root),"ETHUSDT":unlinked_root(eth_study_root)}
    retained=unlinked_root(retained_study_root)
    require(not roots["BTCUSDT"].is_relative_to(roots["ETHUSDT"]) and not roots["ETHUSDT"].is_relative_to(roots["BTCUSDT"]),"BTC and ETH studies must remain separate")
    require(sha(ROOT/"scripts/eth_rl_seed_stability_report_app.py")==ETH_READER_SHA256,"Preserve the completed immutable ETH reader")
    btc_snapshot,btc_evidence=btc.make_reviewed_snapshot(roots["BTCUSDT"],retained)
    eth_snapshot,eth_evidence=eth.make_reviewed_snapshot(roots["ETHUSDT"],retained)
    parts={"BTCUSDT":(btc_snapshot,btc_evidence),"ETHUSDT":(eth_snapshot,eth_evidence)}
    configurations={symbol:snapshot["report"]["studyConfig"] for symbol,(snapshot,_) in parts.items()}
    require(all(configurations["BTCUSDT"].get(k)==configurations["ETHUSDT"].get(k) for k in COMMON_CONTRACT),"Cross-study dates, recipes, budgets, data or funding contract differ")
    require(configurations["BTCUSDT"].get("symbols")==["BTCUSDT"] and configurations["ETHUSDT"].get("symbols")==["ETHUSDT"],"Each study must own one coin only")
    queries=merge_verified_snapshots(parts)
    for query_id,query in queries.items():
        source={"label":"BTC与ETH分别独立的种子复现账户","classification":"modeled","timezone":"UTC","files":[],"fileHashes":{},
            "fileRoots":{s:str(r) for s,r in roots.items()},"perSymbolSources":{},
            "caveats":["每账户独立10,000 USDT；BTC、ETH不共享资金、库存或特征，没有跨币或多模型平均权益。",
                "该年度已被研究观察；新增种子是同一行情上的训练随机性复现，不是新的市场盲测。",
                "分位数和同编号差值只作描述；不同种子不等于不同独立市场样本。",
                "TD3每模型65,536环境步，SAC32,768步，保留原配方，计算预算不等。"],
            "metricDefinitions":[{"label":"扣费收益","definition":"每个实际账户期末现金加本币市值/10000-1；手续费与不利滑点已扣，期末真实持仓不强平。"},
                {"label":"最大回撤","definition":"各自完整8760小时收盘权益相对含期初10000的历史峰值最大跌幅；展示跌幅绝对值。"},
                {"label":"种子分组","definition":"全部30为预设seed0..29；新增25只含seed5..29。各币各算法独立统计，不跨币求平均。"}],
            "evidenceFlow":[{"title":"分别核验后并列","detail":"BTC与ETH各自128年度成本账户、重复执行、60模型及双重审计通过。原ETH来源保持不变。"},
                {"title":"只合并展示行","detail":"每行显式保留symbol及run_id，统计按本币、本算法、本成本计算；没有合成资金或权益曲线。"}]}
        methods=[];periods=[];executed=[]
        for symbol,(snapshot,_) in parts.items():
            original_query=snapshot["queries"][query_id];original_source=deepcopy(original_query["source"])
            source["perSymbolSources"][symbol]=original_source
            periods.append(original_source["period"]);executed.append(original_source["executedAt"])
            for name in original_source["files"]:
                alias=symbol+"/"+name;source["files"].append(alias);source["fileHashes"][alias]=original_source["fileHashes"][name]
            methods.extend({**method,"code":symbol+": "+method["code"]} for method in original_query.get("methods",[]))
        require(periods[0]==periods[1],"Source query periods differ: "+query_id)
        source.update(period=periods[0],executedAt=max(executed))
        query.update(source=source,methods=methods)
        if query_id=="training_records":query["payloadColumns"]=["native_counters_json"]
    external=dict(btc_evidence["external_input_sha256"])
    for name,expected in eth_evidence["external_input_sha256"].items():
        require(name not in external or external[name]==expected,"Original evidence hash conflict")
        external[name]=expected
    for name,expected in eth_evidence["input_sha256"].items():
        path=str(roots["ETHUSDT"]/name);require(path not in external or external[path]==expected,"ETH evidence hash conflict");external[path]=expected
    sources={}
    for evidence in (btc_evidence,eth_evidence):
        for name,expected in evidence["source_sha256"].items():
            require(name not in sources or sources[name]==expected,"Reader dependency hash conflict");sources[name]=expected
    for name in ("scripts/btceth_seed_stability_report_app.py","scripts/btceth_seed_stability_report_app.jsx","scripts/btceth_seed_stability_report_app.css"):
        sources[name]=sha(ROOT/name)
    snapshot={"surface":"report","status":"reviewed","buildStatus":"complete","title":"BTC与ETH分别交易：SAC、TD3的30种子收益与回撤",
        "generatedAt":datetime.now(timezone.utc).isoformat(),"filters":[],"queries":queries,
        "report":{"asOf":"2026-10-22","scope":SCOPE,"studies":configurations,"defaultCohort":"all30","noEnsemble":True,"noPooledCapital":True,
            "sourceEthReaderSha256":ETH_READER_SHA256,"auditStatus":"passed","selectionRole":"comparison_only_no_winner"}}
    evidence={"input_sha256":btc_evidence["input_sha256"],"external_input_sha256":external,"source_sha256":sources,
        "accounts_per_phase":256,"accounts_per_coin_phase":128,"training_models":120,"retained_models":20,"new_models":100,
        "numeric_results_changed":False,"pickle_loaded":False,"policy_constructed":False,"pooled_capital":False,"cross_coin_equity_created":False,
        "verified_roots":{s:str(r) for s,r in roots.items()},"immutable_eth_reader_sha256":ETH_READER_SHA256}
    verify_inputs(roots["BTCUSDT"],evidence)
    return snapshot,evidence


def build_report(study_root,*,eth_study_root,retained_study_root,output_dir=None,build_mode="native",wsl_node=None,wsl_distro="Ubuntu-26.04",node_path=None,plugin_root=None):
    root=unlinked_root(study_root);snapshot,evidence=make_reviewed_snapshot(root,eth_study_root,retained_study_root)
    destination=unlinked_root(output_dir) if output_dir else root/"report"
    project=destination/"report_app";ownership=project/".btceth-seed-stability-authoring.json"
    prior=read(ownership) if ownership.is_file() else None
    if destination.exists() and any(destination.iterdir()):
        require(prior is not None and prior.get("scope")==SCOPE,"Preserve an unowned/different report")
        for name,expected in prior.get("authored_sha256",{}).items():require(sha(relative(project,name))==expected,"Preserve user presentation edit")
        for name,expected in prior.get("delivery_sha256",{}).items():require(sha(relative(destination,name))==expected,"Preserve changed delivered evidence")
    elif project.exists():raise FileExistsError("Unowned app project")
    destination.mkdir(parents=True,exist_ok=True);reviewed=destination/"reviewed.json"
    write(reviewed,snapshot);write(destination/"report_input_manifest.json",evidence)
    runtime=DataRuntime(node_path=node_path,plugin_root=plugin_root,build_mode=build_mode,wsl_node=wsl_node,wsl_distro=wsl_distro)
    if project.exists():preparation=runtime.run("data-app.mjs","prepare","--project-dir",project,cwd=destination)
    else:preparation=runtime.run("prepare-data-app.mjs","--surface","report","--output",project,"--snapshot",reviewed,cwd=destination)
    require("src/content/" in (project/"AGENTS.md").read_text(encoding="utf-8"),"Official authoring boundary changed")
    docs=preparation["documentation"];host_path(docs["entryPoint"]).read_text(encoding="utf-8")
    refs=host_path(docs.get("referenceRoot",str(runtime.plugin/"docs/components")))
    for name in ("narrative/reports.md","charts/charts.md","tables/tables.md","controls/controls.md","sources/sources.md"):(refs/name).read_text(encoding="utf-8")
    content=project/"src/content/report"
    authoring={Path(__file__).with_suffix(".jsx"):content/"ReportContent.jsx",Path(__file__).with_suffix(".css"):content/"btceth-seed-stability.css"}
    for source,target in authoring.items():shutil.copyfile(source,target)
    prepared=read(project/"src/data.json");snapshot["id"]=prepared["id"]
    if "legacyPresentationTitle" in prepared:snapshot["legacyPresentationTitle"]=prepared["legacyPresentationTitle"]
    write(project/"src/data.json",snapshot)
    authored={path.relative_to(project).as_posix():sha(path) for path in authoring.values()};write(ownership,{"scope":SCOPE,"authored_sha256":authored})
    build=runtime.run("data-app.mjs","build","--project-dir",project,"--separate-data",cwd=destination)
    exported=project/".data-app-offline/exports/btceth-seed-stability.html"
    export=runtime.run("data-app.mjs","export-html","--project-dir",project,"--output",exported,cwd=destination)
    verify_inputs(root,evidence);delivered=destination/"report.html";shutil.copyfile(exported,delivered)
    result={"scope":SCOPE,"html_path":str(delivered),"preview_path":str(project/"dist/index.html"),"project_dir":str(project),
        "reviewed_sha256":sha(reviewed),"html_sha256":sha(delivered),"runtime":build,"export":export,"build_mode":build_mode,"published":False,"numeric_results_changed":False,
        "authoring_source_sha256":{p.name:sha(p) for p in (Path(__file__),*authoring)}}
    write(destination/"report_build.json",result)
    write(ownership,{"scope":SCOPE,"authored_sha256":authored,"delivery_sha256":{p.name:sha(p) for p in (reviewed,destination/"report_input_manifest.json",delivered,destination/"report_build.json")}})
    return result


def main():
    import json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--study-root",required=True,type=Path);p.add_argument("--eth-study-root",required=True,type=Path);p.add_argument("--retained-study-root",required=True,type=Path)
    p.add_argument("--output-dir",type=Path);p.add_argument("--build-mode",choices=("native","wsl"),default="native");p.add_argument("--wsl-node");p.add_argument("--wsl-distro",default="Ubuntu-26.04");p.add_argument("--node-path");p.add_argument("--plugin-root")
    a=p.parse_args();print(json.dumps(build_report(a.study_root,eth_study_root=a.eth_study_root,retained_study_root=a.retained_study_root,output_dir=a.output_dir,build_mode=a.build_mode,wsl_node=a.wsl_node,wsl_distro=a.wsl_distro,node_path=a.node_path,plugin_root=a.plugin_root),ensure_ascii=False))


if __name__=="__main__":main()
