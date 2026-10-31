import React, { useMemo, useState } from "react";
import { DataComponent, DataTable, Dropdown, EvidenceChart, ReportSection, RichNarrative, useDataApp } from "../../data-app-public.jsx";
import "./single-pair-rl-report.css";

const MODEL = {sac:"SAC",tqc:"TQC",cash:"USDT现金",buy_hold:"95%买入持有",sma200_trend:"SMA200趋势",momentum180:"180日动量"};
const COST = {base:"主成本",stress:"压力成本"};
const COLORS = ["#2563eb","#b45309","#7c3aed","#087f5b","#64748b"];
const finite = v => typeof v === "number" && Number.isFinite(v);
const pct = v => finite(v) ? `${v>0?"+":""}${(v*100).toFixed(2)}%` : "—";
const magnitude = v => finite(v) ? `${(Math.abs(v)*100).toFixed(2)}%` : "—";
const money = v => finite(v) ? v.toLocaleString("zh-CN",{minimumFractionDigits:2,maximumFractionDigits:2}) : "—";
const retCell = v => <span className={v>0?"single-positive":v<0?"single-negative":""}>{pct(v)}</span>;

export function accountPairs(rows) {
  const pairs=new Map();
  rows.forEach(row=>{
    if(row.split!=="test") throw new Error("Only real annual accounts may enter comparison");
    const key=`${row.symbol}:${row.algorithm}:${row.seed??"baseline"}`;
    if(!pairs.has(key)) pairs.set(key,{key,symbol:row.symbol,algorithm:row.algorithm,seed:row.seed});
    const pair=pairs.get(key);
    if(pair[row.scenario]) throw new Error("Duplicate independent account cost");
    pair[row.scenario]=row;
  });
  return [...pairs.values()].map(pair=>{
    if(!pair.base||!pair.stress) throw new Error("Both costs are required for every account");
    const cash=pair.base.fills===0&&pair.base.average_exposure===0;
    pair.label=`${pair.symbol} · ${MODEL[pair.algorithm]}${pair.seed===null?"":` · seed ${pair.seed}`}${cash&&pair.seed!==null?"（持现金）":""}`;
    return pair;
  });
}

export function pairDisplay(pairs,summary,includeSeedStats=false) {
  return pairs.map(p=>({key:p.key,label:p.label,symbol:p.symbol,algorithm:MODEL[p.algorithm],seed:p.seed,
    base_return:p.base.net_return,base_dd:Math.abs(p.base.max_drawdown),stress_return:p.stress.net_return,stress_dd:Math.abs(p.stress.max_drawdown),
    ...(includeSeedStats?{positive_seeds:summary.filter(r=>r.symbol===p.symbol&&r.algorithm===p.algorithm&&r.scenario==="base"&&r.seed!==null&&r.net_return>0).length}:{}),
    base_fees:p.base.fees,stress_fees:p.stress.fees,base_slippage:p.base.slippage_cost,stress_slippage:p.stress.slippage_cost,
    base_liquidation:p.base.net_liquidation_return,stress_liquidation:p.stress.net_liquidation_return,
    base_exposure:p.base.average_exposure,stress_exposure:p.stress.average_exposure}));
}

function accountScope(pairs,summary,includeSeeds=false) {
  const ids=new Set(pairs.flatMap(p=>[p.base.run_id,p.stress.run_id]));
  if(includeSeeds) pairs.filter(p=>p.seed!==null).forEach(p=>summary.filter(r=>r.symbol===p.symbol&&r.algorithm===p.algorithm&&r.scenario==="base"&&r.seed!==null).forEach(r=>ids.add(r.run_id)));
  return summary.filter(r=>ids.has(r.run_id));
}

function PairTable({id,title,pairs,summary,seeds=false,select,selected=[]}) {
  const rows=pairDisplay(pairs,summary,seeds);
  return <DataComponent id={id} title={title} queryId="account_summary" sourceRows={accountScope(pairs,summary,seeds)} displayRows={rows} kind="table">
    <DataTable rows={rows} rowKey="key" searchable={pairs.length>8} compactNumbers={false} caption={`每项独立10,000 USDT · ${pairs.length}项 · 回撤为跌幅，越小越好`} columns={[
      {field:"label",label:"独立账户",renderCell:v=><span className="single-account-name">{v}</span>},
      {field:"base_return",label:"主成本收益",renderCell:retCell},{field:"base_dd",label:"主成本回撤",renderCell:magnitude},
      {field:"stress_return",label:"压力收益",renderCell:retCell},{field:"stress_dd",label:"压力回撤",renderCell:magnitude},
      ...(seeds?[{field:"positive_seeds",label:"盈利种子",renderCell:v=>`${v}/5`}]:[]),
      ...(select?[{field:"key",label:"同币曲线",renderCell:(key,row)=><button className="single-line-button" aria-pressed={selected.includes(key)} aria-label={`选择${row.label}曲线`} onClick={()=>select(key)}>{selected.includes(key)?"已选":"加入"}</button>}]:[])
    ]}/>
  </DataComponent>;
}

export function ReportContent() {
  const {appTitle,setAppTitle,canEdit,mode,reviewedPeriodRows}=useDataApp();
  const query=id=>reviewedPeriodRows(id,{period:"history"});
  const summary=query("account_summary"),history=query("equity_history"),months=query("monthly_returns"),candidates=query("validation_selection"),training=query("training_records"),gates=query("research_targets"),audits=query("audit_evidence"),validation=query("validation_summary");
  const all=useMemo(()=>accountPairs(summary),[summary]);
  const [symbol,setSymbol]=useState("all"),[curveSymbol,setCurveSymbol]=useState("BTCUSDT"),[scenario,setScenario]=useState("base"),[metric,setMetric]=useState("net_return");
  const defaultKeys=s=>[`${s}:sac:0`,`${s}:tqc:0`,`${s}:buy_hold:baseline`];
  const [selected,setSelected]=useState(defaultKeys("BTCUSDT")),[monthlyKey,setMonthlyKey]=useState("BTCUSDT:sac:0");
  const onSymbol=next=>{setSymbol(next);if(next!=="all"){setCurveSymbol(next);setSelected(defaultKeys(next));setMonthlyKey(`${next}:sac:0`);}};
  const onCurveSymbol=next=>{setCurveSymbol(next);setSelected(defaultKeys(next));setMonthlyKey(`${next}:sac:0`);};
  const scope=r=>symbol==="all"||r.symbol===symbol;
  const scoped=all.filter(scope),primary=scoped.filter(p=>p.seed===0),singles=scoped.filter(p=>p.seed!==null),baselines=scoped.filter(p=>p.seed===null);
  const curveChoices=all.filter(p=>p.symbol===curveSymbol),chosen=curveChoices.filter(p=>selected.includes(p.key));
  const toggle=key=>{
    const pair=all.find(p=>p.key===key);
    if(pair.symbol!==curveSymbol){setCurveSymbol(pair.symbol);setSelected([key]);setMonthlyKey(key);return;}
    setSelected(old=>old.includes(key)?old.filter(x=>x!==key):[...old,key].slice(-5));
  };
  const ids=new Set(chosen.map(p=>p[scenario].run_id)),curveSource=history.filter(r=>ids.has(r.run_id)),plots=new Map();
  chosen.forEach(p=>curveSource.filter(r=>r.run_id===p[scenario].run_id).forEach(r=>{
    if(!plots.has(r.timestamp)) plots.set(r.timestamp,{timestamp:r.timestamp.replace(" ","T")});
    plots.get(r.timestamp)[`${p.label}(%)`]=finite(r[metric])?r[metric]*100:null;
  }));
  const fields=chosen.map(p=>`${p.label}(%)`),chartRows=[...plots.entries()].sort(([a],[b])=>a.localeCompare(b)).map(([,r])=>r);
  const monthlyPair=curveChoices.find(p=>p.key===monthlyKey)??curveChoices[0],monthSource=months.filter(r=>r.run_id===monthlyPair?.[scenario].run_id);
  const selectedCandidates=candidates.filter(scope);
  const selectedPrimary=primary.filter(p=>selectedCandidates.some(c=>c.selected&&c.symbol===p.symbol&&c.algorithm===p.algorithm));
  const sourcePrimary=accountScope(selectedPrimary,summary,true);
  const targetSources=[...sourcePrimary,...summary.filter(r=>scope(r)&&r.algorithm==="buy_hold"&&r.scenario==="base")];
  const overview=["BTCUSDT","ETHUSDT"].filter(s=>symbol==="all"||s===symbol).map(s=>{
    const choice=candidates.find(c=>c.symbol===s&&c.selected),p=all.find(p=>p.symbol===s&&p.algorithm===choice?.algorithm&&p.seed===0);
    const n=summary.filter(r=>r.symbol===s&&r.algorithm===choice?.algorithm&&r.seed!==null&&r.scenario==="base"&&r.net_return>0).length;
    return p?`- **${s}：验证选定 ${MODEL[p.algorithm]}，固定 seed 0。** 主成本收益 ${pct(p.base.net_return)}、最大回撤 ${magnitude(p.base.max_drawdown)}；压力收益 ${pct(p.stress.net_return)}、回撤 ${magnitude(p.stress.max_drawdown)}。该算法主成本 ${n}/5 种子盈利。`:"";
  }).join("\n");
  return <article className="report-content single-pair-rl-report" aria-label="独立单币交易回测">
    <header className="report-hero"><h1 data-data-app-title contentEditable={canEdit&&mode==="edit"} suppressContentEditableWarning onBlur={canEdit&&mode==="edit"?e=>setAppTitle(e.currentTarget.textContent?.trim()||appTitle):undefined}>{appTitle}</h1><RichNarrative id="single-deck" value="2025-10-23—2026-10-23 UTC · 365天 · BTCUSDT与ETHUSDT各自交易"/></header>
    <div className="single-controls"><Dropdown label="查看交易对" showLabel value={symbol} choices={["all","BTCUSDT","ETHUSDT"]} choiceLabels={{all:"两个交易对分别展示",BTCUSDT:"BTCUSDT",ETHUSDT:"ETHUSDT"}} onChange={onSymbol}/></div>
    <ReportSection id="single-overview" title="Executive Summary" queryId="account_summary" queryIds={["account_summary","validation_selection"]} sourceRowsByQuery={{account_summary:sourcePrimary,validation_selection:selectedCandidates}} showHeading={false}><RichNarrative id="single-overview-body" value={`## 结果概览\n\n${overview}\n\n每项都是独立的10,000 USDT账户。下表同时保留两种算法的事前主结果，不按年度收益更换种子。本年度此前已被研究观察，以上是已观察历史上的比较。`}/></ReportSection>
    <PairTable id="single-primary" title="固定 seed 0：两档成本的收益与回撤" pairs={primary} summary={summary} seeds select={toggle} selected={selected}/>
    <RichNarrative id="single-costs" className="single-fine-print" value="主成本：单边手续费0.10%＋不利滑点0.02%；压力成本：0.20%＋0.05%。收益已扣这两项，期末保留持仓市值。回撤按全部8,760个小时估值点计算。盈利种子分母固定5；五种子属于同一行情上的训练差异。"/>
    <section className="single-curves" aria-label="同一交易对的独立账户曲线">
      <div className="single-controls"><Dropdown label="曲线交易对" showLabel value={curveSymbol} choices={symbol==="all"?["BTCUSDT","ETHUSDT"]:[symbol]} onChange={onCurveSymbol}/><Dropdown label="曲线成本" showLabel value={scenario} choices={["base","stress"]} choiceLabels={COST} onChange={setScenario}/><Dropdown label="曲线指标" showLabel value={metric} choices={["net_return","drawdown"]} choiceLabels={{net_return:"累计扣费收益",drawdown:"相对小时峰值的日末回撤"}} onChange={setMetric}/></div>
      <div className="single-choice-list" aria-label="同币独立账户选择">{curveChoices.map(p=><button key={p.key} aria-pressed={selected.includes(p.key)} onClick={()=>toggle(p.key)}>{MODEL[p.algorithm]}{p.seed===null?"":` seed ${p.seed}`}</button>)}</div>
      {chartRows.length?<EvidenceChart id="single-curve" title={`${curveSymbol} · ${COST[scenario]} · ${metric==="net_return"?"累计扣费收益":"日末回撤"}`} queryId="equity_history" sourceRows={curveSource} rows={chartRows} spec={{type:"line",x:"timestamp",y:fields[0],fields,stackable:false,valueDecimals:2,colors:Object.fromEntries(fields.map((f,i)=>[f,COLORS[i]])),yLabel:metric==="net_return"?"累计收益（%）":"回撤（%）"}} height={360}/>:<RichNarrative id="single-curve-empty" value="尚未选择曲线。可从同币账户中选择，最多同时显示5项。"/>}
      <RichNarrative id="single-curve-note" className="single-fine-print" value="每条曲线来自一个真实独立账本；切换交易对后重新选择。图为日末采样，可能看不到小时内的最深回撤。未合成两币收益，也没有中位数或多模型平均交易曲线。"/>
    </section>
    <PairTable id="single-seeds" title="全部种子：保留每份盈利与亏损账户" pairs={singles} summary={summary} select={toggle} selected={selected}/>
    <RichNarrative id="single-seed-note" className="single-fine-print" value={`当前列出 **${singles.length}份单模型**，每币每算法seed 0—4各一份。表格可翻页查看，按训练前固定编号呈现；没有按测试收益筛除。`}/>
    <PairTable id="single-baselines" title="同币固定规则基准：成本和本金一致" pairs={baselines} summary={summary} select={toggle} selected={selected}/>
    <DataComponent id="single-details" title="所选同币账户：费用、仓位与净清算" queryId="account_summary" sourceRows={accountScope(chosen,summary)} displayRows={pairDisplay(chosen,summary)} kind="table"><DataTable rows={pairDisplay(chosen,summary)} rowKey="key" searchable={false} compactNumbers={false} caption="手续费与滑点分列；它们已包含在净收益中。净清算另扣期末假设退出成本。" columns={[{field:"label",label:"独立账户"},{field:"base_fees",label:"主手续费 USDT",renderCell:money},{field:"stress_fees",label:"压力手续费",renderCell:money},{field:"base_slippage",label:"主滑点成本",renderCell:money},{field:"stress_slippage",label:"压力滑点成本",renderCell:money},{field:"base_exposure",label:"主平均币仓",renderCell:magnitude},{field:"stress_exposure",label:"压力平均币仓",renderCell:magnitude},{field:"base_liquidation",label:"主净清算收益",renderCell:retCell},{field:"stress_liquidation",label:"压力净清算收益",renderCell:retCell}]}/></DataComponent>
    <DataComponent id="single-monthly" title="单账户月度扣费收益" queryId="monthly_returns" sourceRows={monthSource} displayRows={monthSource} kind="table"><DataTable rows={monthSource} rowKey="month" searchable={false} caption={`${monthlyPair?.label??""} · ${COST[scenario]} · 首尾月份为实际部分区间`} toolbarControls={<Dropdown label="月度账户" showLabel value={monthlyPair?.key??""} choices={curveChoices.map(p=>p.key)} choiceLabels={Object.fromEntries(curveChoices.map(p=>[p.key,p.label]))} onChange={setMonthlyKey}/>} columns={[{field:"month",label:"UTC月份"},{field:"return",label:"月度扣费收益",renderCell:retCell}]}/></DataComponent>
    <ReportSection id="single-selection-method" title="验证期选择详情" queryId="validation_selection" queryIds={["validation_selection","validation_summary"]} sourceRowsByQuery={{validation_selection:selectedCandidates,validation_summary:validation.filter(scope)}} showHeading={false}><RichNarrative id="single-selection-copy" value="## 验证只选择算法，seed 0在训练前固定\n\n验证期为2025-01-01至2025-10-23 UTC，共295天。每个种子先平均主／压力两档收益回撤评分，再取五种子评分中位数选择算法，同分按算法标识排序。中位评分不是实际交易账户，不对应一条收益曲线。年度不再挑选最好种子。"/></ReportSection>
    <DataComponent id="single-selection" title="每币两种算法的完整验证评分" queryId="validation_selection" sourceRows={selectedCandidates} displayRows={selectedCandidates} kind="table"><DataTable rows={selectedCandidates.map(r=>({...r,key:`${r.symbol}:${r.algorithm}`,decision:r.selected?"验证选定，主seed0":"保留比较"}))} rowKey="key" searchable={false} caption="评分越高越好；逐种子两档财务结果可在上一段来源中查看" columns={[{field:"symbol",label:"交易对"},{field:"algorithm",label:"算法"},{field:"decision",label:"事前选择"},{field:"score_median",label:"五种子中位评分",renderCell:v=>finite(v)?v.toFixed(5):"—"},...[0,1,2,3,4].map(s=>({field:`seed${s}_score`,label:`seed ${s}`,renderCell:v=>finite(v)?v.toFixed(4):"—"}))]}/></DataComponent>
    <DataComponent id="single-gates" title="验证选定算法：事前五项目标" queryId="research_targets" queryIds={["research_targets","account_summary"]} sourceRowsByQuery={{research_targets:gates.filter(scope),account_summary:targetSources}} displayRows={gates.filter(scope)} kind="table"><DataTable rows={gates.filter(scope).map(r=>({...r,key:`${r.symbol}:${r.gate}`}))} rowKey="key" searchable={false} caption="目标达成与账本审计通过分别判断；目标应用于固定seed0主账户" columns={[{field:"symbol",label:"交易对"},{field:"algorithm",label:"验证算法"},{field:"label",label:"研究目标"},{field:"passed",label:"达成",renderCell:v=>v?"是":"否"}]}/></DataComponent>
    <DataComponent id="single-training" title="20份新训练记录" queryId="training_records" sourceRows={training.filter(scope)} displayRows={training.filter(scope)} kind="table"><DataTable rows={training.filter(scope)} rowKey="record_file" caption="每份独立资本10,000 USDT；观测13维、动作1维；未复用旧组合权重" columns={[{field:"symbol",label:"交易对"},{field:"algorithm",label:"算法"},{field:"seed",label:"seed"},{field:"actual_steps",label:"实际环境步"},{field:"gradient_update_rounds",label:"梯度更新轮次"},{field:"maximum_input_hour",label:"训练最大输入小时"}]}/></DataComponent>
    <DataComponent id="single-audits" title="独立财务核验：两阶段各56账户及重复" queryId="audit_evidence" sourceRows={audits} displayRows={audits} kind="table"><DataTable rows={audits} rowKey="phase" searchable={false} caption="审计通过证明保存账本与独立算术重放一致，不保证收益或目标达成" columns={[{field:"phase",label:"阶段"},{field:"passed",label:"核验",renderCell:v=>v?"通过":"失败"},{field:"accounts_checked",label:"账户数"},{field:"repeats_checked",label:"重复数"},{field:"checks_total",label:"检查项"},{field:"failure_count",label:"失败数"}]}/></DataComponent>
    <ReportSection id="single-protocol" title="训练与成交口径" queryId="protocol" sourceRows={query("protocol")} showHeading={false}><RichNarrative id="single-protocol-copy" value="## 训练与成交口径\n\n训练区间为2020-01-01至2025-01-01 UTC，从完整32天片段抽样；不宣称每个种子遍历全部历史。所有本币特征仅由当时已完成数据计算。每模型32,768环境步，前512步收集经验，随后32,256轮实际更新。\n\n每个环境只能持本币和USDT，不读取另一币特征、持仓或资金。动作给出本币目标，上限95%、现金目标至少5%；2.5个百分点调仓门槛，完整退出可穿过门槛。日频决策，00UTC先卖、01UTC再买；逐腿收取研究手续费和不利滑点，全部小时收盘估值计算回撤。95%买入持有基准只在开始买入，随后不再平衡，币仓比例会漂移。\n\n每账户本金独立10,000 USDT，没有跨币总组合、收益加权或多模型平均。本轮仅为研究，不改变机器人配置或实盘权限。"/></ReportSection>
  </article>;
}
