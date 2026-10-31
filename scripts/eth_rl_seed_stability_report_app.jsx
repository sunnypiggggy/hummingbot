import React, { useMemo, useState } from "react";
import { DataComponent, DataTable, Dropdown, EvidenceChart, ReportSection, RichNarrative, useDataApp } from "../../data-app-public.jsx";
import "./eth-seed-stability.css";

const MODEL={sac:"SAC",td3:"TD3",cash:"USDT现金",buy_hold:"95%买入持有",sma200_trend:"SMA200趋势",momentum180:"180日动量"};
const COST={base:"主成本",stress:"压力成本"};
const COHORT={new25:"新增25个种子（seed 5—29）",all30:"全部30个种子（seed 0—29）"};
const finite=v=>typeof v==="number"&&Number.isFinite(v);
const pct=v=>finite(v)?`${v>0?"+":""}${(v*100).toFixed(2)}%`:"—";
const mag=v=>finite(v)?`${(Math.abs(v)*100).toFixed(2)}%`:"—";
const pp=v=>finite(v)?`${v>0?"+":""}${(v*100).toFixed(2)}个百分点`:"—";
const tone=v=><span className={v>0?"eth-seed-positive":v<0?"eth-seed-negative":""}>{pct(v)}</span>;
const colors={SAC:"#2563eb",TD3:"#b45309"};

export function accountPairs(rows){
  const pairs=new Map();
  rows.forEach(r=>{
    if(r.symbol!=="ETHUSDT"||r.split!=="test")throw new Error("Only actual ETH annual accounts may enter this report");
    const key=`${r.algorithm}:${r.seed??"baseline"}`;
    if(!pairs.has(key))pairs.set(key,{key,algorithm:r.algorithm,seed:r.seed});
    const p=pairs.get(key);if(p[r.scenario])throw new Error("Duplicate account cost");p[r.scenario]=r;
  });
  return [...pairs.values()].map(p=>{if(!p.base||!p.stress)throw new Error("Both costs required");return {...p,label:`${MODEL[p.algorithm]}${p.seed===null?"":` · seed ${p.seed}`}`,origin:p.seed===null?"固定规则":p.seed<5?"原5个种子":"新增25个种子"};});
}

function displayPairs(pairs){return pairs.map(p=>({key:p.key,label:p.label,origin:p.origin,algorithm:p.algorithm,seed:p.seed,
  base_return:p.base.net_return,base_dd:Math.abs(p.base.max_drawdown),stress_return:p.stress.net_return,stress_dd:Math.abs(p.stress.max_drawdown),
  base_fees:p.base.fees,stress_fees:p.stress.fees,base_fills:p.base.fills,stress_fills:p.stress.fills}));}
function PairTable({id,title,pairs,summary,choose,seed}){
  const ids=new Set(pairs.flatMap(p=>[p.base.run_id,p.stress.run_id])),rows=displayPairs(pairs);
  return <DataComponent id={id} title={title} queryId="account_summary" sourceRows={summary.filter(r=>ids.has(r.run_id))} displayRows={rows} kind="table"><DataTable rows={rows} rowKey="key" searchable={pairs.length>8} compactNumbers={false} caption={`${pairs.length}份独立10,000 USDT账户；两档成本分别推理和扣费`} columns={[
    {field:"label",label:"算法与种子"},{field:"origin",label:"来源"},{field:"base_return",label:"主成本收益",renderCell:tone},{field:"base_dd",label:"主成本回撤",renderCell:mag},
    {field:"stress_return",label:"压力收益",renderCell:tone},{field:"stress_dd",label:"压力回撤",renderCell:mag},
    ...(choose?[{field:"seed",label:"实际曲线",renderCell:s=><button className="eth-seed-button" aria-pressed={String(s)===seed} onClick={()=>choose(String(s))}>查看 seed {s}</button>}]:[])
  ]}/></DataComponent>;
}

export function ReportContent(){
  const {appTitle,setAppTitle,canEdit,mode,reviewedPeriodRows}=useDataApp();
  const query=id=>reviewedPeriodRows(id,{period:"history"});
  const summary=query("account_summary"),stats=query("cohort_statistics"),pairStats=query("paired_statistics"),pairResults=query("paired_seed_results"),history=query("equity_history"),months=query("monthly_returns"),training=query("training_records"),imports=query("training_import"),audits=query("audit_evidence");
  const all=useMemo(()=>accountPairs(summary),[summary]);
  const [cohort,setCohort]=useState("new25"),[scenario,setScenario]=useState("base"),[seed,setSeed]=useState("5"),[metric,setMetric]=useState("net_return");
  const inCohort=r=>r.seed!==null&&(cohort==="all30"||r.seed>=5);
  const scopedStats=stats.filter(r=>r.cohort===cohort),primaryStats=scopedStats.filter(r=>r.scenario==="base"),pressureStats=scopedStats.filter(r=>r.scenario==="stress"),scopedPairStats=pairStats.filter(r=>r.cohort===cohort),scatterSource=summary.filter(r=>inCohort(r)&&r.scenario===scenario),scatterRows=scatterSource.map(r=>({...r,"算法":MODEL[r.algorithm],"最大回撤(%)":Math.abs(r.max_drawdown)*100,"扣费收益(%)":r.net_return*100,seed_label:`${MODEL[r.algorithm]} · seed ${r.seed}`}));
  const reference=all.filter(p=>p.seed===0),models=all.filter(p=>p.seed!==null),baselines=all.filter(p=>p.seed===null);
  const selectedPairs=all.filter(p=>p.seed===Number(seed)),selectedIds=new Set(selectedPairs.map(p=>p[scenario].run_id)),curveSource=history.filter(r=>selectedIds.has(r.run_id)),plotMap=new Map();
  selectedPairs.forEach(p=>curveSource.filter(r=>r.run_id===p[scenario].run_id).forEach(r=>{if(!plotMap.has(r.timestamp))plotMap.set(r.timestamp,{timestamp:r.timestamp.replace(" ","T")});plotMap.get(r.timestamp)[`${MODEL[p.algorithm]}(%)`]=r[metric]*100;}));
  const curveRows=[...plotMap.values()].sort((a,b)=>a.timestamp.localeCompare(b.timestamp)),curveFields=selectedPairs.map(p=>`${MODEL[p.algorithm]}(%)`);
  const monthSource=months.filter(r=>selectedIds.has(r.run_id)),monthMap=new Map();monthSource.forEach(r=>{if(!monthMap.has(r.month))monthMap.set(r.month,{month:r.month});monthMap.get(r.month)[r.algorithm]=r.return;});
  const monthRows=[...monthMap.values()].sort((a,b)=>a.month.localeCompare(b.month));
  const statsRows=scopedStats.map(r=>({...r,key:`${r.algorithm}:${r.scenario}`,algorithm_label:MODEL[r.algorithm],cost_label:COST[r.scenario],profits:`${r.positive_seeds}/${r.seeds}`,losses:`${r.loss_seeds}/${r.seeds}`,iqr:`${pct(r.q1_return)} ～ ${pct(r.q3_return)}`,range:`${pct(r.min_return)} ～ ${pct(r.max_return)}`}));
  const get=(rows,a)=>rows.find(r=>r.algorithm===a),sac=get(primaryStats,"sac"),td3=get(primaryStats,"td3"),psac=get(pressureStats,"sac"),ptd3=get(pressureStats,"td3");
  const overview=sac&&td3?`## 结果概览\n\n- **${COHORT[cohort]}**：SAC扣费收益中位数 ${pct(sac.median_return)}，${sac.positive_seeds}/${sac.seeds}个种子盈利；TD3中位数 ${pct(td3.median_return)}，${td3.positive_seeds}/${td3.seeds}个盈利。\n- **最大回撤的种子中位数**：SAC ${mag(sac.median_drawdown)}，TD3 ${mag(td3.median_drawdown)}；最差种子分别为 ${mag(sac.worst_drawdown)}、${mag(td3.worst_drawdown)}。\n- **压力成本**：SAC收益中位数 ${pct(psac.median_return)}，亏损 ${psac.loss_seeds}/${psac.seeds}；TD3中位数 ${pct(ptd3.median_return)}，亏损 ${ptd3.loss_seeds}/${ptd3.seeds}。`:"";
  return <article className="report-content eth-seed-stability-report" aria-label="ETH强化学习种子稳定性比较">
    <header className="report-hero"><h1 data-data-app-title contentEditable={canEdit&&mode==="edit"} suppressContentEditableWarning onBlur={canEdit&&mode==="edit"?e=>setAppTitle(e.currentTarget.textContent?.trim()||appTitle):undefined}>{appTitle}</h1><RichNarrative id="eth-seed-deck" value="2025-10-23—2026-10-23 UTC · 365天 · 每个ETHUSDT账户独立10,000 USDT"/></header>
    <div className="eth-seed-controls"><Dropdown label="统计种子范围" showLabel value={cohort} choices={["new25","all30"]} choiceLabels={COHORT} onChange={setCohort}/></div>
    <ReportSection id="eth-seed-overview" title="Executive Summary" queryId="cohort_statistics" sourceRows={scopedStats} showHeading={false}><RichNarrative id="eth-seed-overview-copy" value={overview}/></ReportSection>
    <RichNarrative id="eth-seed-population" className="eth-seed-note" value="默认先看新增25个种子，再切换至包含原5个种子的全部30个。原seed 0—4的10份模型原样保留，seed 5—29新增50份；种子集合在本轮训练前固定。新增种子检验同一段行情上的训练差异，这个年度已被研究观察。"/>
    <DataComponent id="eth-seed-return-distribution" title="收益分布：中位数、四分位和亏损种子" queryId="cohort_statistics" sourceRows={scopedStats} displayRows={statsRows} kind="table"><DataTable rows={statsRows} rowKey="key" searchable={false} compactNumbers={false} caption={`${COHORT[cohort]} · 每行分母为该算法全部预设种子；分位数使用linear插值`} columns={[
      {field:"algorithm_label",label:"算法"},{field:"cost_label",label:"成本"},{field:"median_return",label:"收益中位数",renderCell:tone},{field:"iqr",label:"中间50%种子收益"},{field:"p10_return",label:"收益P10",renderCell:tone},{field:"p90_return",label:"收益P90",renderCell:tone},{field:"range",label:"最差～最好收益"},{field:"profits",label:"盈利种子"},{field:"losses",label:"亏损种子"}
    ]}/></DataComponent>
    <DataComponent id="eth-seed-risk-distribution" title="回撤分布：典型种子与尾部风险" queryId="cohort_statistics" sourceRows={scopedStats} displayRows={statsRows} kind="table"><DataTable rows={statsRows} rowKey="key" searchable={false} compactNumbers={false} caption="最大回撤为跌幅绝对值；P90表示90%的种子不超过该回撤" columns={[
      {field:"algorithm_label",label:"算法"},{field:"cost_label",label:"成本"},{field:"median_drawdown",label:"回撤中位数",renderCell:mag},{field:"p90_drawdown",label:"回撤P90",renderCell:mag},{field:"worst_drawdown",label:"最差种子回撤",renderCell:mag},{field:"drawdown_above_20_seeds",label:"回撤超过20%的种子"}
    ]}/></DataComponent>
    <div className="eth-seed-controls"><Dropdown label="图表与同seed比较成本" showLabel value={scenario} choices={["base","stress"]} choiceLabels={COST} onChange={setScenario}/></div>
    <EvidenceChart id="eth-seed-risk-return" title={`${COHORT[cohort]} · ${COST[scenario]} · 每个点是一份实际账户`} queryId="account_summary" sourceRows={scatterSource} rows={scatterRows} spec={{type:"scatter",x:"最大回撤(%)",y:"扣费收益(%)",series:"算法",labelField:"seed_label",valueDecimals:2,stackable:false,colors}} height={400}/>
    <RichNarrative id="eth-seed-scatter-note" className="eth-seed-note" value="散点越靠左表示回撤越小，越靠上表示收益越高。每个点对应一份独立模型的真实账户；没有对权重、收益或权益曲线求平均。"/>
    <DataComponent id="eth-seed-paired-summary" title="同编号种子的差值：TD3减SAC" queryId="paired_statistics" sourceRows={scopedPairStats} displayRows={scopedPairStats} kind="table"><DataTable rows={scopedPairStats.map(r=>({...r,key:r.scenario,cost_label:COST[r.scenario]}))} rowKey="key" searchable={false} compactNumbers={false} caption="正收益差表示TD3较高；负回撤差表示TD3回撤较小。同编号不是独立市场配对试验" columns={[
      {field:"cost_label",label:"成本"},{field:"median_return_difference",label:"收益差中位数",renderCell:pp},{field:"median_drawdown_difference",label:"回撤差中位数",renderCell:pp},{field:"td3_higher_return_seeds",label:"TD3收益更高数"},{field:"td3_lower_drawdown_seeds",label:"TD3回撤更小数"},{field:"td3_both_better_seeds",label:"TD3两项都较好数"},{field:"seeds",label:"分母"}
    ]}/></DataComponent>
    <DataComponent id="eth-seed-paired-details" title="逐个种子直接比较" queryId="paired_seed_results" sourceRows={pairResults.filter(r=>(cohort==="all30"||r.seed>=5)&&r.scenario===scenario)} kind="table"><DataTable rows={pairResults.filter(r=>(cohort==="all30"||r.seed>=5)&&r.scenario===scenario)} rowKey="seed" searchable compactNumbers={false} caption={`${COHORT[cohort]} · ${COST[scenario]}；所有种子按编号保留`} columns={[
      {field:"seed",label:"seed"},{field:"sac_return",label:"SAC收益",renderCell:tone},{field:"sac_drawdown",label:"SAC回撤",renderCell:mag},{field:"td3_return",label:"TD3收益",renderCell:tone},{field:"td3_drawdown",label:"TD3回撤",renderCell:mag},{field:"return_difference",label:"TD3−SAC收益差",renderCell:pp}
    ]}/></DataComponent>
    <section className="eth-seed-curves" aria-label="实际单模型账户曲线">
      <div className="eth-seed-controls"><Dropdown label="曲线seed" showLabel value={seed} choices={Array.from({length:30},(_,i)=>String(i))} onChange={setSeed}/><Dropdown label="曲线指标" showLabel value={metric} choices={["net_return","drawdown"]} choiceLabels={{net_return:"累计扣费收益",drawdown:"相对小时峰值的日末回撤"}} onChange={setMetric}/></div>
      <EvidenceChart id="eth-seed-actual-curves" title={`ETHUSDT · seed ${seed} · ${COST[scenario]} · ${metric==="net_return"?"累计扣费收益":"日末回撤"}`} queryId="equity_history" sourceRows={curveSource} rows={curveRows} spec={{type:"line",x:"timestamp",y:curveFields[0],fields:curveFields,valueDecimals:2,stackable:false,colors:{"SAC(%)":colors.SAC,"TD3(%)":colors.TD3},yLabel:metric==="net_return"?"累计扣费收益（%）":"回撤（%）"}} height={350}/>
      <RichNarrative id="eth-seed-curve-note" className="eth-seed-note" value="这里只画选定编号的两份实际账户。图为日末采样，表中的最大回撤仍取全部小时收盘，可能比图中最深点更大。"/>
    </section>
    <DataComponent id="eth-seed-monthly" title={`seed ${seed} 的月度扣费收益`} queryId="monthly_returns" sourceRows={monthSource} displayRows={monthRows} kind="table"><DataTable rows={monthRows} rowKey="month" searchable={false} caption={`${COST[scenario]} · 首尾月份为实际部分区间`} columns={[{field:"month",label:"UTC月份"},{field:"sac",label:"SAC",renderCell:tone},{field:"td3",label:"TD3",renderCell:tone}]}/></DataComponent>
    <PairTable id="eth-seed-reference" title="原seed 0保留参考，不替换为本轮最好种子" pairs={reference} summary={summary}/>
    <PairTable id="eth-seed-all-models" title="全部60份模型：两档成本结果完整保留" pairs={models} summary={summary} choose={setSeed} seed={seed}/>
    <PairTable id="eth-seed-baselines" title="同币固定规则基准" pairs={baselines} summary={summary}/>
    <DataComponent id="eth-seed-training" title="训练记录：10份保留与50份新增" queryId="training_records" queryIds={["training_records","training_import"]} sourceRowsByQuery={{training_records:training,training_import:imports}} displayRows={training} kind="table"><DataTable rows={training.map(r=>({...r,key:`${r.algorithm}:${r.seed}`,algorithm_label:MODEL[r.algorithm],origin_label:r.origin==="retained"?"原5个种子":"新增25个种子"}))} rowKey="key" caption="原模型逐字节保留；TD3每模型65,536环境步，SAC32,768步" columns={[{field:"algorithm_label",label:"算法"},{field:"seed",label:"seed"},{field:"origin_label",label:"来源"},{field:"actual_steps",label:"实际环境步"},{field:"maximum_input_hour",label:"训练最大输入小时"}]}/></DataComponent>
    <DataComponent id="eth-seed-audits" title="两阶段128账户及重复执行的独立财务核验" queryId="audit_evidence" sourceRows={audits} kind="table"><DataTable rows={audits} rowKey="phase" searchable={false} caption="账本核验与盈利稳定性分别判断" columns={[{field:"phase",label:"阶段"},{field:"passed",label:"核验",renderCell:v=>v?"通过":"失败"},{field:"accounts_checked",label:"账户数"},{field:"repeats_checked",label:"重复数"},{field:"checks_total",label:"检查项"},{field:"failure_count",label:"失败数"}]}/></DataComponent>
    <ReportSection id="eth-seed-method" title="统计与交易口径" queryId="protocol" sourceRows={query("protocol")} showHeading={false}><RichNarrative id="eth-seed-method-copy" value="## 统计与交易口径\n\n种子范围在本轮训练前固定为0—29，原0—4沿用，新增5—29全部保留。分位数按linear插值。盈利为扣费收益严格大于0，亏损为小于0，持平另计；不同种子反映同一条市场路径上的训练差异，不能视作30个独立市场样本。\n\n资金、数据、观测、网络、奖励、手续费、滑点、调仓和训练步数均沿用原研究。训练至2025-01-01，验证至2025-10-23，年度回测为之后365天。TD3原配方训练步数是SAC的两倍，算法比较不属于等计算预算实验。验证期保留两算法完整比较，不按年度结果挑选种子或更换主模型。\n\n每账户只持ETH与自身USDT：目标币仓上限95%，现金目标至少5%；2.5个百分点调仓门槛。日频决策，00UTC先卖、01UTC再买，逐腿扣费，全部小时收盘估值。主成本单边手续费0.10%与不利滑点0.02%；压力成本0.20%与0.05%。期末真实持仓按市值计入收益，另保留净清算值。研究不改变实盘配置。"/></ReportSection>
  </article>;
}
