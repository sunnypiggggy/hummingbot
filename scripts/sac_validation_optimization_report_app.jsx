import React,{useMemo,useState} from "react";
import {DataComponent,DataTable,Dropdown,EvidenceChart,ReportSection,RichNarrative,useDataApp} from "../../data-app-public.jsx";
import "./sac-validation-optimization.css";

const VARIANT={before:"原SAC",after:"验证选参后SAC",benchmark:"固定规则"},COST={base:"主成本",stress:"压力成本"};
const BASELINE={cash:"USDT现金",buy_hold:"95%买入持有",sma200_trend:"SMA200趋势",momentum180:"180日动量"};
const colors={原SAC:"#2563eb",验证选参后SAC:"#b45309"};
const finite=v=>typeof v==="number"&&Number.isFinite(v),pct=v=>finite(v)?`${v>0?"+":""}${(v*100).toFixed(2)}%`:"—",mag=v=>finite(v)?`${Math.abs(v*100).toFixed(2)}%`:"—",pp=v=>finite(v)?`${v>0?"+":""}${(v*100).toFixed(2)}个百分点`:"—";
const tone=v=><span className={v>0?"sac-positive":v<0?"sac-negative":""}>{pct(v)}</span>;

export function accountPairs(rows){
  const pairs=new Map();
  rows.forEach(r=>{
    if(!["BTCUSDT","ETHUSDT"].includes(r.symbol)||r.split!=="test"||!r.account_id||!r.policy_variant)throw new Error("Qualified actual annual account identity required");
    const key=`${r.symbol}:${r.policy_variant}:${r.algorithm}:${r.seed??"baseline"}`;
    if(!pairs.has(key))pairs.set(key,{key,symbol:r.symbol,policy_variant:r.policy_variant,algorithm:r.algorithm,seed:r.seed,candidate_id:r.candidate_id,model_origin:r.model_origin});
    const p=pairs.get(key);if(p[r.scenario])throw new Error("Duplicate independent account cost");p[r.scenario]=r;
  });
  return [...pairs.values()].map(p=>{if(!p.base||!p.stress)throw new Error("Both costs required");return {...p,label:p.seed===null?`${p.symbol} · ${BASELINE[p.algorithm]}`:`${p.symbol} · ${VARIANT[p.policy_variant]} · seed ${p.seed}`};});
}

export function trainingDisplayRows(rows){return rows.map(r=>({...r,key:r.model_id,stage_label:r.stage==="tuning"?"训练／验证候选":r.policy_variant==="before"?"原模型":r.model_origin==="retained_baseline_fallback"?"回退复用原模型":"选参后模型"}));}

function StatsTable({id,title,source}){
  const rows=source.map(r=>({...r,key:`${r.symbol}:${r.policy_variant}`,variant_label:VARIANT[r.policy_variant],profits:`${r.positive_seeds}/${r.seeds}`,losses:`${r.loss_seeds}/${r.seeds}`,iqr:`${pct(r.q1_return)} ～ ${pct(r.q3_return)}`}));
  return <DataComponent id={id} title={title} queryId="seed_statistics" sourceRows={source} displayRows={rows} kind="table"><DataTable rows={rows} rowKey="key" searchable={false} compactNumbers={false} caption="各币各版本预先固定seed 0—29，每个实际账户独立10,000 USDT" columns={[
    {field:"symbol",label:"交易对"},{field:"variant_label",label:"版本"},{field:"median_return",label:"扣费收益中位数",renderCell:tone},{field:"iqr",label:"中间50%收益"},{field:"p10_return",label:"收益P10",renderCell:tone},{field:"profits",label:"盈利种子"},{field:"losses",label:"亏损种子"},{field:"median_drawdown",label:"回撤中位数",renderCell:mag},{field:"p90_drawdown",label:"回撤P90",renderCell:mag},{field:"worst_drawdown",label:"最差种子回撤",renderCell:mag}
  ]}/></DataComponent>;
}

function PairTable({id,title,pairs,summary,choose}){
  const ids=new Set(pairs.flatMap(p=>[p.base.account_id,p.stress.account_id])),source=summary.filter(r=>ids.has(r.account_id));
  const rows=pairs.map(p=>({key:p.key,label:p.label,symbol:p.symbol,seed:p.seed,model_origin:p.model_origin,base_return:p.base.net_return,base_dd:Math.abs(p.base.max_drawdown),stress_return:p.stress.net_return,stress_dd:Math.abs(p.stress.max_drawdown)}));
  return <DataComponent id={id} title={title} queryId="account_summary" sourceRows={source} displayRows={rows} kind="table"><DataTable rows={rows} rowKey="key" searchable={pairs.length>8} compactNumbers={false} caption="两档成本分别按真实账户状态推理和成交；未对模型或资金求平均" columns={[
    {field:"label",label:"独立账户"},{field:"base_return",label:"主成本收益",renderCell:tone},{field:"base_dd",label:"主成本回撤",renderCell:mag},{field:"stress_return",label:"压力收益",renderCell:tone},{field:"stress_dd",label:"压力回撤",renderCell:mag},
    ...(choose?[{field:"key",label:"同币曲线",renderCell:(_,r)=><button className="sac-button" onClick={()=>choose(r.symbol,String(r.seed))}>查看此seed</button>}]:[])
  ]}/></DataComponent>;
}

export function ReportContent(){
  const {appTitle,setAppTitle,canEdit,mode,reviewedPeriodRows}=useDataApp(),query=id=>reviewedPeriodRows(id,{period:"history"});
  const summary=query("account_summary"),stats=query("seed_statistics"),pairs=query("matched_seed_results"),pairStats=query("matched_statistics"),selection=query("selection"),candidates=query("validation_candidates"),gates=query("validation_gates"),history=query("equity_history"),months=query("monthly_returns"),training=query("training_records"),audits=query("audit_evidence"),provenance=query("reference_provenance"),protocol=query("protocol")[0];
  const all=useMemo(()=>accountPairs(summary),[summary]);
  const [symbol,setSymbol]=useState("BTCUSDT"),[scenario,setScenario]=useState("base"),[seed,setSeed]=useState("0"),[metric,setMetric]=useState("net_return");
  const baseStats=stats.filter(r=>r.scenario==="base"),stressStats=stats.filter(r=>r.scenario==="stress");
  const scatterSource=summary.filter(r=>r.symbol===symbol&&r.scenario===scenario&&r.seed!==null&&["before","after"].includes(r.policy_variant));
  const scatterRows=scatterSource.map(r=>({...r,name:`${r.symbol} · ${VARIANT[r.policy_variant]} · seed ${r.seed}`,"版本":VARIANT[r.policy_variant],"最大回撤(%)":Math.abs(r.max_drawdown)*100,"扣费收益(%)":r.net_return*100}));
  const selectedPairs=all.filter(p=>p.symbol===symbol&&p.seed===Number(seed)),selectedIds=new Set(selectedPairs.map(p=>p[scenario].account_id)),curveSource=history.filter(r=>selectedIds.has(r.account_id)),plotMap=new Map();
  selectedPairs.forEach(p=>curveSource.filter(r=>r.account_id===p[scenario].account_id).forEach(r=>{if(!plotMap.has(r.timestamp))plotMap.set(r.timestamp,{timestamp:r.timestamp.replace(" ","T")});plotMap.get(r.timestamp)[`${VARIANT[p.policy_variant]}(%)`]=r[metric]*100;}));
  const curveRows=[...plotMap.values()].sort((a,b)=>a.timestamp.localeCompare(b.timestamp)),curveFields=selectedPairs.map(p=>`${VARIANT[p.policy_variant]}(%)`);
  const monthSource=months.filter(r=>selectedIds.has(r.account_id)),monthMap=new Map();monthSource.forEach(r=>{if(!monthMap.has(r.month))monthMap.set(r.month,{month:r.month});monthMap.get(r.month)[r.policy_variant]=r.return;});const monthRows=[...monthMap.values()].sort((a,b)=>a.month.localeCompare(b.month));
  const pairedRows=pairStats.map(r=>({...r,key:`${r.symbol}:${r.scenario}`,cost_label:COST[r.scenario]})),details=pairs.filter(r=>r.symbol===symbol&&r.scenario===scenario),selectionRows=selection.map(r=>({...r,key:r.symbol,result_label:r.fallback_to_baseline?"保留原参数，复用原30模型":"采用验证选定参数",recipe_label:`lr=${r.learning_rate}，gamma=${r.gamma}`}));
  const trainingRows=trainingDisplayRows(training),auditRows=audits.map(r=>({...r,key:r.audit_id})),gateRows=gates.map(r=>({...r,passed_label:r.candidate_id==="c00_baseline"?"原参数参考／不适用":r.passed?"通过":"未通过"}));
  const overview=["BTCUSDT","ETHUSDT"].map(s=>{const before=baseStats.find(r=>r.symbol===s&&r.policy_variant==="before"),after=baseStats.find(r=>r.symbol===s&&r.policy_variant==="after"),choice=selection.find(r=>r.symbol===s);return before&&after&&choice?`- **${s}**：原SAC收益中位数 ${pct(before.median_return)}，选参后 ${pct(after.median_return)}；回撤中位数由 ${mag(before.median_drawdown)} 到 ${mag(after.median_drawdown)}，盈利种子 ${before.positive_seeds}/30 到 ${after.positive_seeds}/30。${choice.fallback_to_baseline?"验证候选未达到预设接受条件，继续使用原参数和原模型。":`验证选定 ${choice.candidate_id}。`}`:"";}).join("\n");
  const choose=(s,n)=>{setSymbol(s);setSeed(n);};
  return <article className="report-content sac-optimization-report" aria-label="SAC验证选参前后独立账户比较">
    <header className="report-hero"><h1 data-data-app-title contentEditable={canEdit&&mode==="edit"} suppressContentEditableWarning onBlur={canEdit&&mode==="edit"?e=>setAppTitle(e.currentTarget.textContent?.trim()||appTitle):undefined}>{appTitle}</h1><RichNarrative id="sac-opt-deck" value="2025-10-23—2026-10-23 UTC · 365天 · BTC/ETH分别交易，每账户独立10,000 USDT"/></header>
    <ReportSection id="sac-opt-overview" title="Executive Summary" queryId="seed_statistics" sourceRows={stats} showHeading={false}><RichNarrative id="sac-opt-overview-copy" value={`## 结果概览\n\n${overview}\n\n参数选择只使用本轮训练与验证结果，年度结果未参与本轮选择。该年度行情此前已被研究观察；这不属于新的市场盲测。`}/></ReportSection>
    <StatsTable id="sac-opt-main" title="主成本：BTC/ETH原SAC与选参后SAC直接比较" source={baseStats}/>
    <StatsTable id="sac-opt-stress" title="压力成本：提高费用后的收益与风险" source={stressStats}/>
    <RichNarrative id="sac-opt-costs" className="sac-note" value="主成本单边手续费0.10%＋不利滑点0.02%；压力成本0.20%＋0.05%。收益已扣这两项。回撤按全部小时收盘计算，表中为跌幅；P10和四分位描述30个训练种子的结果，不是未来亏损概率。"/>
    <DataComponent id="sac-opt-selected" title="参数选择结果与回退状态" queryId="selection" sourceRows={selection} displayRows={selectionRows} kind="table"><DataTable rows={selectionRows} rowKey="key" searchable={false} compactNumbers={false} caption="seed100—104用于选参；seed0—29固定用于前后比较，不挑最好种子" columns={[{field:"symbol",label:"交易对"},{field:"candidate_id",label:"验证选择"},{field:"recipe_label",label:"学习率与gamma"},{field:"result_label",label:"执行结果"},{field:"frozen_at",label:"选择冻结时间UTC"}]}/></DataComponent>
    <DataComponent id="sac-opt-matched" title="同一seed的变化：选参后减原SAC" queryId="matched_statistics" sourceRows={pairStats} displayRows={pairedRows} kind="table"><DataTable rows={pairedRows} rowKey="key" searchable={false} compactNumbers={false} caption="收益差为正表示提高；回撤差为负表示回撤减小。每行分母固定30，全部保留" columns={[
      {field:"symbol",label:"交易对"},{field:"cost_label",label:"成本"},{field:"median_return_difference",label:"收益差中位数",renderCell:pp},{field:"median_drawdown_difference",label:"回撤差中位数",renderCell:pp},{field:"higher_return_seeds",label:"收益提高数"},{field:"lower_drawdown_seeds",label:"回撤降低数"},{field:"both_better_seeds",label:"两项都改善数"},{field:"unchanged_seeds",label:"两项均相同数"}
    ]}/></DataComponent>
    <section className="sac-exploration" aria-label="同币同seed真实账户比较"><div className="sac-controls"><Dropdown label="图表交易对" showLabel value={symbol} choices={["BTCUSDT","ETHUSDT"]} onChange={setSymbol}/><Dropdown label="图表与逐seed成本" showLabel value={scenario} choices={["base","stress"]} choiceLabels={COST} onChange={setScenario}/></div>
      <EvidenceChart id="sac-opt-risk-return" title={`${symbol} · ${COST[scenario]} · 30个seed前后各一份账户`} queryId="account_summary" sourceRows={scatterSource} rows={scatterRows} spec={{type:"scatter",x:"最大回撤(%)",y:"扣费收益(%)",series:"版本",valueDecimals:2,stackable:false,colors}} height={400}/>
      <DataComponent id="sac-opt-seed-details" title={`${symbol}：全部30个匹配seed`} queryId="matched_seed_results" sourceRows={details} displayRows={details} kind="table"><DataTable rows={details} rowKey="seed" searchable compactNumbers={false} caption={`${COST[scenario]} · 未按年度结果筛除任何seed`} columns={[{field:"seed",label:"seed"},{field:"before_return",label:"原SAC收益",renderCell:tone},{field:"before_drawdown",label:"原SAC回撤",renderCell:mag},{field:"after_return",label:"选参后收益",renderCell:tone},{field:"after_drawdown",label:"选参后回撤",renderCell:mag},{field:"return_difference",label:"收益差",renderCell:pp},{field:"drawdown_difference",label:"回撤差",renderCell:pp}]}/></DataComponent>
      <div className="sac-controls"><Dropdown label="实际曲线seed" showLabel value={seed} choices={Array.from({length:30},(_,i)=>String(i))} onChange={setSeed}/><Dropdown label="曲线指标" showLabel value={metric} choices={["net_return","drawdown"]} choiceLabels={{net_return:"累计扣费收益",drawdown:"相对小时峰值的日末回撤"}} onChange={setMetric}/></div>
      <EvidenceChart id="sac-opt-actual-curves" title={`${symbol} · seed ${seed} · ${COST[scenario]} · ${metric==="net_return"?"实际账户收益":"日末回撤"}`} queryId="equity_history" sourceRows={curveSource} rows={curveRows} spec={{type:"line",x:"timestamp",y:curveFields[0],fields:curveFields,valueDecimals:2,stackable:false,colors:{"原SAC(%)":colors.原SAC,"验证选参后SAC(%)":colors.验证选参后SAC}}} height={350}/>
      <RichNarrative id="sac-opt-curve-note" className="sac-note" value="每条曲线是一份同币真实账户；没有跨币、跨seed、模型权重平均或中位数权益曲线。图为日末采样，表中的最大回撤取全部小时收盘。若回退原参数，前后曲线重合属于复用原模型，不代表额外训练产生收益。"/>
    </section>
    <DataComponent id="sac-opt-months" title={`${symbol} · seed ${seed} 的月度收益`} queryId="monthly_returns" sourceRows={monthSource} displayRows={monthRows} kind="table"><DataTable rows={monthRows} rowKey="month" searchable={false} caption={`${COST[scenario]} · 首尾月份是实际部分区间`} columns={[{field:"month",label:"UTC月份"},{field:"before",label:"原SAC",renderCell:tone},{field:"after",label:"验证选参后SAC",renderCell:tone}]}/></DataComponent>
    <DataComponent id="sac-opt-candidates" title="完整验证候选：全部4组参数、两币分别比较" queryId="validation_candidates" sourceRows={candidates} displayRows={candidates} kind="table"><DataTable rows={candidates} rowKey="candidate_key" searchable={false} compactNumbers={false} caption="只用seed100—104的训练／验证结果；稳健评分=种子效用中位数−0.25×IQR" columns={[{field:"symbol",label:"交易对"},{field:"candidate_id",label:"候选"},{field:"learning_rate",label:"学习率",renderCell:v=>finite(v)?v.toExponential(0):"—"},{field:"gamma",label:"gamma",renderCell:v=>finite(v)?v.toFixed(3):"—"},{field:"robust_score",label:"验证稳健评分",renderCell:v=>finite(v)?v.toFixed(4):"—"},{field:"base_positive_seeds",label:"主成本盈利数/5"},{field:"stress_median_return",label:"压力收益中位数",renderCell:tone},{field:"stress_p90_drawdown",label:"压力回撤P90",renderCell:mag},{field:"selected",label:"最终采用",renderCell:v=>v?"是":"否"}]}/></DataComponent>
    <DataComponent id="sac-opt-gates" title="候选接受门槛：通过才替换原参数" queryId="validation_gates" sourceRows={gates} displayRows={gateRows} kind="table"><DataTable rows={gateRows} rowKey="gate_key" searchable={false} compactNumbers={false} caption="新候选需通过全部预设门；c00是原参数参考与兜底，不适用替换门槛" columns={[{field:"symbol",label:"交易对"},{field:"candidate_id",label:"候选"},{field:"gate_label",label:"门槛"},{field:"actual_text",label:"实际值"},{field:"threshold_text",label:"固定要求"},{field:"passed_label",label:"门槛判定"}]}/></DataComponent>
    <PairTable id="sac-opt-reference" title="原seed 0保留参考，不换成年度最好seed" pairs={all.filter(p=>p.seed===0)} summary={summary}/>
    <PairTable id="sac-opt-all-models" title="全部原SAC／选参后SAC实际账户" pairs={all.filter(p=>p.seed!==null)} summary={summary} choose={choose}/>
    <PairTable id="sac-opt-baselines" title="同币固定规则基准" pairs={all.filter(p=>p.seed===null)} summary={summary}/>
    <RichNarrative id="sac-opt-training-counts" className="sac-note" value={protocol?`原SAC与选参后共${protocol.before_after_model_records}条前后模型记录，另有${protocol.tuning_models}条候选调参记录。本轮实际新训练${protocol.new_training_models}份：候选调参${protocol.tuning_models}份，最终${protocol.fresh_final_models}份；最终另有${protocol.reused_final_models}份引用原权重。复用引用不计为不同的新模型。`:""}/>
    <DataComponent id="sac-opt-training" title="训练与复用记录：原模型、验证候选、最终模型分开" queryId="training_records" sourceRows={training} displayRows={trainingRows} kind="table"><DataTable rows={trainingRows} rowKey="key" caption="复用原模型不会计为新训练；所有新SAC训练固定32,768环境步" columns={[{field:"symbol",label:"交易对"},{field:"stage_label",label:"用途／来源"},{field:"candidate_id",label:"参数候选"},{field:"seed",label:"seed"},{field:"actual_steps",label:"原生环境步"},{field:"maximum_input_hour",label:"训练最大输入小时"}]}/></DataComponent>
    <DataComponent id="sac-opt-provenance" title="选择冻结与原模型、年度记录的引用证据" queryId="reference_provenance" sourceRows={provenance} displayRows={provenance} kind="table"><DataTable rows={provenance} rowKey="symbol" searchable={false} caption="年度记录仅在本轮选择冻结后导入；原研究及报告不覆盖" columns={[{field:"symbol",label:"交易对"},{field:"selected_candidate",label:"选择"},{field:"fallback_to_baseline",label:"回退原参数",renderCell:v=>v?"是":"否"},{field:"reused_final_models",label:"复用最终模型数"},{field:"fresh_final_models",label:"新增最终模型数"},{field:"annual_imported_at",label:"年度记录导入UTC"}]}/></DataComponent>
    <DataComponent id="sac-opt-audits" title="候选、选择与最终回测的核验" queryId="audit_evidence" sourceRows={audits} displayRows={auditRows} kind="table"><DataTable rows={auditRows} rowKey="key" caption="训练预算、验证选参来源与金融账本分别核验；全部通过后才生成本报告" columns={[{field:"stage",label:"阶段"},{field:"candidate_id",label:"候选"},{field:"audit_kind",label:"核验类型"},{field:"passed",label:"通过",renderCell:v=>v?"是":"否"},{field:"accounts_checked",label:"账户数"},{field:"failure_count",label:"失败数"}]}/></DataComponent>
    <ReportSection id="sac-opt-method" title="固定搜索范围与年度边界" queryId="protocol" sourceRows={query("protocol")} showHeading={false}><RichNarrative id="sac-opt-method-copy" value="## 固定搜索范围与年度边界\n\n本轮只比较4个预先固定的SAC参数：学习率3e−4／1e−4与gamma0.995／0.99的组合，其他网络、更新预算、环境、奖励、资金与成交规则保持原样。候选训练及验证用seed100—104；每币分别评分并按预设门槛选参数，年度对比固定使用seed0—29。未通过替换门槛就回退c00原参数，复用原模型与相同口径记录。\n\n稳健验证评分为五个种子两档成本效用的中位数减0.25倍IQR；效用为log(1+扣费收益)−0.5×回撤−2×超过20%的回撤。候选需至少提高0.01分、至少3/5种子效用更好、主成本至少3/5盈利、压力收益中位数非负、压力回撤P90至多比原候选高1个百分点，并在3个连续验证子窗口中至少2个不差。准确候选成绩、门槛与冻结凭证见上表和Sources。\n\n训练窗口2020-01-01至2025-01-01；验证至2025-10-23；年度窗口随后365天。该年度此前已被研究观察，本轮虽不以年度结果选参，也不是新的盲测。种子分布描述同一行情上的训练随机性，不代表30个独立市场样本。\n\n每账户独立10,000 USDT，只持本币与自身USDT，不跨币调仓或共享库存。目标币仓上限95%，现金目标至少5%，2.5个百分点调仓门槛；日频00UTC先卖、01UTC再买，逐腿扣费，全部小时收盘估值。主成本0.10%手续费＋0.02%不利滑点；压力成本0.20%＋0.05%。期末真实持仓按市值估值，不强制清算。研究不改变机器人配置。"/></ReportSection>
  </article>;
}
