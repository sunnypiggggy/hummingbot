import React,{useMemo,useState} from "react";
import {DataComponent,DataTable,Dropdown,EvidenceChart,ReportSection,RichNarrative,useDataApp} from "../../data-app-public.jsx";
import "./btceth-seed-stability.css";

const MODEL={sac:"SAC",td3:"TD3",cash:"USDT现金",buy_hold:"95%买入持有",sma200_trend:"SMA200趋势",momentum180:"180日动量"};
const COST={base:"主成本",stress:"压力成本"},COHORT={all30:"全部30个种子（seed 0—29）",new25:"新增25个种子（seed 5—29）"};
const colors={SAC:"#2563eb",TD3:"#b45309"};
const finite=v=>typeof v==="number"&&Number.isFinite(v),pct=v=>finite(v)?`${v>0?"+":""}${(v*100).toFixed(2)}%`:"—",mag=v=>finite(v)?`${(Math.abs(v)*100).toFixed(2)}%`:"—",pp=v=>finite(v)?`${v>0?"+":""}${(v*100).toFixed(2)}个百分点`:"—";
const tone=v=><span className={v>0?"seed-positive":v<0?"seed-negative":""}>{pct(v)}</span>;

export function accountPairs(rows){
  const pairs=new Map();
  rows.forEach(r=>{
    if(!["BTCUSDT","ETHUSDT"].includes(r.symbol)||r.split!=="test")throw new Error("Only actual independent annual BTC/ETH accounts may enter this report");
    const key=`${r.symbol}:${r.algorithm}:${r.seed??"baseline"}`;
    if(!pairs.has(key))pairs.set(key,{key,symbol:r.symbol,algorithm:r.algorithm,seed:r.seed});
    const p=pairs.get(key);if(p[r.scenario])throw new Error("Duplicate account cost");p[r.scenario]=r;
  });
  return [...pairs.values()].map(p=>{if(!p.base||!p.stress)throw new Error("Both costs required");return {...p,label:`${p.symbol} · ${MODEL[p.algorithm]}${p.seed===null?"":` · seed ${p.seed}`}`,origin:p.seed===null?"固定规则":p.seed<5?"原5个种子":"新增25个种子"};});
}

export function trainingDisplayRows(training){
  return training.map(r=>({...r,key:`${r.symbol}:${r.algorithm}:${r.seed}`,algorithm_label:MODEL[r.algorithm],origin_label:r.origin==="retained"?"原5个种子":"新增25个种子"}));
}

function StatsTable({id,title,source,cohort}){
  const rows=source.map(r=>({...r,key:`${r.symbol}:${r.algorithm}`,algorithm_label:MODEL[r.algorithm],profits:`${r.positive_seeds}/${r.seeds}`,losses:`${r.loss_seeds}/${r.seeds}`}));
  return <DataComponent id={id} title={title} queryId="cohort_statistics" sourceRows={source} displayRows={rows} kind="table"><DataTable rows={rows} rowKey="key" searchable={false} compactNumbers={false} caption={`${COHORT[cohort]} · 收益与回撤各自在本币、本算法内统计；未合成跨币组合`} columns={[
    {field:"symbol",label:"交易对"},{field:"algorithm_label",label:"算法"},{field:"median_return",label:"扣费收益中位数",renderCell:tone},{field:"p10_return",label:"收益P10",renderCell:tone},
    {field:"profits",label:"盈利种子"},{field:"losses",label:"亏损种子"},{field:"median_drawdown",label:"回撤中位数",renderCell:mag},{field:"worst_drawdown",label:"最差种子回撤",renderCell:mag}
  ]}/></DataComponent>;
}

function PairTable({id,title,pairs,summary,choose}){
  const ids=new Set(pairs.flatMap(p=>[p.base.run_id,p.stress.run_id])),source=summary.filter(r=>ids.has(r.run_id));
  const rows=pairs.map(p=>({key:p.key,label:p.label,symbol:p.symbol,seed:p.seed,origin:p.origin,base_return:p.base.net_return,base_dd:Math.abs(p.base.max_drawdown),stress_return:p.stress.net_return,stress_dd:Math.abs(p.stress.max_drawdown)}));
  return <DataComponent id={id} title={title} queryId="account_summary" sourceRows={source} displayRows={rows} kind="table"><DataTable rows={rows} rowKey="key" searchable={pairs.length>8} compactNumbers={false} caption={`${pairs.length}份模型或基准 · 每个实际成本账户独立10,000 USDT`} columns={[
    {field:"label",label:"独立账户"},{field:"origin",label:"来源"},{field:"base_return",label:"主成本收益",renderCell:tone},{field:"base_dd",label:"主成本回撤",renderCell:mag},{field:"stress_return",label:"压力收益",renderCell:tone},{field:"stress_dd",label:"压力回撤",renderCell:mag},
    ...(choose?[{field:"key",label:"同币曲线",renderCell:(_,row)=><button className="seed-button" onClick={()=>choose(row.symbol,String(row.seed))}>查看此seed</button>}]:[])
  ]}/></DataComponent>;
}

export function ReportContent(){
  const {appTitle,setAppTitle,canEdit,mode,reviewedPeriodRows}=useDataApp(),query=id=>reviewedPeriodRows(id,{period:"history"});
  const summary=query("account_summary"),stats=query("cohort_statistics"),pairStats=query("paired_statistics"),pairResults=query("paired_seed_results"),history=query("equity_history"),months=query("monthly_returns"),training=query("training_records"),imports=query("training_import"),audits=query("audit_evidence");
  const all=useMemo(()=>accountPairs(summary),[summary]);
  const [cohort,setCohort]=useState("all30"),[symbol,setSymbol]=useState("BTCUSDT"),[scenario,setScenario]=useState("base"),[seed,setSeed]=useState("0"),[metric,setMetric]=useState("net_return");
  const inCohort=r=>r.seed!==null&&(cohort==="all30"||r.seed>=5),scopedStats=stats.filter(r=>r.cohort===cohort),baseStats=scopedStats.filter(r=>r.scenario==="base"),stressStats=scopedStats.filter(r=>r.scenario==="stress");
  const scopedPairStats=pairStats.filter(r=>r.cohort===cohort),pairStatsRows=scopedPairStats.map(r=>({...r,key:`${r.symbol}:${r.scenario}`,cost_label:COST[r.scenario]}));
  const detailStats=scopedStats.map(r=>({...r,key:`${r.symbol}:${r.algorithm}:${r.scenario}`,algorithm_label:MODEL[r.algorithm],cost_label:COST[r.scenario],iqr:`${pct(r.q1_return)} ～ ${pct(r.q3_return)}`}));
  const scatterSource=summary.filter(r=>r.symbol===symbol&&inCohort(r)&&r.scenario===scenario),scatterRows=scatterSource.map(r=>({...r,name:`${r.symbol} · ${MODEL[r.algorithm]} · seed ${r.seed}`,"算法":MODEL[r.algorithm],"最大回撤(%)":Math.abs(r.max_drawdown)*100,"扣费收益(%)":r.net_return*100}));
  const sameSeedSource=pairResults.filter(r=>r.symbol===symbol&&(cohort==="all30"||r.seed>=5)&&r.scenario===scenario);
  const selectedPairs=all.filter(p=>p.symbol===symbol&&p.seed===Number(seed)),selectedIds=new Set(selectedPairs.map(p=>p[scenario].run_id)),curveSource=history.filter(r=>selectedIds.has(r.run_id)),plotMap=new Map();
  selectedPairs.forEach(p=>curveSource.filter(r=>r.run_id===p[scenario].run_id).forEach(r=>{if(!plotMap.has(r.timestamp))plotMap.set(r.timestamp,{timestamp:r.timestamp.replace(" ","T")});plotMap.get(r.timestamp)[`${MODEL[p.algorithm]}(%)`]=r[metric]*100;}));
  const curveRows=[...plotMap.values()].sort((a,b)=>a.timestamp.localeCompare(b.timestamp)),curveFields=selectedPairs.map(p=>`${MODEL[p.algorithm]}(%)`);
  const monthSource=months.filter(r=>selectedIds.has(r.run_id)),monthMap=new Map();monthSource.forEach(r=>{if(!monthMap.has(r.month))monthMap.set(r.month,{month:r.month});monthMap.get(r.month)[r.algorithm]=r.return;});
  const monthRows=[...monthMap.values()].sort((a,b)=>a.month.localeCompare(b.month));
  const reference=all.filter(p=>p.seed===0),models=all.filter(p=>p.seed!==null),baselines=all.filter(p=>p.seed===null);
  const trainingRows=trainingDisplayRows(training);
  const overview=["BTCUSDT","ETHUSDT"].map(s=>{const sac=baseStats.find(r=>r.symbol===s&&r.algorithm==="sac"),td3=baseStats.find(r=>r.symbol===s&&r.algorithm==="td3");return sac&&td3?`- **${s}**：SAC扣费收益中位数 ${pct(sac.median_return)}，${sac.positive_seeds}/${sac.seeds}个种子盈利；TD3中位数 ${pct(td3.median_return)}，${td3.positive_seeds}/${td3.seeds}个盈利。回撤中位数分别为 ${mag(sac.median_drawdown)}、${mag(td3.median_drawdown)}。`:"";}).join("\n");
  const choose=(s,nextSeed)=>{setSymbol(s);setSeed(nextSeed);};
  return <article className="report-content btceth-seed-stability-report" aria-label="BTC与ETH分别独立交易的种子稳定性比较">
    <header className="report-hero"><h1 data-data-app-title contentEditable={canEdit&&mode==="edit"} suppressContentEditableWarning onBlur={canEdit&&mode==="edit"?e=>setAppTitle(e.currentTarget.textContent?.trim()||appTitle):undefined}>{appTitle}</h1><RichNarrative id="seed-deck" value="2025-10-23—2026-10-23 UTC · 365天 · BTC与ETH各自交易，每个账户独立10,000 USDT"/></header>
    <div className="seed-controls"><Dropdown label="统计种子范围" showLabel value={cohort} choices={["all30","new25"]} choiceLabels={COHORT} onChange={setCohort}/></div>
    <ReportSection id="seed-overview" title="Executive Summary" queryId="cohort_statistics" sourceRows={scopedStats} showHeading={false}><RichNarrative id="seed-overview-copy" value={`## 结果概览\n\n${overview}\n\n当前比较 **${COHORT[cohort]}**。下表把两个交易对并列，压力成本另列；各币没有共享资金或合成收益。`}/></ReportSection>
    <StatsTable id="seed-main-comparison" title="主成本：BTC与ETH直接并列比较" source={baseStats} cohort={cohort}/>
    <StatsTable id="seed-stress-comparison" title="压力成本：提高费用后的收益与回撤" source={stressStats} cohort={cohort}/>
    <RichNarrative id="seed-cost-note" className="seed-note" value="主成本单边手续费0.10%＋不利滑点0.02%；压力成本0.20%＋0.05%。收益已扣两项成本。P10为收益的第10百分位，不是未来亏损概率；表中回撤为跌幅，越小越好。"/>
    <DataComponent id="seed-return-tail" title="收益区间与回撤尾部：保留最差种子" queryId="cohort_statistics" sourceRows={scopedStats} displayRows={detailStats} kind="table"><DataTable rows={detailStats} rowKey="key" searchable={false} compactNumbers={false} caption="收益四分位与全部范围均在各币、各算法、各成本内计算；linear分位数插值" columns={[
      {field:"symbol",label:"交易对"},{field:"algorithm_label",label:"算法"},{field:"cost_label",label:"成本"},{field:"iqr",label:"中间50%种子收益"},{field:"min_return",label:"最差收益",renderCell:tone},{field:"max_return",label:"最好收益",renderCell:tone},{field:"p90_return",label:"收益P90",renderCell:tone},{field:"p90_drawdown",label:"回撤P90",renderCell:mag},{field:"drawdown_above_20_seeds",label:"回撤超过20%数"}
    ]}/></DataComponent>
    <RichNarrative id="seed-population-note" className="seed-note" value="BTC与ETH各60份模型：各自原seed 0—4的10份模型逐字节保留，seed 5—29新增50份。全部120份模型分别回测两档成本；种子集合在每轮训练前固定，亏损和持平结果全部保留。原ETH研究与报告保持不变。"/>
    <DataComponent id="seed-paired-summary" title="同编号种子的差值：TD3减SAC，各币分别计算" queryId="paired_statistics" sourceRows={scopedPairStats} displayRows={pairStatsRows} kind="table"><DataTable rows={pairStatsRows} rowKey="key" searchable={false} compactNumbers={false} caption="正收益差表示TD3较高；负回撤差表示TD3回撤较小。同编号不是独立市场配对试验" columns={[
      {field:"symbol",label:"交易对"},{field:"cost_label",label:"成本"},{field:"median_return_difference",label:"收益差中位数",renderCell:pp},{field:"median_drawdown_difference",label:"回撤差中位数",renderCell:pp},{field:"td3_higher_return_seeds",label:"TD3收益更高数"},{field:"td3_lower_drawdown_seeds",label:"TD3回撤更小数"},{field:"td3_both_better_seeds",label:"TD3两项都较好数"},{field:"seeds",label:"分母"}
    ]}/></DataComponent>
    <section className="seed-exploration" aria-label="同一交易对的账户分布与实际曲线">
      <div className="seed-controls"><Dropdown label="图表交易对" showLabel value={symbol} choices={["BTCUSDT","ETHUSDT"]} onChange={setSymbol}/><Dropdown label="图表与逐seed成本" showLabel value={scenario} choices={["base","stress"]} choiceLabels={COST} onChange={setScenario}/></div>
      <EvidenceChart id="seed-risk-return" title={`${symbol} · ${COHORT[cohort]} · ${COST[scenario]} · 每个点是一份实际账户`} queryId="account_summary" sourceRows={scatterSource} rows={scatterRows} spec={{type:"scatter",x:"最大回撤(%)",y:"扣费收益(%)",series:"算法",valueDecimals:2,stackable:false,colors}} height={400}/>
      <RichNarrative id="seed-scatter-note" className="seed-note" value="越靠左回撤越小，越靠上收益越高。散点只比较选定交易对，每个点是一份实际模型账户；没有跨币平均、平均模型目标或中位数交易曲线。"/>
      <DataComponent id="seed-paired-details" title={`${symbol}：逐个种子直接比较`} queryId="paired_seed_results" sourceRows={sameSeedSource} displayRows={sameSeedSource} kind="table"><DataTable rows={sameSeedSource} rowKey="seed" searchable compactNumbers={false} caption={`${COHORT[cohort]} · ${COST[scenario]}；所有种子按编号保留`} columns={[
        {field:"seed",label:"seed"},{field:"sac_return",label:"SAC收益",renderCell:tone},{field:"sac_drawdown",label:"SAC回撤",renderCell:mag},{field:"td3_return",label:"TD3收益",renderCell:tone},{field:"td3_drawdown",label:"TD3回撤",renderCell:mag},{field:"return_difference",label:"TD3−SAC收益差",renderCell:pp}
      ]}/></DataComponent>
      <div className="seed-controls"><Dropdown label="实际曲线seed" showLabel value={seed} choices={Array.from({length:30},(_,i)=>String(i))} onChange={setSeed}/><Dropdown label="曲线指标" showLabel value={metric} choices={["net_return","drawdown"]} choiceLabels={{net_return:"累计扣费收益",drawdown:"相对小时峰值的日末回撤"}} onChange={setMetric}/></div>
      <EvidenceChart id="seed-actual-curves" title={`${symbol} · seed ${seed} · ${COST[scenario]} · ${metric==="net_return"?"累计扣费收益":"日末回撤"}`} queryId="equity_history" sourceRows={curveSource} rows={curveRows} spec={{type:"line",x:"timestamp",y:curveFields[0],fields:curveFields,valueDecimals:2,stackable:false,colors:{"SAC(%)":colors.SAC,"TD3(%)":colors.TD3},yLabel:metric==="net_return"?"累计扣费收益（%）":"回撤（%）"}} height={350}/>
      <RichNarrative id="seed-curve-note" className="seed-note" value="这里只画所选币种、所选编号的两份真实账户。图为日末采样，表中最大回撤仍取全部小时收盘，可能比图中最深点更大。"/>
    </section>
    <DataComponent id="seed-monthly" title={`${symbol} · seed ${seed} 的月度扣费收益`} queryId="monthly_returns" sourceRows={monthSource} displayRows={monthRows} kind="table"><DataTable rows={monthRows} rowKey="month" searchable={false} caption={`${COST[scenario]} · 首尾月份为实际部分区间`} columns={[{field:"month",label:"UTC月份"},{field:"sac",label:"SAC",renderCell:tone},{field:"td3",label:"TD3",renderCell:tone}]}/></DataComponent>
    <PairTable id="seed-reference" title="原seed 0保留参考，不替换为最好种子" pairs={reference} summary={summary}/>
    <PairTable id="seed-all-models" title="全部120份模型的独立账户结果" pairs={models} summary={summary} choose={choose}/>
    <PairTable id="seed-baselines" title="同币固定规则基准，分别核算" pairs={baselines} summary={summary}/>
    <DataComponent id="seed-training" title="训练记录：每币10份保留、50份新增" queryId="training_records" queryIds={["training_records","training_import"]} sourceRowsByQuery={{training_records:training,training_import:imports}} displayRows={trainingRows} kind="table"><DataTable rows={trainingRows} rowKey="key" caption="各币各算法独立训练；TD3每模型65,536环境步，SAC32,768步" columns={[{field:"symbol",label:"交易对"},{field:"algorithm_label",label:"算法"},{field:"seed",label:"seed"},{field:"origin_label",label:"来源"},{field:"actual_steps",label:"实际环境步"},{field:"maximum_input_hour",label:"训练最大输入小时"}]}/></DataComponent>
    <DataComponent id="seed-audits" title="每币两阶段128账户及重复执行的独立核验" queryId="audit_evidence" sourceRows={audits} kind="table"><DataTable rows={audits.map(r=>({...r,key:`${r.symbol}:${r.phase}`}))} rowKey="key" searchable={false} caption="审计通过与盈利稳定性分别判断；BTC和ETH分别保留完整证据" columns={[{field:"symbol",label:"交易对"},{field:"phase",label:"阶段"},{field:"passed",label:"核验",renderCell:v=>v?"通过":"失败"},{field:"accounts_checked",label:"账户数"},{field:"repeats_checked",label:"重复数"},{field:"checks_total",label:"检查项"},{field:"failure_count",label:"失败数"}]}/></DataComponent>
    <ReportSection id="seed-method" title="统计与交易口径" queryId="protocol" sourceRows={query("protocol")} showHeading={false}><RichNarrative id="seed-method-copy" value="## 统计与交易口径\n\n每币每算法固定seed 0—29，原0—4保留，新增5—29全部纳入。默认看全部30，可切换只看新增25；分位数用linear插值。盈利是扣费收益严格大于0，亏损为小于0，持平另计。不同种子反映同一条市场路径上的训练差异，不能视作30个独立市场样本。该年度已被多轮研究观察，新增种子不是新的市场盲测。\n\n每份账户只持本币和自身USDT，初始10,000 USDT；BTC和ETH分别核算，未读取对方特征或共享资金。资金、数据、观测、网络、奖励、费用、滑点、调仓与步数保持原研究。TD3每模型65,536步，SAC32,768步，算法比较不是等计算预算。验证保留两算法完整比较，不按年度结果挑选种子或替换主模型。\n\n训练至2025-01-01，验证至2025-10-23，年度回测随后365天。目标币仓上限95%，现金目标至少5%，2.5个百分点调仓门槛；日频决策，00UTC先卖、01UTC再买，逐腿扣费，全部小时收盘估值。主成本单边手续费0.10%与不利滑点0.02%；压力成本0.20%与0.05%。期末真实持仓按市值计入收益，不强制清算。研究不改变实盘配置。"/></ReportSection>
  </article>;
}
