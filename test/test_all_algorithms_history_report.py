"""Historical delivery cannot fabricate or authorize new rolling outcomes."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts import all_algorithms_history_report_app as history


def test_full_parameter_flattening_retains_values_lists_and_identity():
    original = [{"key": "BTC:sac", "symbol": "BTCUSDT", "recipe_id": "sac_default",
        "recipe_json": json.dumps({"learning_rate": .0001, "policy_kwargs": {"net_arch": [128, 128]}, "ent_coef": "auto"})}]
    before = deepcopy(original)
    rows = history.flatten_parameters(original)
    assert original == before
    values = {r["parameter"]: r["value"] for r in rows}
    assert values == {"learning_rate": "0.0001", "policy_kwargs.net_arch": "[128, 128]", "ent_coef": '"auto"'}
    assert all(r["symbol"] == "BTCUSDT" and r["recipe_id"] == "sac_default" for r in rows)
    assert len({r["key"] for r in rows}) == 3


def test_unavailable_read_only_status_stays_unavailable_not_zero(monkeypatch):
    def unavailable(*a, **k):
        raise OSError("unit status unavailable")
    monkeypatch.setattr(history.ops, "remote_request", unavailable)
    status = history.capture_status()
    assert status["status_available"] is False and status["error_type"] == "OSError"
    assert "read_only_status" not in status


def test_status_capture_uses_only_status_action(monkeypatch):
    calls = []
    def status(action, payload, **kwargs):
        calls.append((action, payload, kwargs))
        return {"remote_root": history.ops.REMOTE_ROOT, "models": {"completed": 0, "started": 32}}
    monkeypatch.setattr(history.ops, "remote_request", status)
    result = history.capture_status()
    assert calls == [("status", {}, {"timeout": 45})]
    assert result["status_available"] and result["read_only_status"]["models"]["completed"] == 0


def test_missing_historical_evidence_does_not_create_report(tmp_path):
    with pytest.raises(ValueError):
        history.build_history(tmp_path)
    assert not (tmp_path / "history_report").exists()


def test_actual_historical_sources_and_selection_are_read_only():
    top = history.ROOT / "results/backtests/single_pair_rolling_all_algorithms_20261031"
    if not (top / "protocol.json").is_file():
        pytest.skip("Actual immutable selection evidence unavailable")
    protocol_before = history.rolling.sha(top / "protocol.json")
    selection_before = history.rolling.sha(top / "selection.json")
    spec, inventory, selection, pins, external, candidates, shortlist = history.history_inputs(top)
    assert spec["selected"] == selection["selected"]
    assert len(candidates) == 18 and len(shortlist) == 6 and len(history.flatten_parameters(shortlist)) == 83
    assert len(external) == 99 and len(pins) == 23
    assert history.rolling.sha(top / "protocol.json") == protocol_before
    assert history.rolling.sha(top / "selection.json") == selection_before
    assert all(x["selected_for_experiment"] for x in shortlist)


def test_official_compiler_actual_history_data_and_display_bindings():
    """Actual historical data; source-only component contract, no pixel claim."""
    root = Path(__file__).resolve().parents[1]
    reviewed = root / "results/backtests/single_pair_rolling_all_algorithms_20261031/history_report/reviewed.json"
    if not reviewed.is_file() or shutil.which("wsl.exe") is None:
        pytest.skip("Actual historical reviewed evidence and official compiler required")
    script = r'''
import vm from 'node:vm';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {loadPrebuiltCompiler} from 'file:///mnt/c/Users/sunny/.codex/plugins/cache/openai-curated-remote/data-analytics/1.1.1-exp-components.2/output_formats/html/scripts/data-app-runtime.mjs';
const path='/mnt/h/PycharmProjects/hummingbot/results/backtests/single_pair_rolling_all_algorithms_20261031/history_report/reviewed.json';
const snapshot=JSON.parse(await readFile(path,'utf8')),before=JSON.stringify(snapshot);
assert.equal(snapshot.report.rollingFinancialResultsIncluded,false);
assert.ok(!Object.hasOwn(snapshot.queries,'account_summary'));
const historyDD=snapshot.queries.historical_results.source.metricDefinitions.find(r=>r.label==='最大回撤').definition;
assert.ok(historyDD.includes('原实验记录的采样粒度')&&historyDD.includes('其他以原来源为准'));
assert.ok(!historyDD.includes('原记录全部小时收盘'));
const compiler=await loadPrebuiltCompiler();
const source=await readFile('/mnt/h/PycharmProjects/hummingbot/scripts/all_algorithms_history_report_app.jsx','utf8');
compiler.parseCss(await readFile('/mnt/h/PycharmProjects/hummingbot/scripts/all_algorithms_history_report_app.css','utf8'));
const compiled=compiler.transform(source,{filePath:'ReportContent.jsx'});compiler.parseJavaScript(compiled,{sourceType:'script'});
const element=(type,props)=>({type,props:props??{}}),module={exports:{}};
const names=['DataComponent','DataTable','Dropdown','EvidenceChart','ReportSection','RichNarrative'],components=Object.fromEntries(names.map(n=>[n,n]));
components.useDataApp=()=>({appTitle:snapshot.title,setAppTitle:()=>{},canEdit:false,mode:'view',reviewedPeriodRows:id=>{assert.ok(Object.hasOwn(snapshot.queries,id),id);return snapshot.queries[id].rows;}});
vm.runInNewContext(compiled,{module,exports:module.exports,require:name=>{
 if(name==='react')return {useState:value=>[value,()=>{}]};
 if(name==='react/jsx-runtime')return {jsx:element,jsxs:element,Fragment:'Fragment'};
 if(name==='../../data-app-public.jsx')return components;
 if(name.endsWith('.css'))return {};
 throw Error('Unexpected authored import '+name);
}});
const tables=[],charts=[],sections=[];
function walk(n){
 if(Array.isArray(n)){n.forEach(walk);return;}if(!n||typeof n!=='object')return;
 if(typeof n.type==='function'){walk(n.type(n.props));return;}
 if(n.type==='DataComponent'){
  const p=n.props;assert.ok(Array.isArray(p.sourceRows)&&Array.isArray(p.displayRows));
  assert.equal(p.children.type,'DataTable');assert.equal(p.children.props.rows,p.displayRows);
  assert.equal(new Set(p.displayRows.map(r=>r[p.children.props.rowKey])).size,p.displayRows.length);
  for(const r of p.sourceRows)assert.ok(snapshot.queries[p.queryId].rows.includes(r),'source identity');
  tables.push(n);
 }
 if(n.type==='EvidenceChart'){charts.push(n);for(const r of n.props.sourceRows)assert.ok(snapshot.queries[n.props.queryId].rows.includes(r));}
 if(n.type==='ReportSection')sections.push(n);
 walk(n.props?.children);
}
walk(module.exports.ReportContent());
assert.equal(tables.length,12);assert.equal(charts.length,1);
const table=id=>tables.find(n=>n.props.id===id).props;
assert.equal(table('history-native-BTCUSDT').displayRows.length,6);
assert.equal(table('history-native-ETHUSDT').displayRows.length,6);
assert.equal(table('history-expanded').displayRows.length,8);
assert.equal(table('history-baselines').displayRows.length,8);
const sma=table('history-baselines').displayRows.find(r=>r.symbol==='BTCUSDT'&&r.algorithm==='sma200_trend');
assert.equal((sma.median_return*100).toFixed(2),'10.49');assert.equal((sma.median_drawdown*100).toFixed(2),'9.78');
const nativeSAC=table('history-native-BTCUSDT').displayRows.find(r=>r.algorithm==='sac');
assert.equal(nativeSAC.accounts,5);assert.equal((nativeSAC.median_return*100).toFixed(2),'12.50');
assert.equal(table('history-sac-replication').displayRows.length,2);
assert.equal(table('history-shortlist').displayRows.length,6);
const configurationRows=table('history-family-details').displayRows.filter(r=>['crossq','droq','sac','td3','tqc','recurrent_ppo'].includes(r.algorithm));
assert.ok(configurationRows.length>0);assert.ok(configurationRows.every(r=>r.configuration!=='原固定规则'),'native models must not be relabeled as fixed rules');
assert.equal(charts[0].props.rows.length,9);
const summary=sections.find(n=>n.props.id==='history-summary').props;
assert.ok(summary.children.props.value.includes('尚无本页收益结果'));
assert.ok(summary.children.props.value.includes('10.21%'));
for(const id of summary.queryIds)assert.ok(Array.isArray(summary.sourceRowsByQuery[id]));
assert.equal(JSON.stringify(snapshot),before);
console.log(JSON.stringify({passed:true,sourceOnly:true,tables:tables.length,charts:charts.length,noNewFinancialResults:true}));
'''
    result = subprocess.run(["wsl.exe", "-d", "Ubuntu-26.04", "--exec", "/mnt/c/Users/sunny/.cache/codex-research-tools/btceth-rl-20261029/node-v24.19.0-linux-x64/bin/node", "--input-type=module", "-"], input=script, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {"passed": True, "sourceOnly": True, "tables": 12, "charts": 1, "noNewFinancialResults": True}
