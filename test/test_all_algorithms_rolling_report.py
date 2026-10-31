"""Hand-calculated statistics and fail-closed delivery checks, no model training."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts import all_algorithms_rolling_report_app as report


def accounts():
    rows = []
    for aid, (symbol, recipe, cadence, seed, scenario) in report.expected_accounts().items():
        shift = {"fixed": 0., "every_4w": .01, "every_week": .03}[cadence]
        stress = -.01 if scenario == "stress" else 0.
        rows.append({"account_id": aid, "symbol": symbol, "recipe_id": recipe, "algorithm": report.ALGORITHMS[recipe],
            "cadence": cadence, "seed": seed, "scenario": scenario, "net_return": seed * .01 - .02 + shift + stress,
            "max_drawdown": -(.03 + seed * .01 - shift / 3), "fees": 2 + seed + shift * 100, "turnover": 3 + seed + shift * 10, "average_exposure": .25 + seed * .05})
    return rows


def test_exact_independent_account_population():
    expected = report.expected_accounts()
    assert len(expected) == 180 and len(set(expected)) == 180
    assert sum(v[0] == "BTCUSDT" for v in expected.values()) == 90
    assert sum(v[0] == "ETHUSDT" for v in expected.values()) == 90
    assert all(len([v for v in expected.values() if v[:3] == (s, r, c)]) == 10
               for s in report.SYMBOLS for r in report.EXPECTED[s] for c in report.CADENCES)


def test_manual_seed_statistics_keep_losses_zeroes_and_tails():
    original = accounts()
    before = deepcopy(original)
    result = report.summarize_seeds(original)
    assert len(result) == 36 and original == before
    row = next(r for r in result if r["symbol"] == "BTCUSDT" and r["recipe_id"] == "sac_default" and r["cadence"] == "fixed" and r["scenario"] == "base")
    assert row["median_return"] == 0.
    assert row["q1_return"] == pytest.approx(-.01)
    assert row["q3_return"] == pytest.approx(.01)
    assert row["return_iqr"] == pytest.approx(.02)
    assert row["p10_return"] == pytest.approx(-.016)
    assert row["worst_return"] == pytest.approx(-.02)
    assert (row["positive_seeds"], row["loss_seeds"], row["zero_seeds"]) == (2, 2, 1)
    assert row["median_drawdown"] == pytest.approx(.05)
    assert row["p90_drawdown"] == pytest.approx(.066)
    assert row["worst_drawdown"] == pytest.approx(.07)
    assert row["median_fees"] == 4


@pytest.mark.parametrize("bad", ["missing", "duplicate_seed", "nan", "inf"])
def test_incomplete_or_nonfinite_seed_population_fails(bad):
    rows = accounts()
    if bad == "missing":
        rows.pop(0)
    elif bad == "duplicate_seed":
        # Same group and count, but seed0 appears twice and seed1 is absent.
        rows[2]["seed"] = 0
    else:
        rows[0]["net_return"] = float(bad)
    with pytest.raises(ValueError):
        report.summarize_seeds(rows)


def test_same_seed_cost_differences_are_hand_calculated():
    rows = accounts()
    before = deepcopy(rows)
    paired = report.paired_results(rows)
    assert len(paired) == 120 and rows == before
    weekly = [r for r in paired if r["cadence"] == "every_week"]
    assert all(r["return_difference"] == pytest.approx(.03) for r in weekly)
    assert all(r["drawdown_difference"] == pytest.approx(-.01) for r in weekly)
    assert all(r["fee_difference"] == pytest.approx(3.) for r in weekly)
    assert all(r["fixed_account_id"] != r["updated_account_id"] for r in paired)
    assert all(r["symbol"] in r["fixed_account_id"] and r["scenario"] in r["fixed_account_id"] for r in paired)
    stats = report.paired_statistics(paired)
    assert len(stats) == 24
    assert all(r["both_better_seeds"] == 5 and r["higher_return_seeds"] == 5 and r["lower_drawdown_seeds"] == 5 for r in stats)


def test_paired_median_is_not_replaced_by_difference_of_medians():
    rows = accounts()
    fixed, updated = [0., .01, .02, .03, .04], [.01, .02, .03, .04, 0.]
    for row in rows:
        if row["symbol"] == "BTCUSDT" and row["recipe_id"] == "td3_default" and row["scenario"] == "base":
            if row["cadence"] == "fixed":
                row["net_return"] = fixed[row["seed"]]
            elif row["cadence"] == "every_week":
                row["net_return"] = updated[row["seed"]]
    group = next(r for r in report.paired_statistics(report.paired_results(rows)) if (r["symbol"], r["recipe_id"], r["scenario"], r["cadence"]) == ("BTCUSDT", "td3_default", "base", "every_week"))
    assert group["median_return_difference"] == pytest.approx(.01)
    # The separate medians are both .02, so their difference is zero.
    assert sorted(updated)[2] - sorted(fixed)[2] == 0.


def test_duplicate_actual_account_identity_fails():
    rows = accounts()
    rows.append(deepcopy(rows[0]))
    with pytest.raises(ValueError, match="Duplicate account identity"):
        report.paired_results(rows)


def test_expanded_seed_reference_retains_phase_cost_and_all_thirty():
    sources = []
    for symbol in report.SYMBOLS:
        for phase in ("validation", "test"):
            groups = []
            for algorithm in ("sac", "td3"):
                for scenario in report.SCENARIOS:
                    groups.append({"dimensions": {"algorithm": algorithm, "symbol": symbol, "scenario": scenario}, "statistics": {"accounts": 30, "seeds": list(range(30)), "median_return": .1, "median_drawdown": .2, "return_iqr": .05, "p10_return": -.01, "worst_drawdown": .3, "positive_accounts": 25}})
            sources.append({"study": symbol + "_history", "phase": phase, "path": symbol + "/" + phase + ".csv", "groups": groups})
    inventory = {"expanded_seed_descriptions": sources}
    before = deepcopy(inventory)
    rows = report.expanded_seed_rows(inventory)
    assert len(rows) == 16 and inventory == before
    assert len({r["key"] for r in rows}) == 16
    assert all(r["descriptive_only_not_selection"] and r["seeds"] == 30 for r in rows)
    inventory["expanded_seed_descriptions"][0]["groups"][0]["statistics"]["seeds"].pop()
    with pytest.raises(ValueError, match="all30"):
        report.expanded_seed_rows(inventory)


def test_historical_sources_use_original_capital_and_full_mixed_windows():
    original = {"label": "new funded accounts", "period": {"start": report.TEST[0], "end": report.TEST[1]},
        "metricDefinitions": [{"label": "扣费收益", "definition": "new account equity/10000−1"}],
        "caveats": ["new5seed cadence account"]}
    before = deepcopy(original)
    source = report.query_source_metadata(original, "historical_results", report.TEST)
    assert original == before
    assert source["period"] == {"start": report.VALIDATION[0], "end": report.TEST[1]}
    definitions = {x["label"]: x["definition"] for x in source["metricDefinitions"]}
    assert "initial_capital" in definitions["扣费收益"]
    assert "不能统一改用10000" in definitions["扣费收益"]
    assert "原权益采样粒度" in definitions["最大回撤"]
    assert "资金与账户身份" in definitions
    assert source["caveats"] != original["caveats"]


def test_expanded_thirty_seed_source_does_not_claim_new_five_seed_population():
    original = {"metricDefinitions": [{"label": "seed分布", "definition": "new5seed accounts"}]}
    source = report.query_source_metadata(original, "expanded_seed_results", report.TEST)
    assert source["period"] == {"start": report.VALIDATION[0], "end": report.TEST[1]}
    definitions = {x["label"]: x["definition"] for x in source["metricDefinitions"]}
    assert "seed0—29" in definitions["seed分布"] and "共30个" in definitions["seed分布"]
    assert "/10000" in definitions["扣费收益"]
    assert original["metricDefinitions"][0]["definition"] == "new5seed accounts"


def test_new_rolling_query_source_metadata_is_unchanged():
    original = {"metricDefinitions": [{"label": "扣费收益", "definition": "new independent equity/10000−1"}],
        "caveats": ["exact5seed cadences"], "label": "new funded accounts"}
    source = report.query_source_metadata(original, "account_summary", report.TEST)
    assert source == {**original, "period": {"start": report.TEST[0], "end": report.TEST[1]}}
    assert source["metricDefinitions"] is not original["metricDefinitions"]


@pytest.mark.parametrize("key,value", [("passed", False), ("failure_count", 1), ("failures", ["ledger error"]), ("expected_accounts", 179), ("accounts_checked", 179), ("repeats_checked", 0), ("expected_models", 1589), ("models_checked", 1589)])
def test_failed_or_partial_completion_gate_never_reaches_report_creation(tmp_path, key, value):
    gate = {"passed": True, "failure_count": 0, "failures": [], "expected_accounts": 180, "accounts_checked": 180, "repeats_checked": 180, "expected_models": 1590, "models_checked": 1590}
    gate[key] = value
    report.write(tmp_path / "completion_gate.json", gate)
    destination = tmp_path / "must-not-exist"
    with pytest.raises(ValueError):
        report.build_report(tmp_path, output_dir=destination)
    assert not destination.exists()


def test_no_gate_no_snapshot_or_html(tmp_path):
    destination = tmp_path / "report"
    with pytest.raises(ValueError, match="completion_gate.json"):
        report.build_report(tmp_path, output_dir=destination)
    assert not destination.exists()


def test_actual_frozen_protocol_schema_without_symbols_or_inventory_receipt():
    """Read the actual declared protocol only; no gate, model or annual replay."""
    path = Path(__file__).resolve().parents[1] / "results/backtests/single_pair_rolling_all_algorithms_20261031/protocol.json"
    if not path.exists():
        pytest.skip("The frozen local protocol is not present on this checkout")
    protocol = report.read(path)
    assert "symbols" not in protocol
    assert report.verify_protocol_contract(protocol) is True
    assert protocol["selected"] == {s: list(rs) for s, rs in report.EXPECTED.items()}
    for identifier in set(r for recipes in report.EXPECTED.values() for r in recipes):
        assert protocol["recipes"][identifier]["algorithm"] == report.ALGORITHMS[identifier]


@pytest.mark.parametrize("bad", ["../secret.json", "/absolute.json", "C:/secret.json", "a\\b.json", "a//b.json"])
def test_evidence_cannot_escape_root_or_rename_identity(tmp_path, bad):
    with pytest.raises(ValueError):
        report.relative(tmp_path, bad)


def test_official_compiler_parses_authored_jsx_css_and_pure_curve_contract():
    """Source-only validation. This does not build or claim completed RL results."""
    if shutil.which("wsl.exe") is None:
        pytest.skip("WSL official Data compiler unavailable")
    script = r'''
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFile} from 'node:fs/promises';
import {loadPrebuiltCompiler} from 'file:///mnt/c/Users/sunny/.codex/plugins/cache/openai-curated-remote/data-analytics/1.1.1-exp-components.2/output_formats/html/scripts/data-app-runtime.mjs';
const compiler=await loadPrebuiltCompiler();
const jsx=await readFile('/mnt/h/PycharmProjects/hummingbot/scripts/all_algorithms_rolling_report_app.jsx','utf8');
const css=await readFile('/mnt/h/PycharmProjects/hummingbot/scripts/all_algorithms_rolling_report_app.css','utf8');
const compiled=compiler.transform(jsx,{filePath:'ReportContent.jsx'});
compiler.parseJavaScript(compiled,{sourceType:'script'});compiler.parseCss(css);
const module={exports:{}},element=(type,props)=>({type,props:props??{}});
vm.runInNewContext(compiled,{module,exports:module.exports,require:name=>{
 if(name==='react')return {useState:value=>[value,()=>{}]};
 if(name==='react/jsx-runtime')return {jsx:element,jsxs:element,Fragment:'Fragment'};
 if(name==='../../data-app-public.jsx')return {};
 if(name.endsWith('.css'))return {};
 throw Error('Unexpected authored import '+name);
}});
const matrixSource=['fixed','every_4w','every_week'].map((cadence,i)=>({symbol:'BTCUSDT',recipe_id:'sac_default',cadence,median_return:i*.1,median_drawdown:.2-i*.01}));
const selections=[{symbol:'BTCUSDT',recipe_id:'sac_default',rank:1}];
const original=JSON.stringify({matrixSource,selections});
const matrix=module.exports.matrixRows(matrixSource,selections,'BTCUSDT');
assert.equal(matrix.length,1);assert.equal(matrix[0].fixed_return,0);assert.equal(matrix[0].every_week_return,.2);assert.ok(Math.abs(matrix[0].every_week_drawdown-.18)<1e-12);
assert.equal(JSON.stringify({matrixSource,selections}),original);
assert.throws(()=>module.exports.matrixRows(matrixSource.slice(1),selections,'BTCUSDT'),/complete reviewed/);
// Explicit mathematical unit fixture only; never written as an experiment.
const ids={fixed:'A',every_4w:'B',every_week:'C'},history=[];
for(let day=0;day<366;day++)for(const [cadence,account_id] of Object.entries(ids))history.push({account_id,timestamp:new Date(Date.UTC(2025,9,23+day)).toISOString(),net_return:day/10000,drawdown:-day/20000});
const originalHistory=JSON.stringify(history);
const curves=module.exports.actualCurveRows(history,ids,'net_return');
assert.equal(curves.length,366);assert.equal(Object.keys(curves[0]).length,4);assert.ok(Math.abs(curves[365]['固定模型(%)']-3.65)<1e-12);assert.ok(Math.abs(curves[365]['每周重训(%)']-3.65)<1e-12);
assert.equal(JSON.stringify(history),originalHistory);
assert.throws(()=>module.exports.actualCurveRows(history.slice(1),ids,'net_return'),/366 boundaries/);
console.log(JSON.stringify({passed:true,sourceOnly:true,noReportWritten:true,matrixRows:matrix.length,curveBoundaries:curves.length}));
'''
    result = subprocess.run(["wsl.exe", "-d", "Ubuntu-26.04", "--exec", "/mnt/c/Users/sunny/.cache/codex-research-tools/btceth-rl-20261029/node-v24.19.0-linux-x64/bin/node", "--input-type=module", "-"], input=script, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {"passed": True, "sourceOnly": True, "noReportWritten": True, "matrixRows": 1, "curveBoundaries": 366}
