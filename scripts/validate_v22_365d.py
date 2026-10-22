"""Independent result checks and optional original-runtime prediction parity."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from prepare_v22_365d_inputs import START,END,sha


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__();self.current=None;self.specs={};self.external=[]
    def handle_starttag(self,tag,attributes):
        attributes=dict(attributes)
        if tag=='script':
            self.current=attributes.get('id')
            if attributes.get('src'):self.external.append(attributes['src'])
    def handle_data(self,data):
        if self.current and self.current.startswith('spec-'):self.specs[self.current]=data
    def handle_endtag(self,tag):
        if tag=='script':self.current=None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('results/backtests/v22_365d_20261017'))
    parser.add_argument('--remote-probe',action='store_true')
    args=parser.parse_args();root=args.output
    if args.remote_probe:
        requests=json.loads((root/'native_parity_requests.json').read_text())
        source="""import json,hashlib,joblib,pandas as pd,numpy as np
from pathlib import Path
requests=json.loads(__PAYLOAD__)
package=Path('/workspace/releases/releases/316593786732bae7c0c97676c91d58f4e8668f61e161a351840fa18c39bd95ef/shadow_package')
model=package/'models/xgboost_long_risk_gate_v22_weekly.joblib'
assert hashlib.sha256(model.read_bytes()).hexdigest()==requests['model_sha256']
bundle=joblib.load(model);errors=[];count=0
for case in requests['cases']:
 week=next(w for w in bundle['pairs'][case['pair']]['weeks'] if w['fold']==case['fold'])
 frame=pd.DataFrame(case['rows'],columns=case['features'])
 actual=week['model'].predict_proba(frame)[:,1]
 errors.extend(np.abs(actual-np.array(case['expected'])).tolist());count+=len(actual)
print(json.dumps({'original_weekly_models':len(requests['cases']),'real_feature_vectors':count,'max_abs_probability_error':max(errors),'source_model_sha256':requests['model_sha256']}))
""".replace('__PAYLOAD__',repr(json.dumps(requests)))
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ServerAliveInterval=10',
            '-o','ServerAliveCountMax=2','oci-ubuntu-152-69-205-54','docker','exec','-i','grid-live-fdusd-scheduler','python','-'],
            input=source,text=True,capture_output=True,timeout=60)
        result.check_returncode()
        parity=json.loads(result.stdout)
        assert parity['max_abs_probability_error']<=1e-7
        (root/'original_runtime_parity.json').write_text(json.dumps(parity,indent=2),encoding='utf8')
        print('Original runtime parity:',parity)
    curves=pd.read_csv(root/'continuous_equity_5m.csv.gz',low_memory=False)
    metrics=pd.read_csv(root/'summary.csv')
    checks=[]
    for (strategy,pair,scenario),frame in curves.groupby(['strategy','pair','scenario']):
        assert len(frame)==105120
        assert np.array_equal(frame.ts.to_numpy(),np.arange(START+300,END+300,300))
        assert np.allclose(frame.equity,frame.quote+frame.base*frame.price,rtol=0,atol=1e-8)
        initial=200 if strategy=='Grid' else 190
        values=np.r_[initial,frame.equity.to_numpy()]
        dd=float(np.max(1-values/np.maximum.accumulate(values))*100)
        record=metrics[(metrics.strategy==strategy)&(metrics.pair==pair)&(metrics.scenario==scenario)].iloc[0]
        assert abs(values[-1]-initial-record.net_pnl_quote)<1e-8
        assert abs(dd-record.max_drawdown_pct)<1e-8
        checks.append({'strategy':strategy,'pair':pair,'scenario':scenario,'rows':len(frame),'recomputed_max_dd_pct':dd})
    before=json.loads((root/'reproduction_before.json').read_text())
    same={f:sha(root/f)==value for f,value in before.items()}
    assert all(same.values()),'repeat run differs'
    parsed=ReportParser();parsed.feed((root/'v22_365d_plotly.html').read_text(encoding='utf8'))
    assert len(parsed.specs)==4 and not parsed.external
    shape_counts={}
    for key,raw in parsed.specs.items():
        spec=json.loads(raw.replace('<\\/','</'))
        shapes=[s for s in spec['layout']['shapes'] if s['type']=='rect']
        assert shapes and len(shapes)%4==0
        shape_counts[key]=len(shapes)//4
    result={'metrics_recomputed':checks,'reproduction_identical':same,'external_javascript':parsed.external,
            'risk_off_episodes_by_tab':shape_counts,'copied_model_sha256':sha(root/'inputs/model.joblib')}
    (root/'independent_validation.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    manifest_path=root/'artifact_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    project=Path(__file__).resolve().parents[1]
    manifest['source_hashes'][str(Path(__file__).resolve().relative_to(project))]=sha(__file__)
    manifest['artifacts']={p.name:sha(p) for p in root.iterdir() if p.is_file() and p.name!='artifact_manifest.json'}
    manifest['offline_model_hashes']={str(p.relative_to(root)):sha(p) for p in (root/'offline_models').rglob('*.ubj')}
    manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
