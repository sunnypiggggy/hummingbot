"""Independent ledger/metric/artifact validation for the offline experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def validate(output: Path, compare: Path | None = None):
    summary=json.loads((output/"summary.json").read_text(encoding="utf-8"))
    signals=pd.read_csv(output/"hourly_signals.csv.gz",float_precision="round_trip")
    checks={"offline_only":summary["offline_only"] and not summary["deployment_allowed"]}
    max_error=0.
    for arm in ("A","B"):
        eq=pd.read_csv(output/f"{arm}_equity_5m.csv.gz",float_precision="round_trip")
        trades=pd.read_csv(output/f"{arm}_trades.csv",float_precision="round_trip")
        makers=trades[trades.kind=="MAKER"]
        assert not makers.order_id.duplicated().any(), "duplicate economic fill"
        assert trades.fee.ge(0).all() and makers.fee.eq(0).all()
        assert not (trades.reason=="REENTRY").loc[trades.side=="SELL"].any()
        for pair in ("BTC-FDUSD","ETH-FDUSD"):
            q=eq[eq.pair==pair].sort_values("ts")
            t=trades[trades.pair==pair]
            raw=pd.read_csv(output/"inputs"/f"binance_{pair}_5m.csv")
            start=int(pd.Timestamp(summary["start"]).timestamp())
            first_open=float(raw.loc[raw.timestamp==start,"open"].iloc[0])
            filt=json.loads((output/"inputs/exchange_filters.json").read_text(encoding="utf-8"))["pairs"][pair]
            initial_base=np.floor((100/first_open)/filt["step_size"])*filt["step_size"]
            base=np.zeros(len(q)); cash=np.zeros(len(q))
            for r in t.itertuples():
                # Maker fills precede the recorded bar close; taker at that
                # same clock instant is a NEXT-open execution, after the mark.
                idx=np.searchsorted(q.ts.to_numpy(),r.ts,side="left" if r.kind=="MAKER" else "right")
                assert idx < len(q), "fill outside reporting window"
                base[idx] += r.quantity if r.side=="BUY" else -r.quantity
                cash[idx] += -r.notional-r.fee if r.side=="BUY" else r.notional-r.fee
                if r.side=="BUY":
                    h=signals[(signals.pair==pair)&(signals.arm==arm)&(signals.signal_ts+300<=r.ts-(300 if r.kind=="MAKER" else 0))]
                    assert not bool(h.iloc[-1].risk_off), "Risk-Off new BUY"
            rb=initial_base+np.cumsum(base)
            rq=200-initial_base*first_open+np.cumsum(cash)
            errors=[np.max(np.abs(rb-q.base)),np.max(np.abs(rq-q.quote)),np.max(np.abs(rq+rb*q.price-q.equity))]
            max_error=max(max_error,*[float(x) for x in errors])
            assert max(errors)<1e-6, (arm,pair,errors)
            assert np.min(rb)>-1e-10 and np.min(rq)>-1e-7
            assert len(q)==summary["five_minute_rows_per_pair"] and q.ts.diff().dropna().eq(300).all()
            r=summary["results"][arm][pair]
            assert abs(q.equity.iloc[-1]-200-r["pnl"])<1e-8
            dd=(1-q.equity/np.maximum.accumulate(np.r_[200,q.equity])[1:]).max()*100
            assert abs(dd-r["max_drawdown_pct"])<1e-8
        p=eq.groupby("ts").equity.sum()+20
        assert abs(p.iloc[-1]-420-summary["results"][arm]["PORTFOLIO"]["pnl"])<1e-8
        checks[f"{arm}_ledger_and_summary"]=True
    pivot=signals.pivot(index=["pair","signal_ts"],columns="arm",values=["probability","threshold"])
    assert np.array_equal(pivot["probability"]["A"],pivot["probability"]["B"])
    assert np.array_equal(pivot["threshold"]["A"],pivot["threshold"]["B"])
    checks["identical_probabilities_thresholds"]=True
    html=(output/"comparison.html").read_text(encoding="utf-8")
    assert not re.search(r'<script[^>]+src=',html,re.I)
    decoder=json.JSONDecoder()
    groups=json.loads(re.search(r'const groups=(.*?);function shade',html).group(1))
    for pair in ("BTC_FDUSD","ETH_FDUSD"):
        match=re.search(r'Plotly\.newPlot\(\s*"chart_'+pair+r'"\s*,\s*',html)
        data,used=decoder.raw_decode(html[match.end():])
        remaining=html[match.end()+used:].lstrip()[1:].lstrip()
        layout,_=decoder.raw_decode(remaining)
        g=groups["chart_"+pair]
        assert set(g["A"]).isdisjoint(g["B"])
        assert len(g["A"])+len(g["B"])==len(layout["shapes"])
        assert len(data)==6 and len(layout["shapes"])>0
        assert all(i<len(layout["shapes"]) for i in g["A"]+g["B"])
        with Image.open(output/f"{pair.replace('_','-')}_comparison.png") as image:
            assert image.size==(1440,2400)
    checks["self_contained_html_independent_shape_groups_png_dimensions"]=True
    if compare is not None:
        names=[x.name for x in output.iterdir() if x.is_file() and x.name not in {"artifact_manifest.json","validation.json"}]
        mismatch=[n for n in names if not (compare/n).is_file() or digest(output/n)!=digest(compare/n)]
        assert not mismatch, f"non-deterministic artifacts: {mismatch}"
        checks["full_rerun_artifact_sha256_identical"]=True
        checks["rerun_files_checked"]=len(names)
    result=dict(checks=checks,maximum_ledger_reconstruction_error=max_error,
                browser_interactive_qa="Not executed: browser URL security policy blocked local file. Shape/trace groups validated statically; PNG visually inspected.")
    (output/"validation.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    manifest={x.name:digest(x) for x in sorted(output.iterdir()) if x.is_file() and x.name!="artifact_manifest.json"}
    (output/"artifact_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=Path("results/backtests/grid_v22_recovery_counter_ablation_2026"))
    p.add_argument("--compare",type=Path)
    args=p.parse_args()
    validate(args.output,args.compare)
