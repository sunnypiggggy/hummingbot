"""Attribute the actual peak-to-trough portfolio loss by stable asset identity.

Read-only study inputs; no training, counterfactual strategy or live connection.
Each asset's PNL is its marked-value change plus actual net trading cash flow.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.audit_crypto_rl_top10 import reconstruct


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def component(peak_value, trough_value, fills):
    sign = np.where(fills.side.eq("SELL"), 1., -1.)
    actual = float((sign * fills.qty * fills.price - fills.fee).sum())
    reference = float((sign * fills.qty * fills.reference_price).sum())
    price_pnl = trough_value - peak_value + reference
    net_pnl = trough_value - peak_value + actual
    fees, slippage = float(fills.fee.sum()), float(fills.slippage_cost.sum())
    if abs(net_pnl - (price_pnl - fees - slippage)) > 1e-6:
        raise AssertionError("Price/cost decomposition does not reconcile")
    return net_pnl, price_pnl, fees, slippage


def self_check():
    sale = pd.DataFrame([dict(side="SELL", qty=1., price=89., reference_price=90., fee=1., slippage_cost=1.)])
    buy = pd.DataFrame([dict(side="BUY", qty=1., price=60., reference_price=60., fee=.6, slippage_cost=0.)])
    a, b = component(200., 80., sale), component(0., 50., buy)
    assert np.allclose(a, [-32., -30., 1., 1.], atol=1e-10, rtol=0)
    assert np.allclose(b, [-10.6, -10., .6, 0.], atol=1e-10, rtol=0)
    assert abs(a[0] + b[0] - (957.4 - 1000)) < 1e-10


def attribute(dataset, folder, run_id):
    manifest = read(folder / "manifest.json")
    row = pd.read_csv(folder / "summary.csv", float_precision="round_trip").set_index("run_id").loc[run_id]
    path = folder / "runs" / run_id
    fills = pd.read_csv(path / "fills.csv", float_precision="round_trip", dtype={"asset_key": str})
    fills["timestamp"] = pd.to_datetime(fills.timestamp, utc=True)
    saved = pd.read_csv(path / "equity_1h.csv.gz", index_col="timestamp", float_precision="round_trip")
    saved.index = pd.to_datetime(saved.index, utc=True)
    fingerprints = manifest["run_results"][run_id]["fingerprints"]
    assert sha(path / "fills.csv") == fingerprints["fills"]
    with gzip.open(path / "equity_1h.csv.gz", "rb") as stream:
        assert hashlib.sha256(stream.read()).hexdigest() == fingerprints["equity"]
    equity, quantities, economics = reconstruct(dataset, row.start, row.end_exclusive, fills,
                                                row.initial_capital, row.fee if "fee" in row else manifest["config"]["costs"][row.scenario]["fee"],
                                                manifest["config"]["costs"][row.scenario]["slippage"])
    assert not economics["errors"], economics["errors"]
    assert np.abs(equity.equity - saved.equity).max() < 1e-6
    values = equity.equity.to_numpy()
    initial = float(row.initial_capital)
    peaks = np.maximum.accumulate(np.r_[initial, values])[1:]
    drawdown = values / peaks - 1
    trough = int(np.argmin(drawdown))
    peak_nav, trough_nav = float(peaks[trough]), float(values[trough])
    candidates = np.flatnonzero(values[:trough + 1] == peak_nav)
    peak = int(candidates[-1]) if len(candidates) else -1
    peak_bar = equity.index[peak] if peak >= 0 else None
    window = fills.loc[(fills.timestamp > peak_bar) & (fills.timestamp <= equity.index[trough])] if peak_bar is not None else fills.loc[fills.timestamp <= equity.index[trough]]
    loss = peak_nav - trough_nav
    records = []
    if loss > 1e-8:
        for key, quantity in quantities.items():
            price = dataset["hourly_close"].loc[equity.index, key].to_numpy()
            marked = np.zeros(len(equity))
            owned = np.abs(quantity) > 1e-10
            if (owned & ~np.isfinite(price)).any():
                raise AssertionError("Held asset missing a real mark")
            np.multiply(quantity, price, out=marked, where=owned)
            peak_value = float(marked[peak]) if peak >= 0 else 0.
            trough_value = float(marked[trough])
            trades = window.loc[window.asset_key.eq(key)]
            pnl, market, fee, slip = component(peak_value, trough_value, trades)
            if max(abs(pnl), peak_value, trough_value, fee, slip) < 1e-8:
                continue
            symbol = dataset["assets"][key]["symbol"]
            records.append(dict(run_id=run_id, strategy=row.strategy, scenario=row.scenario,
                                pair=symbol + "/USDT", asset_key=key, net_pnl=pnl,
                                reference_price_pnl=market, fees=fee, slippage_cost=slip,
                                loss_contribution_usdt=-pnl, drawdown_contribution_pp=-pnl / peak_nav * 100,
                                net_loss_share=-pnl / loss, peak_weight=peak_value / peak_nav,
                                average_weight=float(np.mean(marked[peak + 1:trough + 1] / values[peak + 1:trough + 1])),
                                trough_weight=trough_value / trough_nav))
        assert abs(sum(r["net_pnl"] for r in records) + loss) < 1e-6
        assert abs(sum(r["drawdown_contribution_pp"] for r in records) + float(drawdown[trough]) * 100) < 1e-8
        assert abs(sum(r["net_loss_share"] for r in records) - 1.) < 1e-10
    assert abs(float(drawdown[trough]) - row.max_drawdown) < 1e-10
    summary = dict(run_id=run_id, strategy=row.strategy, scenario=row.scenario,
                   peak_valuation_boundary_utc=str(peak_bar + pd.Timedelta(hours=1)) if peak_bar is not None else row.start,
                   trough_valuation_boundary_utc=str(equity.index[trough] + pd.Timedelta(hours=1)),
                   peak_bar_open_utc=str(peak_bar) if peak_bar is not None else None,
                   trough_bar_open_utc=str(equity.index[trough]),
                   peak_equity=peak_nav, trough_equity=trough_nav, peak_to_trough_loss_usdt=loss,
                   max_drawdown=float(drawdown[trough]), attributed_pnl=sum(r["net_pnl"] for r in records),
                   reconciliation_error_usdt=sum(r["net_pnl"] for r in records) + loss,
                   source_fills_sha256=sha(path / "fills.csv"), source_manifest_sha256=sha(folder / "manifest.json"))
    return records, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("base", "stress"), default="base")
    args = parser.parse_args()
    self_check()
    dataset_path = ROOT / "data/crypto_rl_top10/dataset.pkl.gz"
    data_manifest = read(dataset_path.parent / "manifest.json")
    assert sha(dataset_path) == data_manifest["dataset_sha256"]
    dataset = pd.read_pickle(dataset_path)
    old = ROOT / "results/backtests/crypto_rl_top10_20261024"
    new = ROOT / "results/backtests/crypto_rl_a2c_20261024"
    output = ROOT / "results/backtests/rl_framework_comparison_20261024/drawdown_attribution"
    output.mkdir(parents=True, exist_ok=True)
    plans = [(old, "ppo_ensemble"), (new, "a2c_ensemble"), (old, "top10_equal"),
             (old, "top10_cap"), (old, "btc_eth_hold"), (old, "cash")]
    records, summaries = [], []
    for folder, strategy in plans:
        rows, summary = attribute(dataset, folder, f"test__{args.scenario}__{strategy}")
        records.extend(rows)
        summaries.append(summary)
    contributions = pd.DataFrame(records).sort_values(["run_id", "loss_contribution_usdt"], ascending=[True, False])
    contributions.to_csv(output / f"contributions_{args.scenario}.csv", index=False, float_format="%.17g", encoding="utf-8-sig")
    payload = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), scenario=args.scenario,
                   method="asset PNL = MV(trough)-MV(peak)+sell net cash-buy gross cash; fees/slippage already included once",
                   scope="each portfolio's own maximum hourly-close drawdown episode; not full-year PNL, standalone-coin drawdown or a removal counterfactual",
                   time_convention="hour-open labels valued at hour close; report valuation boundaries at label+1h; choose latest tied peak",
                   self_check="passed", dataset_sha256=sha(dataset_path), source_code_sha256=sha(Path(__file__)),
                   runs=summaries)
    (output / f"summary_{args.scenario}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for summary in summaries[:2]:
        print(json.dumps(summary, ensure_ascii=False))
        print(contributions.loc[contributions.run_id.eq(summary["run_id"]), ["pair", "loss_contribution_usdt", "drawdown_contribution_pp", "net_loss_share", "average_weight"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
