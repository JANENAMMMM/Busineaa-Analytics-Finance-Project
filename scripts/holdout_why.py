"""Why does every model score higher on the holdout than on development?

Splits the gap into candidate causes using the four development validation
periods and the holdout period, all scored with the same frozen fold models:

1. label mix      more crash/spike days in the period
2. gap size       more days with a large overnight gap |gap_z|
3. gap strength   the same size of gap calls direction better
4. market moves   symbols moving together, so one market gap explains many rows
5. composition    reweight the holdout to development's |gap_z| mix and rescore

Uses the holdout labels; per the 2026-10-12 decision the holdout is no longer
held back.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import Dataset, score
from src.data import WEIGHT
from lab.folds import load_folds
from lab import tracking

OUT = ROOT / "reports" / "holdout_why"
BINS = [0, 0.5, 1, 2, np.inf]
BIN_NAMES = ["<0.5", "0.5-1", "1-2", ">=2"]


def weighted_score(y, p, w):
    o = np.zeros((5, 5))
    np.add.at(o, (np.asarray(y, int), np.asarray(p, int)), np.asarray(w, float))
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    return float(1 - (WEIGHT * o).sum() / (WEIGHT * e).sum())


def main():
    ds = Dataset()
    folds = [*load_folds(ds), *load_folds(ds, holdout=True)]
    period = {}
    for f in folds:
        for d in f.val:
            period[d.date] = f.name
    t = pd.read_parquet(ROOT / "cache" / "cmp_B0.parquet")
    t = t[t.date.isin(period)].assign(period=lambda z: z.date.map(period))
    pred = pd.read_parquet(ROOT / "reports" / "premarket_compare" / "predictions.parquet")
    t = t.merge(pred.pivot_table(index=["date", "symbol"], columns="kind", values="pred").reset_index(),
                on=["date", "symbol"], how="left", validate="one_to_one")
    t["absgap"] = t.gap_z.abs()
    t["bin"] = pd.cut(t.absgap, BINS, right=False, labels=BIN_NAMES)
    t["moved"] = t.label != 2
    t["gap_hit"] = np.sign(t.gap_z) == np.sign(t.ret_pct)

    rows, binrows = [], []
    for name, z in t.groupby("period", sort=False):
        m = z[z.moved & (z.gap_z != 0) & z.gap_z.notna()]
        lab = z.label.value_counts(normalize=True).reindex(range(5)).fillna(0)
        day = z.groupby("date").agg(mg=("gap_z", "mean"), mr=("ret_pct", "mean"))
        r = {"period": name, "rows": len(z),
             "extreme_share": float(lab[0] + lab[4]), "flat_share": float(lab[2]),
             "mean_abs_ret_pct": float(z.ret_pct.abs().mean()),
             "mean_abs_gap_z": float(z.absgap.mean()), "share_abs_gap_ge1": float((z.absgap >= 1).mean()),
             "share_abs_gap_ge2": float((z.absgap >= 2).mean()),
             "gap_direction_hit_moved": float(m.gap_hit.mean()),
             "corr_gap_ret": float(z[["gap_z", "ret_pct"]].corr().iloc[0, 1]),
             "market_gap_ret_corr": float(day.corr().iloc[0, 1]),
             "market_gap_sign_hit": float((np.sign(day.mg) == np.sign(day.mr)).mean()),
             "mean_abs_market_gap": float(day.mg.abs().mean())}
        for k in ["B0", "pm0928", "pm0929"]:
            if k in z and z[k].notna().all():
                r[f"score_{k}"] = score(z.label, z[k].astype(int))["score"]
        rows.append(r)
        for b, q in m.groupby("bin", observed=False):
            binrows.append({"period": name, "bin": b, "share_of_moved": len(q) / len(m),
                            "gap_direction_hit": float(q.gap_hit.mean()) if len(q) else np.nan, "n": len(q)})
    P = pd.DataFrame(rows).set_index("period")
    B = pd.DataFrame(binrows)

    # Composition counterfactual: give each holdout row the weight that makes its |gap_z| bin
    # share match pooled development, then rescore the same predictions.
    dev = t[t.period != "holdout"]; hol = t[t.period == "holdout"].copy()
    dev_share = dev.bin.value_counts(normalize=True)
    hol_share = hol.bin.value_counts(normalize=True)
    hol["w"] = hol.bin.map(dev_share / hol_share).astype(float)
    cf = {}
    for k in ["B0", "pm0928", "pm0929"]:
        cf[k] = {"holdout": score(hol.label, hol[k].astype(int))["score"],
                 "holdout_reweighted_to_dev_gap_mix": weighted_score(hol.label, hol[k], hol.w),
                 "dev_pooled": score(dev.label, dev[k].astype(int))["score"]}
    # Same idea for direction: per-bin hit rate of holdout under dev bin mix.
    dm = B[B.period != "holdout"].groupby("bin", observed=False).apply(
        lambda q: np.average(q.gap_direction_hit, weights=q.n) if q.n.sum() else np.nan)
    devmix = B[B.period != "holdout"].groupby("bin", observed=False).n.sum()
    devmix = devmix / devmix.sum()
    hb = B[B.period == "holdout"].set_index("bin")
    direction = {"dev_pooled_hit": float((dm * devmix).sum()),
                 "holdout_hit": float((hb.gap_direction_hit * hb.share_of_moved).sum()),
                 "holdout_hit_with_dev_mix": float((hb.gap_direction_hit * devmix).sum()),
                 "dev_hit_with_holdout_mix": float((dm * hb.share_of_moved).sum())}

    OUT.mkdir(parents=True, exist_ok=True)
    P.round(6).to_csv(OUT / "periods.csv"); B.round(6).to_csv(OUT / "gap_bins.csv", index=False)
    res = {"periods": P.round(6).to_dict(orient="index"), "counterfactual_score": cf, "direction_decomposition": direction,
           "bin_share_dev": dev_share.round(4).to_dict(), "bin_share_holdout": hol_share.round(4).to_dict()}
    (OUT / "result.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    tr = tracking.Tracker(name="holdout_why", tags=["holdout", "analysis"], job_type="analysis",
                          config={"periods": "4 dev validation folds + holdout", "bins_abs_gap_z": BIN_NAMES,
                                  "predictions": "reports/premarket_compare/predictions.parquet",
                                  "folds_hash": tracking.folds_info(load_folds(ds))["folds_hash"]})
    if tr.on:
        tr.run.log({"periods": tr.wandb.Table(dataframe=P.reset_index().round(6)),
                    "gap_bins": tr.wandb.Table(dataframe=B.astype({"bin": str}).round(6))})
        for k, v in cf.items():
            for kk, vv in v.items():
                tr.run.summary[f"cf/{k}/{kk}"] = vv
        for k, v in direction.items():
            tr.run.summary[f"direction/{k}"] = v
        tr.run.finish()
    pd.set_option("display.width", 220)
    print(P.round(3).T.to_string())
    print(B.pivot(index="bin", columns="period", values="gap_direction_hit").round(3).to_string())
    print(B.pivot(index="bin", columns="period", values="share_of_moved").round(3).to_string())
    print(json.dumps({"counterfactual_score": cf, "direction": direction}, indent=2))


if __name__ == "__main__":
    main()
