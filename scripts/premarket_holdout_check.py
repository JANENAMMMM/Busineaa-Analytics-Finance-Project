"""One out-of-sample check of the premarket gap on the holdout period.

B0 was frozen and scored on this period on 2026-10-09. The premarket rule was
never chosen with it, so fitting on the holdout fold's train (base dates up to
2026-05-11, 12-day embargo) and scoring its 73 target days once is the only
evaluation of that rule nobody has looked at before choosing.

Every model is fitted the same way (gapz_rule.fit_table on the 40 training
symbols) and scored on all 50 symbols. B0 must reproduce the recorded 0.596716,
otherwise the run stops.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "work" / "b" / "features"))

from lab.cv import feature_table
from lab.folds import load_folds, unseen_symbols
from lab import tracking
from src import Dataset, score
from src.data import WEIGHT
import gapz_rule as G
import premarket_rule as PR

OUT = ROOT / "reports" / "premarket_holdout"
GROUPS = json.loads((ROOT / "reports/oct10_extension/preregistered.json").read_text(encoding="utf-8"))["groups"]
B0_RECORDED = 0.596716404211983


def digest(files):
    h = hashlib.sha256()
    for p in files:
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def cm(y, p):
    return np.bincount(np.asarray(y, int) * 5 + np.asarray(p, int), minlength=25).reshape(5, 5)


def cm_score(c):
    c = np.asarray(c, float)
    e = c.sum(-1)[..., :, None] * c.sum(-2)[..., None, :] / c.sum((-1, -2))[..., None, None]
    return 1 - (c * WEIGHT).sum((-1, -2)) / (e * WEIGHT).sum((-1, -2))


def bootstrap(a, b, seed=2026, n=2000, block=5):
    """Paired 5-trading-day blocks of whole 50-symbol days; difference a - b."""
    days = sorted(a.date.unique())
    ca = np.array([cm(g.label, g.pred) for _, g in a.groupby("date", sort=True)])
    cb = np.array([cm(g.label, g.pred) for _, g in b.groupby("date", sort=True)])
    rng = np.random.default_rng(seed)
    k = int(np.ceil(len(days) / block))
    starts = rng.integers(0, len(days) - block + 1, size=(n, k))
    ix = (starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :len(days)]
    d = cm_score(ca[ix].sum(1)) - cm_score(cb[ix].sum(1))
    return {"lo95": float(np.quantile(d, .025)), "hi95": float(np.quantile(d, .975)),
            "nonpositive_share": float(np.mean(d <= 0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", nargs="+", default=["09:15", "09:28"])
    args = ap.parse_args()
    (fold,) = load_folds(Dataset(), holdout=True)
    days = [*fold.train, *fold.val]
    targets = {str(d.target.date()) for d in days}
    unseen = set(unseen_symbols())
    kinds = {"B0": None}
    inputs = {}
    for a in args.as_of:
        tag = PR._tag(a); store = PR.store_for(tag)
        files = sorted(store.glob("*.parquet"))
        # A missing day is not skipped: the rule itself falls back to the B0 gap, so it is scored
        # exactly as it would run. The list is recorded so nobody mistakes it for full coverage.
        missing = sorted(targets - {p.stem for p in files})
        val_missing = sorted({str(d.target.date()) for d in fold.val} - {p.stem for p in files})
        if missing:
            print(f"{tag}: {len(missing)} target files missing (val {len(val_missing)}): {missing[:5]}", flush=True)
        used = [p for p in files if p.stem in targets]
        kinds[f"premarket_{tag}"] = (store, tag)
        inputs[tag] = {"store": str(store.relative_to(ROOT)), "files": len(used), "sha256": digest(used),
                       "missing_targets": missing, "missing_val_targets": val_missing}
    config = {"fold": "holdout", "train": [str(fold.train[0].date.date()), str(fold.train[-1].date.date())],
              "val_targets": [str(fold.val[0].target.date()), str(fold.val[-1].target.date())],
              "n_val_days": len(fold.val), "models": list(kinds), "inputs": inputs,
              "folds_hash": tracking.folds_info(load_folds(Dataset()))["folds_hash"],
              "unseen_symbols": sorted(unseen), "score_fn": "src.data.score",
              "fit": "gapz_rule.fit_table on 40 training symbols; 50 symbols scored",
              "bootstrap": "paired 5-trading-day blocks of 50-symbol days, 2000 draws, seed 2026",
              "note": "holdout used once for the premarket rule; B0 was opened on 2026-10-09"}
    tracker = tracking.Tracker(name="premarket_holdout_check", tags=["premarket", "holdout", "test"],
                               config=config, job_type="holdout")
    tables = {}
    for kind, spec in kinds.items():
        build = G.build_gapz if spec is None else (lambda day, s=spec: PR.build_gapz_pm(day, s[0], s[1]))
        tables[kind] = feature_table(days, build, name=None, verbose=False)
    preds, rows = {}, []
    tr_dates = {d.date for d in fold.train}; va_dates = {d.date for d in fold.val}
    for kind, t in tables.items():
        tr = t[t.date.isin(tr_dates) & ~t.symbol.isin(unseen)]
        va = t[t.date.isin(va_dates)].sort_values(["date", "symbol"])
        params = G.fit_table(tr)
        z = va[["date", "symbol", "label"]].assign(pred=G.predict_table(va, params), kind=kind)
        if "pm" in va:
            z["pm_used"] = va.pm.to_numpy()
        preds[kind] = z
        for g, syms in {"all": None, **GROUPS}.items():
            q = z if syms is None else z[z.symbol.isin(syms)]
            s = score(q.label, q.pred)
            rows.append({"kind": kind, "group": g, **s,
                         "flat_share": float((q.pred == 2).mean()),
                         "pm_used_share": float(q.pm_used.mean()) if "pm_used" in q else None,
                         **{f"param_{k}": v for k, v in params.items() if isinstance(v, (int, float))}})
    m = pd.DataFrame(rows)
    b0 = float(m[(m.kind == "B0") & (m.group == "all")].score.iloc[0])
    assert abs(b0 - B0_RECORDED) < 1e-9, f"B0 holdout {b0} != recorded {B0_RECORDED}"
    boot = {k: bootstrap(preds[k], preds["B0"]) for k in preds if k != "B0"}
    OUT.mkdir(parents=True, exist_ok=True)
    m.to_csv(OUT / "metrics.csv", index=False)
    pd.concat(preds.values(), ignore_index=True).to_parquet(OUT / "predictions.parquet", index=False)
    (OUT / "result.json").write_text(json.dumps({"config": config, "bootstrap_vs_B0": boot,
        "all": m[m.group == "all"].set_index("kind").score.round(6).to_dict()}, indent=2), encoding="utf-8")
    if tracker.on:
        for _, r in m.iterrows():
            for k in ["score", "accuracy", "direction", "big_recall", "big_prec", "flat_share"]:
                tracker.run.summary[f"holdout/{r.kind}/{r.group}/{k}"] = r[k]
        for k, v in boot.items():
            for kk, vv in v.items():
                tracker.run.summary[f"bootstrap/{k}/{kk}"] = vv
        tracker.run.log({"holdout_table": tracker.wandb.Table(dataframe=m.round(6))})
        tracker.run.summary["score"] = b0
        tracker.run.finish()
    print(m[["kind", "group", "score", "accuracy", "direction", "flat_share", "pm_used_share"]].round(4).to_string(index=False))
    print(json.dumps(boot, indent=2))


if __name__ == "__main__":
    main()
