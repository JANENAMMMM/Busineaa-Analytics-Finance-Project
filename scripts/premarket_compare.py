"""One yardstick for every premarket variant: four development folds plus the holdout fold.

No variant is filtered by an adoption bar; the table is for people to read. All
variants share gapz_rule.fit_table (40 training symbols) and differ only in the
gap price (store/cutoff) or in which training dates they fit on.

    uv run python scripts/premarket_compare.py --variants B0 pm0915 pm0928 pm0929 pm0928_recent250
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
from scripts.oct12_metric_robustness import chance_corrected, weights
from scripts.premarket_holdout_check import bootstrap
import gapz_rule as G
import premarket_rule as PR

OUT = ROOT / "reports" / "premarket_compare"
GROUPS = json.loads((ROOT / "reports/oct10_extension/preregistered.json").read_text(encoding="utf-8"))["groups"]
RULERS = weights()


def variant(name):
    """name -> (cutoff or None for B0, recent window in base dates or None)."""
    base, _, win = name.partition("_recent")
    tag = None if base == "B0" else f"{base[2:4]}:{base[4:6]}"
    return tag, (int(win) if win else None)


def table(days, tag):
    if tag is None:
        return feature_table(days, G.build_gapz, name="cmp_B0", verbose=False)
    store = PR.store_for(tag)
    return feature_table(days, lambda d: PR.build_gapz_pm(d, store, tag),
                         name=f"cmp_pm{tag.replace(':', '')}", verbose=False)


def evaluate(z):
    out = {}
    for g, syms in {"all": None, **GROUPS}.items():
        q = z if syms is None else z[z.symbol.isin(syms)]
        s = score(q.label, q.pred)
        out[g] = {**s, "flat_share": float((q.pred == 2).mean())}
        if g == "all":
            for r, w in RULERS.items():
                out[g][f"ruler_{r}"] = chance_corrected(q.label, q.pred, w)
            out[g]["pred_dist"] = (q.pred.value_counts(normalize=True).reindex(range(5)).fillna(0).round(4).tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True)
    args = ap.parse_args()
    ds = Dataset()
    folds = [*load_folds(ds), *load_folds(ds, holdout=True)]
    days = sorted({d.date: d for f in folds for d in (*f.train, *f.val)}.values(), key=lambda d: d.date)
    unseen = set(unseen_symbols())
    fh = tracking.folds_info(load_folds(ds))["folds_hash"]
    OUT.mkdir(parents=True, exist_ok=True)
    tables, preds, rows = {}, {}, []
    for name in ["B0", *[v for v in args.variants if v != "B0"]]:
        tag, win = variant(name)
        if tag not in tables:
            tables[tag] = table(days, tag)
        t = tables[tag]
        store_info = None
        if tag:
            store = PR.store_for(tag)
            files = sorted(store.glob("*.parquet"))
            need = {str(d.target.date()) for d in days}
            h = hashlib.sha256()
            for p in files:
                if p.stem in need:
                    h.update(p.name.encode()); h.update(p.read_bytes())
            store_info = {"store": str(store.relative_to(ROOT)), "missing_targets": len(need - {p.stem for p in files}),
                          "sha256": h.hexdigest()}
        cfg = {"variant": name, "as_of": tag or "B0 (professor hourly bars)", "recent_window_base_dates": win,
               "fit": "gapz_rule.fit_table on 40 training symbols", "scored_symbols": 50,
               "folds": "lab/folds.json 4 dev folds + holdout fold", "folds_hash": fh,
               "unseen_symbols": sorted(unseen), "groups": GROUPS, "rulers": list(RULERS),
               "bootstrap": "paired 5-trading-day blocks, 2000 draws, seed 2026, vs B0",
               "premarket_input": store_info, "score_fn": "src.data.score",
               "selection_rule": "none; descriptive comparison"}
        tr_ = tracking.Tracker(name=f"premarket_compare_{name}", tags=["premarket", "compare", name], config=cfg,
                               job_type="compare")
        zs = []
        for fold in folds:
            trd = sorted({d.date for d in fold.train})
            if win:
                trd = trd[-win:]
            tr = t[t.date.isin(set(trd)) & ~t.symbol.isin(unseen)]
            va = t[t.date.isin({d.date for d in fold.val})].sort_values(["date", "symbol"])
            p = G.fit_table(tr)
            z = va[["date", "symbol", "label"]].assign(pred=G.predict_table(va, p), fold=fold.name, kind=name)
            zs.append(z)
            ev = evaluate(z)
            for g, m in ev.items():
                rows.append({"kind": name, "fold": fold.name, "group": g,
                             **{k: v for k, v in m.items() if k != "pred_dist"},
                             "pred_dist": json.dumps(m.get("pred_dist")) if "pred_dist" in m else None,
                             "lam": p["lam"], "a": p["a"], "b": p["b"]})
                if tr_.on:
                    for k, v in m.items():
                        if isinstance(v, (int, float)):
                            tr_.run.summary[f"{fold.name}/{g}/{k}"] = v
        preds[name] = pd.concat(zs, ignore_index=True)
        m = pd.DataFrame([r for r in rows if r["kind"] == name])
        dev = m[(m.group == "all") & (m.fold != "holdout")]
        if tr_.on:
            tr_.run.summary["score"] = float(dev.score.mean())
            tr_.run.summary["score_main"] = float(dev[dev.fold == "fold4"].score.iloc[0])
            tr_.run.summary["holdout_score"] = float(m[(m.group == "all") & (m.fold == "holdout")].score.iloc[0])
            tr_.run.log({"metrics": tr_.wandb.Table(dataframe=m.drop(columns=["pred_dist"]).round(6))})
            if name != "B0":
                for part in ("dev", "holdout"):
                    a = preds[name]; b = preds["B0"]
                    sel = (a.fold != "holdout") if part == "dev" else (a.fold == "holdout")
                    bs = bootstrap(a[sel], b[(b.fold != "holdout") if part == "dev" else (b.fold == "holdout")])
                    for k, v in bs.items():
                        tr_.run.summary[f"bootstrap_{part}/{k}"] = v
            tr_.run.finish()
        print(name, "dev", round(float(dev.score.mean()), 6), "holdout",
              round(float(m[(m.group == "all") & (m.fold == "holdout")].score.iloc[0]), 6), flush=True)
    M = pd.DataFrame(rows)
    M.to_csv(OUT / "metrics.csv", index=False)
    pd.concat(preds.values(), ignore_index=True).to_parquet(OUT / "predictions.parquet", index=False)
    # Bootstrap on dev uses all four folds pooled day by day; it is a spread, not a gate.
    boots = {}
    for name in preds:
        if name == "B0":
            continue
        a, b = preds[name], preds["B0"]
        boots[name] = {"dev": bootstrap(a[a.fold != "holdout"], b[b.fold != "holdout"]),
                       "holdout": bootstrap(a[a.fold == "holdout"], b[b.fold == "holdout"])}
    allg = M[M.group == "all"]
    summary = allg.pivot(index="kind", columns="fold", values="score")
    summary["dev_mean"] = summary[["fold1", "fold2", "fold3", "fold4"]].mean(axis=1)
    summary.round(6).to_csv(OUT / "score_table.csv")
    (OUT / "bootstrap.json").write_text(json.dumps(boots, indent=2), encoding="utf-8")
    print(summary.round(4).to_string())
    print(json.dumps(boots, indent=2))


if __name__ == "__main__":
    main()
