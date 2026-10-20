"""Compare B0 and a fixed premarket cutoff on the common four development folds.

Every invocation creates one W&B run and writes a compact local result.  It is
deliberately limited to the public/development folds; later test evaluation is
kept as a separate, explicitly labelled experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "work" / "b" / "features"))

from lab.cv import feature_table
from lab.folds import load_folds, unseen_symbols
from lab import tracking
from src import Dataset, score
import gapz_rule as G
import premarket_rule as PR


def digest(files):
    h = hashlib.sha256()
    for p in files:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default="09:15")
    args = ap.parse_args()
    as_of = PR._tag(args.as_of)
    store = PR.store_for(as_of)
    folds = load_folds(Dataset())
    days = sorted({d.date: d for f in folds for d in (*f.train, *f.val)}.values(), key=lambda d: d.date)
    files = sorted(store.glob("*.parquet"))
    targets = {str(d.target.date()) for d in days}
    missing = sorted(targets - {p.stem for p in files})
    if missing:
        raise SystemExit(f"{as_of} premarket files incomplete: {len(missing)} missing, first={missing[0]}")
    out = ROOT / "reports" / "premarket_cutoff" / as_of.replace(":", "")
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "as_of": as_of, "store": str(store.relative_to(ROOT)),
        "files": len(files), "files_sha256": digest(files),
        "folds": tracking.folds_info(folds), "unseen_symbols": unseen_symbols(),
        "score": "src.data.score", "models": ["B0", f"premarket_{as_of}"],
        "rule": "px_asof replaces gap only when px_0830 agrees with B0 price within 2%; otherwise B0 fallback",
    }
    tracker = tracking.Tracker(name=f"premarket_cutoff_{as_of.replace(':', '')}", tags=["premarket", "cutoff", as_of], config=config)
    if tracker.on:
        print("W&B run:", tracker.run.url, flush=True)
    b0 = feature_table(days, G.build_gapz, name=None, verbose=True)
    pm = feature_table(days, lambda day: PR.build_gapz_pm(day, store, as_of), name=None, verbose=True)
    # feature_table supplies the base-date key; target is not needed for scoring.
    keys = ["date", "symbol", "label"]
    b0 = b0[keys + ["gap_z", "ext_range_z"]]
    pm = pm[keys + ["gap_z", "ext_range_z", "pm"]]
    rows, metric_rows = [], []
    unseen = set(unseen_symbols())
    for i, fold in enumerate(folds, 1):
        tr_dates = {d.date for d in fold.train}; va_dates = {d.date for d in fold.val}
        btr = b0[b0.date.isin(tr_dates) & ~b0.symbol.isin(unseen)]
        ptr = pm[pm.date.isin(tr_dates) & ~pm.symbol.isin(unseen)]
        bva, pva = b0[b0.date.isin(va_dates)], pm[pm.date.isin(va_dates)]
        pred = {"B0": G.predict_table(bva, G.fit_table(btr)), f"premarket_{as_of}": G.predict_table(pva, G.fit_table(ptr))}
        for kind, p in pred.items():
            z = (bva if kind == "B0" else pva)[keys].assign(pred=p, fold=fold.name, kind=kind)
            rows.append(z)
            metrics = {}
            for group, q in {"all": z, "seen": z[~z.symbol.isin(unseen)], "unseen": z[z.symbol.isin(unseen)]}.items():
                metrics[group] = score(q.label, q.pred)
                metric_rows.append({"fold": fold.name, "kind": kind, "group": group, **metrics[group]})
            if tracker.on:
                safe_kind = kind.replace(":", "")
                tracker.fold((i - 1) * 2 + (kind != "B0"), f"{safe_kind}_{fold.name}", metrics, z.label, z.pred)
    pred = pd.concat(rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    table = metrics[metrics.group.eq("all")].pivot(index="kind", columns="fold", values="score")
    table["mean"] = table.mean(axis=1)
    pred.to_parquet(out / "predictions.parquet", index=False)
    metrics.to_csv(out / "metrics.csv", index=False)
    table.to_csv(out / "scores.csv")
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    if tracker.on:
        for kind in table.index:
            tracker.run.summary[f"official/{kind}/mean"] = float(table.loc[kind, "mean"])
        tracker.finish()
    print(table.round(6).to_string())


if __name__ == "__main__":
    main()
