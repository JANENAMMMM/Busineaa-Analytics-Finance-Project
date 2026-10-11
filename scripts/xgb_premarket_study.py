"""Frozen, development-only B0/premarket/XGBoost comparison.  Never opens holdout."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import Dataset, score
from lab.cv import feature_table
from lab.folds import load_folds, unseen_symbols
from lab.leak import guard
from lab import tracking

OUT = ROOT / "reports" / "xgb_premarket"
CFG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
HOLDOUT = pd.Timestamp(CFG["holdout_from"])
BASE = CFG["base_features"]
PM = CFG["premarket_features"]
Q = np.asarray(CFG["decoder"]["quantiles"], float)

sys.path.insert(0, str(ROOT / "work" / "b" / "features"))
import gapz_rule as G  # noqa: E402
import premarket_rule as PR  # noqa: E402

# --as-of picks the premarket store; 09:15 keeps the original output folder, others write a subfolder.
AS_OF = "09:15"
STORE = PR.store_for(AS_OF)
RUN_OUT = OUT


def configure(as_of):
    global AS_OF, STORE, RUN_OUT
    AS_OF = PR._tag(as_of)
    STORE = PR.store_for(AS_OF)
    RUN_OUT = OUT if AS_OF == "09:15" else OUT / AS_OF.replace(":", "")


def write(name, obj):
    path = RUN_OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def pm_manifest():
    files = []
    for p in sorted(STORE.glob("*.parquet")):
        target = pd.Timestamp(p.stem)
        if target >= HOLDOUT:
            continue
        files.append({"target": p.stem, "sha256": sha(p), "bytes": p.stat().st_size})
    assert files and max(x["target"] for x in files) < str(HOLDOUT.date())
    return {"path": f"{STORE.relative_to(ROOT)}/YYYY-MM-DD.parquet", "as_of": AS_OF,
            "used_target_range": [files[0]["target"], files[-1]["target"]],
            "n_files": len(files), "files": files,
            "rule": "px_asof only when px_0830 agrees with professor B0 gap price within 2%; otherwise B0 fallback"}


def pm_build(day):
    """Read only target dates strictly before holdout and derive declared PM fields."""
    if pd.Timestamp(day.target) >= HOLDOUT:
        raise RuntimeError("holdout premarket read blocked")
    base = PR.build_gapz_pm(guard(day), STORE, AS_OF)
    if base.empty:
        return pd.DataFrame(columns=["symbol", "pm_gap_z", "pm_move_z", "log1p_vol_pre", "pm_available"])
    d = day.daily(days=40, columns=["symbol", "date_et", "ret"])
    r = d.pivot_table(index="date_et", columns="symbol", values="ret", aggfunc="last").sort_index()
    vol20 = r.rolling(20, min_periods=10).std().loc[r.index[-1]]
    raw = PR.load_store(day.target, STORE)
    x = base.rename(columns={"gap_z": "pm_gap_z", "pm": "pm_available"}).set_index("symbol")
    x["pm_move_z"] = np.nan
    x["log1p_vol_pre"] = np.nan
    if raw is not None:
        raw = raw.set_index("symbol").reindex(x.index)
        good = x.pm_available.eq(1) & raw.px_0830.notna() & raw.px_asof.notna()
        x.loc[good, "pm_move_z"] = ((raw.loc[good, "px_asof"] / raw.loc[good, "px_0830"] - 1) /
                                      vol20.reindex(x.index)[good])
        x.loc[good, "log1p_vol_pre"] = np.log1p(raw.loc[good, "vol_pre"])
    return x.reset_index()[["symbol", "pm_gap_z", "pm_move_z", "log1p_vol_pre", "pm_available"]]


def cut(v, a, b):
    p = np.full(len(v), 2, dtype=int)
    p[v >= a], p[v >= b], p[v <= -a], p[v <= -b] = 3, 4, 1, 0
    p[~np.isfinite(v)] = 2
    return p


def inner_splits(t):
    dates = np.sort(t.date.unique())
    for i in range(3):
        start = len(dates) - (3 - i) * 40
        tr = t[t.date.isin(dates[:start - 12])]
        va = t[t.date.isin(dates[start:start + 40])]
        assert tr.date.max() < va.date.min()
        yield tr, va


def fit_xgb(t, features, constraint):
    from xgboost import XGBRegressor
    args = dict(CFG["xgboost"])
    for k in ("version", "target", "monotone_constraints"):
        args.pop(k, None)
    args["monotone_constraints"] = tuple(constraint)
    valid = np.isfinite(t[features[0]]) & np.isfinite(t.vol20) & (t.vol20 > 0)
    return XGBRegressor(**args).fit(t.loc[valid, features], t.loc[valid, "label"].to_numpy(float) - 2)


def decoder(t, features, constraint):
    vs, ys = [], []
    for tr, va in inner_splits(t):
        est = fit_xgb(tr, features, constraint)
        v = est.predict(va[features]).astype(float)
        v[~(np.isfinite(va[features[0]]) & np.isfinite(va.vol20) & (va.vol20 > 0))] = np.nan
        vs.append(v); ys.append(va.label.to_numpy(int))
    v, y = np.concatenate(vs), np.concatenate(ys)
    q = np.unique(np.quantile(np.abs(v[np.isfinite(v)]), Q))
    choices = [(score(y, cut(v, a, b))["score"], float(b), float(a)) for i, a in enumerate(q) for b in q[i + 1:] if a < b]
    _, b, a = max(choices)
    return a, b


def predict_xgb(t, val, features, constraint):
    a, b = decoder(t, features, constraint)
    est = fit_xgb(t, features, constraint)
    v = est.predict(val[features]).astype(float)
    v[~(np.isfinite(val[features[0]]) & np.isfinite(val.vol20) & (val.vol20 > 0))] = np.nan
    return cut(v, a, b), {"a": a, "b": b}


def alternate_score(y, p, w):
    o = np.bincount(np.asarray(y, int) * 5 + np.asarray(p, int), minlength=25).reshape(5, 5).astype(float)
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    d = (w * e).sum()
    return float(1 - (w * o).sum() / d) if d else float("nan")


def rulers():
    levels = np.arange(5); dist = levels[:, None] - levels[None, :]
    extreme = np.abs(levels - 2)[:, None] ** 2
    official = (dist ** 2) * extreme
    flat = official.copy(); flat[2] = dist[2] ** 2
    return {"official": official, "flat_credit": flat, "linear": np.abs(dist) * extreme,
            "plain_qwk": dist ** 2, "extreme_only": (dist ** 2) * (np.abs(levels - 2)[:, None] == 2)}


def add_metrics(pred, groups):
    rows = []
    ws = rulers()
    for (kind, fold), z in pred.groupby(["kind", "fold"], sort=True):
        for group, symbols in groups.items():
            q = z if group == "all" else z[z.symbol.isin(symbols)]
            r = score(q.label, q.pred)
            r.update({name: alternate_score(q.label, q.pred, w) for name, w in ws.items()})
            dist = q.pred.value_counts(normalize=True).reindex(range(5), fill_value=0)
            r.update({f"pred_share_{i}": float(dist[i]) for i in range(5)})
            r["flat_share"] = float(dist[2])
            rows.append({"kind": kind, "fold": fold, "group": group, **r})
    return pd.DataFrame(rows)


def bootstrap(pred, kind):
    rng = np.random.default_rng(2026); out = np.zeros(2000)
    a = pred[pred.kind.eq(kind)]; b = pred[pred.kind.eq("B0")]
    j = a.merge(b[["fold", "date", "symbol", "pred"]], on=["fold", "date", "symbol"], suffixes=("_a", "_b"), validate="one_to_one")
    for _, f in j.groupby("fold", sort=True):
        ca, cb = [], []
        for _, d in f.groupby("date", sort=True):
            ca.append(np.bincount(d.label.to_numpy(int) * 5 + d.pred_a.to_numpy(int), minlength=25).reshape(5, 5))
            cb.append(np.bincount(d.label.to_numpy(int) * 5 + d.pred_b.to_numpy(int), minlength=25).reshape(5, 5))
        starts = rng.integers(0, len(ca) - 5 + 1, size=(2000, 12))
        ix = (starts[:, :, None] + np.arange(5)).reshape(2000, 60)
        out += (np.array([alternate_score_from_cm(np.asarray(ca)[x].sum(0), rulers()["official"]) for x in ix]) -
                np.array([alternate_score_from_cm(np.asarray(cb)[x].sum(0), rulers()["official"]) for x in ix])) / 4
    return {"mean": float(out.mean()), "lo95": float(np.quantile(out, .025)), "hi95": float(np.quantile(out, .975)),
            "nonpositive_fraction": float((out <= 0).mean())}


def alternate_score_from_cm(o, w):
    e = np.outer(o.sum(1), o.sum(0)) / o.sum(); d = (w * e).sum()
    return 1 - (w * o).sum() / d if d else np.nan


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--as-of", default="09:15")
    configure(ap.parse_args().as_of)
    RUN_OUT.mkdir(parents=True, exist_ok=True)
    ds, folds = Dataset(), load_folds(Dataset())
    all_days = sorted({d.date: d for f in folds for d in (*f.train, *f.val)}.values(), key=lambda d: d.date)
    assert max(d.target for d in all_days) < HOLDOUT
    source = pd.read_parquet(ROOT / "reports" / "final_study" / "dev_features.parquet")
    source.date = pd.to_datetime(source.date); source.target = pd.to_datetime(source.target)
    source = source[source.date.isin([d.date for d in all_days])].copy()
    assert len(source) == len(all_days) * 50 and source.target.max() < HOLDOUT
    pm = feature_table(all_days, pm_build, name=None, verbose=True)
    t = source.merge(pm[["date", "symbol", "pm_gap_z", "pm_move_z", "log1p_vol_pre", "pm_available"]], on=["date", "symbol"], validate="one_to_one")
    assert len(t) == len(source) and t.date.max() < HOLDOUT
    groups = json.loads((ROOT / "reports" / "oct10_extension" / "preregistered.json").read_text(encoding="utf-8"))["groups"]
    groups = {"all": sorted(ds.symbols), **groups}
    manifest = {"config_sha256": sha(OUT / "config.json"), "folds": tracking.folds_info(folds),
                "unseen_symbols": unseen_symbols(), "premarket": pm_manifest(), "input_sha256": sha(ROOT / "reports" / "final_study" / "dev_features.parquet"),
                "xgboost_import_version": __import__("xgboost").__version__}
    write("manifest.json", manifest)
    tracker = tracking.Tracker(name=f"xgb_premarket_compare_{AS_OF.replace(':', '')}",
                               tags=["development-only", "xgboost", "premarket", AS_OF], config={**CFG, **manifest, "as_of": AS_OF})
    if tracker.on: print("W&B run:", tracker.run.url, flush=True)
    records, params = [], []
    for fi, f in enumerate(folds):
        tr = t[t.date.isin([d.date for d in f.train]) & ~t.symbol.isin(unseen_symbols())].copy()
        va = t[t.date.isin([d.date for d in f.val])].copy()
        tr_pm = tr.assign(gap_z=tr.pm_gap_z)[["gap_z", "ext_range_z", "label"]]
        va_pm = va.assign(gap_z=va.pm_gap_z)[["gap_z", "ext_range_z", "label"]]
        b0par = G.fit_table(tr[["gap_z", "ext_range_z", "label"]]); pmpar = G.fit_table(tr_pm)
        predictions = {"B0": G.predict_table(va, b0par), "B0_premarket": G.predict_table(va_pm, pmpar)}
        p, d = predict_xgb(tr, va, BASE, CFG["xgboost"]["monotone_constraints"]["XGB_X1"]); predictions["XGB_X1"] = p; params.append({"fold": f.name, "kind": "XGB_X1", **d})
        p, d = predict_xgb(tr, va, PM, CFG["xgboost"]["monotone_constraints"]["XGB_PM_X1"]); predictions["XGB_PM_X1"] = p; params.append({"fold": f.name, "kind": "XGB_PM_X1", **d})
        params.extend([{"fold": f.name, "kind": "B0", **b0par}, {"fold": f.name, "kind": "B0_premarket", **pmpar}])
        for kind, p in predictions.items():
            records.append(va[["date", "target", "symbol", "label", "ret_pct"]].assign(fold=f.name, kind=kind, pred=p))
        print(f"{f.name}: completed", flush=True)
    pred = pd.concat(records, ignore_index=True); met = add_metrics(pred, groups)
    boot = {k: bootstrap(pred, k) for k in CFG["models"] if k != "B0"}
    fold_all = met[met.group.eq("all")].pivot(index="kind", columns="fold", values="score").reindex(CFG["models"])
    fold_all["mean"] = fold_all.mean(axis=1); fold_all["fold4"] = fold_all["fold4"]
    write("reproduction.json", {"B0_fold_scores": fold_all.loc["B0", ["fold1", "fold2", "fold3", "fold4"]].to_dict(),
                                  "B0_mean": float(fold_all.loc["B0", "mean"]),
                                  "B0_premarket_mean": float(fold_all.loc["B0_premarket", "mean"]),
                                  "expected_premarket_mean": {"09:15": 0.4635, "09:28": 0.4875}.get(AS_OF)})
    pred.to_parquet(RUN_OUT / "predictions.parquet", index=False); met.to_csv(RUN_OUT / "fold_metrics.csv", index=False)
    pd.DataFrame(params).to_csv(RUN_OUT / "fold_parameters.csv", index=False); fold_all.to_csv(RUN_OUT / "score_table.csv")
    write("bootstrap_vs_B0.json", boot)
    if tracker.on:
        for i, (kind, z) in enumerate(pred.groupby("kind", sort=False)):
            for fold, q in z.groupby("fold", sort=True):
                mm = met[(met.kind == kind) & (met.fold == fold)].set_index("group")
                tracker.fold(i * 4 + int(fold[-1]), f"{kind}_{fold}",
                             {g: mm.loc[g].drop(labels=["kind", "fold", "group"], errors="ignore").to_dict() for g in mm.index},
                             q.label, q.pred)
        tracker.run.summary.update({f"official/{k}/mean": float(fold_all.loc[k, "mean"]) for k in fold_all.index})
        tracker.run.summary["bootstrap_vs_B0"] = boot
    tracker.finish()
    print(fold_all.round(6).to_string())


if __name__ == "__main__":
    main()
