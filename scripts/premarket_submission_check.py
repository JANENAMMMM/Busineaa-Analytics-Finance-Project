"""Checks on the packaged premarket model, logged to W&B with the final fit.

1. src/model.py is a standalone re-implementation (the grader overwrites src/
   except model.py and artifacts/, so it cannot import work/b). It must give the
   same labels as B's premarket_rule.predict with the same constants.
2. With no premarket file at all it must give exactly the B0 labels.
3. No stored bar may end after the cutoff as_of (t_asof is the bar start).
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "work" / "b" / "features"))

from src import Dataset
import src.model as SM
from lab import tracking
from lab.folds import load_folds
import premarket_rule as PM

SAMPLE = 60


def main():
    model = SM.load_model()
    params = model.params
    assert model.premarket and params.get("as_of"), params
    as_of = params["as_of"]
    prov = json.loads((SM.ARTIFACTS / "premarket_rule.provenance.json").read_text(encoding="utf-8"))
    ds = Dataset()
    days = list(ds.days())
    rng = np.random.default_rng(2026)
    pick = [days[i] for i in sorted(rng.choice(len(days), SAMPLE, replace=False))]

    same, n = 0, 0
    for d in pick:
        a = model.predict(d).set_index("symbol").label
        b = PM.predict(d, params).set_index("symbol").label.reindex(a.index)
        same += int((a == b).sum()); n += len(a)

    # Fallback: point the model at an empty artifacts folder copy (rule json only).
    tmp = Path(tempfile.mkdtemp())
    shutil.copy(SM.ARTIFACTS / "premarket_rule.json", tmp / "premarket_rule.json")
    real = SM.ARTIFACTS
    SM.ARTIFACTS = tmp
    try:
        fb_same, fb_n = 0, 0
        b0 = SM.Model(params, premarket=False)
        for d in pick[:20]:
            a = SM.Model(params, premarket=True).predict(d).label.to_numpy()
            b = b0.predict(d).label.to_numpy()
            fb_same += int((a == b).sum()); fb_n += len(a)
    finally:
        SM.ARTIFACTS = real
        shutil.rmtree(tmp, ignore_errors=True)

    store = PM.store_for(as_of)
    late, rows = 0, 0
    for p in sorted(store.glob("*.parquet")):
        z = pd.read_parquet(p)
        t = pd.to_datetime(z.t_asof)
        hh, mm = map(int, as_of.split(":"))
        lim = pd.Timestamp(p.stem) + pd.Timedelta(hours=hh, minutes=mm)
        late += int(((t + pd.Timedelta(minutes=1)) > lim).sum()); rows += len(z)

    result = {"equal_to_B_rule": same / n, "rows_compared": n,
              "fallback_equal_to_B0": fb_same / fb_n, "fallback_rows": fb_n,
              "bars_ending_after_cutoff": late, "store_rows": rows,
              "params": params, "provenance": prov}
    assert same == n, "src/model.py disagrees with premarket_rule"
    assert fb_same == fb_n, "fallback does not reproduce B0"
    assert late == 0, f"a stored bar ends after {as_of}"

    folds = load_folds(ds)
    cfg = {"stage": "final_fit_and_submission_check", "as_of": params["as_of"],
           "params": params, "provenance": prov, "folds_hash": tracking.folds_info(folds)["folds_hash"],
           "fit_data": "all public labels, targets 2024-08-14..2026-09-14, all 50 symbols",
           "checks": ["model==B rule", "no file -> B0", "bar end <= as_of"]}
    tr = tracking.Tracker(name=f"premarket_{as_of.replace(':', '')}_final_fit", tags=["premarket", as_of, "submission", "final_fit"],
                          config=cfg, job_type="final_fit")
    if tr.on:
        for k, v in result.items():
            if isinstance(v, (int, float)):
                tr.run.summary[k] = v
        tr.run.summary["train_score"] = params["train_score"]
        tr.run.finish()
    out = ROOT / "reports" / "premarket_submission_check.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("params", "provenance")}, indent=2))


if __name__ == "__main__":
    main()
