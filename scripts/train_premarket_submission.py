"""Fit the submitted premarket rule once, on every public labelled day.

The cutoff was chosen before this refit on one yardstick (reports/premarket_compare/):
09:29 beat 09:28 on development and holdout and in every symbol group, and 09:28
had beaten 09:15 and B0 on both (reports/premarket_holdout/result.json). With the cutoff fixed, the final
constants use all public labels, holdout included, as is usual for a final fit.

Writes artifacts/premarket_rule.json, the only file src/model.py reads, plus a
provenance sidecar. Run once before packaging; never during evaluation.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "work" / "b" / "features"))

from src import Dataset
from src.paths import ARTIFACTS
import premarket_rule as PM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default="09:29")
    ap.add_argument("--last-target", default="2026-09-14",
                    help="last target date whose label is used (default: every public label)")
    args = ap.parse_args()
    as_of = PM._tag(args.as_of)
    store = PM.store_for(as_of)
    last = pd.Timestamp(args.last_target)
    days = [d for d in Dataset().days() if pd.Timestamp(d.target) <= last]
    assert days
    targets = {str(d.target.date()) for d in days}
    files = sorted(p for p in store.glob("*.parquet") if p.stem in targets)
    missing = sorted(targets - {p.stem for p in files})
    params = PM.fit(days, store, as_of)
    assert params["as_of"] == as_of
    (ARTIFACTS / "premarket_rule.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    h = hashlib.sha256()
    for p in files:
        h.update(p.name.encode()); h.update(p.read_bytes())
    prov = {"as_of": as_of, "store": str(store.relative_to(ROOT)), "days": len(days),
            "first_target": str(min(d.target for d in days).date()),
            "last_target": str(max(d.target for d in days).date()),
            "store_files_used": len(files), "store_sha256": h.hexdigest(),
            "missing_targets_fall_back_to_B0": missing,
            "symbols": "all public symbols in day.symbols",
            "selection_evidence": ["reports/premarket_holdout/result.json", "reports/premarket_compare/0929_vs_0928.json"]}
    (ARTIFACTS / "premarket_rule.provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    print(json.dumps({"params": params, **prov}, indent=2))


if __name__ == "__main__":
    main()
