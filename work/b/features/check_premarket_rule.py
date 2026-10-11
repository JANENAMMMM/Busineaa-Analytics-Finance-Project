"""premarket_rule.py 확인 — 받기·줄이기가 맞는지, 9:15 이후 가격이 없는지, 실험 결과(4폴드 0.464)가 그대로 나오는지,
분봉이 없을 때 B0 와 똑같은지.

    uv run python work/b/features/check_premarket_rule.py          (cache/premarket/ 의 원본 1분봉 필요: work/b/review/premarket/fetch.py)

검사용 저장소는 cache/premarket_0915/ (git 제외) 에 원본 1분봉을 reduce_bars 로 줄여서 만든다 (제출 때 fetch 와 같은 함수).
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "exp"))

from lab.cv import feature_table  # noqa: E402
from lab.folds import unseen_symbols  # noqa: E402
from lab.leak import guard  # noqa: E402
from common import load_table  # noqa: E402
import gapz_rule as G  # noqa: E402
import premarket_rule as PR  # noqa: E402

RAW = ROOT / "cache" / "premarket"
STORE = ROOT / "cache" / "premarket_0915"


def make_store():
    STORE.mkdir(parents=True, exist_ok=True)
    fs = sorted(RAW.glob("*.parquet"))
    for f in fs:
        PR.reduce_bars(pd.read_parquet(f), f.stem).to_parquet(STORE / f.name, index=False)
    return len(fs)


def check_leak():
    worst, n = pd.Timedelta(-10**6, "s"), 0
    for f in sorted(STORE.glob("*.parquet")):
        s = pd.read_parquet(f)
        lim = pd.Timestamp(f.stem) + PR.AS_OF
        over = (s.t_0915 + pd.Timedelta(minutes=1)) - lim
        worst = max(worst, over.max())
        n += len(s)
    return n, worst


def cv(folds, unseen, store):
    all_days = sorted({d.date: d for f in folds for d in (*f.train, *f.val)}.values(), key=lambda d: d.date)
    t = feature_table(all_days, lambda day: PR.build_gapz_pm(day, store), name=None, verbose=False)
    t["label"] = t.label.astype(int)
    rows = []
    for f in folds:
        tr = t[t.date.isin({d.date for d in f.train}) & ~t.symbol.isin(unseen)]
        va = t[t.date.isin({d.date for d in f.val})]
        prm = G.fit_table(tr)
        pr = G.predict_table(va, prm)
        u = va.symbol.isin(unseen).values
        rows.append({"fold": f.name, "score": G.score(va.label, pr), "unseen": G.score(va.label[u], pr[u]),
                     "분봉 갭 비율": va.pm.mean(), "λ": prm["lam"], "a": round(prm["a"], 3), "b": round(prm["b"], 3)})
    return t, pd.DataFrame(rows).set_index("fold")


def main():
    table, folds = load_table()
    unseen = set(unseen_symbols())

    # 1) 저장소 만들기 (원본 1분봉 → 종목당 한 줄)
    print(f"저장소: 대상일 {make_store()}개 → {STORE}")

    # 2) 받기: 이틀을 새로 받아 줄인 값이 원본 캐시를 줄인 값과 같은지 (키 있을 때만)
    days2 = [pd.Timestamp("2025-03-12"), pd.Timestamp("2026-04-15")]
    with tempfile.TemporaryDirectory() as tmp:
        syms = sorted(table.symbol.unique()) + ["SPY", "QQQ"]
        if PR.fetch(days2, syms, tmp, verbose=False):
            for d in days2:
                a = PR.load_store(d, tmp).set_index("symbol").sort_index()
                b = PR.load_store(d, STORE).set_index("symbol").sort_index()
                j = a.join(b, rsuffix="_c", how="outer")
                print(f"받기 비교 {d.date()}: 종목 {len(a)} vs {len(b)}, px_0915 최대 차이 "
                      f"{np.nanmax(np.abs(j.px_0915 - j.px_0915_c)):.2e}")

    # 3) look-ahead: 저장된 마지막 봉이 모두 09:15 전에 끝났는지
    n, worst = check_leak()
    print(f"look-ahead: {n}행, (마지막 봉 끝 − 대상일 09:15) 최댓값 {worst} → {'OK' if worst <= pd.Timedelta(0) else '위반'}")

    # 4) 피처: 분봉 없는 저장소 → gapz_rule 과 완전히 같은지 (guard 아래)
    with tempfile.TemporaryDirectory() as empty:
        for day in [folds[0].train[100], folds[3].val[-1]]:
            a = PR.build_gapz_pm(guard(day), empty).set_index("symbol")
            b = G.build_gapz(guard(day)).set_index("symbol")
            print(f"분봉 없음 {day.date.date()}: gap_z 최대 차이 {np.nanmax(np.abs(a.gap_z - b.gap_z)):.1e}, "
                  f"pm 사용 {int(a.pm.sum())}")

    # 5) 4폴드 성능 (실험 exp.py 의 9:15 결과 0.4635 와 같아야 함)
    t, r = cv(folds, unseen, STORE)
    print("\n", r.round(4).to_string())
    print(f"4폴드 평균 {r.score.mean():.4f} (기대값 0.4635, B0 0.4469), unseen {r.unseen.mean():.4f}")
    m = t.merge(table[["date", "symbol", "gap_z"]], on=["date", "symbol"], suffixes=("", "_b0"))
    same = np.isclose(m.gap_z, m.gap_z_b0, equal_nan=True)
    print(f"분봉 안 쓴 행이 B0 갭과 같은지: {bool(same[m.pm == 0].all())} (분봉 쓴 행 {m.pm.mean():.1%})")

    # 6) predict(day) 형태
    day = folds[3].val[0]
    prm = G.fit_table(t[t.date.isin({d.date for d in folds[3].train}) & ~t.symbol.isin(unseen)])
    out = PR.predict(guard(day), prm, STORE)
    print(f"\npredict({day.date.date()}): {len(out)}행 / day.symbols {len(day.symbols)}개, "
          f"label 분포 {out.label.value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
