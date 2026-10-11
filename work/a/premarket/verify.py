"""A 독립 검증: 민영 후보(9:15 분봉 갭)를 제출 저장소 artifacts/premarket/ 로 재현.

    uv run python work/a/premarket/verify.py

민영 검사(work/b/features/check_premarket_rule.py)는 실험 캐시 cache/premarket/ 의 원본 1분봉을
reduce_bars 로 다시 줄여서 cache/premarket_0915/ 를 만들어 쓴다. 여기서는 실제 제출 경로인
artifacts/premarket/ (premarket_rule.py fetch 가 만든 것) 를 그대로 검사한다.

보는 것
    1. look-ahead   저장된 마지막 봉 끝 ≤ 대상일 09:15 (dev 기간만)
    2. 4폴드        기대값 0.4635 (B0 0.4469)
    3. 분봉 없음    빈 저장소 → gapz_rule(B0) 과 완전히 같은 예측
    4. pm == 0 행   갭이 B0 갭과 같은지
홀드아웃(대상일 2026-06-01~)은 폴드에 들어가지 않으므로 점수를 보지 않는다.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "work/b/features"))

from lab.cv import feature_table  # noqa: E402
from lab.folds import load_folds, unseen_symbols  # noqa: E402
from lab.leak import guard  # noqa: E402
from src import Dataset  # noqa: E402

import gapz_rule as G  # noqa: E402
import premarket_rule as PR  # noqa: E402

STORE = ROOT / "artifacts" / "premarket"
DEV_END = "2026-05-28"


def check_leak():
    """dev 기간 저장 파일 전체: (마지막 봉 끝 - 09:15) 의 최댓값."""
    worst, n, fs = pd.Timedelta(-10**6, "s"), 0, []
    for f in sorted(STORE.glob("*.parquet")):
        if f.stem > DEV_END:
            continue
        s = pd.read_parquet(f)
        fs.append(f)
        n += len(s)
        if len(s):
            over = (s.t_0915 + pd.Timedelta(minutes=1)) - (pd.Timestamp(f.stem) + PR.AS_OF)
            worst = max(worst, over.max())
    return len(fs), n, worst


def cv(folds, unseen, store, build):
    days = sorted({d.date: d for f in folds for d in (*f.train, *f.val)}.values(), key=lambda d: d.date)
    t = feature_table(days, lambda day: build(day, store) if store else build(day), name=None, verbose=False)
    t["label"] = t.label.astype(int)
    rows = []
    for f in folds:
        tr = t[t.date.isin({d.date for d in f.train}) & ~t.symbol.isin(unseen)]
        va = t[t.date.isin({d.date for d in f.val})]
        prm = G.fit_table(tr)
        pr = G.predict_table(va, prm)
        u = va.symbol.isin(unseen).to_numpy()
        rows.append({"fold": f.name, "score": G.score(va.label, pr), "unseen": G.score(va.label[u], pr[u]),
                     "분봉갭비율": va.pm.mean() if "pm" in va else 0.0,
                     "lam": prm["lam"], "a": round(prm["a"], 3), "b": round(prm["b"], 3)})
    return t, pd.DataFrame(rows).set_index("fold")


def main():
    ds = Dataset()
    folds = load_folds(ds)
    unseen = set(unseen_symbols())

    nf, n, worst = check_leak()
    print(f"1) look-ahead: dev {nf}일 {n}행, (마지막 봉 끝 - 09:15) 최댓값 {worst} "
          f"→ {'OK' if worst <= pd.Timedelta(0) else '위반'}")

    print("\n2) 4폴드 (제출 저장소)")
    t_pm, r_pm = cv(folds, unseen, STORE, PR.build_gapz_pm)
    print(r_pm.round(4).to_string())
    print(f"   평균 {r_pm.score.mean():.4f} (민영 기대값 0.4635), unseen {r_pm.unseen.mean():.4f}")

    print("\n   B0 (gapz_rule, 같은 폴드·같은 코드)")
    t_b0, r_b0 = cv(folds, unseen, None, lambda day: G.build_gapz(day))
    print(r_b0.round(4).to_string())
    print(f"   평균 {r_b0.score.mean():.4f} (기대값 0.4469), unseen {r_b0.unseen.mean():.4f}")

    d = r_pm.score - r_b0.score
    print(f"\n   차이: 폴드별 {d.round(4).to_dict()}")
    print(f"   평균 {d.mean():+.4f} / 상승 폴드 {int((d > 0).sum())}/4 / fold4 {d.iloc[-1]:+.4f}")
    ok = (d.iloc[-1] >= 0) and (d > 0).sum() >= 3 and d.mean() >= 0.005
    print(f"   채택 기준(fold4 하락 없음 + 3폴드 이상 상승 + 평균 +0.005): {'통과' if ok else '미달'}")

    print("\n3) 분봉 없을 때 B0 와 같은지")
    with tempfile.TemporaryDirectory() as empty:
        for day in [folds[0].train[100], folds[3].val[-1]]:
            a = PR.build_gapz_pm(guard(day), empty).set_index("symbol")
            b = G.build_gapz(guard(day)).set_index("symbol")
            j = a.join(b, rsuffix="_b0", how="outer")
            print(f"   {day.date.date()}: gap_z 최대 차이 {np.nanmax(np.abs(j.gap_z - j.gap_z_b0)):.1e}, "
                  f"행 {len(a)} vs {len(b)}, pm 사용 {int(a.pm.sum())}")

    print("\n4) 분봉 안 쓴 행(pm==0)의 갭이 B0 갭과 같은지")
    m = t_pm.merge(t_b0[["date", "symbol", "gap_z"]], on=["date", "symbol"], suffixes=("", "_b0"))
    same = np.isclose(m.gap_z, m.gap_z_b0, equal_nan=True)
    print(f"   pm==0 전부 일치: {bool(same[m.pm == 0].all())} / 분봉 쓴 행 {m.pm.mean():.1%} "
          f"/ pm==1 중 갭이 바뀐 행 {1 - same[m.pm == 1].mean():.1%}")


if __name__ == "__main__":
    main()
