"""프리마켓 분봉으로 갭 가격을 더 최신으로 — 시점별 실험.

    uv run python work/b/review/premarket/exp.py      (먼저 fetch.py 로 cache/premarket/ 수집)

사전 등록 (실행 전에 고정)
---------------------------
이용 가능 시각 = 봉 시작 + 1분(완성) + 피드 지연. Alpaca 무료 SIP 는 최근 15분 지연 → 기본안은 9:15 에 확정되는 값까지.
- 갭 가격 px(c) = 봉 끝 시각이 c 이하인 마지막 1분봉 종가 (= 봉 시작 ≤ c − 1분)
- 시점 c: 09:00 · 09:05 · **09:15 (판정)** · 09:20 · 09:25 · 09:29.  09:20 이후는 실시간 피드가 있어야 쓸 수 있어 참고만
- gap_z(c) = (px(c) ÷ 전일 종가 − 1) ÷ vol20. ext_range_z 는 그대로. 분봉이 없으면 지금 갭(08:00 봉)으로 채움
- λ, a, b 는 폴드마다 train 으로 다시 고름. A 폴드 4개 val 채점
- 판정 (09:15): 4폴드 평균 +0.005 이상, 3폴드 이상 상승, fold4 하락 없음
- 확인: Alpaca 08:30 가격 vs 교수님 08:00 봉 종가 → 단위·분할 일치. 2% 넘게 다르면 그 행은 분봉을 안 씀
  (10/11 수정: 처음엔 Alpaca 09:00 가격과 0.5% 로 비교했으나, 교수님 08:00 봉 종가가 실제로는 약 08:30 가격이라
   08:30~09:00 에 실제로 움직인 행 14% 를 버리는 문제가 있었음. 판정 시점·기준은 바꾸지 않음)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
for p in ["work/b/features", "work/b/review/explain", "work/b/review/news_gap"]:
    sys.path.insert(0, str(ROOT / p))
import gapz_rule as G  # noqa: E402
import news_gap as NG  # noqa: E402
import premarket as PM  # noqa: E402  (build(): 교수님 08:00·09:00 봉 가격)

CACHE = ROOT / "cache" / "premarket"
OUT = HERE / "out"
CUTS = ["08:30", "09:00", "09:05", "09:15", "09:20", "09:25", "09:29"]
CHECK = "08:30"       # 교수님 08:00 봉 종가와 같은 시점 (10/11 대조로 확인)
MAIN = "09:15"
TOL = 0.02           # 단위·분할 오류만 거름 (2%)


def minute_prices(cache=CACHE):
    fs = sorted(cache.glob("*.parquet"))
    m = pd.concat([pd.read_parquet(f).assign(target=pd.Timestamp(f.stem)) for f in fs], ignore_index=True)
    m["t"] = pd.to_datetime(m.t, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    m["end"] = m.t + pd.Timedelta(minutes=1)
    out = []
    for c in CUTS:
        hh, mm = map(int, c.split(":"))
        lim = m.target + pd.Timedelta(hours=hh, minutes=mm)
        x = m[m.end <= lim].sort_values("t").groupby(["target", "symbol"]).agg(px=("c", "last"), last_t=("t", "last"))
        out.append(x.rename(columns={"px": f"px_{c}", "last_t": f"t_{c}"}))
    pv = pd.concat(out, axis=1).reset_index()
    vol = m[m.end <= m.target + pd.Timedelta(hours=9, minutes=15)].groupby(["target", "symbol"]).v.sum().rename("vol_pre")
    return pv.merge(vol.reset_index(), on=["target", "symbol"], how="left")


def main():
    OUT.mkdir(exist_ok=True)
    t = PM.build()
    pv = minute_prices()
    t = t.merge(pv, on=["target", "symbol"], how="left")
    t = t[t.target <= pd.Timestamp("2026-05-29")]

    # 확인: 같은 시점(09:00) 가격 비교
    r = t[f"px_{CHECK}"] / t.px_0800 - 1
    bad = r.abs() > TOL
    print(f"분봉 있는 행 {t['px_09:15'].notna().mean():.1%} | 08:30 가격 차이 중앙값 {r.abs().median():.5f}, "
          f"{TOL:.0%} 넘는 행 {bad.mean():.2%}")
    for c in CUTS:
        t.loc[bad, f"px_{c}"] = np.nan

    cols = {"지금 (08:00 봉)": "gap_z", "교수님 9:00 봉 (9:30, 진단)": "gap_z_0900"}
    for c in CUTS:
        col = f"gz_{c}"
        t[col] = ((t[f"px_{c}"] / t.prev_close - 1) / t.vol20).fillna(t.gap_z)
        cols[f"분봉 {c}"] = col

    rows = []
    for name, tr, va, _, _ in NG.folds_of(t):
        y = va.label.to_numpy(int)
        un = va.unseen.to_numpy()
        for lab, col in cols.items():
            P = G.fit_table(tr.assign(gap_z=tr[col]))
            p = G.predict_table(va.assign(gap_z=va[col]), P)
            rows.append({"fold": name, "설정": lab, "score": G.score(y, p), "unseen": G.score(y[un], p[un]),
                         "방향 적중": PM.dir_hit(y, p), "급등락 비율": np.isin(p, [0, 4]).mean(), "lam": P["lam"]})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "exp.csv", index=False, encoding="utf-8-sig")
    order = list(cols)
    pv2 = res.pivot_table(index="설정", columns="fold", values="score").reindex(order)
    base = pv2.loc["지금 (08:00 봉)"]
    d = pv2 - base
    pv2["평균"] = pv2.mean(1)
    pv2["변화"] = pv2["평균"] - pv2.loc["지금 (08:00 봉)", "평균"]
    extra = res.groupby("설정")[["unseen", "방향 적중", "급등락 비율"]].mean().reindex(order)
    pd.set_option("display.width", 200)
    print(pd.concat([pv2, extra], axis=1).round(3).to_string())

    dm = d.loc[f"분봉 {MAIN}"]
    up = int((dm > 0).sum())
    ok = dm.mean() >= 0.005 and up >= 3 and dm["fold4"] >= 0
    print(f"\n판정 ({MAIN}): 평균 {dm.mean():+.4f}, 상승 {up}/4, fold4 {dm['fold4']:+.4f} → {'통과' if ok else '미통과'}")
    pd.concat([pv2, extra], axis=1).to_csv(OUT / "exp_summary.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
