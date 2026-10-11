"""무료 실시간 IEX 피드로 더 늦은 시점을 쓸 수 있나 — SIP 9:15(판정안)와 비교.

    uv run python work/b/review/premarket/exp_iex.py      (먼저 fetch.py iex 로 cache/premarket_iex/ 수집)

사전 등록 (IEX 결과를 보기 전에 고정, 2026-10-11)
- 배경: SIP 9:15 는 판정 통과(+0.017). 9:20 이후는 실시간 SIP(유료)가 있어야 해서 참고로만 둠.
  무료 요금제의 IEX 웹소켓은 실시간 → 9:30 직전 IEX 가격은 무료로 받을 수 있음
- 이용 가능 시각: IEX 는 실시간이므로 봉 끝 시각. 봉 끝 ≤ c 인 마지막 1분봉 종가
- 비교안 (무료로 9:30 전에 실제로 받을 수 있는 조합)
    H  "무료 혼합": IEX 9:29 가격이 있으면 그것, 없으면 SIP 9:15, 없으면 지금 갭(08:00 봉)
  참고: IEX 단독 9:15 · 9:25 · 9:29 (IEX 가 없으면 지금 갭), SIP 9:29 (유료 실시간 상한)
- 기준: H 가 SIP 9:15 보다 4폴드 평균 +0.005 이상 & 3폴드 이상 높으면 "무료로 더 늦은 시점을 쓸 근거 있음"
- 단위 확인: IEX 08:30 가격 vs 교수님 08:00 봉 종가, 2% 넘으면 그 행의 IEX 는 안 씀 (SIP 와 같은 방식)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
import exp as X  # noqa: E402  (minute_prices, G, NG, PM)

IEX = ROOT / "cache" / "premarket_iex"


def main():
    t = X.PM.build()
    sip = X.minute_prices()
    iex = X.minute_prices(IEX)
    iex = iex.rename(columns={c: c.replace("px_", "iex_").replace("t_", "it_") for c in iex.columns if c.startswith(("px_", "t_"))})
    iex = iex.rename(columns={"vol_pre": "iex_vol"})
    t = t.merge(sip, on=["target", "symbol"], how="left").merge(iex, on=["target", "symbol"], how="left")
    t = t[t.target <= pd.Timestamp("2026-05-29")]

    for pre, lab in [("px_", "SIP"), ("iex_", "IEX")]:
        r = t[f"{pre}08:30"] / t.px_0800 - 1
        bad = r.abs() > X.TOL
        cols = [c for c in t.columns if c.startswith(pre) and c[len(pre):][:2].isdigit()]
        t.loc[bad, cols] = np.nan
        print(f"{lab}: 9:29 가격 있는 행 {t[pre + '09:29'].notna().mean():.1%}, 08:30 단위 확인 2% 초과 {bad.mean():.2%}")
    m = t["iex_09:29"].notna() & t["px_09:29"].notna()
    print(f"IEX vs SIP 9:29 가격 차이 중앙값 {(t.loc[m, 'iex_09:29'] / t.loc[m, 'px_09:29'] - 1).abs().median():.5f}, "
          f"IEX 마지막 체결이 9:20 이전인 행 {(t.loc[t['iex_09:29'].notna(), 'it_09:29'].dt.time < pd.Timestamp('09:20').time()).mean():.1%}")

    def gz(px):
        return (px / t.prev_close - 1) / t.vol20

    t["c_now"] = t.gap_z
    t["c_sip915"] = gz(t["px_09:15"]).fillna(t.gap_z)
    t["c_H"] = gz(t["iex_09:29"]).fillna(gz(t["px_09:15"])).fillna(t.gap_z)
    for c in ["09:15", "09:25", "09:29"]:
        t[f"c_iex{c}"] = gz(t[f"iex_{c}"]).fillna(t.gap_z)
    t["c_sip929"] = gz(t["px_09:29"]).fillna(t.gap_z)
    cols = {"지금 B0": "c_now", "SIP 9:15 (판정안)": "c_sip915", "H 무료 혼합 (IEX 9:29 → SIP 9:15)": "c_H",
            "IEX 9:15": "c_iex09:15", "IEX 9:25": "c_iex09:25", "IEX 9:29": "c_iex09:29", "SIP 9:29 (유료 상한)": "c_sip929"}

    rows = []
    for name, tr, va, _, _ in X.NG.folds_of(t):
        y, un = va.label.to_numpy(int), va.unseen.to_numpy()
        for lab, col in cols.items():
            P = X.G.fit_table(tr.assign(gap_z=tr[col]))
            p = X.G.predict_table(va.assign(gap_z=va[col]), P)
            rows.append({"fold": name, "설정": lab, "score": X.G.score(y, p), "unseen": X.G.score(y[un], p[un]),
                         "방향 적중": X.PM.dir_hit(y, p)})
    res = pd.DataFrame(rows)
    res.to_csv(X.OUT / "exp_iex.csv", index=False, encoding="utf-8-sig")
    pv = res.pivot_table(index="설정", columns="fold", values="score").reindex(list(cols))
    d = pv.sub(pv.loc["SIP 9:15 (판정안)"])
    pv["평균"] = pv.mean(1)
    pv["B0 대비"] = pv["평균"] - pv.loc["지금 B0", "평균"]
    pv["SIP 9:15 대비"] = pv["평균"] - pv.loc["SIP 9:15 (판정안)", "평균"]
    ex = res.groupby("설정")[["unseen", "방향 적중"]].mean().reindex(list(cols))
    pd.set_option("display.width", 220)
    print(pd.concat([pv, ex], axis=1).round(3).to_string())
    dh = d.loc["H 무료 혼합 (IEX 9:29 → SIP 9:15)"]
    ok = dh.mean() >= 0.005 and (dh > 0).sum() >= 3
    print(f"\n기준 (H vs SIP 9:15): 평균 {dh.mean():+.4f}, 높은 폴드 {(dh > 0).sum()}/4 → {'무료로 더 늦은 시점 근거 있음' if ok else '근거 없음 (SIP 9:15 유지)'}")


if __name__ == "__main__":
    main()
