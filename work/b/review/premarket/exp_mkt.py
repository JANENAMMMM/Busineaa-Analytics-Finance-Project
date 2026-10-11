"""분봉 9:15 갭 위에 시장(SPY·QQQ) 움직임과 프리마켓 거래량을 더하면 더 오르나.

    uv run python work/b/review/premarket/exp_mkt.py

사전 등록 (실행 전에 고정, 2026-10-11)
- 기준선: 분봉 SIP 9:15 갭 규칙 (exp.py 판정안, dev 0.464). 아래 피처는 이 규칙 위에 붙임
- 피처 (모두 9:15 에 확정된 값만)
    mkt_mv   시장 움직임 = SPY·QQQ 의 8:30 → 9:15 수익률 평균 (그날 모든 종목에 같은 값)       형태 D · T · S
    mkt_with 시장이 갭과 같은 방향인가 = mkt_mv × sign(gap_z)                                   형태 T · S
    vol_abn  프리마켓 거래량 이상치 = log(1 + 8:00~9:15 거래량) − 종목별 직전 20 대상일 평균      형태 T · S
  형태는 batch.py 와 같음 (T 갭 신뢰도, S 급등락 경계, D 방향 더하기). 파라미터는 폴드마다 train 으로
- 우연 기준선: 가짜 피처 각 10개. mkt_mv·mkt_with 는 날짜끼리 섞음(하루 안에서는 같은 값이라), vol_abn 은 같은 날 안에서 종목끼리 섞음
- 채택: 4폴드 평균 +0.005 이상 & 3폴드 이상 상승 & fold4 하락 없음 & 가짜 95 분위 초과 (기준선 = 9:15 규칙)
- 업종 ETF 는 여기서 하나라도 채택될 때만 받아서 확인
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "work/b/review/batch"))
import exp as X  # noqa: E402
import batch as B  # noqa: E402

N_FAKE = 10
TESTS = [("mkt_mv", ["D", "T", "S"]), ("mkt_with", ["T", "S"]), ("vol_abn", ["T", "S"])]
NAMES = {"mkt_mv": "시장 움직임 (SPY·QQQ 8:30→9:15)", "mkt_with": "시장이 갭과 같은 방향", "vol_abn": "프리마켓 거래량 이상치"}


def build():
    t = X.PM.build()
    pv = X.minute_prices()
    t = t.merge(pv, on=["target", "symbol"], how="left")
    t = t[t.target <= pd.Timestamp("2026-05-29")].copy()
    bad = (t[f"px_{X.CHECK}"] / t.px_0800 - 1).abs() > X.TOL
    for c in [c for c in t.columns if c.startswith("px_") and c[3:5].isdigit()]:
        t.loc[bad, c] = np.nan
    t["vol_pre"] = t.vol_pre.where(~bad)
    t["gap_z"] = ((t["px_09:15"] / t.prev_close - 1) / t.vol20).fillna(t.gap_z)     # 기준선 = 9:15 규칙

    m = pv[pv.symbol.isin(["SPY", "QQQ"])].copy()
    m["mv"] = m["px_09:15"] / m["px_08:30"] - 1
    mk = m.groupby("target").mv.mean().rename("mkt_mv")
    t = t.merge(mk.reset_index(), on="target", how="left")
    t["mkt_mv"] = t.mkt_mv.fillna(0.0)
    t["mkt_with"] = t.mkt_mv * np.sign(t.gap_z)

    t = t.sort_values(["symbol", "target"])
    lv = np.log1p(t.vol_pre)
    t["vol_abn"] = lv - lv.groupby(t.symbol).transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    t["vol_abn"] = t.vol_abn.fillna(0.0)
    print(f"시장 움직임 있는 날 {(t.mkt_mv != 0).mean():.1%}, |중앙값| {t.mkt_mv.abs().median():.4f} · "
          f"거래량 이상치 있는 행 {(t.vol_abn != 0).mean():.1%}")
    return t.reset_index(drop=True)


def fake(df, src, k, rng):
    if src == "vol_abn":
        return df.groupby("date")[src].transform(lambda s: rng.permutation(s.to_numpy()))
    days = df.target.drop_duplicates().to_numpy()
    perm = dict(zip(days, rng.permutation(days)))
    val = df.groupby("target")[src].first()
    return df.target.map(perm).map(val).fillna(0.0)


def main():
    rng = np.random.default_rng(2026)
    t = build()
    for src, _ in TESTS:                    # 가짜는 전체 표에서 한 번 만들고 폴드마다 같이 자름
        for k in range(N_FAKE):
            t[f"fk_{src}_{k}"] = fake(t, src, k, rng)
    rows = []
    for name, tr, va, _, _ in X.NG.folds_of(t):
        P = X.G.fit_table(tr)
        y, u = va.label.to_numpy(int), va.unseen.to_numpy()
        base = X.G.predict_table(va, P)
        rows.append({"fold": name, "test": "기준선 (9:15)", "form": "-", "fake": False, "param": P["lam"],
                     "score": X.G.score(y, base), "unseen": X.G.score(y[u], base[u])})
        for src, forms in TESTS:
            for form in forms:
                for feat, is_fake in [(src, False)] + [(f"fk_{src}_{k}", True) for k in range(N_FAKE)]:
                    p, prm = B.fit_form(tr, va, feat, form, P)
                    rows.append({"fold": name, "test": src, "form": form, "fake": is_fake, "param": prm,
                                 "score": X.G.score(y, p), "unseen": X.G.score(y[u], p[u])})
    r = pd.DataFrame(rows)
    bs = r[r.test == "기준선 (9:15)"].set_index("fold")
    r["delta"] = r.score - r.fold.map(bs.score)
    r["d_unseen"] = r.unseen - r.fold.map(bs.unseen)
    r.to_csv(X.OUT / "exp_mkt.csv", index=False, encoding="utf-8-sig")
    print("기준선 (9:15):", bs.score.round(3).to_dict(), "평균", round(bs.score.mean(), 4))

    out = []
    for (src, form), g in r[(r.test != "기준선 (9:15)")].groupby(["test", "form"], sort=False):
        real = g[~g.fake].set_index("fold")
        fakes = g[g.fake].assign(k=lambda d: d.groupby("fold").cumcount()).groupby("k").delta.mean()
        null95 = float(np.quantile(fakes, 0.95))
        crit = real.delta.mean() >= 0.005 and (real.delta > 0).sum() >= 3 and real.delta["fold4"] >= 0
        out.append({"피처": NAMES[src], "형태": form, **{f: round(real.delta[f], 4) for f in real.index},
                    "평균": round(real.delta.mean(), 4), "상승": int((real.delta > 0).sum()),
                    "unseen": round(real.d_unseen.mean(), 4), "가짜95": round(null95, 4),
                    "train 선택": " / ".join(str(x) for x in real.param),
                    "채택": bool(crit and real.delta.mean() > null95)})
    j = pd.DataFrame(out)
    j.to_csv(X.OUT / "exp_mkt_judge.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(j.to_string(index=False))
    print("\n채택:", int(j.채택.sum()), "개")


if __name__ == "__main__":
    main()
