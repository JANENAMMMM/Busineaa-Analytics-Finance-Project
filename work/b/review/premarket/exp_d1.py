"""D1 프리마켓 후반 움직임 — 개장 직전 움직임이 되돌려지는지/이어지는지.

    uv run python work/b/review/premarket/exp_d1.py

사전 등록 (실행 전에 고정, 2026-10-12)
- 피처 late = (9:15 가격 ÷ 8:30 가격 − 1) ÷ vol20. 분봉 못 쓰는 행은 0.
- 기준선: 9:15 규칙 (exp_more.py 와 같은 표). 붙이는 방식 두 가지, β 는 폴드마다 train score 최대:
    더하기   s = gz_915 + β·σ(gz_915)·latẽ
    같은방향 s = gz_915 · exp(β·latẽ·sign(gz_915))
  (latẽ = train 중앙값·표준편차로 표준화, ±5 자름)
- 판정: 4폴드 평균 +0.005 이상 · 3폴드 이상 상승 · fold4 하락 없음 · 같은 날 안에서 섞은 가짜 10개 Δ평균 95 분위 초과
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp_more as X  # noqa: E402
from exp_more import G, NG  # noqa: E402

BETAS = [-0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5]


def apply(form, b, g, x, sd):
    return g + b * sd * x if form == "더하기" else g * np.exp(b * x * np.sign(g))


def run(t, col):
    rows = []
    for name, tr, va, _, _ in NG.folds_of(t):
        med, sdx = np.nanmedian(tr[col]), np.nanstd(tr[col])
        xt = np.nan_to_num(np.clip((tr[col] - med) / sdx, -5, 5))
        xv = np.nan_to_num(np.clip((va[col] - med) / sdx, -5, 5))
        sd = np.nanstd(tr["gz_09:15"])
        y = va.label.to_numpy(int)
        base_p = G.predict_table(va.assign(gap_z=va["gz_09:15"]), G.fit_table(tr.assign(gap_z=tr["gz_09:15"])))
        base = G.score(y, base_p)
        for form in ["더하기", "같은방향"]:
            best = None
            for b in BETAS:
                gt = apply(form, b, tr["gz_09:15"].to_numpy(), xt, sd)
                P = G.fit_table(tr.assign(gap_z=gt))
                if best is None or P["train_score"] > best[0]:
                    best = (P["train_score"], b, P)
            gv = apply(form, best[1], va["gz_09:15"].to_numpy(), xv, sd)
            p = G.predict_table(va.assign(gap_z=gv), best[2])
            rows.append({"방식": form, "fold": name, "β": best[1], "Δ": G.score(y, p) - base})
    return pd.DataFrame(rows)


def summary(r):
    o = r.pivot(index="방식", columns="fold", values="Δ")
    o["Δ평균"] = o.mean(1)
    o["상승"] = (o[["fold1", "fold2", "fold3", "fold4"]] > 0).sum(1)
    o["β"] = r.groupby("방식").β.apply(lambda s: "/".join(f"{v:g}" for v in s))
    return o


def main():
    t = X.build()
    t["late"] = ((t["px_09:15"] / t["px_08:30"] - 1) / t.vol20).where(t.has_min, 0.0)
    print(f"late: 분봉 있는 행 {t.has_min.mean():.1%}, |late| 중앙값 {t.late.abs().median():.3f}")
    resid = (1 + t.ret_pct / 100) / (t["px_09:15"] / t.prev_close) - 1          # 9:15 가격 → 종가
    ok = t.has_min & resid.notna()
    same = (np.sign(t.late[ok]) == np.sign(resid[ok]))[t.late[ok].abs() > 0.1].mean()
    print(f"후반 움직임 방향 = 9:15 이후 움직임 방향: {same:.3f} (|late|>0.1, 0.5 보다 작으면 되돌림)\n")

    real = summary(run(t, "late"))
    pd.set_option("display.width", 200)
    print("== D1 (9:15 규칙 대비) ==")
    print(real.round(4).to_string())

    rng = np.random.default_rng(0)
    fakes = []
    for _ in range(X.N_FAKE):
        sh = t.copy()
        sh["late_f"] = sh.groupby("target").late.transform(lambda s: rng.permutation(s.to_numpy()))
        fakes.append(summary(run(sh, "late_f"))["Δ평균"].max())
    f95 = float(np.quantile(fakes, 0.95))
    print(f"\n섞은 가짜 {X.N_FAKE}개 Δ평균(두 방식 중 큰 것) 95 분위 {f95:+.4f}")
    for form, row in real.iterrows():
        ok = row["Δ평균"] >= 0.005 and row["상승"] >= 3 and row["fold4"] >= 0 and row["Δ평균"] > f95
        print(f"판정 {form}: {'채택' if ok else '불채택'}")
    real.to_csv(X.OUT / "exp_d1.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
