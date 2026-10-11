"""새 가설 H1·H2·H3 — 9:15 규칙 위에서.

    uv run python work/b/review/premarket/exp_h.py

사전 등록 (실행 전에 고정, 2026-10-12). 기준선 = 9:15 규칙. 판정: 4폴드 평균 +0.005 이상 · 3폴드 이상 상승 · fold4 하락 없음.
H1 최근 5분 VWAP: 갭 가격 = 9:10 < 봉 끝 ≤ 9:15 분봉의 거래량 가중 평균가 (전형가격). 없으면 9:15 가격.
   (근거: 갭이 최신일수록 좋음 + 15분 VWAP 이 처음 보는 종목 +0.018. 15분 평균은 신선도를 잃었을 수 있음)
H2 분봉 없는 행 따로: 분봉 있는 행은 9:15 갭으로 학습한 λ·a·b, 분봉 없는 행은 B0(08:00 갭)으로 학습한 λ·a·b.
   (근거: 분봉 없는 13.5% 행이 B0 0.333 → 0.325 로 손해)
H3 유동성 섞기: 9:00~9:15 체결 수가 train 의 분봉 있는 행 하위 25% 미만이면
   gz = w·gz_915 + (1−w)·gz_0800, w ∈ {0, .25, .5, .75, 1} 를 train score 로 고름. 나머지 행은 gz_915.
   우연 기준선: 체결 수를 같은 날 안에서 섞은 가짜 10개의 Δ평균 95 분위 초과.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp_more as X  # noqa: E402
from exp_more import G, NG  # noqa: E402

WS = [0.0, 0.25, 0.5, 0.75, 1.0]


def extra(m):
    k = ["target", "symbol"]
    w5 = m[(m.mins > X.at(9, 10)) & (m.mins <= X.at(9, 15))].copy()
    w5["tp"] = (w5.h + w5.l + w5.c) / 3 * w5.v
    g = w5.groupby(k)
    vw5 = (g.tp.sum() / g.v.sum().replace(0, np.nan)).rename("px_vwap5")
    nt = m[(m.mins > X.at(9, 0)) & (m.mins <= X.at(9, 15))].groupby(k).n.sum().rename("n_pre")
    return pd.concat([vw5, nt], axis=1).reset_index()


def score_rows(va, p, y):
    return G.score(y, p), G.score(y[va.unseen.to_numpy()], p[va.unseen.to_numpy()])


def run_h2(t):
    rows = []
    for name, tr, va, _, _ in NG.folds_of(t):
        y = va.label.to_numpy(int)
        P_min = G.fit_table(tr[tr.has_min].assign(gap_z=tr.loc[tr.has_min, "gz_09:15"]))
        P_fb = G.fit_table(tr.assign(gap_z=tr.gap_z))
        p = np.where(va.has_min, G.predict_table(va.assign(gap_z=va["gz_09:15"]), P_min),
                     G.predict_table(va.assign(gap_z=va.gap_z), P_fb))
        s, u = score_rows(va, p, y)
        rows.append({"fold": name, "score": s, "unseen": u})
    return pd.DataFrame(rows).set_index("fold")


def run_h3(t, ncol):
    rows = []
    for name, tr, va, _, _ in NG.folds_of(t):
        y = va.label.to_numpy(int)
        thr = np.nanquantile(tr.loc[tr.has_min, ncol], 0.25)
        lo_tr = tr.has_min & (tr[ncol] < thr)
        lo_va = va.has_min & (va[ncol] < thr)
        best = None
        for w in WS:
            g = np.where(lo_tr, w * tr["gz_09:15"] + (1 - w) * tr.gap_z, tr["gz_09:15"])
            P = G.fit_table(tr.assign(gap_z=g))
            if best is None or P["train_score"] > best[0]:
                best = (P["train_score"], w, P)
        g = np.where(lo_va, best[1] * va["gz_09:15"] + (1 - best[1]) * va.gap_z, va["gz_09:15"])
        s, u = score_rows(va, G.predict_table(va.assign(gap_z=g), best[2]), y)
        rows.append({"fold": name, "score": s, "unseen": u, "w": best[1]})
    return pd.DataFrame(rows).set_index("fold")


def verdict(label, r, base, extra_ok=True):
    d = r.score - base.score
    up = int((d > 0).sum())
    ok = d.mean() >= 0.005 and up >= 3 and d["fold4"] >= 0 and extra_ok
    X.line(label, r, base)
    return {"가설": label, **{f: round(v, 4) for f, v in d.items()}, "Δ평균": round(d.mean(), 4), "상승": up,
            "unseen Δ": round(r.unseen.mean() - base.unseen.mean(), 4), "채택": ok}


def main():
    t = X.build()
    t = t.merge(extra(X.load_minutes()), on=["target", "symbol"], how="left")
    t.loc[~t.has_min, ["px_vwap5", "n_pre"]] = np.nan
    t["gz_vwap5"] = ((t.px_vwap5.fillna(t["px_09:15"]) / t.prev_close - 1) / t.vol20).fillna(t["gz_09:15"])
    print(f"5분 VWAP 있는 행 {t.px_vwap5.notna().mean():.1%}, 체결 수 중앙값 {t.n_pre.median():.0f}\n")
    print(f"{'':34s} fold1  fold2  fold3  fold4")
    base = X.run(t, "gz_09:15")
    X.line("9:15 규칙 (기준선)", base)

    out = [verdict("H1 최근 5분 VWAP", X.run(t, "gz_vwap5"), base),
           verdict("H2 분봉 없는 행 따로 학습", run_h2(t), base)]
    h3 = run_h3(t, "n_pre")
    print(f"   H3 train 이 고른 w (1 = 9:15 그대로): {h3.w.tolist()}")
    rng = np.random.default_rng(0)
    fakes = []
    for _ in range(X.N_FAKE):
        sh = t.copy()
        sh["n_fake"] = sh.groupby("target").n_pre.transform(lambda s: rng.permutation(s.to_numpy()))
        fakes.append((run_h3(sh, "n_fake").score - base.score).mean())
    f95 = float(np.quantile(fakes, 0.95))
    print(f"   H3 섞은 가짜 {X.N_FAKE}개 Δ평균 95 분위 {f95:+.4f}")
    out.append(verdict("H3 유동성 낮은 행 섞기", h3, base, (h3.score - base.score).mean() > f95))

    j = pd.DataFrame(out).set_index("가설")
    pd.set_option("display.width", 200)
    print("\n== 판정 ==")
    print(j.to_string())
    j.to_csv(X.OUT / "exp_h.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
