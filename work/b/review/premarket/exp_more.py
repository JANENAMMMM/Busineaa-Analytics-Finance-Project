"""분봉 추가 실험 — 강건성 확인(E) + 새 실험 3개(A1·B1·C1).

    uv run python work/b/review/premarket/exp_more.py      (cache/premarket/ 분봉 필요)

사전 등록 (실행 전에 고정, 2026-10-12)
---------------------------------------
기준선: 9:15 규칙 (갭 가격 = 봉 끝 ≤ 9:15 인 마지막 1분봉 종가, exp.py 와 같음). 참고로 B0(08:00 봉)도 같이 출력.
공통: λ, a, b 는 폴드마다 train 으로 다시 고름. A 폴드 4개 val 채점. 분봉이 없거나 08:30 가격이 교수님 08:00 봉과
      2% 넘게 다르면 그 행은 분봉 값을 안 쓰고 원래 값으로 채움 (exp.py 와 같음).

E 강건성 (모델을 바꾸지 않음, 9:15 결과를 믿을 수 있는지)
  E1 판정 시각 흔들기: 9:10 · 9:12 · 9:15 — 셋 다 B0 보다 높으면 "시각에 민감하지 않음"
  E2 지연 가정 20분: 9:10 가격 (= E1 의 9:10)
  E3 월별 점수 (val 월마다 B0 vs 9:15), 처음 보는 종목 폴드별
  E4 분봉 대체 비율: 분봉 값을 못 쓴 행 비율, 그 행과 나머지 행의 점수

새 실험 (각각 9:15 규칙 대비 판정: 4폴드 평균 +0.005 이상 · 3폴드 이상 상승 · fold4 하락 없음)
  A1 VWAP 갭: 갭 가격 = 9:00 < 봉 끝 ≤ 9:15 인 분봉의 거래량 가중 평균가 (전형가격 (고+저+종)/3).
     거래량 0 이면 9:15 가격.
  B1 분봉 ext_range: 시간외 고저 = 교수님 시간외 봉(기준일 16:00 ~ 08:00 봉)과 분봉(봉 끝 ≤ 9:15)을 합친
     최고 − 최저, ÷ 전일 종가 ÷ vol20. 갭은 9:15. 우연 기준선: 같은 날 안에서 종목끼리 섞은 가짜 10개의 95 분위보다
     커야 함.
  C1 프리마켓 변동성 분모: gap_z = 9:15 갭 ÷ (그 종목 지난 20번 9:15 갭의 표준편차, 대상일 전 것만, 최소 10번).
     없으면 9:15 gap_z.
"""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
for p in ["work/b/features", "work/b/review/explain", "work/b/review/news_gap"]:
    sys.path.insert(0, str(ROOT / p))
import gapz_rule as G  # noqa: E402
import news_gap as NG  # noqa: E402
import premarket as PM  # noqa: E402

CACHE = ROOT / "cache" / "premarket"
OUT = HERE / "out"
TOL = 0.02
N_FAKE = 10


def load_minutes():
    fs = sorted(CACHE.glob("*.parquet"))
    m = pd.concat([pd.read_parquet(f).assign(target=pd.Timestamp(f.stem)) for f in fs], ignore_index=True)
    m["t"] = pd.to_datetime(m.t, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    m["end"] = m.t + pd.Timedelta(minutes=1)
    m["mins"] = (m.end - m.target).dt.total_seconds() / 60          # 대상일 0시부터 봉 끝까지 분
    return m.sort_values("t")


def at(hh, mm):
    return hh * 60 + mm


def minute_features(m):
    k = ["target", "symbol"]
    out = []
    for name, lim in [("09:10", at(9, 10)), ("09:12", at(9, 12)), ("09:15", at(9, 15)), ("08:30", at(8, 30))]:
        out.append(m[m.mins <= lim].groupby(k).c.last().rename(f"px_{name}"))
    w = m[(m.mins > at(9, 0)) & (m.mins <= at(9, 15))].copy()
    w["tp"] = (w.h + w.l + w.c) / 3 * w.v
    g = w.groupby(k)
    out.append((g.tp.sum() / g.v.sum().replace(0, np.nan)).rename("px_vwap"))
    u = m[m.mins <= at(9, 15)].groupby(k)
    out += [u.h.max().rename("min_hi"), u.l.min().rename("min_lo")]
    return pd.concat(out, axis=1).reset_index()


def prof_ext_hilo(t):
    w = t[["date", "target"]].drop_duplicates().dropna().copy()
    w["since"] = w.date + pd.Timedelta(hours=16)
    w["cut"] = w.target + pd.Timedelta(hours=9, minutes=30)
    con = duckdb.connect()
    con.register("w", w)
    x = con.execute(f"""
        SELECT w.date, p.symbol, max(p.high) AS ext_hi, min(p.low) AS ext_lo
        FROM read_parquet('{ROOT / "dataset" / "price.parquet"}') p
        JOIN w ON p.datetime >= w.since AND p.known_at < w.cut
        WHERE p.session IN ('post', 'pre') GROUP BY 1, 2""").df()
    con.close()
    return x


def build():
    t = PM.build()
    t = t.merge(minute_features(load_minutes()), on=["target", "symbol"], how="left")
    t = t[t.target <= pd.Timestamp("2026-05-29")].copy()
    t = t.merge(prof_ext_hilo(t), on=["date", "symbol"], how="left")
    bad = (t["px_08:30"] / t.px_0800 - 1).abs() > TOL
    t["has_min"] = t["px_09:15"].notna() & ~bad
    for c in ["px_09:10", "px_09:12", "px_09:15", "px_vwap", "min_hi", "min_lo"]:
        t.loc[~t.has_min, c] = np.nan

    gz = lambda px: ((px / t.prev_close - 1) / t.vol20).fillna(t.gap_z)  # noqa: E731
    for c in ["09:10", "09:12", "09:15"]:
        t[f"gz_{c}"] = gz(t[f"px_{c}"])
    t["gz_vwap"] = gz(t.px_vwap.fillna(t["px_09:15"]))
    hi = np.fmax(t.ext_hi, t.min_hi)
    lo = np.fmin(t.ext_lo, t.min_lo)
    t["ext_min"] = ((hi - lo) / t.prev_close / t.vol20).where(t.has_min, t.ext_range_z)
    g915 = t["px_09:15"] / t.prev_close - 1
    t["g915"] = g915.where(t.has_min, t.gap)
    t = t.sort_values(["symbol", "target"])
    sd = t.groupby("symbol").g915.transform(lambda s: s.shift(1).rolling(20, min_periods=10).std())
    t["gz_pmvol"] = (t.g915 / sd).replace([np.inf, -np.inf], np.nan).fillna(t["gz_09:15"])
    return t.reset_index(drop=True)


def run(t, gap_col, ext_col="ext_range_z", keep_pred=False):
    rows, preds = [], []
    for name, tr, va, _, _ in NG.folds_of(t):
        y = va.label.to_numpy(int)
        un = va.unseen.to_numpy()
        P = G.fit_table(tr.assign(gap_z=tr[gap_col], ext_range_z=tr[ext_col]))
        p = G.predict_table(va.assign(gap_z=va[gap_col], ext_range_z=va[ext_col]), P)
        rows.append({"fold": name, "score": G.score(y, p), "unseen": G.score(y[un], p[un]), "lam": P["lam"]})
        if keep_pred:
            preds.append(va[["target", "symbol", "has_min"]].assign(y=y, p=p, fold=name))
    r = pd.DataFrame(rows).set_index("fold")
    return (r, pd.concat(preds)) if keep_pred else r


def judge(r, base):
    d = r.score - base.score
    up = int((d > 0).sum())
    ok = d.mean() >= 0.005 and up >= 3 and d["fold4"] >= 0
    return d, up, ok


def line(label, r, base=None):
    s = r.score
    txt = f"{label:34s} " + " ".join(f"{v:.4f}" for v in s) + f" | 평균 {s.mean():.4f} | unseen {r.unseen.mean():.4f}"
    if base is not None:
        d = s - base.score
        txt += f" | Δ {d.mean():+.4f} (상승 {int((d > 0).sum())}/4, fold4 {d['fold4']:+.4f})"
    print(txt)


def main():
    OUT.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)
    t = build()
    print(f"행 {len(t):,} | 분봉 값을 쓴 행 {t.has_min.mean():.1%} | VWAP 있는 행 {t.px_vwap.notna().mean():.1%}\n")
    print(f"{'':34s} fold1  fold2  fold3  fold4")

    b0, p0 = run(t, "gap_z", keep_pred=True)
    b915, p915 = run(t, "gz_09:15", keep_pred=True)
    line("B0 (08:00 봉)", b0)
    line("9:15 규칙 (기준선)", b915, b0)

    # E1·E2
    print("\n== E1·E2 시각 흔들기 / 지연 20분 (B0 대비) ==")
    e = {}
    for c in ["09:10", "09:12", "09:15"]:
        e[c] = run(t, f"gz_{c}")
        line(f"분봉 {c}" + (" (지연 20분 = E2)" if c == "09:10" else ""), e[c], b0)
    stable = all(e[c].score.mean() > b0.score.mean() for c in e)
    print(f"→ 세 시각 모두 B0 보다 높음: {stable}")

    # E3
    print("\n== E3 월별 점수 (val) ==")
    mm = p0[["target", "symbol", "y", "p"]].rename(columns={"p": "p0"}).merge(
        p915[["target", "symbol", "p", "has_min"]].rename(columns={"p": "p915"}), on=["target", "symbol"])
    mm["월"] = mm.target.dt.strftime("%Y-%m")
    mon = mm.groupby("월").apply(lambda g: pd.Series({"n": len(g), "B0": G.score(g.y, g.p0), "9:15": G.score(g.y, g.p915)}),
                                include_groups=False)
    mon["차이"] = mon["9:15"] - mon["B0"]
    print(mon.round(4).to_string())
    print(f"→ 9:15 가 높은 달 {int((mon['차이'] > 0).sum())}/{len(mon)}")
    print("처음 보는 종목 폴드별 (B0 → 9:15):", {f: f"{b0.unseen[f]:.3f} → {b915.unseen[f]:.3f}" for f in b0.index})

    # E4
    print("\n== E4 분봉 대체 ==")
    for flag, g in mm.groupby("has_min"):
        print(f"  분봉 {'있음' if flag else '없음(B0 로 대체)'}: {len(g):,}행 ({len(g) / len(mm):.1%}) | "
              f"B0 {G.score(g.y, g.p0):.4f} → 9:15 {G.score(g.y, g.p915):.4f}")

    # 새 실험
    print("\n== 새 실험 (9:15 규칙 대비) ==")
    res = {}
    for key, label, gcol, ecol in [("A1", "A1 VWAP 갭 (9:00~9:15)", "gz_vwap", "ext_range_z"),
                                   ("B1", "B1 분봉 ext_range (9:15)", "gz_09:15", "ext_min"),
                                   ("C1", "C1 프리마켓 변동성 분모", "gz_pmvol", "ext_range_z")]:
        r = run(t, gcol, ecol)
        res[key] = r
        line(label, r, b915)

    rng = np.random.default_rng(0)
    fake = []
    for _ in range(N_FAKE):
        sh = t.copy()
        sh["ext_fake"] = sh.groupby("target").ext_min.transform(lambda s: rng.permutation(s.to_numpy()))
        fake.append((run(sh, "gz_09:15", "ext_fake").score - b915.score).mean())
    f95 = float(np.quantile(fake, 0.95))
    print(f"\nB1 우연 기준선: 섞은 가짜 {N_FAKE}개 Δ평균 95 분위 {f95:+.4f}")

    print("\n== 판정 (사전 등록 기준) ==")
    rows = []
    for k, r in res.items():
        d, up, ok = judge(r, b915)
        if k == "B1":
            ok = ok and d.mean() > f95
        rows.append({"실험": k, **{f: round(v, 4) for f, v in d.items()}, "Δ평균": round(d.mean(), 4), "상승": up,
                     "unseen Δ": round(r.unseen.mean() - b915.unseen.mean(), 4), "채택": ok})
    j = pd.DataFrame(rows).set_index("실험")
    print(j.to_string())
    j.to_csv(OUT / "exp_more_judge.csv", encoding="utf-8-sig")
    mon.to_csv(OUT / "exp_more_month.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
