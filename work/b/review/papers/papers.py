"""논문 기반 실험 A·B·C — 갭이 장중(09:00 → 종가)에 되돌려지는 조건.

    uv run python work/b/review/papers/papers.py

분석용 (제출 모델은 바꾸지 않음. 홀드아웃은 이미 열렸으므로 결과로 모델을 고르지 않음).

사전 등록 (실행 전에 고정)
--------------------------
공통
- 대상: |gap_z| ≥ 0.25, post ≠ 0. post = (1 + ret) / (1 + gap) - 1 (09:00 시간외 가격 → 종가)
- 지표: 갭 방향 유지율 = P(sign(post) = sign(gap)). 낮을수록 장중에 되돌려짐
- 발견 기간 = 대상일 ≤ 2026-02-12, 확인 기간 = fold4 val (기준일 2026-03-04 ~ 05-28)
- 통과: 발견 기간 차이가 논문이 예측한 방향이고 날짜 묶음 부트스트랩(1,000회) 95% 구간이 0 을 안 걸침,
        그리고 확인 기간 차이가 같은 부호

A  관심이 몰린 종목의 위로 갭은 더 되돌려지나 (Berkman et al. 2012)
   관심 급증 = 밤사이 태그 기사 수 또는 Reddit 언급 글 수(채민 v4 후보 글 수)의 log 가
               그 종목 직전 20거래일 평균보다 발견 기간 90 분위 이상 높음
   예측: 위로 갭에서 (급증 - 급증 아님) 유지율 < 0. 아래로 갭은 같은 계산을 참고로만

B  시장·업종 따라 생긴 갭은 되돌려지나 (Wang 2025)
   업종 갭 = 같은 GICS 업종의 다른 종목 gap_z 평균 (업종은 아래 SECTOR, 공개 정보로 고정)
   남 따라 생긴 갭 = 업종 갭과 같은 방향이고 |gap_z - 업종 갭| < 0.5·|gap_z|, 그리고 제목에 기업명이 나온 기사 없음
   비교: 자기 갭 (|gap_z - 업종 갭| ≥ 0.5·|gap_z|)
   예측: (남 따라 - 자기) 유지율 < 0.  참고로 업종 대신 시장 전체 평균으로도 같은 계산

C  나쁜 뉴스 갭은 과하게 반응하나 (UWA, 시간외 정보와 밤사이 수익률)
   대상: 실적 발표 밤 (earnings.known_at 이 기준일 16:00 ~ 대상일 09:30)
   예측: (아래로 갭 - 위로 갭) 유지율 < 0.  참고로 실적 밤이 아닌 날도 같은 계산
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for p in ["work/b/features", "work/b/review/explain", "work/b/review/news_gap"]:
    sys.path.insert(0, str(ROOT / p))
import explain_rule as E  # noqa: E402
import news_gap as NG  # noqa: E402

OUT, FIGS = HERE / "out", HERE / "figs"
SEED, N_BOOT, GAP_MIN = 2026, 1000, 0.25
DISC_END = pd.Timestamp("2026-02-12")
SECTOR = {
    "기술": ["ADI", "AMD", "AVGO", "CSCO", "DELL", "GLW", "INTC", "MSFT", "NOW", "ORCL", "QCOM", "CRM"],
    "통신": ["NFLX", "TMUS", "VZ"],
    "경기소비재": ["AMZN", "BKNG", "MCD", "TJX", "UBER"],
    "필수소비재": ["COST", "KO", "PM", "WMT"],
    "헬스케어": ["AMGN", "CVS", "DHR", "GILD", "JNJ", "LLY", "PFE", "TMO"],
    "금융": ["AXP", "BRK-B", "CB", "GS", "IBKR", "MS", "V"],
    "산업재": ["BA", "DE", "GE", "RTX", "UNP"],
    "에너지": ["COP", "XOM"],
    "소재": ["LIN", "NEM"],
    "부동산": ["PLD", "WELL"],
}
SYM2SEC = {s: k for k, v in SECTOR.items() for s in v}


def build():
    t, _ = E.load()
    t, _ = NG.build(t)                     # n_tag, n_name, post
    assert set(t.symbol) == set(SYM2SEC), set(t.symbol) ^ set(SYM2SEC)
    rd = pd.read_csv(ROOT / "cache" / "llm_reddit_v4" / "daily_features.csv", usecols=["symbol", "target", "candidate_count"])
    rd["target"] = pd.to_datetime(rd.target, format="mixed")
    t = t.merge(rd, on=["symbol", "target"], how="left")
    t["rd_n"] = t.candidate_count.fillna(0)
    t = t.sort_values(["symbol", "date"])
    for c, raw in [("abn_news", "n_tag"), ("abn_rd", "rd_n")]:
        lg = np.log1p(t[raw])
        t[c] = lg - lg.groupby(t.symbol).transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    t["sector"] = t.symbol.map(SYM2SEC)
    g = t.groupby(["date", "sector"]).gap_z
    s_sum, s_n = g.transform("sum"), g.transform("count")
    t["sec_gap"] = (s_sum - t.gap_z) / (s_n - 1)            # 자기 빼고 업종 평균
    m = t.groupby("date").gap_z
    t["mkt_gap"] = (m.transform("sum") - t.gap_z) / (m.transform("count") - 1)
    t["period"] = np.where(t.target <= DISC_END, "발견", "기타")
    f4 = NG.folds_of(t)[3]
    t.loc[t.date.between(f4[3], f4[4]), "period"] = "확인"
    d = t[(t.gap_z.abs() >= GAP_MIN) & (t.post != 0) & t.gap_z.notna()].copy()
    d["cont"] = (np.sign(d.post) == np.sign(d.gap)).astype(float)
    d["up"] = d.gap_z > 0
    return t, d


def boot_diff(x, mask_a, mask_b, rng):
    """날짜 묶음 부트스트랩: 유지율(a) - 유지율(b)."""
    est = x[mask_a].cont.mean() - x[mask_b].cont.mean()
    dates = x.date.unique()
    idx = {k: v for k, v in x.groupby("date").indices.items()}
    ma, mb, c = mask_a.to_numpy(), mask_b.to_numpy(), x.cont.to_numpy()
    bs = []
    for _ in range(N_BOOT):
        ii = np.concatenate([idx[k] for k in rng.choice(dates, len(dates))])
        a, b = ma[ii], mb[ii]
        if a.sum() and b.sum():
            bs.append(c[ii][a].mean() - c[ii][b].mean())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi, int(mask_a.sum()), int(mask_b.sum())


def test(d, name, a_fn, b_fn, rng, expect="neg", sub=None):
    rows = []
    for per in ["발견", "확인"]:
        x = d[d.period == per]
        if sub is not None:
            x = x[sub(x)]
        x = x.reset_index(drop=True)
        est, lo, hi, na, nb = boot_diff(x, a_fn(x), b_fn(x), rng)
        rows.append({"실험": name, "기간": per, "유지율 A": x[a_fn(x)].cont.mean(), "유지율 B": x[b_fn(x)].cont.mean(),
                     "차이": est, "95% 하한": lo, "95% 상한": hi, "n A": na, "n B": nb})
    r = pd.DataFrame(rows)
    disc, conf = r.iloc[0], r.iloc[1]
    ok_dir = disc["차이"] < 0 if expect == "neg" else disc["차이"] > 0
    ok_ci = disc["95% 상한"] < 0 if expect == "neg" else disc["95% 하한"] > 0
    r["통과"] = bool(ok_dir and ok_ci and np.sign(conf["차이"]) == np.sign(disc["차이"]))
    return r


def main():
    OUT.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    pd.set_option("display.width", 220)
    rng = np.random.default_rng(SEED)
    t, d = build()
    disc = d[d.period == "발견"]
    thr_n, thr_r = disc.abn_news.quantile(0.9), disc.abn_rd.quantile(0.9)
    d["spike"] = (d.abn_news >= thr_n) | (d.abn_rd >= thr_r)
    print(f"관심 급증 기준: 기사 log 차이 ≥ {thr_n:.2f}, Reddit log 차이 ≥ {thr_r:.2f} | 급증 행 비율 {d.spike.mean():.3f}")
    resid = (d.gap_z - d.sec_gap).abs()
    d["follow_sec"] = (np.sign(d.sec_gap) == np.sign(d.gap_z)) & (resid < 0.5 * d.gap_z.abs()) & (d.n_name == 0)
    d["own_sec"] = resid >= 0.5 * d.gap_z.abs()
    residm = (d.gap_z - d.mkt_gap).abs()
    d["follow_mkt"] = (np.sign(d.mkt_gap) == np.sign(d.gap_z)) & (residm < 0.5 * d.gap_z.abs()) & (d.n_name == 0)
    d["own_mkt"] = residm >= 0.5 * d.gap_z.abs()

    res = pd.concat([
        test(d, "A 위로 갭: 관심 급증 - 아님", lambda x: x.spike, lambda x: ~x.spike, rng, sub=lambda x: x.up),
        test(d, "A(참고) 아래로 갭: 관심 급증 - 아님", lambda x: x.spike, lambda x: ~x.spike, rng, sub=lambda x: ~x.up),
        test(d, "B 업종 따라 생긴 갭 - 자기 갭", lambda x: x.follow_sec, lambda x: x.own_sec, rng),
        test(d, "B(참고) 시장 따라 생긴 갭 - 자기 갭", lambda x: x.follow_mkt, lambda x: x.own_mkt, rng),
        test(d, "C 실적 밤: 아래로 갭 - 위로 갭", lambda x: ~x.up, lambda x: x.up, rng, sub=lambda x: x.earn),
        test(d, "C(참고) 실적 아닌 밤: 아래로 - 위로", lambda x: ~x.up, lambda x: x.up, rng, sub=lambda x: ~x.earn),
    ], ignore_index=True)
    res.to_csv(OUT / "papers.csv", index=False, encoding="utf-8-sig")
    print(res.round(3).to_string(index=False))

    # 그림: 실험별 발견·확인 차이와 95% 구간
    names = list(dict.fromkeys(res.실험))[::-1]
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    for i, n in enumerate(names):
        for per, off, c in [("발견", 0.12, E.C1), ("확인", -0.12, E.C2)]:
            r = res[(res.실험 == n) & (res.기간 == per)].iloc[0]
            ax.hlines(i + off, r["95% 하한"] * 100, r["95% 상한"] * 100, color=c, alpha=0.5, linewidth=3)
            ax.scatter([r["차이"] * 100], [i + off], color=c, s=36, zorder=3, edgecolors=E.SURF,
                       label=("발견 기간 (~2026-02)" if per == "발견" else "확인 기간 (fold4)") if i == 0 else None)
    ax.axvline(0, color=E.INK2, linewidth=0.8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("갭 방향 유지율 차이 (%p, 음수 = 더 되돌려짐) · 선 = 날짜 묶음 부트스트랩 95% 구간")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("논문 기반 실험 A·B·C — 어떤 갭이 장중에 더 되돌려지나", loc="left")
    fig.tight_layout()
    fig.savefig(FIGS / "papers.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
