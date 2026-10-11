"""프리마켓 9:00~9:30 가격의 가치 — 외부 분봉을 구할 가치가 있는지 먼저 재는 진단.

    uv run python work/b/review/premarket/premarket.py

⚠️ 진단 전용. 교수님 데이터의 9:00 봉(9:00~9:30)은 known_at 이 정확히 9:30 = cutoff 라 규칙상 못 씀.
   여기서는 "외부 분봉으로 9:28 가격을 받으면 얼마나 오를 수 있나"의 상한을 재려고 일부러 씀. 제출 모델에는 안 들어감.

사전 등록 (실행 전에 고정)
- 비교: 지금 규칙(갭 가격 = 08:00 봉 종가) vs 갭 가격만 9:00 봉 종가로 바꾼 규칙. ext_range_z 는 그대로
- 파라미터(λ, a, b)는 폴드마다 train 으로 다시 고름. A 폴드 4개, val 만 채점
- 진행 기준: 4폴드 평균 +0.005 이상 & 3폴드 이상 상승 → 외부 분봉 수집으로 넘어감. 아니면 중단
- 참고 지표: 방향 적중(보합 아닌 예측·정답), 9:00 이후 30분 움직임이 갭과 같은 방향인 비율
"""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for p in ["work/b/features", "work/b/review/explain", "work/b/review/news_gap"]:
    sys.path.insert(0, str(ROOT / p))
import explain_rule as E  # noqa: E402
import gapz_rule as G  # noqa: E402
import news_gap as NG  # noqa: E402

OUT = HERE / "out"


def build():
    t, _ = E.load()
    d = pd.read_parquet(ROOT / "dataset" / "daily.parquet", columns=["symbol", "date_et", "close"])
    t = t.merge(d.rename(columns={"date_et": "date", "close": "prev_close"}), on=["symbol", "date"], how="left")

    w = t[["date", "target"]].drop_duplicates().dropna().copy()
    w["since"] = w.date + pd.Timedelta(hours=16)
    w["cut"] = w.target + pd.Timedelta(hours=9, minutes=30)
    con = duckdb.connect()
    con.register("w", w)
    # 진단용: known_at <= cutoff (9:00 봉 포함). 지금 규칙은 known_at < cutoff
    p = con.execute(f"""
        SELECT w.date, p.symbol, p.datetime, p.known_at, p.close, w.cut
        FROM read_parquet('{ROOT / "dataset" / "price.parquet"}') p
        JOIN w ON p.datetime >= w.since AND p.known_at <= w.cut
        WHERE p.session IN ('post', 'pre')""").df().sort_values("datetime")
    con.close()
    before = p[p.known_at < p.cut].groupby(["date", "symbol"]).close.last().rename("px_0800")
    at930 = p[p.known_at == p.cut].groupby(["date", "symbol"]).close.last().rename("px_0900")
    t = t.merge(pd.concat([before, at930], axis=1).reset_index(), on=["date", "symbol"], how="left")

    d0 = ((t.px_0800 / t.prev_close - 1) / t.vol20 - t.gap_z).abs()
    print(f"확인: 08:00 봉으로 다시 계산한 gap_z 와 기존 값 최대 차이 {d0.max():.1e}")
    print(f"9:00 봉이 있는 행 {t.px_0900.notna().mean():.1%}")
    t["gap_z_0900"] = (t.px_0900 / t.prev_close - 1) / t.vol20
    t["gap_z_0900"] = t.gap_z_0900.fillna(t.gap_z)            # 9:00 봉이 없으면 지금 갭
    t["move30"] = t.px_0900 / t.px_0800 - 1                    # 9:00 ~ 9:30 움직임
    return t


def dir_hit(y, p):
    m = (y != 2) & (p != 2)
    return float((np.sign(y[m] - 2) == np.sign(p[m] - 2)).mean())


def main():
    OUT.mkdir(exist_ok=True)
    t = build()
    rows = []
    for name, tr, va, _, _ in NG.folds_of(t):
        y = va.label.to_numpy(int)
        for lab, col in [("지금 (08:00 봉)", "gap_z"), ("9:00 봉 (진단)", "gap_z_0900")]:
            P = G.fit_table(tr.assign(gap_z=tr[col]))
            p = G.predict_table(va.assign(gap_z=va[col]), P)
            rows.append({"fold": name, "설정": lab, "score": G.score(y, p), "방향 적중": dir_hit(y, p),
                         "급등락 비율": np.isin(p, [0, 4]).mean(), "lam": P["lam"],
                         "unseen": G.score(y[va.unseen.to_numpy()], p[va.unseen.to_numpy()])})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "diag.csv", index=False, encoding="utf-8-sig")
    pv = r.pivot_table(index="설정", columns="fold", values="score")
    pv["평균"] = pv.mean(1)
    print(pv.round(3).to_string())
    diff = pv.loc["9:00 봉 (진단)"] - pv.loc["지금 (08:00 봉)"]
    print("\n차이:", diff.round(3).to_dict())
    print(r.groupby("설정")[["방향 적중", "급등락 비율", "unseen"]].mean().round(3).to_string())
    up = int((diff.drop("평균") > 0).sum())
    go = diff["평균"] >= 0.005 and up >= 3
    print(f"\n판정: 평균 {diff['평균']:+.3f}, 상승 폴드 {up}/4 → {'외부 분봉 수집으로 진행' if go else '중단'}")

    m = t.move30.notna() & (t.gap_z.abs() >= 0.25)
    same = (np.sign(t.move30[m]) == np.sign(t.gap_z[m])).mean()
    print(f"\n9:00~9:30 움직임: |중앙값| {t.move30.abs().median():.4f}, vol20 대비 {(t.move30.abs() / t.vol20).median():.3f}, "
          f"갭과 같은 방향 {same:.1%} (|gap_z|≥0.25)")
    flip = (np.sign(t.gap_z) != np.sign(t.gap_z_0900)) & (t.gap_z.abs() >= 0.25)
    print(f"9:30 에 갭 방향이 뒤집힌 행 (|gap_z|≥0.25): {flip.mean():.1%}")


if __name__ == "__main__":
    main()
