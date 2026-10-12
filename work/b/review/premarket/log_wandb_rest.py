"""10/12 분봉 실험 중 나머지(모두 사전 기준 미달)를 팀 W&B 에 run_cv 로 올림. 기록용.

    uv run python work/b/review/premarket/log_wandb_rest.py

run 7개 (태그 b_review, premarket, not_adopted, hypothesis)
    premarket_0915_vwap15      A1 갭 가격 = 9:00~9:15 VWAP
    premarket_0915_extmin      B1 ext_range = 교수님 시간외 봉 + 분봉(9:15 까지) 고저
    premarket_0915_pmvol       C1 갭 분모 = 지난 20번 9:15 갭 표준편차
    premarket_0915_late_add    D1 후반 움직임(8:30→9:15) 더하기, β train
    premarket_0915_late_same   D1 후반 움직임 같은방향, β train
    premarket_0915_liqmix      H3 체결 적은 행(하위 25%)은 9:15·08:00 갭 섞기, w train
    premarket_0915_vwap5_fb    H4 = H1(최근 5분 VWAP) + H2(분봉 없는 행 따로)
피처는 exp_more.build() 표(대상일 전 정보로 계산)에서 그 날짜 행을 꺼내 씀. 학습은 폴드 train(처음 보는 종목 제외),
예측·채점은 run_cv. 숫자는 exp_more.py · exp_d1.py · exp_h.py 와 같아야 함.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp_more as X  # noqa: E402
import exp_h as H  # noqa: E402
from exp_more import G, ROOT  # noqa: E402

sys.path.insert(0, str(ROOT))
from lab.cv import run_cv  # noqa: E402

BETAS = [-0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5]
WS = [0.0, 0.25, 0.5, 0.75, 1.0]


def table():
    t = X.build()
    t = t.merge(H.extra(X.load_minutes()), on=["target", "symbol"], how="left")
    t.loc[~t.has_min, ["px_vwap5", "n_pre"]] = np.nan
    t["gz_vwap5"] = ((t.px_vwap5.fillna(t["px_09:15"]) / t.prev_close - 1) / t.vol20).fillna(t["gz_09:15"])
    t["late"] = ((t["px_09:15"] / t["px_08:30"] - 1) / t.vol20).where(t.has_min, 0.0)
    t["label"] = t.label.astype(int)
    return t


class TableModel:
    """day.date 의 표 행으로 score s 를 만들고 경계로 자름. 행이 없는 종목은 보합."""

    def __init__(self, t, score_fn, params):
        self.t, self.score_fn, self.params = t, score_fn, params

    def predict(self, day):
        x = self.t[self.t.date == day.date]
        out = pd.DataFrame({"symbol": day.symbols})
        if len(x):
            lab = self.score_fn(x, self.params)
            out = out.merge(x[["symbol"]].assign(label=lab), on="symbol", how="left")
        else:
            out["label"] = 2
        out["label"] = out.label.fillna(2).astype(int)
        return out


def train_rows(t, train_days, exclude):
    d = {x.date for x in train_days}
    return t[t.date.isin(d) & ~t.symbol.isin(exclude)]


def fit_cols(t, gap_col, ext_col="ext_range_z"):
    def fit(train_days, exclude):
        tr = train_rows(t, train_days, exclude)
        P = G.fit_table(tr.assign(gap_z=tr[gap_col], ext_range_z=tr[ext_col]))
        return TableModel(t, lambda x, P: G.predict_table(x.assign(gap_z=x[gap_col], ext_range_z=x[ext_col]), P), P)
    return fit


def fit_late(t, form):
    def apply(b, g, xs, sd):
        return g + b * sd * xs if form == "더하기" else g * np.exp(b * xs * np.sign(g))

    def fit(train_days, exclude):
        tr = train_rows(t, train_days, exclude)
        med, sdx = np.nanmedian(tr.late), np.nanstd(tr.late)
        std = lambda v: np.nan_to_num(np.clip((v - med) / sdx, -5, 5))  # noqa: E731
        sd = np.nanstd(tr["gz_09:15"])
        best = None
        for b in BETAS:
            P = G.fit_table(tr.assign(gap_z=apply(b, tr["gz_09:15"].to_numpy(), std(tr.late.to_numpy()), sd)))
            if best is None or P["train_score"] > best[0]:
                best = (P["train_score"], b, P)
        b, P = best[1], best[2]
        sc = lambda x, P: G.predict_table(x.assign(gap_z=apply(b, x["gz_09:15"].to_numpy(), std(x.late.to_numpy()), sd)), P)  # noqa: E731
        return TableModel(t, sc, P)
    return fit


def fit_liqmix(t):
    def fit(train_days, exclude):
        tr = train_rows(t, train_days, exclude)
        thr = np.nanquantile(tr.loc[tr.has_min, "n_pre"], 0.25)
        mix = lambda x, w: np.where(x.has_min & (x.n_pre < thr), w * x["gz_09:15"] + (1 - w) * x.gap_z, x["gz_09:15"])  # noqa: E731
        best = None
        for w in WS:
            P = G.fit_table(tr.assign(gap_z=mix(tr, w)))
            if best is None or P["train_score"] > best[0]:
                best = (P["train_score"], w, P)
        w, P = best[1], best[2]
        return TableModel(t, lambda x, P: G.predict_table(x.assign(gap_z=mix(x, w)), P), P)
    return fit


def fit_vwap5_fb(t):
    def fit(train_days, exclude):
        tr = train_rows(t, train_days, exclude)
        P = {"min": G.fit_table(tr[tr.has_min].assign(gap_z=tr.loc[tr.has_min, "gz_vwap5"])),
             "fb": G.fit_table(tr.assign(gap_z=tr.gap_z))}
        sc = lambda x, P: np.where(x.has_min, G.predict_table(x.assign(gap_z=x.gz_vwap5), P["min"]),  # noqa: E731
                                   G.predict_table(x.assign(gap_z=x.gap_z), P["fb"]))
        return TableModel(t, sc, P)
    return fit


def main():
    t = table()
    base = {"rule": "gapz_rule (gap_z + ext_range_z, λ·a·b train)", "baseline_compare": "premarket_0915 (0.4635)",
            "feature_source": "exp_more.build() 표 (대상일 전 정보)", "fallback": "분봉 없거나 단위 2% 초과 어긋나면 B0 갭"}
    runs = [
        ("premarket_0915_vwap15", fit_cols(t, "gz_vwap"), "A1: 갭 가격 = 9:00~9:15 VWAP (전형가격)", "exp_more.py", "+0.0045, 3/4, fold4 +0.004 → 평균 미달"),
        ("premarket_0915_extmin", fit_cols(t, "gz_09:15", "ext_min"), "B1: ext_range = 교수님 시간외 봉 + 분봉(9:15) 고저", "exp_more.py", "+0.001, 2/4 → 미달"),
        ("premarket_0915_pmvol", fit_cols(t, "gz_pmvol"), "C1: 갭 ÷ 지난 20번 9:15 갭 표준편차", "exp_more.py", "−0.010, 0/4 → 손해"),
        ("premarket_0915_late_add", fit_late(t, "더하기"), "D1: 후반 움직임(8:30→9:15)/vol20 더하기, β train", "exp_d1.py", "0.000 (train 이 β=0) → 미달"),
        ("premarket_0915_late_same", fit_late(t, "같은방향"), "D1: 후반 움직임 같은방향, β train", "exp_d1.py", "+0.001, 2/4 → 미달"),
        ("premarket_0915_liqmix", fit_liqmix(t), "H3: 체결 하위 25% 행은 9:15·08:00 갭 섞기, w train", "exp_h.py", "+0.000 (train 이 w=1) → 미달"),
        ("premarket_0915_vwap5_fb", fit_vwap5_fb(t), "H4: 최근 5분 VWAP + 분봉 없는 행 B0 경계 따로 (사후 조합)", "exp_h.py", "+0.003, 2/4, fold4 −0.003 → 미달"),
    ]
    for name, fit, what, src, result in runs:
        print(f"\n=== {name}", flush=True)
        res = run_cv(fit, name=name, tags=["b_review", "premarket", "not_adopted", "hypothesis"],
                     config={**base, "experiment": what, "code": f"work/b/review/premarket/{src}", "result_vs_0915": result})
        s = res.table[res.table.group == "all"].score
        u = res.table[res.table.group == "unseen"].score
        print(f"  폴드 {s.round(4).tolist()} | 평균 {s.mean():.4f}, unseen {u.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
