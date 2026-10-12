"""10/11 분봉 실험 중 W&B 에 없던 것(IEX 피드, 시장·거래량 피처)을 팀 W&B 에 run_cv 로 올림. 기록용.

    uv run python work/b/review/premarket/log_wandb_1011.py      (cache/premarket/, cache/premarket_iex/ 필요)

방법: 실험 코드(exp_iex.py, exp_mkt.py)와 같은 표·같은 학습으로 폴드마다 val 예측을 먼저 만들고,
run_cv 는 그 예측을 날짜별로 꺼내 채점 (숫자가 실험과 같음). 피처는 모두 대상일 9:30 전 값.

run (태그 b_review, premarket, not_adopted)
    IEX (태그 iex):  premarket_iex_mix (IEX 9:29 → 없으면 SIP 9:15), premarket_iex_0915 / 0925 / 0929
    시장·거래량 (태그 market): premarket_0915_mktmv_D/T/S, premarket_0915_mktwith_T/S, premarket_0915_volabn_T/S
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "work/b/review/batch"))
import exp as X  # noqa: E402
import exp_mkt as M  # noqa: E402
import batch as B  # noqa: E402
from lab.cv import run_cv  # noqa: E402

IEX = ROOT / "cache" / "premarket_iex"


class PredModel:
    def __init__(self, pred):
        self.pred = pred

    def predict(self, day):
        x = self.pred[self.pred.date == day.date]
        out = pd.DataFrame({"symbol": day.symbols}).merge(x[["symbol", "label"]], on="symbol", how="left")
        out["label"] = out.label.fillna(2).astype(int)
        return out


def fit_from(preds):
    """preds: {(train 첫날, train 마지막날): val 예측 표}. run_cv 가 넘기는 train_days 로 폴드를 찾음."""
    def fit(train_days, exclude):
        k = (min(d.date for d in train_days), max(d.date for d in train_days))
        return PredModel(preds[k])
    return fit


def fold_key(tr):
    return (tr.date.min(), tr.date.max())


def iex_runs():
    t = X.PM.build()
    sip = X.minute_prices()
    iex = X.minute_prices(IEX)
    iex = iex.rename(columns={c: c.replace("px_", "iex_").replace("t_", "it_") for c in iex.columns if c.startswith(("px_", "t_"))})
    iex = iex.rename(columns={"vol_pre": "iex_vol"})
    t = t.merge(sip, on=["target", "symbol"], how="left").merge(iex, on=["target", "symbol"], how="left")
    t = t[t.target <= pd.Timestamp("2026-05-29")].copy()
    for pre in ["px_", "iex_"]:
        bad = (t[f"{pre}08:30"] / t.px_0800 - 1).abs() > X.TOL
        t.loc[bad, [c for c in t.columns if c.startswith(pre) and c[len(pre):][:2].isdigit()]] = np.nan
    gz = lambda px: (px / t.prev_close - 1) / t.vol20  # noqa: E731
    t["c_H"] = gz(t["iex_09:29"]).fillna(gz(t["px_09:15"])).fillna(t.gap_z)
    for c in ["09:15", "09:25", "09:29"]:
        t[f"c_iex{c}"] = gz(t[f"iex_{c}"]).fillna(t.gap_z)
    specs = {"premarket_iex_mix": ("c_H", "무료 혼합: IEX 9:29 → 없으면 SIP 9:15 → 없으면 B0", "SIP 9:15 대비 +0.001, 2/4 → 미달"),
             "premarket_iex_0915": ("c_iex09:15", "IEX 단독 9:15 (없으면 B0)", "참고"),
             "premarket_iex_0925": ("c_iex09:25", "IEX 단독 9:25 (없으면 B0)", "참고"),
             "premarket_iex_0929": ("c_iex09:29", "IEX 단독 9:29 (없으면 B0)", "B0 대비 +0.008. IEX 는 프리마켓 체결이 거의 없음")}
    out = []
    for name, (col, what, result) in specs.items():
        preds = {}
        for _, tr, va, _, _ in X.NG.folds_of(t):
            P = X.G.fit_table(tr.assign(gap_z=tr[col]))
            preds[fold_key(tr)] = va[["date", "symbol"]].assign(label=X.G.predict_table(va.assign(gap_z=va[col]), P))
        out.append((name, preds, ["iex"], {"experiment": what, "code": "work/b/review/premarket/exp_iex.py", "result": result}))
    return out


def mkt_runs():
    t = M.build()
    tag = {"mkt_mv": "mktmv", "mkt_with": "mktwith", "vol_abn": "volabn"}
    out = []
    for src, forms in M.TESTS:
        for form in forms:
            preds = {}
            for _, tr, va, _, _ in X.NG.folds_of(t):
                P = X.G.fit_table(tr)
                p, _ = B.fit_form(tr, va, src, form, P)
                preds[fold_key(tr)] = va[["date", "symbol"]].assign(label=np.asarray(p, int))
            out.append((f"premarket_0915_{tag[src]}_{form}", preds, ["market"],
                        {"experiment": f"{M.NAMES[src]} — 형태 {form} (T 갭 신뢰도, S 급등락 경계, D 방향 더하기)",
                         "code": "work/b/review/premarket/exp_mkt.py", "result": "9:15 규칙 대비 모두 미달 (exp_mkt_judge.csv)"}))
    return out


def main():
    base = {"rule": "gapz_rule (gap_z + ext_range_z, λ·a·b train)", "baseline_compare": "premarket_0915 (0.4635)",
            "eval": "실험 코드와 같은 학습으로 폴드별 val 예측을 만든 뒤 run_cv 로 채점"}
    for name, preds, tags, cfg in [*iex_runs(), *mkt_runs()]:
        print(f"\n=== {name}", flush=True)
        res = run_cv(fit_from(preds), name=name, tags=["b_review", "premarket", "not_adopted", *tags], config={**base, **cfg})
        s = res.table[res.table.group == "all"].score
        u = res.table[res.table.group == "unseen"].score
        print(f"  폴드 {s.round(4).tolist()} | 평균 {s.mean():.4f}, unseen {u.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
