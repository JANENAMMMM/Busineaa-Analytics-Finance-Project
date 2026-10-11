"""10/12 분봉 추가 실험 중 근거로 쓸 만한 것을 팀 W&B 에 run_cv 로 올림.

    uv run python work/b/review/premarket/log_wandb_more.py

run 5개 (태그 b_review, premarket + 아래)
    premarket_0910            강건성: 봉 끝 ≤ 9:10 (무료 SIP 지연 20분 가정과 같음)        robustness
    premarket_0912            강건성: 봉 끝 ≤ 9:12                                        robustness
    premarket_0928            참고: 봉 끝 ≤ 9:28 (실시간 예측이면 유료 피드 필요, 채점 방식 확인 후) reference
    premarket_0915_vwap5      H1: 갭 가격 = 9:10~9:15 분봉 거래량 가중 평균가 (사전 기준 미달)   not_adopted
    premarket_0915_fallback   H2: 분봉 없는 행은 B0 갭·B0 로 학습한 λ·a·b 로 따로 (사전 기준 미달)  not_adopted
비교 기준: premarket_0915 (0.4635), b0_gapz (0.4469) — 10/11 에 log_wandb.py 로 올림.
분봉 저장소: cache/premarket/ 원본 1분봉 → premarket_rule.reduce_bars (git 제외).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import log_wandb as L  # noqa: E402  (경로 설정, RuleModel, make_fit)
from log_wandb import G, PR, ROOT, RAW, feature_table, run_cv  # noqa: E402

BASE_CFG = {"rule": "gapz_rule (gap_z + ext_range_z, λ·a·b train)", "doc": "work/b/review/premarket/exp_more.py, exp_h.py",
            "fallback": "분봉 없거나 단위 2% 초과 어긋나면 B0 갭"}


def store_at(hhmm):
    store = ROOT / "cache" / f"premarket_{hhmm}"
    if store.exists() and len(list(store.glob("*.parquet"))) >= 449:
        return store
    store.mkdir(parents=True, exist_ok=True)
    for f in sorted(RAW.glob("*.parquet")):
        PR.reduce_bars(pd.read_parquet(f), f.stem, f"{hhmm[:2]}:{hhmm[2:]}").to_parquet(store / f.name, index=False)
    return store


def store_vwap5():
    """px_asof 를 9:10~9:15 VWAP 으로 바꾼 저장소 (없으면 9:15 마지막 가격). 나머지 칼럼은 9:15 와 같음."""
    store = ROOT / "cache" / "premarket_vwap5"
    if store.exists() and len(list(store.glob("*.parquet"))) >= 449:
        return store
    store.mkdir(parents=True, exist_ok=True)
    for f in sorted(RAW.glob("*.parquet")):
        bars = pd.read_parquet(f)
        base = PR.reduce_bars(bars, f.stem, "09:15")
        b = bars.copy()
        b["t"] = pd.to_datetime(b.t, utc=True).dt.tz_convert(PR.ET).dt.tz_localize(None)
        tgt = pd.Timestamp(f.stem)
        end = b.t + pd.Timedelta(minutes=1)
        w = b[(end > tgt + pd.Timedelta(hours=9, minutes=10)) & (end <= tgt + pd.Timedelta(hours=9, minutes=15))].copy()
        w["tp"] = (w.h + w.l + w.c) / 3 * w.v
        g = w.groupby("symbol")
        vw = (g.tp.sum() / g.v.sum().replace(0, np.nan)).rename("vw")
        base = base.merge(vw.reset_index(), on="symbol", how="left")
        base["px_asof"] = base.vw.fillna(base.px_asof)
        base.drop(columns="vw").to_parquet(store / f.name, index=False)
    return store


class FallbackModel:
    def __init__(self, p_min, p_fb, store):
        self.p_min, self.p_fb, self.store = p_min, p_fb, store

    def predict(self, day):
        x = PR.build_gapz_pm(day, self.store)
        out = pd.DataFrame({"symbol": day.symbols})
        if len(x):
            lab = np.where(x.pm == 1, G.predict_table(x, self.p_min), G.predict_table(x, self.p_fb))
            out = out.merge(x[["symbol"]].assign(label=lab), on="symbol", how="left")
        else:
            out["label"] = 2
        out["label"] = out.label.fillna(2).astype(int)
        return out


def make_fit_fallback(store):
    cache_pm, cache_b0 = {}, {}

    def fit(train_days, exclude):
        for cache, build in [(cache_pm, lambda day: PR.build_gapz_pm(day, store)), (cache_b0, G.build_gapz)]:
            need = [d for d in train_days if d.date not in cache]
            if need:
                for dt, g in feature_table(need, build, name=None, verbose=False).groupby("date"):
                    cache[pd.Timestamp(dt)] = g
        pick = lambda c: pd.concat([c[d.date] for d in train_days if d.date in c], ignore_index=True)  # noqa: E731
        pm, b0 = pick(cache_pm), pick(cache_b0)
        pm, b0 = pm[~pm.symbol.isin(exclude)].copy(), b0[~b0.symbol.isin(exclude)].copy()
        for t in (pm, b0):
            t["label"] = t.label.astype(int)
        return FallbackModel(G.fit_table(pm[pm.pm == 1]), G.fit_table(b0), store)
    return fit


def main():
    s915 = store_at("0915")
    runs = []
    for hhmm, tag, note in [("0910", "robustness", "강건성 (무료 SIP 지연 20분 가정)"),
                            ("0912", "robustness", "강건성"),
                            ("0928", "reference", "참고: 실시간 예측이면 유료 피드 필요, 기간 뒤 한꺼번에 채점하면 무료로 받아짐 → 교수님 확인 후")]:
        st = store_at(hhmm)
        runs.append((f"premarket_{hhmm}", L.make_fit(lambda day, s=st: PR.build_gapz_pm(day, s)), [tag],
                     {"gap_price": f"Alpaca SIP 1분봉, 봉 끝 ≤ {hhmm[:2]}:{hhmm[2:]}", "as_of": f"{hhmm[:2]}:{hhmm[2:]}", "note": note}))
    sv = store_vwap5()
    runs.append(("premarket_0915_vwap5", L.make_fit(lambda day, s=sv: PR.build_gapz_pm(day, s)), ["not_adopted", "hypothesis"],
                 {"gap_price": "9:10~9:15 분봉 거래량 가중 평균가 (전형가격), 없으면 9:15 가격", "as_of": "09:15",
                  "note": "H1 사전 기준 미달 (vs 0915: +0.006, 3/4, fold4 −0.0005)"}))
    runs.append(("premarket_0915_fallback", make_fit_fallback(s915), ["not_adopted", "hypothesis"],
                 {"gap_price": "분봉 있는 행 9:15 갭 + 그 행으로 학습한 λ·a·b / 없는 행 B0 갭 + B0 로 학습한 λ·a·b",
                  "as_of": "09:15", "note": "H2 사전 기준 미달 (vs 0915: +0.004, 4/4). 분봉 끊길 때 안전장치 후보"}))
    for name, fit, tags, cfg in runs:
        print(f"\n=== {name}", flush=True)
        res = run_cv(fit, name=name, tags=["b_review", "premarket", *tags], config={**BASE_CFG, **cfg})
        s = res.table[res.table.group == "all"].score
        u = res.table[res.table.group == "unseen"].score
        print(f"  폴드 {s.round(4).tolist()} | 평균 {s.mean():.4f}, unseen {u.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
