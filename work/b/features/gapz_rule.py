"""B 최종안: gap_z 규칙 + ext_range_z. 자급 파일 (numpy, pandas 만 씀) — src/model.py 에 그대로 붙여 넣을 수 있음.

    feats  = build_gapz(day)                 # symbol, gap_z, ext_range_z
    params = fit(train_days)                 # 학습 결과 숫자 5개 (dict, JSON 으로 저장 가능)
    pred   = predict(day, params)            # symbol, label (0~4)

규칙
    gap         = 마지막 시간외 봉 종가 ÷ 기준일 종가 − 1     (기준일 16:00 이후 post/pre 봉, cutoff 전까지)
    vol20       = 최근 20거래일 일간 수익률 표준편차
    gap_z       = gap ÷ vol20
    ext_range_z = (시간외 봉 최고가 − 최저가) ÷ 기준일 종가 ÷ vol20
    x̃ = (ext_range_z − 중앙값) ÷ 표준편차, ±5 로 자름, 결측 0       (중앙값·표준편차는 train)
    s = gap_z · exp(−λ·x̃)                                         (λ 는 train score 최대)
    |s| 분위 후보 중 train score 최대인 경계 a < b
    s ≥ b → 4 / s ≥ a → 3 / s ≤ −a → 1 / s ≤ −b → 0 / 그 밖·결측 → 2

A 폴드(lab/folds.json) 4폴드 평균 score 0.447 (gap_z 규칙만 0.444, 원래 갭 규칙 0.433 = eda/cv_check_ovn_gap.py).
"""

import json

import numpy as np
import pandas as pd

CLOSE = pd.Timedelta(hours=16)
QS = [.1, .2, .3, .4, .5, .6, .7, .8, .85, .9, .95, .975, .99]   # 경계 후보 분위 (eda/cv_check_ovn_gap.py 와 같음)
LAMS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]   # λ 후보
_LV = np.arange(5)
WEIGHT = ((_LV[:, None] - _LV[None, :]) ** 2) * (np.abs(_LV - 2)[:, None] ** 2)   # src.data.WEIGHT 와 같음


# ------------------------------------------------------------------ 피처
def build_gapz(day):
    """symbol / gap_z / ext_range_z. 기준일 일봉이 없으면 빈 표."""
    cols = ["symbol", "gap_z", "ext_range_z"]
    d = day.daily(days=40, columns=["symbol", "date_et", "close", "ret"])
    if not len(d) or pd.Timestamp(d.date_et.max()) != day.date:
        return pd.DataFrame(columns=cols)
    R = d.pivot_table(index="date_et", columns="symbol", values="ret", aggfunc="last").sort_index()
    C = d.pivot_table(index="date_et", columns="symbol", values="close", aggfunc="last").sort_index()
    last = R.index[-1]
    vol20 = R.rolling(20, min_periods=10).std().loc[last]
    close = C.loc[last]

    p = day.price(since=day.date, columns=["symbol", "datetime", "high", "low", "close", "session"])
    ext = p[(p.datetime >= day.date + CLOSE) & p.session.isin(["post", "pre"])].sort_values("datetime")
    g = ext.groupby("symbol")
    out = pd.DataFrame(index=pd.Index(close.index, name="symbol"))
    out["gap_z"] = (g.close.last() / close - 1) / vol20
    out["ext_range_z"] = (g.high.max() - g.low.min()) / close / vol20
    out = out.replace([np.inf, -np.inf], np.nan).reset_index()
    return out[cols]


# ------------------------------------------------------------------ 점수·규칙
def score(y, p):
    """src.score 의 score 와 같은 값."""
    o = np.bincount(np.asarray(y, int) * 5 + np.asarray(p, int), minlength=25).reshape(5, 5).astype(float)
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    return 1 - (WEIGHT * o).sum() / (WEIGHT * e).sum()


def cut(s, a, b):
    o = np.full(len(s), 2)
    o[s >= a], o[s >= b], o[s <= -a], o[s <= -b] = 3, 4, 1, 0
    o[np.isnan(s)] = 2
    return o


def _fit_ab(s, y):
    ok = ~np.isnan(s)
    q = np.quantile(np.abs(s[ok]), QS)
    return max((score(y[ok], cut(s[ok], a, b)), a, b) for i, a in enumerate(q) for b in q[i + 1:])


def _score_s(gz, er, med, sd, lam):
    x = np.nan_to_num(np.clip((er - med) / sd, -5, 5))
    return gz * np.exp(-lam * x)


# ------------------------------------------------------------------ 학습·예측
def fit_table(t):
    """t: gap_z, ext_range_z, label 열이 있는 표 → params."""
    gz, er, y = t.gap_z.to_numpy(float), t.ext_range_z.to_numpy(float), t.label.to_numpy(int)
    med, sd = float(np.nanmedian(er)), float(np.nanstd(er))
    best = None
    for lam in LAMS:
        sc, a, b = _fit_ab(_score_s(gz, er, med, sd, lam), y)
        if best is None or sc > best[0]:
            best = (sc, lam, a, b)
    sc, lam, a, b = best
    return {"lam": lam, "med": med, "sd": sd, "a": float(a), "b": float(b), "train_score": float(sc)}


def build_table(days):
    """학습용 표: build_gapz + day.y 의 label."""
    rows = []
    for day in days:
        x = build_gapz(day)
        if len(x):
            rows.append(x.merge(day.y[["symbol", "label"]], on="symbol"))
    t = pd.concat(rows, ignore_index=True)
    t["label"] = t.label.astype(int)
    return t


def fit(days):
    return fit_table(build_table(days))


def predict_table(x, params):
    s = _score_s(x.gap_z.to_numpy(float), x.ext_range_z.to_numpy(float), params["med"], params["sd"], params["lam"])
    return cut(s, params["a"], params["b"])


def predict(day, params):
    """day.symbols 전부에 label. 피처가 없는 종목은 보합(2)."""
    x = build_gapz(day)
    out = pd.DataFrame({"symbol": day.symbols})
    if len(x):
        x = x.assign(label=predict_table(x, params))
        out = out.merge(x[["symbol", "label"]], on="symbol", how="left")
    else:
        out["label"] = 2
    out["label"] = out.label.fillna(2).astype(int)
    return out


def save(params, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ model.py 에 붙일 때 예시
#
#   class Model:
#       def __init__(self, params):
#           self.params = params
#       def predict(self, day):
#           return predict(day, self.params)
#
#   def load_model():
#       return Model(load(ARTIFACTS / "gapz_rule.json"))
#
#   학습 (한 번): save(fit(train_days), ARTIFACTS / "gapz_rule.json")
