"""제출 모델 v1: gap_z 규칙 + ext_range_z (B 최종안, work/a/gapz_rule.py 를 옮김).

    load_model()      artifacts/gapz_rule.json (학습 결과 숫자 5개) 만 읽음
    Model.predict()   Day 하나 → symbol / label (0~4)

규칙
    gap         = cutoff 전 마지막 시간외 봉 종가 ÷ 기준일 종가 − 1   (기준일 16:00 이후 post/pre 봉)
    vol20       = 최근 20거래일 일간 수익률 표준편차
    gap_z       = gap ÷ vol20
    ext_range_z = (시간외 봉 최고가 − 최저가) ÷ 기준일 종가 ÷ vol20
    x̃ = (ext_range_z − med) ÷ sd, ±5 로 자름, 결측 0              (med·sd 는 train)
    s = gap_z · exp(−λ·x̃)                                        (λ 는 train score 최대)
    s ≥ b → 4 / s ≥ a → 3 / s ≤ −a → 1 / s ≤ −b → 0 / 그 밖 → 2

갭을 못 구한 종목(시간외 봉 누락, 기준일 일봉 없음 등)은 보합(2). 방향 정보가 없을 때 급등락을 찍으면
반대로 틀렸을 때 벌점이 커서 보합이 가장 안전함.
학습은 scripts/a_train_v1.py (채점 때 다시 돌리지 않음). numpy, pandas, json 만 씀.
"""

import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from .data import FLAT
from .paths import ARTIFACTS

PARAMS_FILE = "gapz_rule.json"
PM_PARAMS_FILE = "premarket_rule.json"
DEFAULT_AS_OF = "09:15"
CLOSE = pd.Timedelta(hours=16)
PM_TOL = 0.02


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


def _premarket_store(as_of):
    """Return the compact input folder for this fixed cutoff."""
    return "premarket" if as_of == DEFAULT_AS_OF else f"premarket_{as_of.replace(':', '')}"


def _read_premarket_cache(path, target, as_of):
    """Validate external cache rows before allowing them into inference."""
    def warn(reason):
        warnings.warn(f"Premarket cache {path.name}: {reason}; using after-hours gap for rejected rows",
                      RuntimeWarning, stacklevel=3)

    try:
        pm = pd.read_parquet(path).rename(columns={"px_0915": "px_asof", "t_0915": "t_asof"})
    except Exception as exc:
        warn(f"unreadable parquet ({type(exc).__name__})")
        return None
    required = {"symbol", "px_0830", "px_asof", "t_asof"}
    if not pm.columns.is_unique or not required.issubset(pm.columns):
        warn("missing or duplicate required columns")
        return None
    pm = pm.copy()
    target = pd.Timestamp(target)
    if target.tzinfo is not None:
        target = target.tz_convert("America/New_York").tz_localize(None)
    target = target.normalize()
    hh, mm = map(int, str(as_of).split(":"))
    limit = target + pd.Timedelta(hours=hh, minutes=mm)

    def valid_time(value):
        try:
            if not isinstance(value, (str, datetime, np.datetime64)):
                return False
            stamp = pd.Timestamp(value)
            if pd.isna(stamp):
                return False
            if stamp.tzinfo is not None:
                stamp = stamp.tz_convert("America/New_York").tz_localize(None)
            return (stamp.normalize() == target and stamp >= target + pd.Timedelta(hours=8)
                    and stamp == stamp.floor("min") and stamp + pd.Timedelta(minutes=1) <= limit)
        except (ValueError, TypeError, OverflowError):
            return False

    good_symbol = pm.symbol.map(lambda v: isinstance(v, str) and bool(v.strip()))
    duplicates = pm.symbol.where(good_symbol, None).duplicated(keep=False)
    asof_price = pd.to_numeric(pm.px_asof, errors="coerce")
    check_price = pd.to_numeric(pm.px_0830, errors="coerce")
    # No trade before 08:30 is a legitimate missing unit-check price in legacy caches.
    good_check = pm.px_0830.isna() | (np.isfinite(check_price) & (check_price > 0))
    good = (good_symbol & ~duplicates & np.isfinite(asof_price) & (asof_price > 0)
            & good_check & pm.t_asof.map(valid_time)).fillna(False)
    if not good.all():
        rejected = list(dict.fromkeys(str(v) for v in pm.loc[~good, "symbol"]))
        warn(f"invalid price/time/symbol or duplicate symbol: {', '.join(rejected[:10])} ({len(pm) - int(good.sum())} rows)")
    pm["px_asof"], pm["px_0830"] = asof_price, check_price
    return pm.loc[good].set_index("symbol")


def build_gapz_premarket(day, as_of=DEFAULT_AS_OF):
    """Use a supplied cutoff-price when it passes the 08:30 unit check.

    Missing files, symbols, or failed checks deliberately retain the B0 gap.
    """
    out = build_gapz(day)
    if out.empty:
        return out
    path = ARTIFACTS / _premarket_store(as_of) / f"{pd.Timestamp(day.target):%Y-%m-%d}.parquet"
    if not path.exists():
        return out
    pm = _read_premarket_cache(path, day.target, as_of)
    if pm is None or pm.empty:
        return out
    d = day.daily(days=40, columns=["symbol", "date_et", "close", "ret"])
    r = d.pivot_table(index="date_et", columns="symbol", values="ret", aggfunc="last").sort_index()
    c = d.pivot_table(index="date_et", columns="symbol", values="close", aggfunc="last").sort_index()
    if r.empty or c.empty:
        return out
    vol20 = r.rolling(20, min_periods=10).std().loc[r.index[-1]]
    close = c.loc[c.index[-1]]
    x = out.set_index("symbol")
    pm = pm.reindex(x.index)
    # Recover B0's unstandardized price from its gap for the same comparison.
    px_b0 = close.reindex(x.index) * (1 + x.gap_z * vol20.reindex(x.index))
    bad = (pm.px_0830 / px_b0 - 1).abs() > PM_TOL
    gz = ((pm.px_asof / close.reindex(x.index) - 1) / vol20.reindex(x.index)).where(~bad)
    ok = gz.notna() & np.isfinite(gz)
    x.loc[ok, "gap_z"] = gz[ok]
    return x.reset_index()


def strength(gap_z, ext_range_z, params):
    x = np.nan_to_num(np.clip((ext_range_z - params["med"]) / params["sd"], -5, 5))
    return gap_z * np.exp(-params["lam"] * x)


def cut(s, a, b):
    o = np.full(len(s), FLAT)
    o[s >= a], o[s >= b], o[s <= -a], o[s <= -b] = 3, 4, 1, 0
    o[np.isnan(s)] = FLAT
    return o


class Model:
    def __init__(self, params, premarket=False):
        self.params = params
        self.premarket = premarket

    def predict(self, day):
        """day.symbols 전부에 label. 피처가 없는 종목은 보합."""
        out = pd.DataFrame({"symbol": list(day.symbols)})
        x = build_gapz_premarket(day, self.params.get("as_of", DEFAULT_AS_OF)) if self.premarket else build_gapz(day)
        if len(x):
            s = strength(x.gap_z.to_numpy(float), x.ext_range_z.to_numpy(float), self.params)
            x = x.assign(label=cut(s, self.params["a"], self.params["b"]))
            out = out.merge(x[["symbol", "label"]], on="symbol", how="left")
        else:
            out["label"] = FLAT
        out["label"] = out.label.fillna(FLAT).astype(int)
        return out


def load_model():
    path = ARTIFACTS / PM_PARAMS_FILE
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return Model(json.load(f), premarket=True)
    with open(ARTIFACTS / PARAMS_FILE, encoding="utf-8") as f:
        return Model(json.load(f))
