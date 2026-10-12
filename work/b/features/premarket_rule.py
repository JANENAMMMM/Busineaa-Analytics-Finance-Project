"""B 후보: 분봉 갭 규칙 = gapz_rule + Alpaca 프리마켓 1분봉으로 갭 가격만 더 최신으로 (기본 09:15).

⚠️ 제출 후보입니다. 아직 제출 모델(src/model.py = B0, gapz_rule)이 아닙니다. 채택은 팀 결정 → docs/review/16_프리마켓_분봉.md

    받기 (사전 단계, 인터넷·키 필요)
        uv run python work/b/features/premarket_rule.py fetch --since 2024-08-13 --until 2026-05-28 [--as-of 09:28]
    예측 (오프라인, 받아 둔 파일만 읽음)
        params = fit(train_days, as_of="09:15")  /  pred = predict(day, params)   ← params 에 as_of 기록, 예측도 같은 시각

설정값 as_of (몇 시까지 끝난 봉을 쓸지, 기본 "09:15", 09:29 까지만 허용)
    09:15       : 무료 SIP(15분 지연)로 대상일 09:30 전에 실시간으로도 받을 수 있는 마지막 시각
    09:16~09:29 : 실시간 예측이면 유료 실시간 피드 필요. 지난 날짜를 나중에 받으면 무료로도 받아짐
                  → 채점 방식(실시간 / 기간 뒤 한꺼번에)에 따라 교수님 확인 후 정함
    저장 폴더는 시각마다 따로: 09:15 = artifacts/premarket/, 그 밖 = artifacts/premarket_HHMM/

규칙 (gapz_rule 과 같고 갭 가격만 다름)
    gap_z = (px_asof ÷ 기준일 종가 − 1) ÷ vol20
      px_asof = 대상일 08:00 ~ as_of SIP 1분봉 중 "봉 끝 ≤ as_of" 인 마지막 봉 종가
    대체: 분봉 파일이 없거나, 그 종목 봉이 없거나, 단위 확인(px_0830 vs 교수님 08:00 봉 종가) 이 2% 넘게 어긋나면
          gapz_rule 의 갭(교수님 데이터) 그대로 → 키·인터넷이 없어도 B0 와 같은 예측
    ext_range_z, λ, a, b 는 gapz_rule 과 같음 (학습은 이 갭으로 다시)

저장 형식 (저장 폴더/YYYY-MM-DD.parquet, 대상일별, 종목당 한 줄)
    symbol, px_0830, px_asof, t_asof (마지막 봉 시작, ET), vol_pre (08:00~as_of 거래량)
    원본 1분봉은 저장하지 않음. as_of 이후 가격은 파일에 아예 없음 (예전 이름 px_0915 / t_0915 파일도 읽음)

dev 4폴드: 0.447 (B0) → 0.464 (work/b/review/premarket/exp.py, 판정 통과)
의존: numpy, pandas, json, 표준 라이브러리 (urllib, os) + 같은 폴더의 gapz_rule
"""

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gapz_rule as G  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
STORE = ROOT / "artifacts" / "premarket"           # git 제외 (.gitignore)
KEY_FILE = ROOT / "artifacts" / "alpaca_key.json"  # git 제외
URL = "https://data.alpaca.markets/v2/stocks/bars"
AS_OF = "09:15"                                    # 기본 설정값: 이 시각까지 끝난 봉만
CHECK_AT = pd.Timedelta(hours=8, minutes=30)       # 교수님 08:00 봉 종가와 같은 시점 (2026-10-11 대조)
START = pd.Timedelta(hours=8)
TOL = 0.02
ET = "America/New_York"


def _td(as_of):
    """'09:15' 또는 Timedelta → Timedelta. 08:00 이하·09:30 이상은 막음 (cutoff 09:30 미만)."""
    if isinstance(as_of, pd.Timedelta):
        td = as_of
    else:
        hh, mm = str(as_of).split(":")
        td = pd.Timedelta(hours=int(hh), minutes=int(mm))
    if not (START < td < pd.Timedelta(hours=9, minutes=30)):
        raise ValueError(f"as_of 는 08:00 초과 09:30 미만이어야 함: {as_of}")
    return td


def _tag(as_of):
    c = _td(as_of).components
    return f"{c.hours:02d}:{c.minutes:02d}"


def store_for(as_of=AS_OF):
    """시각마다 저장 폴더를 따로 (09:15 는 예전 폴더 그대로)."""
    tag = _tag(as_of)
    return STORE if tag == "09:15" else ROOT / "artifacts" / f"premarket_{tag.replace(':', '')}"


# ------------------------------------------------------------------ 받기 (사전 단계)
def _keys():
    names = ["APCA_API_KEY_ID", "APCA_API_SECRET_KEY"]
    vals = [os.environ.get(n) for n in names]
    if not all(vals) and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key.strip() in names:
                index = names.index(key.strip())
                vals[index] = vals[index] or value.strip().strip('"').strip("'")
    if not all(vals) and sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                vals = [v or winreg.QueryValueEx(k, n)[0] for v, n in zip(vals, names)]
        except OSError:
            pass
    if not all(vals) and KEY_FILE.exists():
        d = json.loads(KEY_FILE.read_text(encoding="utf-8"))
        vals = [d.get(n) for n in names]
    return {"APCA-API-KEY-ID": vals[0], "APCA-API-SECRET-KEY": vals[1]} if all(vals) else None


def _get(hdr, symbols, start, end):
    rows, token = [], None
    while True:
        q = {"symbols": ",".join(s.replace("-", ".") for s in symbols), "timeframe": "1Min",
             "start": start, "end": end, "feed": "sip", "adjustment": "raw", "limit": 10000}
        if token:
            q["page_token"] = token
        req = urllib.request.Request(URL + "?" + urllib.parse.urlencode(q), headers=hdr)
        for k in range(5):
            try:
                d = json.load(urllib.request.urlopen(req, timeout=60))
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and k < 4:
                    time.sleep(10 * (k + 1))
                    continue
                raise
        for sym, bars in (d.get("bars") or {}).items():
            rows += [{"symbol": sym.replace(".", "-"), "t": b["t"], "c": b["c"], "v": b["v"]} for b in bars]
        token = d.get("next_page_token")
        if not token:
            return pd.DataFrame(rows, columns=["symbol", "t", "c", "v"])
        time.sleep(0.35)


def reduce_bars(bars, target, as_of=AS_OF):
    """1분봉(symbol, t[UTC 또는 ET], c, v) → 종목당 한 줄. 봉 끝 ≤ 대상일 as_of 인 봉만 씀."""
    target = pd.Timestamp(target).normalize()
    lim = target + _td(as_of)
    b = bars.copy()
    t = pd.to_datetime(b.t, utc=True)
    b["t"] = t.dt.tz_convert(ET).dt.tz_localize(None)
    end = b.t + pd.Timedelta(minutes=1)
    b = b[(b.t >= target + START) & (end <= lim)].sort_values("t")
    g = b.groupby("symbol")
    out = pd.DataFrame({"px_asof": g.c.last(), "t_asof": g.t.last(), "vol_pre": g.v.sum()})
    out["px_0830"] = b[b.t + pd.Timedelta(minutes=1) <= target + CHECK_AT].groupby("symbol").c.last()
    return out.reset_index()[["symbol", "px_0830", "px_asof", "t_asof", "vol_pre"]]


def fetch(targets, symbols, store=None, skip_existing=True, verbose=True, as_of=AS_OF):
    """대상일마다 08:00~as_of 분봉을 받아 줄인 표를 store(기본: as_of 폴더)에 저장. 키가 없으면 아무것도 안 하고 False."""
    store = store_for(as_of) if store is None else store
    hdr = _keys()
    if hdr is None:
        print("Alpaca 키 없음 → 분봉 없이 기존 갭(B0)으로 예측합니다")
        return False
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    for i, d in enumerate(sorted({pd.Timestamp(x).normalize() for x in targets})):
        f = store / f"{d:%Y-%m-%d}.parquet"
        if skip_existing and f.exists():
            continue
        s = d.tz_localize(ET)
        try:
            bars = _get(hdr, list(symbols), (s + START).isoformat(), (s + _td(as_of)).isoformat())
        except Exception as e:  # noqa: BLE001  받기 실패는 그 날만 B0 로
            print(f"  {d.date()} 받기 실패: {type(e).__name__}")
            continue
        reduce_bars(bars, d, as_of).to_parquet(f, index=False)
        if verbose and i % 50 == 0:
            print(f"  {d.date()} 저장 ({i + 1})", flush=True)
        time.sleep(0.35)
    return True


# ------------------------------------------------------------------ 피처
def load_store(target, store=STORE):
    f = Path(store) / f"{pd.Timestamp(target):%Y-%m-%d}.parquet"
    if not f.exists():
        return None
    return pd.read_parquet(f).rename(columns={"px_0915": "px_asof", "t_0915": "t_asof"})   # 예전 이름도 읽음


def build_gapz_pm(day, store=None, as_of=AS_OF):
    """symbol / gap_z / ext_range_z / pm (분봉 갭을 썼으면 1). 분봉이 없으면 gapz_rule 과 같은 값."""
    store = store_for(as_of) if store is None else store
    cols = ["symbol", "gap_z", "ext_range_z", "pm"]
    d = day.daily(days=40, columns=["symbol", "date_et", "close", "ret"])
    if not len(d) or pd.Timestamp(d.date_et.max()) != day.date:
        return pd.DataFrame(columns=cols)
    R = d.pivot_table(index="date_et", columns="symbol", values="ret", aggfunc="last").sort_index()
    C = d.pivot_table(index="date_et", columns="symbol", values="close", aggfunc="last").sort_index()
    last = R.index[-1]
    vol20 = R.rolling(20, min_periods=10).std().loc[last]
    close = C.loc[last]

    p = day.price(since=day.date, columns=["symbol", "datetime", "high", "low", "close", "session"])
    ext = p[(p.datetime >= day.date + G.CLOSE) & p.session.isin(["post", "pre"])].sort_values("datetime")
    g = ext.groupby("symbol")
    out = pd.DataFrame(index=pd.Index(close.index, name="symbol"))
    px_ext = g.close.last()
    out["gap_z"] = (px_ext / close - 1) / vol20
    out["ext_range_z"] = (g.high.max() - g.low.min()) / close / vol20
    out["pm"] = 0

    pm = load_store(day.target, store)
    if pm is not None and len(pm):
        pm = pm.set_index("symbol").reindex(out.index)
        bad = (pm.px_0830 / px_ext.reindex(out.index) - 1).abs() > TOL      # NaN 비교는 False → 확인값 없으면 그대로 씀
        gz = ((pm.px_asof / close - 1) / vol20).where(~bad)
        ok = gz.notna() & np.isfinite(gz)
        out.loc[ok, "gap_z"] = gz[ok]
        out.loc[ok, "pm"] = 1
    out = out.replace([np.inf, -np.inf], np.nan).reset_index()
    return out[cols]


# ------------------------------------------------------------------ 학습·예측 (규칙은 gapz_rule 그대로)
def build_table(days, store=None, as_of=AS_OF):
    rows = []
    for day in days:
        x = build_gapz_pm(day, store, as_of)
        if len(x):
            rows.append(x.merge(day.y[["symbol", "label"]], on="symbol"))
    t = pd.concat(rows, ignore_index=True)
    t["label"] = t.label.astype(int)
    return t


def fit(days, store=None, as_of=AS_OF):
    """params 에 as_of 를 같이 기록 → predict 가 학습과 같은 시각을 씀."""
    return {**G.fit_table(build_table(days, store, as_of)), "as_of": _tag(as_of)}


def predict(day, params, store=None):
    """day.symbols 전부에 label. 피처가 없는 종목은 보합(2). 시각은 params["as_of"] (없으면 09:15)."""
    x = build_gapz_pm(day, store, params.get("as_of", AS_OF))
    out = pd.DataFrame({"symbol": day.symbols})
    if len(x):
        x = x.assign(label=G.predict_table(x, params))
        out = out.merge(x[["symbol", "label"]], on="symbol", how="left")
    else:
        out["label"] = 2
    out["label"] = out.label.fillna(2).astype(int)
    return out


# ------------------------------------------------------------------ 명령
def main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch"])
    ap.add_argument("--since")
    ap.add_argument("--until")
    ap.add_argument("--as-of", default=AS_OF, help="몇 시까지 끝난 봉을 쓸지 (기본 09:15, 09:29 까지)")
    ap.add_argument("--store", default=None, help="기본: as_of 별 폴더")
    a = ap.parse_args(argv)
    store = a.store or str(store_for(a.as_of))
    sys.path.insert(0, str(ROOT))
    from src import Dataset
    ds = Dataset()
    days = list(ds.days(since=a.since, until=a.until))
    targets = [d.target for d in days]
    syms = sorted(set(ds.symbols))
    print(f"대상일 {len(targets)}개 ({min(targets).date()} ~ {max(targets).date()}), 종목 {len(syms)}개, as_of {_tag(a.as_of)} → {store}")
    fetch(targets, syms, store, as_of=a.as_of)


if __name__ == "__main__":
    main(sys.argv[1:])
