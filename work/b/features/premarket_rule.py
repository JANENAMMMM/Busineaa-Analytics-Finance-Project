"""B 후보: 분봉 9:15 갭 규칙 = gapz_rule + Alpaca 프리마켓 1분봉으로 갭 가격만 더 최신으로.

⚠️ 제출 후보입니다. 아직 제출 모델(src/model.py = B0, gapz_rule)이 아닙니다. 채택은 팀 결정 → docs/review/16_프리마켓_분봉.md

    받기 (사전 단계, 인터넷·키 필요)
        uv run python work/b/features/premarket_rule.py fetch --since 2024-08-13 --until 2026-05-28
    예측 (오프라인, 받아 둔 파일만 읽음)
        params = fit(train_days)  /  pred = predict(day, params)

규칙 (gapz_rule 과 같고 갭 가격만 다름)
    gap_z = (px_0915 ÷ 기준일 종가 − 1) ÷ vol20
      px_0915 = 대상일 08:00 ~ 09:15 SIP 1분봉 중 "봉 끝 ≤ 09:15" 인 마지막 봉 종가
      (Alpaca 무료 SIP 는 최근 15분 지연 → 09:15 확정 값은 09:30 전에 받을 수 있음)
    대체: 분봉 파일이 없거나, 그 종목 봉이 없거나, 단위 확인(px_0830 vs 교수님 08:00 봉 종가) 이 2% 넘게 어긋나면
          gapz_rule 의 갭(교수님 데이터) 그대로 → 키·인터넷이 없어도 B0 와 같은 예측
    ext_range_z, λ, a, b 는 gapz_rule 과 같음 (학습은 이 갭으로 다시)

저장 형식 (STORE/YYYY-MM-DD.parquet, 대상일별, 종목당 한 줄)
    symbol, px_0830, px_0915, t_0915 (마지막 봉 시작, ET), vol_pre (08:00~09:15 거래량)
    원본 1분봉은 저장하지 않음. 09:15 이후 가격은 파일에 아예 없음

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
KEY_FILE = ROOT / "artifacts" / "alpaca_key.json"  # git 제외 (예전 방식)
ENV_FILE = ROOT / ".env"                           # git 제외, 팀 공용 키 보관 위치
URL = "https://data.alpaca.markets/v2/stocks/bars"
AS_OF = pd.Timedelta(hours=9, minutes=15)          # 이 시각까지 끝난 봉만
CHECK_AT = pd.Timedelta(hours=8, minutes=30)       # 교수님 08:00 봉 종가와 같은 시점 (2026-10-11 대조)
START = pd.Timedelta(hours=8)
TOL = 0.02
ET = "America/New_York"


# ------------------------------------------------------------------ 받기 (사전 단계)
def _keys():
    names = ["APCA_API_KEY_ID", "APCA_API_SECRET_KEY"]
    vals = [os.environ.get(n) for n in names]
    if not all(vals) and sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                vals = [v or winreg.QueryValueEx(k, n)[0] for v, n in zip(vals, names)]
        except OSError:
            pass
    if not all(vals) and ENV_FILE.exists():
        d = dict(
            line.split("=", 1)
            for line in ENV_FILE.read_text(encoding="utf-8").splitlines()
            if "=" in line and not line.lstrip().startswith("#")
        )
        vals = [v or d.get(n, "").strip().strip("\"'") or None for v, n in zip(vals, names)]
    if not all(vals) and KEY_FILE.exists():
        d = json.loads(KEY_FILE.read_text(encoding="utf-8"))
        vals = [v or d.get(n) for v, n in zip(vals, names)]
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


def reduce_bars(bars, target):
    """1분봉(symbol, t[UTC 또는 ET], c, v) → 종목당 한 줄. 봉 끝 ≤ 대상일 09:15 인 봉만 씀."""
    target = pd.Timestamp(target).normalize()
    b = bars.copy()
    t = pd.to_datetime(b.t, utc=True)
    b["t"] = t.dt.tz_convert(ET).dt.tz_localize(None)
    end = b.t + pd.Timedelta(minutes=1)
    b = b[(b.t >= target + START) & (end <= target + AS_OF)].sort_values("t")
    g = b.groupby("symbol")
    out = pd.DataFrame({"px_0915": g.c.last(), "t_0915": g.t.last(), "vol_pre": g.v.sum()})
    out["px_0830"] = b[b.t + pd.Timedelta(minutes=1) <= target + CHECK_AT].groupby("symbol").c.last()
    return out.reset_index()[["symbol", "px_0830", "px_0915", "t_0915", "vol_pre"]]


def fetch(targets, symbols, store=STORE, skip_existing=True, verbose=True):
    """대상일마다 08:00~09:15 분봉을 받아 줄인 표를 store 에 저장. 키가 없으면 아무것도 안 하고 False."""
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
            bars = _get(hdr, list(symbols), (s + START).isoformat(), (s + AS_OF).isoformat())
        except Exception as e:  # noqa: BLE001  받기 실패는 그 날만 B0 로
            print(f"  {d.date()} 받기 실패: {type(e).__name__}")
            continue
        reduce_bars(bars, d).to_parquet(f, index=False)
        if verbose and i % 50 == 0:
            print(f"  {d.date()} 저장 ({i + 1})", flush=True)
        time.sleep(0.35)
    return True


# ------------------------------------------------------------------ 피처
def load_store(target, store=STORE):
    f = Path(store) / f"{pd.Timestamp(target):%Y-%m-%d}.parquet"
    return pd.read_parquet(f) if f.exists() else None


def build_gapz_pm(day, store=STORE):
    """symbol / gap_z / ext_range_z / pm (분봉 갭을 썼으면 1). 분봉이 없으면 gapz_rule 과 같은 값."""
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
        gz = ((pm.px_0915 / close - 1) / vol20).where(~bad)
        ok = gz.notna() & np.isfinite(gz)
        out.loc[ok, "gap_z"] = gz[ok]
        out.loc[ok, "pm"] = 1
    out = out.replace([np.inf, -np.inf], np.nan).reset_index()
    return out[cols]


# ------------------------------------------------------------------ 학습·예측 (규칙은 gapz_rule 그대로)
def build_table(days, store=STORE):
    rows = []
    for day in days:
        x = build_gapz_pm(day, store)
        if len(x):
            rows.append(x.merge(day.y[["symbol", "label"]], on="symbol"))
    t = pd.concat(rows, ignore_index=True)
    t["label"] = t.label.astype(int)
    return t


def fit(days, store=STORE):
    return G.fit_table(build_table(days, store))


def predict(day, params, store=STORE):
    """day.symbols 전부에 label. 피처가 없는 종목은 보합(2)."""
    x = build_gapz_pm(day, store)
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
    ap.add_argument("--store", default=str(STORE))
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    from src import Dataset
    ds = Dataset()
    days = list(ds.days(since=a.since, until=a.until))
    targets = [d.target for d in days]
    syms = sorted(set(ds.symbols))
    print(f"대상일 {len(targets)}개 ({min(targets).date()} ~ {max(targets).date()}), 종목 {len(syms)}개 → {a.store}")
    fetch(targets, syms, a.store)


if __name__ == "__main__":
    main(sys.argv[1:])
