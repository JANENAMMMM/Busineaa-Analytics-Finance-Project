"""Alpaca 프리마켓 1분봉 수집 (dev 기간). 결과는 cache/premarket/ (git 무시).

    uv run python work/b/review/premarket/fetch.py test      # 한 종목 하루로 연결 확인
    uv run python work/b/review/premarket/fetch.py           # 전체 수집 (이미 받은 날짜는 건너뜀)
    uv run python work/b/review/premarket/fetch.py iex       # IEX 피드 (무료 실시간) → cache/premarket_iex/

- 키: 환경변수 APCA_API_KEY_ID / APCA_API_SECRET_KEY → 없으면 Windows 사용자 환경변수 → 없으면 artifacts/alpaca_key.json
- 범위: 대상일마다 ET 08:00 ~ 09:30 (봉 시작 시각 기준), 1분봉, feed=sip, 분할·배당 조정 없음(raw)
  · raw 로 받는 이유: 교수님 price 데이터의 전일 종가와 같은 기준으로 비교하려고 (분할은 겹치는 08:00 봉으로 대조)
- 대상일: cache/gapz_rule.parquet 의 target 중 2026-05-29 이하 (홀드아웃 기간은 받지 않음)
- 종목: 50종목 + SPY, QQQ
"""

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FEED = "iex" if "iex" in sys.argv[1:] else "sip"
CACHE = ROOT / "cache" / ("premarket_iex" if FEED == "iex" else "premarket")
URL = "https://data.alpaca.markets/v2/stocks/bars"
DEV_END = pd.Timestamp("2026-05-29")
EXTRA = ["SPY", "QQQ"]


def keys():
    """premarket_rule._keys() 와 같은 순서로 찾음 (환경변수 → winreg → .env → alpaca_key.json)."""
    sys.path.insert(0, str(ROOT / "work/b/features"))
    import premarket_rule as PR
    hdr = PR._keys()
    if hdr is None:
        raise SystemExit("Alpaca 키가 없습니다 (.env 의 APCA_API_KEY_ID / APCA_API_SECRET_KEY 확인)")
    return hdr


def get_bars(hdr, symbols, start, end):
    """start, end: tz 포함 ISO 문자열. 페이지를 끝까지 넘김."""
    rows, token = [], None
    while True:
        q = {"symbols": ",".join(s.replace("-", ".") for s in symbols), "timeframe": "1Min", "start": start, "end": end,
             "feed": FEED, "adjustment": "raw", "limit": 10000}
        if token:
            q["page_token"] = token
        req = urllib.request.Request(URL + "?" + urllib.parse.urlencode(q), headers=hdr)
        for k in range(5):
            try:
                d = json.load(urllib.request.urlopen(req, timeout=60))
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    time.sleep(10 * (k + 1))
                    continue
                raise SystemExit(f"HTTP {e.code}: {e.read()[:300]!r}")
        for sym, bars in (d.get("bars") or {}).items():
            for b in bars:
                rows.append({"symbol": sym.replace(".", "-"), "t": b["t"], "o": b["o"], "h": b["h"], "l": b["l"],
                             "c": b["c"], "v": b["v"], "n": b.get("n")})
        token = d.get("next_page_token")
        if not token:
            return rows
        time.sleep(0.35)


def window(target):
    s = pd.Timestamp(target).tz_localize("America/New_York")
    return (s + pd.Timedelta(hours=8)).isoformat(), (s + pd.Timedelta(hours=9, minutes=30)).isoformat()


def main():
    hdr = keys()
    sys.path.insert(0, str(ROOT / "work/b/review/explain"))
    sys.path.insert(0, str(ROOT / "work/b/features"))
    import explain_rule as E
    t, _ = E.load()
    syms = sorted(t.symbol.unique()) + EXTRA
    targets = sorted(d for d in pd.to_datetime(t.target.dropna().unique()) if d <= DEV_END)
    CACHE.mkdir(parents=True, exist_ok=True)

    if sys.argv[1:] == ["test"]:
        s, e = window(targets[-1])
        r = pd.DataFrame(get_bars(hdr, ["AMD", "SPY"], s, e))
        print(f"연결 OK · {targets[-1].date()} AMD·SPY {len(r)}봉")
        print(r.groupby("symbol").agg(첫봉=("t", "min"), 끝봉=("t", "max"), 봉수=("t", "size"), 거래량=("v", "sum")).to_string())
        return

    done = {p.stem for p in CACHE.glob("*.parquet")}
    todo = [d for d in targets if d.strftime("%Y-%m-%d") not in done]
    print(f"대상일 {len(targets)}개 중 남은 것 {len(todo)}개, 종목 {len(syms)}개")
    t0 = time.time()
    for i, d in enumerate(todo):
        s, e = window(d)
        r = pd.DataFrame(get_bars(hdr, syms, s, e))
        r.to_parquet(CACHE / f"{d.strftime('%Y-%m-%d')}.parquet", index=False)
        if i % 25 == 0:
            print(f"  {i + 1}/{len(todo)} {d.date()} {len(r)}봉 · {time.time() - t0:.0f}초", flush=True)
        time.sleep(0.35)
    print("끝")


if __name__ == "__main__":
    main()
