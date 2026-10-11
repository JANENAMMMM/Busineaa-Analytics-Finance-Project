"""16 문서 그림 2장. exp.py · exp_iex.py 결과(out/)와 cache/premarket/ 원본 분봉을 읽음.

    uv run python work/b/review/premarket/figs.py

1. timing.png  교수님 시간봉 종가가 실제로 몇 시 가격인가 (Alpaca 분봉과 차이 중앙값, 대상일 4일에 1일 표본)
2. curve.png   갭 가격 시점별 4폴드 평균 점수 (SIP · IEX · B0 · 교수님 9:00 봉 진단)
"""

import sys
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "work/b/review/explain"))
sys.path.insert(0, str(ROOT / "work/b/features"))
import explain_rule as E  # noqa: E402

OUT, FIGS = HERE / "out", HERE / "figs"
TIMES = ["08:15", "08:30", "08:45", "09:00", "09:15", "09:29"]


def timing():
    fs = sorted((ROOT / "cache" / "premarket").glob("*.parquet"))[::4]
    m = pd.concat([pd.read_parquet(f).assign(target=pd.Timestamp(f.stem)) for f in fs], ignore_index=True)
    m["t"] = pd.to_datetime(m.t, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    m["end"] = m.t + pd.Timedelta(minutes=1)
    con = duckdb.connect()
    p = con.execute(f"""SELECT symbol, datetime, close FROM read_parquet('{ROOT / "dataset" / "price.parquet"}')
                        WHERE session = 'pre' AND hour(datetime) IN (8, 9)""").df()
    con.close()
    p["target"], p["bar"] = p.datetime.dt.normalize(), p.datetime.dt.strftime("%H:%M")
    rows = []
    for tm in TIMES:
        hh, mm = map(int, tm.split(":"))
        a = m[m.end <= m.target + pd.Timedelta(hours=hh, minutes=mm)].sort_values("t").groupby(["target", "symbol"]).c.last()
        for bar in ["08:00", "09:00"]:
            z = p[p.bar == bar].set_index(["target", "symbol"]).close
            j = pd.concat([a.rename("a"), z.rename("z")], axis=1, join="inner")
            rows.append({"시각": tm, "봉": bar, "차이": float((j.a / j.z - 1).abs().median() * 100)})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "timing.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    for bar, c, lab in [("08:00", E.C1, "교수님 08:00 봉 종가 (공개 09:00, 지금 B0 의 갭 가격)"),
                        ("09:00", E.C2, "교수님 09:00 봉 종가 (공개 09:30, 규칙상 못 씀)")]:
        d = r[r.봉 == bar]
        ax.plot(d.시각, d.차이, marker="o", color=c, label=lab)
        k = d.차이.idxmin()
        ax.annotate(f"가장 비슷: {d.시각[k]}", (d.시각[k], d.차이[k]), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color=c)
    ax.set_xlabel("Alpaca SIP 1분봉 가격 시각 (이 시각까지 끝난 마지막 봉, ET)")
    ax.set_ylabel("가격 차이 중앙값 (%)")
    ax.legend(fontsize=8, loc="upper center")
    ax.set_title("교수님 시간봉 종가는 봉 끝보다 약 30분 이른 가격", loc="left")
    fig.tight_layout()
    fig.savefig(FIGS / "timing.png", dpi=160)
    plt.close(fig)
    return r


def curve():
    s = pd.read_csv(OUT / "exp_summary.csv", encoding="utf-8-sig").set_index("설정")
    iex = pd.read_csv(OUT / "exp_iex.csv", encoding="utf-8-sig").groupby("설정").score.mean()
    b0 = s.loc["지금 (08:00 봉)", "평균"]
    diag = s.loc["교수님 9:00 봉 (9:30, 진단)", "평균"]
    xs = ["08:30", "09:00", "09:05", "09:15", "09:20", "09:25", "09:29"]
    ys = [s.loc[f"분봉 {x}", "평균"] for x in xs]
    xi = ["09:15", "09:25", "09:29"]
    yi = [iex[f"IEX {x.lstrip('0')}"] for x in xi]

    def mins(x):
        h, m = map(int, x.split(":"))
        return h * 60 + m - 8 * 60

    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.axvspan(mins("09:15"), mins("09:30"), color=E.GRID, alpha=0.8, linewidth=0)
    ax.text(mins("09:16"), b0 - 0.0015, "무료 SIP 로는 9:30 전에\n못 받는 구간 (15분 지연)", fontsize=7.5, color=E.INK2, va="top")
    ax.axhline(b0, color=E.INK2, linestyle="--", linewidth=1)
    ax.text(mins("08:30"), b0 + 0.001, f"지금 B0 {b0:.3f}", fontsize=8, color=E.INK2, va="bottom")
    ax.plot([mins(x) for x in xs], ys, marker="o", color=E.C1, label="SIP 1분봉 (전체 거래소)")
    ax.plot([mins(x) for x in xi], yi, marker="s", color=E.C3, label="IEX 1분봉 (무료 실시간, 체결 적음)")
    ax.scatter([mins("09:15")], [diag], marker="*", s=120, color=E.C2, zorder=4, label="교수님 09:00 봉 (진단 전용)")
    k = xs.index("09:15")
    ax.scatter([mins("09:15")], [ys[k]], s=140, facecolors="none", edgecolors=E.C1, linewidths=2, zorder=5)
    ax.annotate(f"판정안 9:15\n{ys[k]:.3f} (+{ys[k] - b0:.3f})", (mins("09:15"), ys[k]), textcoords="offset points",
                xytext=(-58, 8), fontsize=8, color=E.C1)
    ax.set_xticks([mins(x) for x in xs], xs)
    ax.set_xlabel("갭 가격 시점 (이 시각까지 끝난 마지막 1분봉 종가, ET)")
    ax.set_ylabel("4폴드 평균 점수")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("갭 가격이 최신일수록 점수가 오른다", loc="left")
    fig.tight_layout()
    fig.savefig(FIGS / "curve.png", dpi=160)
    plt.close(fig)


def main():
    FIGS.mkdir(exist_ok=True)
    print(timing().pivot_table(index="시각", columns="봉", values="차이").round(3))
    curve()


if __name__ == "__main__":
    main()
