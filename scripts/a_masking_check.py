"""기업명이 가려지면 무엇이 깨지는지 센다 (2026-10-08).

    uv run python scripts/a_masking_check.py

배경: 평가 데이터에 기업명이 마스킹돼 있을 수 있음 (10/8 회의, 재은).
채민 LLM 파이프라인은 제목 안의 기업명 문자열로 기사를 종목에 연결하므로 이름이 사라지면 후보가 0 이 됨.
데이터가 주는 `symbols` 열은 구조화된 값이라 영향이 없음 — 둘의 커버리지를 비교한다.

해설: docs/worklog/2026-10-08_A_마스킹_점검.md
dev 기간(기준일 ≤ 2026-05-28)만 본다. 홀드아웃은 보지 않음. API 호출 없음.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lab.cv import CACHE  # noqa: E402

DEV_END = pd.Timestamp("2026-05-28")      # dev 마지막 기준일
N_CELLS = 22450                            # dev 종목·날짜 칸 (50 x 449)


def main():
    spec = json.loads((CACHE / "llm_news_v6_dev" / "extraction_spec.json").read_text(encoding="utf-8"))
    al = spec["aliases"]
    print(f"채민 별칭 사전: 종목 {len(al)}개")

    va = pd.read_csv(CACHE / "llm_news_v6_dev" / "valid_articles.csv", encoding="utf-8-sig",
                     usecols=["symbol", "title"])
    pat = {s: re.compile("|".join(r"\b" + re.escape(a) + r"\b" for a in v), re.I) for s, v in al.items()}
    hit = np.array([bool(pat[s].search(t)) if s in pat else False for s, t in zip(va.symbol, va.title)])
    print(f"\n[A] 채민 검증 통과 기사 {len(va):,}건 중 제목에 기업명·티커가 있는 것: "
          f"{hit.sum():,} ({hit.mean():.1%})")
    print("    -> 이름이 가려지면 남는 것: 0건 (후보 선별 조건 자체가 이름 매칭)")

    t = pd.read_parquet(CACHE / "news_titles.parquet", columns=["date", "symbol"])
    cov = t[t.date <= DEV_END].groupby(["date", "symbol"]).size()
    nf = pd.read_csv(CACHE / "llm_news_v6_dev" / "daily_features.csv", encoding="utf-8-sig",
                     usecols=["feature_relevant_share"])
    n_llm = int(nf.feature_relevant_share.notna().sum())
    print(f"\n[B] dev {N_CELLS:,} 종목·날짜 중 기사가 연결된 칸")
    print(f"    symbols 열 기반 (마스킹돼도 작동): {len(cov):,} ({len(cov) / N_CELLS:.1%})")
    print(f"    채민 LLM 피처 있는 칸 (이름 필요):  {n_llm:,} ({n_llm / N_CELLS:.1%})")
    print(f"    -> 배수 {len(cov) / n_llm:.1f}x")
    print(f"\n[C] 기사 수: symbols 열 {len(t[t.date <= DEV_END]):,}건 vs 채민 검증 통과 {len(va):,}건 "
          f"({len(t[t.date <= DEV_END]) / len(va):.0f}x)")


if __name__ == "__main__":
    main()
