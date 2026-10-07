"""채민 LLM 표(뉴스 v6, Reddit v4)를 symbol + target 으로 붙일 때 기준일/대상일이 하루 어긋나지 않는지 확인.
dataset/ 과 cache/llm_news_v6_dev/, cache/llm_reddit_v4/ 가 있어야 돌아감 (scripts/a_llm_news.py 참고)."""
import pandas as pd
import pytest

from lab.folds import HOLDOUT
from src import DATASET, Dataset
from src.paths import ROOT

NEWS = ROOT / "cache" / "llm_news_v6_dev" / "daily_features.csv"
REDDIT = ROOT / "cache" / "llm_reddit_v4" / "daily_features.csv"
pytestmark = pytest.mark.skipif(not (DATASET.exists() and NEWS.exists() and REDDIT.exists()),
                                reason="dataset/ 또는 LLM 표 없음")


def _read(path):
    x = pd.read_csv(path, encoding="utf-8-sig", nrows=None)
    x["target"] = pd.to_datetime(x["target"])
    if "date" in x:
        x["date"] = pd.to_datetime(x["date"], format="ISO8601")   # 추가 기간 행은 "2026-02-12 00:00:00" 형식
    return x


@pytest.fixture(scope="module")
def cal():
    dev = Dataset().split(HOLDOUT)[0]
    return pd.DataFrame({"date": [d.date for d in dev], "target_ours": [d.target for d in dev]})


@pytest.mark.parametrize("path", [NEWS, REDDIT])
def test_no_holdout_and_unique(path):
    x = _read(path)
    assert x.target.max() < pd.Timestamp(HOLDOUT)
    assert not x.duplicated(["symbol", "target"]).any()


@pytest.mark.parametrize("path", [NEWS, REDDIT])
def test_targets_are_our_targets(path, cal):
    assert _read(path).target.isin(cal.target_ours).all()


def test_news_date_target_matches_calendar(cal):
    j = _read(NEWS)[["date", "target"]].drop_duplicates().merge(cal, on="date", how="left")
    assert j.target_ours.notna().all()
    assert (j.target == j.target_ours).all()      # CSV 의 date 는 기준일, target 은 우리 day.target 과 같음
