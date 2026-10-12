"""09:29 제출 모델의 독립 구현 일치, 결측 대체, 누수 검사."""
from pathlib import Path

import pytest

from src import DATASET, Dataset
from src.model import load_model
from lab.leak import check_no_leak
from work.a import gapz_rule
from work.b.features import premarket_rule

pytestmark = pytest.mark.skipif(not DATASET.exists(), reason="dataset/ 없음")


@pytest.fixture(scope="module")
def days():
    ds = Dataset()
    return [ds.day("2025-11-14"), ds.day("2026-01-16"), ds.day("2026-01-30")]   # 1/30 → 2/2 는 갭 결측일


def test_no_leak(days):
    assert check_no_leak(load_model().predict, days, verbose=False)


def test_same_as_premarket_reference(days):
    m = load_model()
    assert m.premarket and m.params["as_of"] == "09:29"
    for d in days:
        a = m.predict(d).sort_values("symbol").reset_index(drop=True)
        b = premarket_rule.predict(d, m.params).sort_values("symbol").reset_index(drop=True)
        assert a.equals(b)


def test_missing_premarket_uses_b0_gap(days, tmp_path, monkeypatch):
    m = load_model()
    monkeypatch.setattr("src.model.ARTIFACTS", Path(tmp_path))
    out = m.predict(days[2]).sort_values("symbol").reset_index(drop=True)
    expected = gapz_rule.predict(days[2], m.params).sort_values("symbol").reset_index(drop=True)
    assert set(out.symbol) == set(days[2].symbols)
    assert out.equals(expected)
