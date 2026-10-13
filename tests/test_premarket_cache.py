"""Malformed external caches must never enter inference or destroy saved inputs."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import DATASET, Dataset
import src.model as model
from work.b.features import premarket_rule as collector


def row(symbol="AMD"):
    return dict(symbol=symbol, px_0830=100., px_asof=101.,
                t_asof="2026-09-11 09:28:00", vol_pre=10.)


@pytest.mark.parametrize("field,value", [
    ("px_asof", 0), ("px_asof", -1), ("px_asof", np.inf),
    ("px_asof", np.nan), ("px_asof", "bad"),
    ("px_0830", -1), ("px_0830", np.inf), ("px_0830", "bad"),
    ("t_asof", "2026-09-10 09:28:00"),
    ("t_asof", "2026-09-11 09:29:00"),
    ("t_asof", "2026-09-11 07:59:00"),
    ("t_asof", "2026-09-11 09:28:01"),
    ("t_asof", "bad"), ("t_asof", None), ("t_asof", 123),
    ("symbol", ""), ("symbol", None),
])
def test_bad_row_rejected_without_losing_good_row(tmp_path, field, value):
    bad = row(); bad[field] = value
    path = tmp_path / "2026-09-11.parquet"
    frame = pd.DataFrame([bad, row("MSFT")])
    if (field in {"px_asof", "px_0830"} and isinstance(value, str)) or (field == "t_asof" and isinstance(value, int)):
        frame[field] = frame[field].astype(str)
    frame.to_parquet(path, index=False)
    with pytest.warns(RuntimeWarning):
        valid = model._read_premarket_cache(path, "2026-09-11", "09:29")
    assert list(valid.index) == ["MSFT"]


@pytest.mark.parametrize("stamp", ["2026-09-11 09:28:00", "2026-09-11T13:28:00Z",
                                   "2026-09-11T09:28:00-04:00"])
def test_equivalent_eastern_and_utc_times(tmp_path, stamp):
    data = row(); data["t_asof"] = stamp
    path = tmp_path / "cache.parquet"
    pd.DataFrame([data]).to_parquet(path, index=False)
    assert list(model._read_premarket_cache(path, "2026-09-11", "09:29").index) == ["AMD"]


def test_duplicate_symbol_rejects_all_ambiguous_rows(tmp_path):
    path = tmp_path / "cache.parquet"
    pd.DataFrame([row(), row(), row("MSFT")]).to_parquet(path, index=False)
    with pytest.warns(RuntimeWarning):
        valid = model._read_premarket_cache(path, "2026-09-11", "09:29")
    assert list(valid.index) == ["MSFT"]


@pytest.mark.skipif(not DATASET.exists(), reason="dataset/ unavailable")
@pytest.mark.parametrize("problem", ["corrupt", "missing_column", "duplicate", "late", "bad_price"])
def test_inference_fallback_preserves_other_symbols_and_parameters(tmp_path, monkeypatch, problem):
    day = Dataset().day("2026-09-10")
    m = model.load_model()
    params = m.params.copy()
    normal = model.build_gapz_premarket(day, "09:29").set_index("symbol")
    base = model.build_gapz(day).set_index("symbol")
    cache = pd.read_parquet(model.ARTIFACTS / "premarket_0929" / "2026-09-11.parquet")
    symbol = cache.symbol.iloc[0]
    path = tmp_path / "premarket_0929" / "2026-09-11.parquet"
    path.parent.mkdir()
    if problem == "corrupt":
        path.write_bytes(b"not parquet")
    else:
        if problem == "missing_column":
            cache = cache.drop(columns="px_asof")
        elif problem == "duplicate":
            cache = pd.concat([cache, cache.iloc[[0]]], ignore_index=True)
        elif problem == "late":
            cache.loc[cache.symbol == symbol, "t_asof"] = pd.Timestamp("2026-09-11 09:29")
        elif problem == "bad_price":
            cache.loc[cache.symbol == symbol, "px_asof"] = -1
        cache.to_parquet(path, index=False)
    monkeypatch.setattr(model, "ARTIFACTS", tmp_path)
    with pytest.warns(RuntimeWarning):
        actual = model.build_gapz_premarket(day, "09:29").set_index("symbol")
        output = m.predict(day)
    expected = base if problem in {"corrupt", "missing_column"} else normal.copy()
    if problem not in {"corrupt", "missing_column"}:
        expected.loc[symbol] = base.loc[symbol]
    pd.testing.assert_frame_equal(actual, expected)
    assert set(output.symbol) == set(day.symbols)
    assert output.label.between(0, 4).all()
    assert m.params == params


def test_fetch_adds_only_missing_symbols_and_is_idempotent(tmp_path, monkeypatch):
    path = tmp_path / "2026-09-11.parquet"
    pd.DataFrame([row()]).to_parquet(path, index=False)
    calls = []
    monkeypatch.setattr(collector, "_keys", lambda: {"key": "test"})
    monkeypatch.setattr(collector.time, "sleep", lambda _: None)
    def get(headers, symbols, start, end):
        calls.append(symbols)
        return pd.DataFrame()
    monkeypatch.setattr(collector, "_get", get)
    monkeypatch.setattr(collector, "reduce_bars", lambda *a: pd.DataFrame([row("MSFT"), row("MSFT")]))
    assert collector.fetch(["2026-09-11"], ["AMD", "MSFT"], store=tmp_path)
    assert calls == [["MSFT"]]
    saved = pd.read_parquet(path)
    assert list(saved.symbol) == ["AMD", "MSFT"]
    pd.testing.assert_frame_equal(saved.iloc[[0]].reset_index(drop=True), pd.DataFrame([row()]))
    assert collector.fetch(["2026-09-11"], ["AMD", "MSFT"], store=tmp_path)
    assert calls == [["MSFT"]]


def test_fetch_failure_preserves_cache_bytes(tmp_path, monkeypatch):
    path = tmp_path / "2026-09-11.parquet"
    pd.DataFrame([row()]).to_parquet(path, index=False)
    original = path.read_bytes()
    monkeypatch.setattr(collector, "_keys", lambda: {"key": "test"})
    def fail(*args):
        raise RuntimeError("simulated failure")
    monkeypatch.setattr(collector, "_get", fail)
    assert not collector.fetch(["2026-09-11"], ["AMD", "MSFT"], store=tmp_path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("problem", ["corrupt", "missing_column"])
def test_fetch_invalid_existing_cache_preserved(tmp_path, monkeypatch, problem):
    path = tmp_path / "2026-09-11.parquet"
    if problem == "corrupt":
        path.write_bytes(b"broken parquet")
    else:
        pd.DataFrame([row()]).drop(columns="px_asof").to_parquet(path, index=False)
    before = path.read_bytes()
    monkeypatch.setattr(collector, "_keys", lambda: {"key": "test"})
    def unexpected(*args):
        pytest.fail("invalid cache must be preserved without an API request")
    monkeypatch.setattr(collector, "_get", unexpected)
    assert not collector.fetch(["2026-09-11"], ["AMD", "MSFT"], store=tmp_path)
    assert path.read_bytes() == before


def test_fetch_removes_existing_duplicates_without_api(tmp_path, monkeypatch):
    path = tmp_path / "2026-09-11.parquet"
    pd.DataFrame([row(), row()]).to_parquet(path, index=False)
    monkeypatch.setattr(collector, "_keys", lambda: {"key": "test"})
    monkeypatch.setattr(collector.time, "sleep", lambda _: None)
    def unexpected(*args):
        pytest.fail("complete cache must not issue an API request")
    monkeypatch.setattr(collector, "_get", unexpected)
    assert collector.fetch(["2026-09-11"], ["AMD"], store=tmp_path)
    assert len(pd.read_parquet(path)) == 1


def test_fetch_write_failure_preserves_cache_bytes(tmp_path, monkeypatch):
    path = tmp_path / "2026-09-11.parquet"
    pd.DataFrame([row()]).to_parquet(path, index=False)
    before = path.read_bytes()
    monkeypatch.setattr(collector, "_keys", lambda: {"key": "test"})
    monkeypatch.setattr(collector, "_get", lambda *a: pd.DataFrame())
    monkeypatch.setattr(collector, "reduce_bars", lambda *a: pd.DataFrame([row("MSFT")]))
    def fail(*args, **kwargs):
        raise OSError("simulated disk failure")
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
    assert not collector.fetch(["2026-09-11"], ["AMD", "MSFT"], store=tmp_path)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_legacy_missing_unit_check_price_is_preserved(tmp_path):
    data = row(); data["px_0830"] = np.nan
    path = tmp_path / "cache.parquet"
    pd.DataFrame([data]).to_parquet(path, index=False)
    valid = model._read_premarket_cache(path, "2026-09-11", "09:29")
    assert list(valid.index) == ["AMD"]


def test_utc_winter_timestamp_uses_eastern_standard_time(tmp_path):
    data = row(); data["t_asof"] = "2026-01-20T14:28:00Z"
    path = tmp_path / "cache.parquet"
    pd.DataFrame([data]).to_parquet(path, index=False)
    assert list(model._read_premarket_cache(path, "2026-01-20", "09:29").index) == ["AMD"]
