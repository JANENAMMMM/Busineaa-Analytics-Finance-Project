"""Leakage and bookkeeping checks for the oct12 follow-up candidates."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/followup_checks'
SPEC = json.loads((ROOT / 'lab/folds.json').read_text(encoding='utf-8'))
CUTOFF = pd.Timedelta(hours=9, minutes=30)
pytestmark = pytest.mark.skipif(not OUT.exists(), reason='follow-up run not present')


def test_price_recency_uses_only_bars_known_before_the_open():
    """Every bar behind the features is published strictly before the target 09:30."""
    r = pd.read_parquet(ROOT / 'cache/oct12_price_recency.parquet')
    assert len(r) == 22450 and not r.duplicated(['date', 'target', 'symbol']).any()
    # price_age_min is cutoff minus the last used known_at, so a positive age proves the bar predates the open.
    assert r.price_age_min.dropna().gt(0).all()
    p = pq.read_table(ROOT / 'dataset/price.parquet',
                      columns=['symbol', 'datetime', 'session', 'known_at'],
                      filters=[('symbol', '=', 'AMD')]).to_pandas()
    one = r[(r.symbol == 'AMD') & r.price_age_min.notna()].iloc[0]
    window = p[(p.datetime >= one.date + pd.Timedelta(hours=16)) & (p.known_at < one.target + CUTOFF)
               & p.session.isin(['post', 'pre'])]
    assert len(window) and window.known_at.max() == one.target + CUTOFF - pd.Timedelta(minutes=one.price_age_min)


def test_adding_later_days_does_not_change_earlier_price_features():
    """Recomputing the window from a truncated price file reproduces the earlier rows."""
    r = pd.read_parquet(ROOT / 'cache/oct12_price_recency.parquet')
    end = pd.Timestamp(SPEC['folds'][0]['train'][1])
    early = r[r.date <= end].sort_values(['date', 'symbol']).reset_index(drop=True)
    cal = r[['date', 'target']].drop_duplicates().sort_values('date').reset_index(drop=True)
    cal = cal[cal.date <= end]
    p = pq.read_table(ROOT / 'dataset/price.parquet',
                      columns=['symbol', 'datetime', 'close', 'session', 'known_at'],
                      filters=[('known_at', '<', end + pd.Timedelta(days=5))]).to_pandas()
    p = p[p.session.isin(['post', 'pre'])].sort_values('datetime')
    start = (cal.date + pd.Timedelta(hours=16)).to_numpy()
    i = np.searchsorted(start, p.datetime.to_numpy(), side='right') - 1
    keep = i >= 0
    p = p.loc[keep].copy()
    i = i[keep]
    p['date'] = cal.date.to_numpy()[i]
    p['target'] = cal.target.to_numpy()[i]
    day = p.datetime.dt.normalize().to_numpy()
    on = (((p.session.to_numpy() == 'post') & (day == p.date.to_numpy()))
          | ((p.session.to_numpy() == 'pre') & (day == p.target.to_numpy())))
    cut = (p.target + CUTOFF).to_numpy()
    p = p[on & (p.datetime.to_numpy() < cut) & (p.known_at.to_numpy() < cut)].copy()
    p['cutoff'] = p.target + CUTOFF
    last = p.sort_values('datetime').groupby(['date', 'target', 'symbol'], sort=False).tail(1)
    age = ((last.cutoff - last.known_at).dt.total_seconds() / 60)
    age.index = pd.MultiIndex.from_frame(last[['date', 'target', 'symbol']])
    recomputed = age.reindex(pd.MultiIndex.from_frame(early[['date', 'target', 'symbol']]))
    pd.testing.assert_series_equal(recomputed.reset_index(drop=True), early.price_age_min,
                                   check_names=False)


def test_missing_zero_and_not_computed_are_separate_codes():
    z = pd.read_csv(OUT / 'novelty_panel_coded.csv')
    codes = set(z.novelty_code.unique())
    assert {'no_articles', 'no_prior_comparison', 'not_computed_out_of_screen_scope'} <= codes
    assert z.novelty_code.notna().all()
    assert z.loc[z.novelty_code == 'no_articles', 'news_count'].eq(0).all()
    assert z.loc[z.novelty_code == 'no_prior_comparison', 'novelty'].isna().all()
    assert z.loc[z.novelty_code == 'computed_positive', 'novelty'].gt(0).all()
    # Gap codes keep missing and exact zero out of the disagreement tally.
    assert set(z.gap_code.unique()) <= {'usable', 'gap_missing', 'gap_exact_zero'}
    assert z.loc[z.gap_code == 'gap_exact_zero', 'gap_z'].eq(0).all()


def test_t1_novelty_threshold_uses_only_earlier_days():
    """A held day has fewer than 20 prior observations and no high/low verdict."""
    z = pd.read_csv(OUT / 't1_screen_panel.csv', parse_dates=['date'])
    assert z.prior_days.lt(20).eq(z.high.isna() | z.novelty.isna()).all() or \
        z.loc[z.prior_days < 20, 'high'].isna().all()
    assert z.loc[z.high.notna(), 'prior_days'].ge(20).all()
    for _, g in z.groupby('symbol'):
        g = g.sort_values('date')
        assert g.prior_days.is_monotonic_increasing


def test_symbol_exclusion_and_evaluation_universe_hold():
    pred = pd.read_parquet(OUT / 'p1_predictions.parquet')
    for (fold, kind), g in pred.groupby(['fold', 'kind']):
        assert len(g) == 3000 and g.symbol.nunique() == 50
    panel = pd.read_csv(OUT / 't1_screen_panel.csv')
    assert panel.symbol.nunique() == 40
    assert not set(panel.symbol) & set(SPEC['unseen_symbols'])


def test_b0_reproduces_the_frozen_reference_mean():
    d = json.loads((OUT / 'p1_decision.json').read_text(encoding='utf-8'))
    assert abs(d['B0_mean'] - d['B0_mean_reference']) < 1e-6


def test_neither_candidate_was_adopted():
    assert json.loads((OUT / 'p1_decision.json').read_text(encoding='utf-8'))['passed'] is False
    assert json.loads((OUT / 't1_decision.json').read_text(encoding='utf-8'))['passed'] is False


def test_expected_cost_decoder_is_sane():
    from scripts.oct12_cost_decode import decode, base_cost
    import numpy as np
    # A degenerate probability decodes to its own class; the flat row needs the tie-break,
    # because row 2 of the score weights is all zeros and every answer costs the same.
    assert list(decode(np.eye(5), np.zeros(5), 0.0)) == [0, 1, 2, 3, 4]
    # A larger tau buys more extreme answers, never fewer.
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(5), size=500)
    base = base_cost(np.array([0] * 7 + [1] * 24 + [2] * 34 + [3] * 27 + [4] * 8))
    extremes = [int(np.isin(decode(p, base, t), [0, 4]).sum()) for t in [0.0, 0.3, 0.55, 0.7]]
    assert extremes == sorted(extremes)


def test_cost_decode_reproduces_the_earlier_lightgbm_scores():
    import json
    d = json.loads((OUT / 'cost_decode_decision.json').read_text(encoding='utf-8'))
    assert abs(d['B0_mean'] - d['B0_mean_reference']) < 1e-6
    for fold, expected in d['lgb_aggr_reference'].items():
        got = [c for c in d['candidates'] if c['kind'] == 'lgb_aggr'][0]['fold_score'][fold]
        assert abs(got - expected) < 1e-6, (fold, got, expected)
    assert all(c['passed'] is False for c in d['candidates'])


def test_official_variant_reproduces_the_shipped_score():
    """The 'official' ruler must be the real one, or the whole comparison is off."""
    from scripts.oct12_metric_robustness import chance_corrected, weights
    from src import score
    import numpy as np
    pred = pd.read_parquet(OUT / 'p1_predictions.parquet')
    W = weights()['official']
    for (fold, kind), g in pred.groupby(['fold', 'kind']):
        assert abs(chance_corrected(g.label, g.pred, W) - score(g.label, g.pred)['score']) < 1e-12
    # Every ruler keeps 0 as chance: answering one class for everything scores exactly 0.
    y = pred[pred.kind == 'B0'].label.to_numpy(int)
    for name, w in weights().items():
        for k in range(5):
            v = chance_corrected(y, np.full(len(y), k), w)
            assert np.isnan(v) or abs(v) < 1e-12, (name, k, v)
