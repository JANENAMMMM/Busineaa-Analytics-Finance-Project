# Exploratory: fixed P1 direction overlay on B0 + premarket

This is a descriptive follow-up, not a new selected submission candidate. It does not read the holdout.

The already frozen P1 fold decisions and predictions from `reports/followup_checks/p1_predictions.parquet` were applied only where that fixed P1 prediction differed from its paired B0 prediction. Otherwise the B0 + premarket prediction was retained. No P1 threshold, feature, inner split, or PM-specific on/off rule was retuned.

| Fold | B0 + premarket | Fixed P1 overlay | Difference | Changed rows |
|---|---:|---:|---:|---:|
| fold1 | 0.469611 | 0.469369 | -0.000242 | 1 |
| fold2 | 0.424387 | 0.424387 | +0.000000 | 0 |
| fold3 | 0.411349 | 0.419971 | +0.008622 | 46 |
| fold4 | 0.549233 | 0.549408 | +0.000175 | 7 |
| mean | 0.463645 | 0.465784 | +0.002138 | 54 |

Paired 5-trading-day block bootstrap (2,000 draws, seed 2026) versus B0 + premarket is [-0.000141, +0.006736], with 18.85% nonpositive draws. The gain is driven primarily by fold3 and is not stable enough to describe as a reliable improvement.

The submission feasibility constraint is also unchanged: premarket artifacts stop at 2026-09-11 and are git-ignored, so this development-only overlay cannot be activated for an unknown later evaluation date without a permitted, reproducible data delivery route. The submitted B0 remains unchanged.
