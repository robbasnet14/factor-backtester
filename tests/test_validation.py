"""Sanity checks for Step 7: walk-forward folds + purge/embargo stitching."""
import numpy as np
import pandas as pd
import pytest

from src.backtest.validation import make_walk_forward_folds, walk_forward_backtest


def test_folds_expand_and_embargo_gap_is_excluded():
    folds = make_walk_forward_folds(
        "2010-01-01", "2014-01-01", initial_train_months=24, test_months=12, embargo_months=1
    )

    assert len(folds) >= 2
    for fold in folds:
        assert fold["train_end"] < fold["test_start"]
        assert fold["test_start"] < fold["test_end"]
        # the embargo gap itself must be excluded from both train and test
        gap_days = (fold["test_start"] - fold["train_end"]).days
        assert gap_days >= 28  # ~1 month

    # expanding window: each fold's training window absorbs the prior fold's test block
    for prev, cur in zip(folds, folds[1:]):
        assert cur["train_end"] == prev["test_end"]

    # consecutive test blocks never overlap
    for prev, cur in zip(folds, folds[1:]):
        assert cur["test_start"] > prev["test_end"] or cur["test_start"] == prev["test_end"]


def test_make_folds_returns_empty_when_range_too_short():
    folds = make_walk_forward_folds("2010-01-01", "2011-06-01", initial_train_months=24, test_months=12)
    assert folds == []


def test_walk_forward_backtest_raises_when_no_folds_fit():
    dates = pd.date_range("2010-01-31", periods=6, freq="ME")
    weights = pd.DataFrame({"A": [1.0] * 6}, index=dates)
    forward_returns = pd.DataFrame({"A": [0.01] * 6}, index=dates)

    with pytest.raises(ValueError):
        walk_forward_backtest(weights, forward_returns, cost_bps=0, start="2010-01-01", end="2010-06-30")


def test_stitched_series_excludes_embargo_dates_and_has_no_duplicates():
    dates = pd.date_range("2010-01-31", periods=48, freq="ME")
    weights = pd.DataFrame({"A": [1.0] * 48}, index=dates)
    forward_returns = pd.DataFrame({"A": [0.01] * 48}, index=dates)

    stitched, folds = walk_forward_backtest(
        weights, forward_returns, cost_bps=0,
        start="2010-01-01", end="2014-01-01",
        initial_train_months=24, test_months=12, embargo_months=1,
    )

    assert not stitched.index.duplicated().any()

    # every date in the stitched series must fall inside some fold's [test_start, test_end)
    for dt in stitched.index:
        assert any(f["test_start"] <= dt < f["test_end"] for f in folds)

    # no date inside a fold's embargo gap (train_end, test_start) should appear
    for f in folds:
        embargo_dates = [dt for dt in dates if f["train_end"] < dt < f["test_start"]]
        assert not any(dt in stitched.index for dt in embargo_dates)


def test_first_period_of_each_fold_costed_from_flat():
    # A big weight swing right before a fold boundary must not show up as
    # turnover cost inside the next fold — each fold starts "from cash."
    dates = pd.date_range("2010-01-31", periods=48, freq="ME")
    weights = pd.DataFrame({"A": [1.0] * 48, "B": [-1.0] * 48}, index=dates)
    forward_returns = pd.DataFrame({"A": [0.0] * 48, "B": [0.0] * 48}, index=dates)

    stitched, folds = walk_forward_backtest(
        weights, forward_returns, cost_bps=100,
        start="2010-01-01", end="2014-01-01",
        initial_train_months=24, test_months=12, embargo_months=1,
    )
    first_test_date = folds[1]["test_start"]
    # weights are unchanged across the whole series, so if the fold were NOT
    # treated as starting from flat, turnover (and thus cost) at the first
    # date of fold 1's test block would be zero, not a full-book cost.
    first_actual_date = stitched.index[stitched.index >= first_test_date][0]
    assert stitched.loc[first_actual_date] == pytest.approx(-0.5 * 2.0 * 100 / 1e4)


def test_stitched_metrics_computed_only_on_oos_series():
    from src.analytics.metrics import sharpe

    dates = pd.date_range("2010-01-31", periods=48, freq="ME")
    rng = np.random.default_rng(1)
    weights = pd.DataFrame({"A": [1.0] * 48}, index=dates)
    forward_returns = pd.DataFrame({"A": rng.normal(0.01, 0.02, 48)}, index=dates)

    stitched, _ = walk_forward_backtest(
        weights, forward_returns, cost_bps=0,
        start="2010-01-01", end="2014-01-01",
        initial_train_months=24, test_months=12, embargo_months=1,
    )
    full_sample_sharpe = sharpe(forward_returns["A"], periods_per_year=12)
    oos_sharpe = sharpe(stitched, periods_per_year=12)

    # Not a numeric equivalence claim — just confirms the OOS series is a
    # strict subset of the full series, so its Sharpe is computed on fewer,
    # later observations rather than silently falling back to the full sample.
    assert len(stitched) < len(forward_returns)
    assert oos_sharpe != full_sample_sharpe
