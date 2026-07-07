"""Walk-forward, out-of-sample validation.

Momentum/value/quality are computed point-in-time already (Steps 2-3), so
there's no model being *fit* to an in-sample window here — no lookback,
weighting, or decile choice is estimated from data the way a regression
coefficient would be. What a naive single backtest can still get wrong is
subtler: factor lookback windows (12 months of trailing prices) and lagged
fundamentals settle in near a train/test boundary, so a rebalance decision
made right at that boundary can lean on information that was still
"in-flight." Walk-forward validation here exists to (a) prove performance
holds up across multiple independent expanding folds rather than one lucky
full-sample run, and (b) give a clean seam — the embargo gap — to later plug
in genuine in-sample parameter search (factor weights, lookbacks, deciles)
without leaking into the block being scored.

Because nothing is fit to labels in this pipeline, "purge" and "embargo"
collapse into a single gap between the end of each expanding in-sample
window and the start of the next out-of-sample test block: no rebalance is
scored or traded during that gap at all.
"""
import pandas as pd

from src.backtest.engine import run_backtest


def make_walk_forward_folds(
    start: str,
    end: str,
    initial_train_months: int = 24,
    test_months: int = 12,
    embargo_months: int = 1,
) -> list[dict]:
    """Build expanding-window walk-forward fold boundaries.

    Fold 0 trains on [start, start + initial_train_months), embargoes the
    next `embargo_months`, then tests on the following `test_months`. Each
    subsequent fold's training window expands to absorb the prior fold's
    test block (that period is now historical, known data), then applies a
    fresh embargo before its own test block. Stops once there's no room left
    for another test block before `end`.
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    folds = []
    train_end = start_ts + pd.DateOffset(months=initial_train_months)

    while train_end + pd.DateOffset(months=embargo_months) < end_ts:
        test_start = train_end + pd.DateOffset(months=embargo_months)
        test_end = min(test_start + pd.DateOffset(months=test_months), end_ts)
        folds.append({"train_start": start_ts, "train_end": train_end, "test_start": test_start, "test_end": test_end})
        train_end = test_end

    return folds


def walk_forward_backtest(
    weights: pd.DataFrame,
    forward_returns: pd.DataFrame,
    cost_bps: float,
    start: str,
    end: str,
    initial_train_months: int = 24,
    test_months: int = 12,
    embargo_months: int = 1,
    prices: pd.DataFrame | None = None,
) -> tuple[pd.Series, list[dict]]:
    """Run the backtest engine independently on each fold's out-of-sample
    block only, then stitch the results into one continuous OOS series.

    `weights` and `forward_returns` should already be computed over the full
    period (they're point-in-time correct by construction — Steps 2-3) —
    this function's job is purely to decide, per fold, which dates are
    eligible to be scored as out-of-sample, and to keep folds from leaking
    into each other. `prices`, if given, is passed straight through to
    `run_backtest` so delisted names get exited at their last available
    price in every fold, same as the full-period run — see
    `engine.run_backtest`.

    Each fold's test block is costed as if establishing positions from cash
    (the embargo month means no position was carried into it) — slightly
    conservative on turnover at each fold seam, but keeps folds independent
    and avoids the skipped embargo rebalance smearing cost accounting across
    the gap.

    Returns (stitched out-of-sample return series, the fold boundaries used)
    so the caller can report exactly which windows produced the headline
    number.
    """
    folds = make_walk_forward_folds(start, end, initial_train_months, test_months, embargo_months)
    if not folds:
        raise ValueError(
            "walk_forward_backtest: no folds fit in the given range with "
            f"initial_train_months={initial_train_months}, test_months={test_months}, "
            f"embargo_months={embargo_months} — shorten the training window or the range is too short"
        )

    oos_segments = []
    for fold in folds:
        test_dates = weights.index[(weights.index >= fold["test_start"]) & (weights.index < fold["test_end"])]
        if test_dates.empty:
            continue
        fold_prices = prices.reindex(test_dates) if prices is not None else None
        oos_segments.append(run_backtest(weights.loc[test_dates], forward_returns.reindex(test_dates), cost_bps, prices=fold_prices))

    if not oos_segments:
        raise ValueError("walk_forward_backtest: folds were computed but none contained any rebalance dates")

    stitched = pd.concat(oos_segments).sort_index()
    if stitched.index.duplicated().any():
        raise AssertionError("walk_forward_backtest: fold test windows overlapped — this should be impossible")

    return stitched, folds
