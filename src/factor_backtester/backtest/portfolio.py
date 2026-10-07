"""Portfolio construction from factor scores."""
import pandas as pd


def tradable_on_rebalance(daily_prices: pd.DataFrame, rebalance_dates: pd.Index) -> pd.DataFrame:
    """Which tickers could actually be traded on each rebalance date.

    `daily_prices` is a wide (date x ticker) panel of daily prices. A ticker
    is tradable for a month-end rebalance date if it has a price on that
    month's last trading day (the last date in `daily_prices` falling in the
    month) — that close is the entry price the backtest uses. A name that
    stopped trading earlier in the month, or never traded in it, is not
    tradable even if it still has a factor score: quality comes from
    forward-filled fundamentals and momentum from prices up to the prior
    month, so a delisted name can keep scoring after it's gone.

    Deliberately entry-only: requiring a price at the NEXT rebalance date
    too would drop exactly the names that go on to delist during the
    holding period, which isn't knowable at the rebalance date (look-ahead).

    Returns a boolean (rebalance date x ticker) frame; a rebalance date with
    no price data in its month is all-False.
    """
    months = daily_prices.index.to_period("M")
    # groupby().last() on a boolean frame is simply the value on each month's last trading day.
    priced_on_last_day = daily_prices.notna().groupby(months).last()
    rebalance_dates = pd.DatetimeIndex(rebalance_dates)
    tradable = priced_on_last_day.reindex(rebalance_dates.to_period("M"), fill_value=False)
    tradable.index = rebalance_dates
    return tradable.astype(bool)


def decile_portfolios(
    scores: pd.DataFrame,
    n_deciles: int = 10,
    long_short: bool = True,
    tradable: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Turn a wide (date x ticker) score panel into target portfolio weights.

    Each rebalance date, tickers are ranked into `n_deciles` equal-sized
    buckets by score. The top decile is held equal-weight long, summing to
    +1.0; if `long_short`, the bottom decile is held equal-weight short,
    summing to -1.0 (so the book is dollar-neutral with gross exposure 2.0 —
    the standard long-short decile-spread construction). If `long_short` is
    False, only the long leg is built.

    If `tradable` (a boolean date x ticker frame, e.g. from
    `tradable_on_rebalance`) is given, names that aren't tradable on a date
    are dropped BEFORE ranking, so decile boundaries are formed only from
    names that could actually be bought or sold then. Cells missing from
    `tradable` count as not tradable.

    Dates with fewer than `n_deciles` non-NaN scores (not enough names to
    form deciles yet — e.g. before any factor has enough history) get an
    all-zero weight row rather than raising.
    """
    if tradable is not None:
        tradable = tradable.reindex(index=scores.index, columns=scores.columns, fill_value=False).astype(bool)
        scores = scores.where(tradable)

    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)

    for dt, row in scores.iterrows():
        valid = row.dropna()
        if len(valid) < n_deciles:
            continue

        buckets = pd.qcut(valid.rank(method="first"), n_deciles, labels=False, duplicates="drop")
        top = valid.index[buckets == buckets.max()]
        weights.loc[dt, top] = 1.0 / len(top)

        if long_short:
            bottom = valid.index[buckets == buckets.min()]
            weights.loc[dt, bottom] = -1.0 / len(bottom)

    return weights
