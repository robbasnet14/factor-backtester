"""Portfolio construction from factor scores."""
import pandas as pd


def decile_portfolios(scores: pd.DataFrame, n_deciles: int = 10, long_short: bool = True) -> pd.DataFrame:
    """Turn a wide (date x ticker) score panel into target portfolio weights.

    Each rebalance date, tickers are ranked into `n_deciles` equal-sized
    buckets by score. The top decile is held equal-weight long, summing to
    +1.0; if `long_short`, the bottom decile is held equal-weight short,
    summing to -1.0 (so the book is dollar-neutral with gross exposure 2.0 —
    the standard long-short decile-spread construction). If `long_short` is
    False, only the long leg is built.

    Dates with fewer than `n_deciles` non-NaN scores (not enough names to
    form deciles yet — e.g. before any factor has enough history) get an
    all-zero weight row rather than raising.
    """
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
