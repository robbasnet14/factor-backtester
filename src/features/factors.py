"""Cross-sectional factor computation.

All factors are returned as wide monthly panels: index=month-end date,
columns=ticker. Factors can have different date ranges and NaN patterns
(momentum needs `lookback_m` months of price history before it produces its
first value; a ticker with no fundamentals yet is NaN until its first
report) — `transforms.combine_factors` is responsible for aligning and
averaging across those gaps, not this module.
"""
import numpy as np
import pandas as pd


def momentum(prices: pd.DataFrame, lookback_m: int = 12, skip_m: int = 1) -> pd.DataFrame:
    """12-1 style momentum: cumulative return from t-lookback_m to t-skip_m.

    `prices` is the long DataFrame produced by `load_prices`
    (columns: date, ticker, adj_close). Monthly price is the last available
    adjusted close in each calendar month. The most recent `skip_m` month(s)
    are excluded from the lookback window to avoid the well-documented
    short-term reversal effect that contaminates naive 12-0 momentum.
    """
    monthly = _pivot_prices_wide(prices).resample("ME").last()
    return monthly.shift(skip_m) / monthly.shift(lookback_m) - 1


def value(fundamentals: pd.DataFrame, prices: pd.DataFrame, metric: str = "earnings_yield") -> pd.DataFrame:
    """Value factor. Currently only `metric="earnings_yield"` (EPS / price).

    `earnings_yield` needs a market price, so unlike the Step-2 stub this
    takes `prices` (long, from `load_prices`) alongside `fundamentals` (long,
    from `load_fundamentals`). `earnings` is the latest reported quarterly
    EPS as of each month, not trailing-twelve-month EPS — a simplification
    worth revisiting if the factor looks noisy in later steps.
    """
    if metric != "earnings_yield":
        raise NotImplementedError(f"value: metric {metric!r} is not implemented (only 'earnings_yield' is)")

    monthly_price = _pivot_prices_wide(prices).resample("ME").last()
    monthly_eps = _fundamentals_metric_to_monthly(fundamentals, "earnings", monthly_price.index)
    monthly_eps, monthly_price = monthly_eps.align(monthly_price, join="outer")

    yield_ = monthly_eps / monthly_price
    return yield_.replace([np.inf, -np.inf], np.nan)


def quality(fundamentals: pd.DataFrame, metric: str = "roe") -> pd.DataFrame:
    """Quality factor. Currently only `metric="roe"`.

    `fundamentals` is the long DataFrame from `load_fundamentals`; ROE is
    carried forward from each ticker's last reported (lagged) value until
    the next report, on a month-end calendar spanning the input data.
    """
    if metric != "roe":
        raise NotImplementedError(f"quality: metric {metric!r} is not implemented (only 'roe' is)")

    monthly_index = pd.date_range(fundamentals["date"].min(), fundamentals["date"].max(), freq="ME")
    return _fundamentals_metric_to_monthly(fundamentals, "roe", monthly_index)


def _pivot_prices_wide(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()


def _fundamentals_metric_to_monthly(fundamentals: pd.DataFrame, column: str, monthly_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Point-in-time carry-forward of a fundamentals column onto `monthly_index`.

    Fundamentals arrive on irregular (already look-ahead-lagged) dates.
    Forward-filling across the union of those dates and the target monthly
    calendar means every month picks up the most recent value known as of
    that month, and never a value from the future.
    """
    pivoted = fundamentals.pivot_table(index="date", columns="ticker", values=column, aggfunc="last")
    combined_index = pivoted.index.union(monthly_index).sort_values()
    filled = pivoted.reindex(combined_index).ffill()
    return filled.reindex(monthly_index)
