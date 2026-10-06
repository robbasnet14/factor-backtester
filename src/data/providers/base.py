"""Data-source interfaces.

A provider is anything that can answer "what's the data for this ticker?" —
Yahoo, Tiingo and SEC EDGAR ship with the engine, and `load_prices` /
`load_fundamentals` accept any other object with the same shape (a paid
vendor, a local CSV dump, an internal feed).

Contract, the same for every provider:

- **No data is a normal answer.** If the source simply doesn't have the
  ticker (unknown, delisted, never filed), return an empty DataFrame with the
  documented columns. Don't raise.
- **Raise only when the source couldn't answer**: a network or transport
  error, rate limiting that outlasted retries, missing credentials.

The loader relies on that split: an empty answer from every provider means a
ticker is genuinely unavailable, while an exception means "unknown, ask again
next run". A provider that turns a failure into an empty result (or the
reverse) breaks that distinction.

Prices carry two adjusted series with different jobs. `adj_close` is split-
and dividend-adjusted, for total returns (momentum, forward returns). `close`
is split-adjusted only, for anything compared with per-share accounting
figures (earnings yield, market cap). A split adjustment is a change of units
fixed by the splits the source knew about when it answered, so a price
provider must return every split its `close` is adjusted for, as
`split_ratio` on the split date (4.0 for a 4-for-1), even ones after `end`,
so per-share figures can be restated onto exactly the same basis.
"""
from typing import Protocol

import pandas as pd

PRICE_COLUMNS = ["date", "adj_close", "close", "split_ratio"]
FUNDAMENTALS_COLUMNS = ["report_date", "period_end", "period", "eps", "book_value", "roe", "shares"]


def empty_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "adj_close": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "split_ratio": pd.Series(dtype="float64"),
        }
    )


def empty_fundamentals() -> pd.DataFrame:
    return pd.DataFrame(columns=FUNDAMENTALS_COLUMNS)


class PriceProvider(Protocol):
    name: str  # shown in logs and used to namespace the cache for a custom chain

    def fetch(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Daily prices for `ticker` from `start` through at least `end`:
        columns [date, adj_close, close, split_ratio], `date` tz-naive.
        `split_ratio` is 1.0 except on split dates. Rows after `end` may be
        included, and must be if `close` is adjusted for splits after `end`.
        Empty if the source has no data for the ticker."""
        ...


class FundamentalsProvider(Protocol):
    name: str

    def fetch(self, ticker: str) -> pd.DataFrame:
        """The ticker's full history of EPS facts, one row per fiscal period:
        columns [report_date, period_end, period, eps, book_value, roe,
        shares], with `period` "quarter" or "year". `report_date` is when the
        figures first became public (no look-ahead); `eps` and `shares`
        (shares outstanding, NaN if not reported) are exactly as filed, on
        the share basis in effect at `report_date`. The loader
        restates EPS across later splits, derives any quarter reported only
        inside a fiscal-year total, and builds the trailing twelve months;
        those steps have to happen after restatement, so a provider mustn't
        do them itself. Empty if the source has no data."""
        ...
