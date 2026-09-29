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
"""
from typing import Protocol

import pandas as pd

PRICE_COLUMNS = ["date", "adj_close"]
FUNDAMENTALS_COLUMNS = ["report_date", "earnings", "book_value", "roe"]


def empty_prices() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "adj_close": pd.Series(dtype="float64")})


def empty_fundamentals() -> pd.DataFrame:
    return pd.DataFrame(columns=FUNDAMENTALS_COLUMNS)


class PriceProvider(Protocol):
    name: str  # shown in logs and used to namespace the cache for a custom chain

    def fetch(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Daily split/dividend-adjusted closes for `ticker` between `start`
        and `end` inclusive: columns [date, adj_close], `date` tz-naive.
        Empty if the source has no data for the ticker."""
        ...


class FundamentalsProvider(Protocol):
    name: str

    def fetch(self, ticker: str) -> pd.DataFrame:
        """The ticker's full fundamentals history: columns [report_date,
        earnings, book_value, roe], where `report_date` is when each figure
        became public (no look-ahead). Empty if the source has no data."""
        ...
