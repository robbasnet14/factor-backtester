"""In-memory providers implementing the data-source Protocols, for tests."""
import pandas as pd

from src.data.providers.base import empty_fundamentals, empty_prices


class FakePriceProvider:
    """Serves `prices` ({ticker: {date: adj_close}}), records every call, and
    raises `error` instead of answering if one is given. A ticker it doesn't
    know is an empty answer, per the provider contract."""

    def __init__(self, name: str, prices: dict | None = None, error: Exception | None = None):
        self.name = name
        self.prices = prices or {}
        self.error = error
        self.calls: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []

    def fetch(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        self.calls.append((ticker, start, end))
        if self.error is not None:
            raise self.error
        series = self.prices.get(ticker.upper())
        if not series:
            return empty_prices()
        df = pd.DataFrame({"date": pd.to_datetime(list(series)), "adj_close": [float(v) for v in series.values()]})
        return df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)


class FakeFundamentalsProvider:
    """Serves `rows` ({ticker: [{report_date, earnings, book_value, roe}, ...]})
    and records every call; raises `error` instead if one is given."""

    def __init__(self, name: str = "fake fundamentals", rows: dict | None = None, error: Exception | None = None):
        self.name = name
        self.rows = rows or {}
        self.error = error
        self.calls: list[str] = []

    def fetch(self, ticker: str) -> pd.DataFrame:
        self.calls.append(ticker)
        if self.error is not None:
            raise self.error
        rows = self.rows.get(ticker.upper())
        if not rows:
            return empty_fundamentals()
        df = pd.DataFrame(rows)
        df["report_date"] = pd.to_datetime(df["report_date"])
        return df[["report_date", "earnings", "book_value", "roe"]]
