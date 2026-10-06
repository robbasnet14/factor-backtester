"""In-memory providers implementing the data-source Protocols, for tests."""
import pandas as pd

from src.data.providers.base import empty_fundamentals, empty_prices


class FakePriceProvider:
    """Serves `prices` ({ticker: {date: price}}, used as both `adj_close` and
    `close`) with split events from `splits` ({ticker: {date: ratio}}),
    records every call, and raises `error` instead of answering if one is
    given (an exception for every ticker, or {ticker: exception} for some). A
    ticker it doesn't know is an empty answer, per the provider contract."""

    def __init__(
        self, name: str, prices: dict | None = None, error: Exception | dict | None = None, splits: dict | None = None
    ):
        self.name = name
        self.prices = prices or {}
        self.splits = splits or {}
        self.error = error
        self.calls: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []

    def fetch(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        self.calls.append((ticker, start, end))
        error = self.error.get(ticker.upper()) if isinstance(self.error, dict) else self.error
        if error is not None:
            raise error
        series = self.prices.get(ticker.upper())
        if not series:
            return empty_prices()
        values = [float(v) for v in series.values()]
        df = pd.DataFrame({"date": pd.to_datetime(list(series)), "adj_close": values, "close": values})
        split_dates = {pd.Timestamp(d): r for d, r in self.splits.get(ticker.upper(), {}).items()}
        df["split_ratio"] = df["date"].map(split_dates).fillna(1.0)
        return df[df["date"] >= start].reset_index(drop=True)


class FakeFundamentalsProvider:
    """Serves `rows` ({ticker: [{report_date, eps, book_value, roe,
    period_end (defaults to report_date), period (defaults to "quarter")},
    ...]}) and records every call; raises `error` instead if one is given
    (an exception for every ticker, or {ticker: exception} for some)."""

    def __init__(self, name: str = "fake fundamentals", rows: dict | None = None, error: Exception | dict | None = None):
        self.name = name
        self.rows = rows or {}
        self.error = error
        self.calls: list[str] = []

    def fetch(self, ticker: str) -> pd.DataFrame:
        self.calls.append(ticker)
        error = self.error.get(ticker.upper()) if isinstance(self.error, dict) else self.error
        if error is not None:
            raise error
        rows = self.rows.get(ticker.upper())
        if not rows:
            return empty_fundamentals()
        df = pd.DataFrame(rows)
        df["report_date"] = pd.to_datetime(df["report_date"])
        df["period_end"] = pd.to_datetime(df["period_end"]) if "period_end" in df else df["report_date"]
        if "period" not in df:
            df["period"] = "quarter"
        for optional in ("shares", "public_float"):
            if optional not in df:
                df[optional] = float("nan")
        return df[["report_date", "period_end", "period", "eps", "book_value", "roe", "shares", "public_float"]]


def flat_split_history(tickers: list[str], start: str = "2010-01-01", end: str = "2022-12-31") -> pd.DataFrame:
    """A `splits=` argument for `load_fundamentals` saying these tickers had
    no splits over [start, end]: every business day with split_ratio 1.0."""
    days = pd.bdate_range(start, end)
    return pd.DataFrame(
        [{"date": d, "ticker": t.upper(), "split_ratio": 1.0} for t in tickers for d in days]
    )
