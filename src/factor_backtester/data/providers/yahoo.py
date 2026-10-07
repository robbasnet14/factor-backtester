"""Yahoo Finance prices via `yfinance` — free, no API key."""
import time

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFTickerMissingError

from factor_backtester.data.providers.base import empty_prices

_MAX_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 1.0
_POLITE_DELAY_SECONDS = 0.3


def to_yahoo_symbol(ticker: str) -> str:
    """Yahoo uses `-` for share classes where most other sources use `.` (BRK.B -> BRK-B)."""
    return ticker.upper().replace(".", "-")


class YahooProvider:
    """Daily prices from Yahoo Finance: `Adj Close` (split- and
    dividend-adjusted) as `adj_close`, `Close` (split-adjusted only) as
    `close`, and `Stock Splits` as `split_ratio`. Yahoo adjusts `Close` for
    every split up to today, so the download always runs through today —
    otherwise a split after `end` would be baked into `close` but missing
    from `split_ratio`. An unknown or delisted symbol comes back empty;
    errors are retried with backoff and re-raised if they persist (see
    `_history` for why that takes more than calling `yf.download`)."""

    name = "yfinance"

    def fetch(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        symbol = to_yahoo_symbol(ticker)
        first_day = start.date().isoformat()
        # Through today, not `end` (see the class docstring); yfinance's end is exclusive.
        after_last_day = (max(end, pd.Timestamp.today().normalize()) + pd.Timedelta(days=1)).date().isoformat()
        df = None
        for attempt in range(_MAX_RETRIES):
            try:
                df = _history(symbol, first_day, after_last_day)
                break
            except Exception as e:
                if _means_no_data(e):
                    break
                if attempt == _MAX_RETRIES - 1:
                    raise
                time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
        # Polite delay only after a real request, so cache hits aren't slowed down.
        time.sleep(_POLITE_DELAY_SECONDS)

        if df is None or df.empty:
            return empty_prices()

        out = df[["Adj Close", "Close", "Stock Splits"]].rename(
            columns={"Adj Close": "adj_close", "Close": "close", "Stock Splits": "split_ratio"}
        )
        out["split_ratio"] = out["split_ratio"].replace(0.0, 1.0)  # Yahoo uses 0 for "no split"
        out.index.name = "date"
        out = out.reset_index()
        out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
        return out[["date", "adj_close", "close", "split_ratio"]]


def _history(symbol: str, first_day: str, after_last_day: str) -> pd.DataFrame:
    """One symbol's daily history, with yfinance's errors raised.

    By default yfinance catches every error per symbol (a rate limit, a
    dropped connection), logs it and returns an empty frame, which looks
    exactly like "Yahoo has no such ticker": `yf.download` does this
    regardless of settings. Asked this way it raises instead, so a failure
    can be told apart from no data (`_means_no_data`)."""
    hide = yf.config.debug.hide_exceptions
    yf.config.debug.hide_exceptions = False
    try:
        return yf.Ticker(symbol).history(start=first_day, end=after_last_day, auto_adjust=False, actions=True)
    finally:
        yf.config.debug.hide_exceptions = hide


def _means_no_data(error: Exception) -> bool:
    """Yahoo's ways of saying it has nothing: a symbol it marks "possibly
    delisted" (no timezone, or no prices in the window), or one it has
    never heard of (HTTP 404)."""
    if isinstance(error, YFTickerMissingError):
        return True
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None) == 404
