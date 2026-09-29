"""Yahoo Finance prices via `yfinance` — free, no API key."""
import time

import pandas as pd
import yfinance as yf

from src.data.providers.base import empty_prices

_MAX_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 1.0
_POLITE_DELAY_SECONDS = 0.3


def to_yahoo_symbol(ticker: str) -> str:
    """Yahoo uses `-` for share classes where most other sources use `.` (BRK.B -> BRK-B)."""
    return ticker.upper().replace(".", "-")


class YahooProvider:
    """Adjusted closes from Yahoo Finance (`auto_adjust=True`, so `Close` is
    already split/dividend-adjusted). An unknown or delisted symbol comes back
    empty; errors are retried with backoff and re-raised if they persist."""

    name = "yfinance"

    def fetch(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        symbol = to_yahoo_symbol(ticker)
        df = None
        for attempt in range(_MAX_RETRIES):
            try:
                df = yf.download(
                    symbol,
                    start=start.date().isoformat(),
                    end=(end + pd.Timedelta(days=1)).date().isoformat(),  # yfinance's end is exclusive
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
                break
            except Exception:
                if attempt == _MAX_RETRIES - 1:
                    raise
                time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
        # Polite delay only after a real request, so cache hits aren't slowed down.
        time.sleep(_POLITE_DELAY_SECONDS)

        if df is None or df.empty:
            return empty_prices()

        if isinstance(df.columns, pd.MultiIndex):  # yfinance always uses (field, ticker) columns
            df.columns = df.columns.get_level_values(0)

        out = df[["Close"]].rename(columns={"Close": "adj_close"})
        out.index.name = "date"
        out = out.reset_index()
        out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
        return out[["date", "adj_close"]]
