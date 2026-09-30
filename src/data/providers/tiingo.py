"""Tiingo daily prices — much better coverage of long-delisted tickers than
Yahoo, but needs an API key (`TIINGO_KEY` env var)."""
import logging
import os
import time

import pandas as pd
import requests

from src.data.providers.base import empty_prices

_logger = logging.getLogger(__name__)

_BASE_URL = "https://api.tiingo.com"
_REQUEST_TIMEOUT = 30
_MAX_RETRIES = 5
_RETRY_BASE_SECONDS = 60  # 429 backoff: 60s, 120s, 240s, ...
_POLITE_DELAY_SECONDS = 0.5


class TiingoProvider:
    """Daily prices from Tiingo's Daily Prices API. Tiingo returns the raw,
    as-traded `close` plus a `splitFactor` on each split date, so `close`
    here is split-adjusted using exactly the splits in the same response (to
    the basis at the last returned date). A 404 (Tiingo has no such ticker)
    is an empty answer; a missing key, exhausted 429 retries, or any other
    HTTP error raises."""

    name = "Tiingo"

    def fetch(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        url = f"{_BASE_URL}/tiingo/daily/{ticker}/prices"
        params = {"startDate": start.date().isoformat(), "endDate": end.date().isoformat(), "format": "json"}
        rows = _get(url, params)
        if not rows:
            return empty_prices()
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
        df = df.sort_values("date").reset_index(drop=True)
        df = df.rename(columns={"adjClose": "adj_close", "splitFactor": "split_ratio"})
        # Undo every split that happens after each date: raw close / product of later split ratios.
        later_splits = df["split_ratio"][::-1].cumprod()[::-1].shift(-1, fill_value=1.0)
        df["close"] = df["close"] / later_splits
        return df[["date", "adj_close", "close", "split_ratio"]]


def _headers() -> dict:
    key = os.environ.get("TIINGO_KEY")
    if not key:
        raise EnvironmentError(
            "TIINGO_KEY is not set — required for the Tiingo fallback used when yfinance has no "
            "data for a ticker. Sign up at https://www.tiingo.com and run: export TIINGO_KEY=your_key_here"
        )
    return {"Content-Type": "application/json", "Authorization": f"Token {key}"}


def _get(url: str, params: dict) -> list:
    """GET a Tiingo URL. 404 (no data for this ticker) returns `[]` — a real,
    non-retried "nothing here." 429 (rate limited) waits with exponential
    backoff (60s, 120s, 240s, ...) and retries; any other error raises.
    """
    headers = _headers()
    last_error = None
    for attempt in range(_MAX_RETRIES):
        resp = requests.get(url, headers=headers, params=params, timeout=_REQUEST_TIMEOUT)
        if resp.status_code == 404:
            return []
        if resp.status_code == 429:
            last_error = requests.HTTPError(f"429 rate limited fetching {url}")
            if attempt < _MAX_RETRIES - 1:
                backoff = _RETRY_BASE_SECONDS * (2**attempt)
                _logger.info("Tiingo rate limit hit; waiting %ds before retry %d/%d", backoff, attempt + 2, _MAX_RETRIES)
                time.sleep(backoff)
                continue
            raise last_error
        resp.raise_for_status()  # other 4xx/5xx fail immediately, no retry
        time.sleep(_POLITE_DELAY_SECONDS)
        return resp.json()
    raise last_error  # pragma: no cover — loop always returns or raises above
