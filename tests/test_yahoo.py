"""Yahoo's failures have to reach the loader as exceptions: yfinance's
default is to log them and return an empty frame, which reads as "no data"
and gets a throttled ticker marked unavailable for good."""
import time

import pandas as pd
import pytest
import yfinance as yf
from yfinance.exceptions import YFPricesMissingError, YFTzMissingError

from src.data.providers import yahoo
from src.data.providers.yahoo import YahooProvider


def _no_waiting(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)


def test_yahoo_asks_yfinance_to_raise_and_restores_its_setting(monkeypatch):
    _no_waiting(monkeypatch)
    seen = []

    class FakeTicker:
        def __init__(self, symbol):
            pass

        def history(self, **kwargs):
            seen.append(yf.config.debug.hide_exceptions)
            raise ConnectionError("dropped")

    monkeypatch.setattr(yf, "Ticker", FakeTicker)
    before = yf.config.debug.hide_exceptions

    with pytest.raises(ConnectionError):
        YahooProvider().fetch("AAPL", pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-06"))

    assert seen == [False] * 3  # raised on every attempt, so retried and then re-raised
    assert yf.config.debug.hide_exceptions == before


class _NotFound(Exception):
    """The shape of yfinance's HTTP error for a symbol Yahoo has never heard of."""

    class response:
        status_code = 404


@pytest.mark.parametrize("no_data", [YFTzMissingError("TWTR"), YFPricesMissingError("ABNB", ""), _NotFound()])
def test_yahoo_treats_its_no_data_answers_as_empty_without_retrying(monkeypatch, no_data):
    _no_waiting(monkeypatch)
    attempts = []

    def history(symbol, *dates):
        attempts.append(symbol)
        raise no_data

    monkeypatch.setattr(yahoo, "_history", history)

    df = YahooProvider().fetch("TWTR", pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-06"))

    assert df.empty and len(attempts) == 1
