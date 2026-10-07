"""Every shipped provider keeps the contract in `providers.base`: when the
transport fails, `fetch` raises; it never answers with an empty frame,
which would mean "no data" and get the ticker skipped, or skiplisted for
good.

The transport is failed underneath each provider's own code (and, for
Yahoo, underneath yfinance's), so a change in how a dependency handles
errors breaks these tests instead of quietly reintroducing the bug.
yfinance did exactly that: `yf.download` turns every per-ticker error into
an empty frame.
"""
import json
import time

import pandas as pd
import pytest
import requests
from curl_cffi import requests as curl_requests

from src.data.providers.sec_edgar import SecEdgarProvider
from src.data.providers.tiingo import TiingoProvider
from src.data.providers.yahoo import YahooProvider

START, END = pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-06")


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)


class _Response:
    def __init__(self, status_code):
        self.status_code = status_code

    def raise_for_status(self):
        raise requests.HTTPError(f"{self.status_code} error")


def _transport_down(*args, **kwargs):
    raise ConnectionError("simulated: connection refused")


def test_yahoo_raises_when_the_transport_fails(monkeypatch):
    # yfinance talks to Yahoo through a curl_cffi session.
    monkeypatch.setattr(curl_requests.Session, "request", _transport_down)
    monkeypatch.setattr(curl_requests.Session, "get", _transport_down)

    with pytest.raises(Exception, match="connection refused"):
        YahooProvider().fetch("AAPL", START, END)


@pytest.mark.parametrize("failure", [_transport_down, lambda *a, **k: _Response(503)])
def test_tiingo_raises_when_the_transport_fails(monkeypatch, failure):
    monkeypatch.setenv("TIINGO_KEY", "dummy")
    monkeypatch.setattr(requests, "get", failure)

    with pytest.raises(Exception, match="connection refused|503"):
        TiingoProvider().fetch("AAPL", START, END)


@pytest.mark.parametrize("failure", [_transport_down, lambda *a, **k: _Response(503)])
def test_sec_edgar_raises_when_the_transport_fails(monkeypatch, tmp_path, failure):
    cik_map = tmp_path / "sec_company_tickers.json"
    cik_map.write_text(json.dumps({"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}))
    monkeypatch.setattr(requests, "get", failure)

    with pytest.raises(Exception, match="connection refused|503"):
        SecEdgarProvider(cik_map).fetch("AAPL")


def test_sec_edgar_raises_when_it_cannot_fetch_its_ticker_map(monkeypatch, tmp_path):
    # Without the map every ticker would look like "no CIK", an empty answer.
    monkeypatch.setattr(requests, "get", _transport_down)

    with pytest.raises(Exception, match="connection refused"):
        SecEdgarProvider(tmp_path / "sec_company_tickers.json").fetch("AAPL")
