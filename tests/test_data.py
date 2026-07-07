"""Sanity checks for the data layer: Yahoo Finance (with a Tiingo fallback)
prices, SEC EDGAR fundamentals, and the point-in-time universe.

Network calls are mocked so these run offline, deterministically, and
without hitting Yahoo Finance, Tiingo, or SEC EDGAR — except for one
explicitly-marked integration check against real SEC EDGAR data for a known
filer, which skips itself if the network isn't reachable rather than
failing. Every test that can reach the Tiingo fallback path explicitly
unsets TIINGO_KEY (or sets a dummy one) so behavior never depends on
whatever happens to be in the host environment.
"""
import json
import logging
import warnings

import pandas as pd
import pytest
import requests

from src.data import loader as loader_module
from src.data.loader import load_fundamentals, load_prices
from src.data.universe import build_universe


class _FakeResponse:
    def __init__(self, json_data=None, content=b"", status_code=200):
        self._json_data = json_data
        self.content = content
        self.status_code = status_code

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


def _fake_yf_frame(dates: list[str], closes: list[float], symbol: str) -> pd.DataFrame:
    index = pd.DatetimeIndex(dates, name="Date")
    columns = pd.MultiIndex.from_tuples([("Close", symbol)], names=["Price", "Ticker"])
    return pd.DataFrame([[c] for c in closes], index=index, columns=columns)


def test_load_prices_fetches_caches_and_slices(monkeypatch, tmp_path):
    calls = []

    def fake_download(symbol, start=None, end=None, auto_adjust=None, progress=None, threads=None):
        calls.append(symbol)
        return _fake_yf_frame(["2020-01-02", "2020-01-03", "2020-01-06"], [10.0, 10.5, 11.0], symbol)

    monkeypatch.setattr(loader_module.yf, "download", fake_download)
    monkeypatch.setattr(loader_module.time, "sleep", lambda s: None)

    df = load_prices(["aapl"], "2020-01-02", "2020-01-06", cache_dir=str(tmp_path))

    assert list(df.columns) == ["date", "ticker", "adj_close"]
    assert (df["ticker"] == "AAPL").all()
    assert len(df) == 3
    assert (tmp_path / "prices" / "AAPL.parquet").exists()

    # Second call for the same (already cached) range must not hit the network again.
    load_prices(["aapl"], "2020-01-02", "2020-01-06", cache_dir=str(tmp_path))
    assert len(calls) == 1


def test_load_prices_maps_dot_ticker_to_yahoo_dash_symbol(monkeypatch, tmp_path):
    seen_symbols = []

    def fake_download(symbol, **kwargs):
        seen_symbols.append(symbol)
        return _fake_yf_frame(["2020-01-02"], [10.0], symbol)

    monkeypatch.setattr(loader_module.yf, "download", fake_download)
    monkeypatch.setattr(loader_module.time, "sleep", lambda s: None)

    df = load_prices(["BRK.B"], "2020-01-02", "2020-01-06", cache_dir=str(tmp_path))

    assert seen_symbols == ["BRK-B"]         # Yahoo's symbol was used for the actual call
    assert set(df["ticker"]) == {"BRK.B"}    # but the original spelling comes back to the caller


def test_load_prices_skips_ticker_with_no_data(monkeypatch, tmp_path):
    # BADTICKER has nothing on yfinance NOR Tiingo (no key -> fallback fails fast)
    # so this must resolve as "truly unavailable" without ever touching a real network.
    monkeypatch.delenv("TIINGO_KEY", raising=False)

    def fake_download(symbol, **kwargs):
        if symbol == "BADTICKER":
            return pd.DataFrame()  # yfinance's response for an unknown/delisted symbol
        return _fake_yf_frame(["2020-01-02"], [10.0], symbol)

    monkeypatch.setattr(loader_module.yf, "download", fake_download)
    monkeypatch.setattr(loader_module.time, "sleep", lambda s: None)

    with pytest.warns(UserWarning):
        df = load_prices(["aapl", "badticker"], "2020-01-02", "2020-01-06", cache_dir=str(tmp_path))

    assert set(df["ticker"]) == {"AAPL"}


def test_load_prices_retries_transient_network_error(monkeypatch, tmp_path):
    attempts = {"n": 0}

    def flaky_download(symbol, **kwargs):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("simulated transient network error")
        return _fake_yf_frame(["2020-01-02"], [10.0], symbol)

    monkeypatch.setattr(loader_module.yf, "download", flaky_download)
    monkeypatch.setattr(loader_module.time, "sleep", lambda s: None)

    df = load_prices(["aapl"], "2020-01-02", "2020-01-06", cache_dir=str(tmp_path))

    assert attempts["n"] == 3
    assert len(df) == 1


def test_load_prices_gives_up_after_max_retries(monkeypatch, tmp_path):
    # yfinance never succeeds here, so this also exercises (and must not
    # silently succeed via) the Tiingo fallback path with no key configured.
    monkeypatch.delenv("TIINGO_KEY", raising=False)

    def always_fails(symbol, **kwargs):
        raise ConnectionError("simulated persistent network error")

    monkeypatch.setattr(loader_module.yf, "download", always_fails)
    monkeypatch.setattr(loader_module.time, "sleep", lambda s: None)

    with pytest.warns(UserWarning):
        df = load_prices(["aapl"], "2020-01-02", "2020-01-06", cache_dir=str(tmp_path))

    assert df.empty


def test_load_prices_skiplists_unavailable_ticker_and_skips_network_on_rerun(monkeypatch, tmp_path):
    monkeypatch.delenv("TIINGO_KEY", raising=False)
    calls = {"n": 0}

    def always_fails(symbol, **kwargs):
        calls["n"] += 1
        raise ConnectionError("simulated persistent network error")

    monkeypatch.setattr(loader_module.yf, "download", always_fails)
    monkeypatch.setattr(loader_module.time, "sleep", lambda s: None)

    with pytest.warns(UserWarning):
        df = load_prices(["aapl"], "2020-01-02", "2020-01-06", cache_dir=str(tmp_path))
    assert df.empty
    calls_after_first = calls["n"]
    assert calls_after_first > 0

    skiplist_path = tmp_path / "unavailable_prices.json"
    assert skiplist_path.exists()
    assert "AAPL" in json.loads(skiplist_path.read_text())

    # A second call must skip AAPL with no network call and no warning at all.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df2 = load_prices(["aapl"], "2020-01-02", "2020-01-06", cache_dir=str(tmp_path))
    assert calls["n"] == calls_after_first  # no additional yfinance calls
    assert df2.empty

    # force_refresh=True bypasses the skiplist and re-attempts the network call.
    with pytest.warns(UserWarning):
        load_prices(["aapl"], "2020-01-02", "2020-01-06", cache_dir=str(tmp_path), force_refresh=True)
    assert calls["n"] > calls_after_first


def test_load_prices_falls_back_to_tiingo_when_yfinance_empty(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("TIINGO_KEY", "dummy")
    monkeypatch.setattr(loader_module.yf, "download", lambda symbol, **kwargs: pd.DataFrame())  # yfinance: nothing

    tiingo_calls = []

    def fake_tiingo_get(url, headers=None, params=None, timeout=None):
        tiingo_calls.append(url)
        rows = [{"date": "2020-01-02T00:00:00.000Z", "close": 10.0, "adjClose": 42.0}]
        return _FakeResponse(json_data=rows)

    monkeypatch.setattr(requests, "get", fake_tiingo_get)
    monkeypatch.setattr(loader_module.time, "sleep", lambda s: None)

    caplog.set_level(logging.INFO)
    df = load_prices(["delistedco"], "2020-01-01", "2020-01-05", cache_dir=str(tmp_path))

    assert len(tiingo_calls) == 1
    assert "tiingo.com" in tiingo_calls[0]
    assert not df.empty
    assert df.iloc[0]["adj_close"] == pytest.approx(42.0)
    assert (tmp_path / "prices" / "DELISTEDCO.parquet").exists()  # same cache regardless of source
    assert any("Tiingo fallback" in r.message for r in caplog.records)
    assert any("load_prices summary" in r.message for r in caplog.records)


def test_load_prices_tiingo_fallback_not_needed_when_yfinance_has_data(monkeypatch, tmp_path):
    # The common case: yfinance satisfies the request, so TIINGO_KEY is never required.
    monkeypatch.delenv("TIINGO_KEY", raising=False)
    monkeypatch.setattr(loader_module.yf, "download", lambda symbol, **kwargs: _fake_yf_frame(["2020-01-02"], [10.0], symbol))
    monkeypatch.setattr(loader_module.time, "sleep", lambda s: None)

    df = load_prices(["aapl"], "2020-01-02", "2020-01-06", cache_dir=str(tmp_path))

    assert not df.empty


def test_tiingo_429_retries_with_exponential_backoff(monkeypatch, tmp_path):
    monkeypatch.setenv("TIINGO_KEY", "dummy")
    monkeypatch.setattr(loader_module.yf, "download", lambda symbol, **kwargs: pd.DataFrame())

    sleep_calls = []
    monkeypatch.setattr(loader_module.time, "sleep", lambda s: sleep_calls.append(s))

    attempts = {"n": 0}

    def fake_get(url, headers=None, params=None, timeout=None):
        attempts["n"] += 1
        if attempts["n"] < 3:
            return _FakeResponse(status_code=429)
        return _FakeResponse(json_data=[{"date": "2020-01-02T00:00:00.000Z", "close": 10.0, "adjClose": 55.0}])

    monkeypatch.setattr(requests, "get", fake_get)

    df = load_prices(["someticker"], "2020-01-01", "2020-01-05", cache_dir=str(tmp_path))

    assert attempts["n"] == 3
    backoff_waits = [s for s in sleep_calls if s >= 60]
    assert backoff_waits == [60, 120]  # exponential: 60s, then 120s, before the 3rd attempt succeeds
    assert df.iloc[0]["adj_close"] == pytest.approx(55.0)


def test_tiingo_404_is_a_real_skip_without_retry(monkeypatch, tmp_path):
    monkeypatch.setenv("TIINGO_KEY", "dummy")
    monkeypatch.setattr(loader_module.yf, "download", lambda symbol, **kwargs: pd.DataFrame())

    attempts = {"n": 0}

    def fake_get(url, headers=None, params=None, timeout=None):
        attempts["n"] += 1
        return _FakeResponse(status_code=404)

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(loader_module.time, "sleep", lambda s: None)

    with pytest.warns(UserWarning):
        df = load_prices(["nosuchticker"], "2020-01-01", "2020-01-05", cache_dir=str(tmp_path))

    assert attempts["n"] == 1  # no retry on 404 — it's a definitive "nothing here"
    assert df.empty


def test_load_fundamentals_lags_report_date(monkeypatch, tmp_path):
    company_tickers = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
    company_facts = {
        "facts": {
            "us-gaap": {
                "EarningsPerShareDiluted": {
                    "units": {
                        "USD/shares": [
                            {"end": "2019-03-31", "start": "2019-01-01", "filed": "2019-05-01", "val": 1.0, "form": "10-Q"},
                            {"end": "2019-06-30", "start": "2019-04-01", "filed": "2019-08-01", "val": 1.1, "form": "10-Q"},
                            {"end": "2019-09-30", "start": "2019-07-01", "filed": "2019-11-01", "val": 1.2, "form": "10-Q"},
                            {"end": "2019-12-31", "start": "2019-10-01", "filed": "2020-02-01", "val": 1.3, "form": "10-K"},
                            # a 9-month YTD duration for the same concept must NOT be treated as a quarter
                            {"end": "2019-09-30", "start": "2019-01-01", "filed": "2019-11-01", "val": 3.3, "form": "10-Q"},
                        ]
                    }
                },
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {"end": "2019-03-31", "start": "2019-01-01", "filed": "2019-05-01", "val": 100.0, "form": "10-Q"},
                            {"end": "2019-06-30", "start": "2019-04-01", "filed": "2019-08-01", "val": 110.0, "form": "10-Q"},
                            {"end": "2019-09-30", "start": "2019-07-01", "filed": "2019-11-01", "val": 120.0, "form": "10-Q"},
                            {"end": "2019-12-31", "start": "2019-10-01", "filed": "2020-02-01", "val": 130.0, "form": "10-K"},
                        ]
                    }
                },
                "StockholdersEquity": {
                    "units": {"USD": [{"end": "2019-12-31", "filed": "2020-02-01", "val": 1000.0, "form": "10-K"}]}
                },
            }
        }
    }

    def fake_get(url, headers=None, timeout=None):
        if "company_tickers.json" in url:
            return _FakeResponse(json_data=company_tickers)
        return _FakeResponse(json_data=company_facts)

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(loader_module.time, "sleep", lambda s: None)

    df = load_fundamentals(["aapl"], "2020-01-01", "2020-12-31", lag_days=90, cache_dir=str(tmp_path))

    # Two report events land inside the lagged date window: the Q3 2019 filing
    # (not enough trailing quarters yet -> NaN TTM) and the Q4/FY 10-K (the
    # first with a full 4-quarter trailing window).
    assert len(df) == 2
    assert (df["date"] > df["report_date"]).all()

    q3_row = df[df["report_date"] == pd.Timestamp("2019-11-01")].iloc[0]
    assert q3_row["earnings"] != q3_row["earnings"]  # NaN: fewer than 4 trailing quarters

    row = df[df["report_date"] == pd.Timestamp("2020-02-01")].iloc[0]
    assert row["date"] == pd.Timestamp("2020-02-01") + pd.Timedelta(days=90)
    assert row["earnings"] == pytest.approx(1.0 + 1.1 + 1.2 + 1.3)  # TTM, not the 3.3 YTD figure
    assert row["book_value"] == pytest.approx(1000.0)
    assert row["roe"] == pytest.approx((100.0 + 110.0 + 120.0 + 130.0) / 1000.0)


def test_load_fundamentals_skips_ticker_with_no_cik(monkeypatch, tmp_path):
    def fake_get(url, headers=None, timeout=None):
        return _FakeResponse(json_data={"0": {"cik_str": 1, "ticker": "SOMEOTHERTICKER", "title": "X"}})

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(loader_module.time, "sleep", lambda s: None)

    with pytest.warns(UserWarning):
        df = load_fundamentals(["nosuchticker"], "2020-01-01", "2020-12-31", lag_days=90, cache_dir=str(tmp_path))

    assert df.empty


def test_load_fundamentals_never_caches_an_empty_result_as_parquet(monkeypatch, tmp_path):
    def fake_get(url, headers=None, timeout=None):
        if "company_tickers.json" in url:
            return _FakeResponse(json_data={"0": {"cik_str": 1, "ticker": "AAA", "title": "AAA Inc"}})
        return _FakeResponse(json_data={"facts": {"us-gaap": {}}})  # always empty

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(loader_module.time, "sleep", lambda s: None)

    with pytest.warns(UserWarning):
        load_fundamentals(["aaa"], "2020-01-01", "2020-12-31", lag_days=90, cache_dir=str(tmp_path))

    assert not (tmp_path / "fundamentals" / "AAA.parquet").exists()  # a failed fetch must not be cached


def test_load_fundamentals_skiplists_permanent_failure_and_skips_network_on_rerun(monkeypatch, tmp_path):
    calls = {"n": 0}

    def fake_get(url, headers=None, timeout=None):
        if "company_tickers.json" in url:
            return _FakeResponse(json_data={"0": {"cik_str": 1, "ticker": "AAA", "title": "AAA Inc"}})
        calls["n"] += 1
        return _FakeResponse(json_data={"facts": {"us-gaap": {}}})  # permanently empty -> no usable EPS

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(loader_module.time, "sleep", lambda s: None)

    with pytest.warns(UserWarning):
        load_fundamentals(["aaa"], "2020-01-01", "2020-12-31", lag_days=90, cache_dir=str(tmp_path))
    assert calls["n"] == 1
    skiplist_path = tmp_path / "unavailable_fundamentals.json"
    assert skiplist_path.exists()
    assert "AAA" in json.loads(skiplist_path.read_text())

    # A second call must NOT hit the network at all — no warning either, since
    # nothing was even attempted this time.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df = load_fundamentals(["aaa"], "2020-01-01", "2020-12-31", lag_days=90, cache_dir=str(tmp_path))
    assert calls["n"] == 1
    assert df.empty

    # force_refresh=True bypasses the skiplist and re-attempts the network call.
    with pytest.warns(UserWarning):
        load_fundamentals(["aaa"], "2020-01-01", "2020-12-31", lag_days=90, cache_dir=str(tmp_path), force_refresh=True)
    assert calls["n"] == 2


def test_load_fundamentals_dedupes_multiple_periods_sharing_one_filing_date(monkeypatch, tmp_path):
    # A single 10-K can bundle several fiscal years' worth of quarterly EPS
    # (a "selected quarterly data" table), all sharing the same `filed` date
    # but different period `end`s — this must collapse to ONE row per
    # report_date (the most recent period), not one row per period.
    company_tickers = {"0": {"cik_str": 1, "ticker": "AAA", "title": "AAA Inc"}}
    eps_facts = [
        {"end": "2015-08-31", "start": "2015-06-01", "filed": "2016-10-20", "val": 1.0, "form": "10-K"},
        {"end": "2015-11-30", "start": "2015-09-01", "filed": "2016-10-20", "val": 1.0, "form": "10-K"},
        {"end": "2016-02-29", "start": "2015-12-01", "filed": "2016-10-20", "val": 1.0, "form": "10-K"},
        {"end": "2016-05-31", "start": "2016-03-01", "filed": "2016-10-20", "val": 1.0, "form": "10-K"},
        {"end": "2016-08-31", "start": "2016-06-01", "filed": "2016-10-20", "val": 1.0, "form": "10-K"},
    ]
    company_facts = {
        "facts": {
            "us-gaap": {
                "EarningsPerShareDiluted": {"units": {"USD/shares": eps_facts}},
                "NetIncomeLoss": {"units": {"USD": []}},
                "StockholdersEquity": {"units": {"USD": []}},
            }
        }
    }

    def fake_get(url, headers=None, timeout=None):
        if "company_tickers.json" in url:
            return _FakeResponse(json_data=company_tickers)
        return _FakeResponse(json_data=company_facts)

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(loader_module.time, "sleep", lambda s: None)

    df = load_fundamentals(["aaa"], "2016-01-01", "2020-12-31", lag_days=0, cache_dir=str(tmp_path))

    # All 5 rows share report_date=2016-10-20; only one may survive.
    matches = df[df["report_date"] == pd.Timestamp("2016-10-20")]
    assert len(matches) == 1
    # TTM as of the most recent (2016-08-31) quarter: the 4 quarters ending there.
    assert matches.iloc[0]["earnings"] == pytest.approx(4.0)


def test_resolve_cik_uses_manual_overrides_for_known_sec_gaps():
    # MMC and WBA are real tickers SEC's own company_tickers.json omits/aliases
    # (see _CIK_OVERRIDES) — they must resolve even from an otherwise-empty map.
    ticker_to_cik = {"AAPL": 320193}
    resolved = {t: loader_module._resolve_cik(t, {**ticker_to_cik, **loader_module._CIK_OVERRIDES}) for t in ["AAPL", "MMC", "WBA"]}
    assert resolved == {"AAPL": 320193, "MMC": 62709, "WBA": 1618921}


def test_load_sec_ticker_to_cik_map_applies_overrides(monkeypatch, tmp_path):
    def fake_get(url, headers=None, timeout=None):
        return _FakeResponse(json_data={"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}})

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(loader_module.time, "sleep", lambda s: None)

    mapping = loader_module._load_sec_ticker_to_cik_map(str(tmp_path))

    assert mapping["AAPL"] == 320193
    assert mapping["MMC"] == 62709  # not in the fetched JSON at all — only via the override
    assert mapping["WBA"] == 1618921


def test_debug_print_resolved_ciks_returns_all_four(monkeypatch, tmp_path, capsys):
    def fake_get(url, headers=None, timeout=None):
        return _FakeResponse(
            json_data={
                "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
                "1": {"cik_str": 1403161, "ticker": "V", "title": "VISA INC."},
            }
        )

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(loader_module.time, "sleep", lambda s: None)

    resolved = loader_module.debug_print_resolved_ciks(cache_dir=str(tmp_path))

    assert resolved == {"AAPL": 320193, "V": 1403161, "MMC": 62709, "WBA": 1618921}
    out = capsys.readouterr().out
    assert "AAPL: CIK 320193" in out
    assert "WBA: CIK 1618921" in out


def test_load_fundamentals_falls_back_through_eps_concept_chain(monkeypatch, tmp_path):
    # No EarningsPerShareDiluted at all; EarningsPerShareBasicAndDiluted has the data instead.
    company_tickers = {"0": {"cik_str": 1, "ticker": "AAA", "title": "AAA Inc"}}
    quarters = [
        {"end": "2019-03-31", "start": "2019-01-01", "filed": "2019-05-01", "val": 1.0, "form": "10-Q"},
        {"end": "2019-06-30", "start": "2019-04-01", "filed": "2019-08-01", "val": 1.0, "form": "10-Q"},
        {"end": "2019-09-30", "start": "2019-07-01", "filed": "2019-11-01", "val": 1.0, "form": "10-Q"},
        {"end": "2019-12-31", "start": "2019-10-01", "filed": "2020-01-01", "val": 1.0, "form": "10-K"},
    ]
    company_facts = {
        "facts": {
            "us-gaap": {
                "EarningsPerShareDiluted": {"units": {"USD/shares": []}},
                "EarningsPerShareBasicAndDiluted": {"units": {"USD/shares": quarters}},
                "NetIncomeLoss": {"units": {"USD": []}},
                "StockholdersEquity": {"units": {"USD": []}},
            }
        }
    }

    def fake_get(url, headers=None, timeout=None):
        if "company_tickers.json" in url:
            return _FakeResponse(json_data=company_tickers)
        return _FakeResponse(json_data=company_facts)

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(loader_module.time, "sleep", lambda s: None)

    df = load_fundamentals(["aaa"], "2019-01-01", "2020-12-31", lag_days=0, cache_dir=str(tmp_path))

    assert not df.empty
    row = df[df["report_date"] == pd.Timestamp("2020-01-01")].iloc[0]
    assert row["earnings"] == pytest.approx(4.0)  # TTM = 1.0 * 4 quarters


def test_load_fundamentals_derives_missing_q4_from_annual_minus_first_three_quarters(monkeypatch, tmp_path):
    # Only Q1-Q3 are tagged as discrete quarters; Q4 only shows up baked into the annual 10-K total.
    company_tickers = {"0": {"cik_str": 1, "ticker": "AAA", "title": "AAA Inc"}}
    eps_facts = [
        {"end": "2019-03-31", "start": "2019-01-01", "filed": "2019-05-01", "val": 1.0, "form": "10-Q"},
        {"end": "2019-06-30", "start": "2019-04-01", "filed": "2019-08-01", "val": 1.0, "form": "10-Q"},
        {"end": "2019-09-30", "start": "2019-07-01", "filed": "2019-11-01", "val": 1.0, "form": "10-Q"},
        # annual (FY) total for the full year, filed with the 10-K — no discrete Q4 entry at all
        {"end": "2019-12-31", "start": "2019-01-01", "filed": "2020-01-01", "val": 5.0, "form": "10-K"},
    ]
    company_facts = {
        "facts": {
            "us-gaap": {
                "EarningsPerShareDiluted": {"units": {"USD/shares": eps_facts}},
                "NetIncomeLoss": {"units": {"USD": []}},
                "StockholdersEquity": {"units": {"USD": []}},
            }
        }
    }

    def fake_get(url, headers=None, timeout=None):
        if "company_tickers.json" in url:
            return _FakeResponse(json_data=company_tickers)
        return _FakeResponse(json_data=company_facts)

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(loader_module.time, "sleep", lambda s: None)

    df = load_fundamentals(["aaa"], "2019-01-01", "2020-12-31", lag_days=0, cache_dir=str(tmp_path))

    # Derived Q4 = 5.0 (FY) - (1.0 + 1.0 + 1.0) = 2.0; TTM at the 10-K's report_date = 1+1+1+2 = 5.0.
    row = df[df["report_date"] == pd.Timestamp("2020-01-01")].iloc[0]
    assert row["earnings"] == pytest.approx(5.0)


def test_load_fundamentals_logs_when_sec_returns_no_facts(monkeypatch, tmp_path, caplog):
    def fake_get(url, headers=None, timeout=None):
        if "company_tickers.json" in url:
            return _FakeResponse(json_data={"0": {"cik_str": 1, "ticker": "EMPTYCO", "title": "Empty Co"}})
        return _FakeResponse(json_data={"facts": {"us-gaap": {}}})  # a real CIK, but SEC has no facts on file

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(loader_module.time, "sleep", lambda s: None)

    caplog.set_level(logging.INFO)
    with pytest.warns(UserWarning):
        df = load_fundamentals(["emptyco"], "2020-01-01", "2020-12-31", lag_days=90, cache_dir=str(tmp_path))

    assert df.empty  # an empty result here is never silent — it's logged, not just a quiet empty frame
    assert any("no us-gaap facts" in r.message for r in caplog.records)


def test_load_fundamentals_date_filter_excludes_rows_lagged_past_end(monkeypatch, tmp_path):
    company_tickers = {"0": {"cik_str": 1, "ticker": "AAA", "title": "AAA Inc"}}
    company_facts = {
        "facts": {
            "us-gaap": {
                "EarningsPerShareDiluted": {
                    "units": {
                        "USD/shares": [
                            {"end": "2019-03-31", "start": "2019-01-01", "filed": "2019-05-01", "val": 1.0, "form": "10-Q"},
                            {"end": "2019-06-30", "start": "2019-04-01", "filed": "2019-08-01", "val": 1.0, "form": "10-Q"},
                            {"end": "2019-09-30", "start": "2019-07-01", "filed": "2019-11-01", "val": 1.0, "form": "10-Q"},
                            {"end": "2019-12-31", "start": "2019-10-01", "filed": "2020-01-01", "val": 1.0, "form": "10-K"},
                            {"end": "2020-03-31", "start": "2020-01-01", "filed": "2020-02-01", "val": 1.0, "form": "10-Q"},
                        ]
                    }
                },
                "NetIncomeLoss": {"units": {"USD": []}},
                "StockholdersEquity": {"units": {"USD": []}},
            }
        }
    }

    def fake_get(url, headers=None, timeout=None):
        if "company_tickers.json" in url:
            return _FakeResponse(json_data=company_tickers)
        return _FakeResponse(json_data=company_facts)

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(loader_module.time, "sleep", lambda s: None)

    # lag_days=10: the 2020-01-01 filing lands at date=2020-01-11 (<= end, kept);
    # the 2020-02-01 filing lands at date=2020-02-11 (> end, must be excluded).
    end = "2020-01-15"
    df = load_fundamentals(["aaa"], "2019-01-01", end, lag_days=10, cache_dir=str(tmp_path))

    assert (df["date"] <= pd.Timestamp(end)).all()
    assert pd.Timestamp("2020-02-01") not in df["report_date"].values
    assert pd.Timestamp("2020-01-01") in df["report_date"].values


def test_load_fundamentals_real_aapl_returns_nonempty_2019_2020(tmp_path):
    """Integration check against real SEC EDGAR for a known filer (AAPL).

    Everything else in this file is mocked; this one intentionally hits the
    real API to confirm the whole pipeline actually works against SEC's real
    data shape, not just our assumptions about it. Skips rather than fails
    if the network is unreachable.
    """
    try:
        df = load_fundamentals(["AAPL"], "2019-01-01", "2020-12-31", lag_days=90, cache_dir=str(tmp_path))
    except (requests.exceptions.RequestException, OSError) as e:
        pytest.skip(f"network unavailable for SEC EDGAR integration check: {e}")

    assert not df.empty
    assert (df["date"] <= pd.Timestamp("2020-12-31")).all()
    assert (df["date"] > df["report_date"]).all()


def test_build_universe_includes_delisted_names(monkeypatch, tmp_path):
    csv_bytes = (
        b"date,tickers\n"
        b'2010-01-04,"AAA,BBB-201006"\n'
        b'2010-07-01,"AAA,CCC"\n'
    )

    monkeypatch.setattr(requests, "get", lambda url, timeout=None: _FakeResponse(content=csv_bytes))

    universe = build_universe("SP500", "2010-01-04", "2010-08-02", cache_dir=str(tmp_path))

    assert bool(universe.loc["2010-01-05", "AAA"]) is True
    # BBB was delisted mid-2010 but must still show up as a historical member
    # (no survivorship bias) up until the snapshot that drops it.
    assert bool(universe.loc["2010-06-01", "BBB"]) is True
    assert bool(universe.loc["2010-07-01", "BBB"]) is False
    assert bool(universe.loc["2010-07-01", "CCC"]) is True
    assert bool(universe.loc["2010-01-05", "CCC"]) is False
