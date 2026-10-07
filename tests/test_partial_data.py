"""A source failing for some tickers stops the load unless the caller opts
into partial data; a source having no data for a ticker doesn't."""
import warnings

import pytest

from factor_backtester.data.loader import load_fundamentals, load_prices
from factor_backtester.data.partial import PartialDataError
from tests.fakes import FakeFundamentalsProvider, FakePriceProvider, flat_split_history

JAN = {"2020-01-02": 10.0, "2020-01-03": 10.5, "2020-01-06": 11.0}
QUARTERS = [{"report_date": d, "eps": 0.25, "book_value": 10.0, "roe": 0.1}
            for d in ["2019-04-15", "2019-07-15", "2019-10-15", "2020-01-15"]]


def _flaky_prices():
    return FakePriceProvider("vendor", {"AAA": JAN, "BBB": JAN}, error={"BBB": ConnectionError("read timed out")})


def test_a_failed_ticker_stops_the_price_load_and_is_named(tmp_path):
    provider = _flaky_prices()

    with pytest.warns(UserWarning), pytest.raises(PartialDataError) as caught:
        load_prices(["aaa", "bbb", "ccc"], "2020-01-02", "2020-01-06", cache_dir=str(tmp_path), providers=[provider])

    assert caught.value.failures == {"bbb": "vendor: read timed out"}  # CCC has no data: that's not a failure
    assert [call[0] for call in provider.calls] == ["aaa", "bbb", "ccc"]  # every ticker was still tried...

    provider.error = None
    load_prices(["aaa", "bbb"], "2020-01-02", "2020-01-06", cache_dir=str(tmp_path), providers=[provider])
    assert [call[0] for call in provider.calls][3:] == ["bbb"]  # ...and AAA was cached, so a re-run only asks for BBB


def test_allow_partial_continues_without_the_failed_ticker_and_records_it(tmp_path):
    with pytest.warns(UserWarning, match="PARTIAL DATA: continuing without 1 ticker.*bbb"):
        df = load_prices(
            ["aaa", "bbb"], "2020-01-02", "2020-01-06", cache_dir=str(tmp_path), providers=[_flaky_prices()],
            allow_partial=True,
        )

    assert set(df["ticker"]) == {"AAA"}
    assert df.attrs["failed_tickers"] == {"bbb": "vendor: read timed out"}


def test_a_complete_load_records_no_failures_and_says_nothing(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df = load_prices(["aaa"], "2020-01-02", "2020-01-06", cache_dir=str(tmp_path),
                         providers=[FakePriceProvider("vendor", {"AAA": JAN})])

    assert df.attrs["failed_tickers"] == {}


def test_a_later_provider_with_data_makes_an_earlier_failure_harmless(tmp_path):
    down = FakePriceProvider("down", error=ConnectionError("refused"))
    up = FakePriceProvider("up", {"AAA": JAN})

    with pytest.warns(UserWarning, match="down error for aaa"):
        df = load_prices(["aaa"], "2020-01-02", "2020-01-06", cache_dir=str(tmp_path), providers=[down, up])

    assert len(df) == 3 and df.attrs["failed_tickers"] == {}


def test_a_failed_ticker_stops_the_fundamentals_load_unless_partial_is_allowed(tmp_path):
    provider = FakeFundamentalsProvider(rows={"AAA": QUARTERS, "BBB": QUARTERS}, error={"BBB": ConnectionError("SEC 503")})
    load = dict(start="2020-01-01", end="2020-12-31", lag_days=0, cache_dir=str(tmp_path), provider=provider,
                splits=flat_split_history(["aaa", "bbb"]))

    with pytest.raises(PartialDataError) as caught:
        load_fundamentals(["aaa", "bbb"], **load)
    assert caught.value.failures == {"bbb": "fake fundamentals: SEC 503"}

    with pytest.warns(UserWarning, match="PARTIAL DATA"):
        df = load_fundamentals(["aaa", "bbb"], **load, allow_partial=True)
    assert set(df["ticker"]) == {"AAA"}
    assert df.attrs["failed_tickers"] == {"bbb": "fake fundamentals: SEC 503"}
