"""The public `providers=` / `provider=` extension point: any object meeting the
provider contract can be plugged in, and a custom chain is isolated from the
default chain's cache and skiplist."""
import json
import warnings

import pandas as pd
import pytest

from src.data.loader import load_fundamentals, load_prices
from tests.fakes import FakeFundamentalsProvider, FakePriceProvider

JAN = {"2020-01-02": 10.0, "2020-01-03": 10.5, "2020-01-06": 11.0}


def test_chain_is_walked_in_order_and_the_first_provider_with_data_wins(tmp_path):
    first = FakePriceProvider("first")                       # has nothing
    second = FakePriceProvider("second", {"AAA": JAN})
    third = FakePriceProvider("third", {"AAA": {"2020-01-02": 99.0}})

    df = load_prices(["aaa"], "2020-01-02", "2020-01-06", cache_dir=str(tmp_path), providers=[first, second, third])

    assert df["adj_close"].tolist() == [10.0, 10.5, 11.0]
    assert len(first.calls) == 1 and len(second.calls) == 1
    assert third.calls == []  # never asked once an earlier provider answered


def test_a_single_provider_chain_needs_no_fallback(tmp_path):
    only = FakePriceProvider("only", {"AAA": JAN})

    df = load_prices(["aaa"], "2020-01-02", "2020-01-06", cache_dir=str(tmp_path), providers=[only])

    assert set(df["ticker"]) == {"AAA"} and len(df) == 3


def test_empty_chain_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        load_prices(["aaa"], "2020-01-02", "2020-01-06", cache_dir=str(tmp_path), providers=[])


def test_custom_chain_is_not_served_from_the_default_chains_cache(tmp_path):
    # The default chain's cache already holds AAA (as if Yahoo had supplied it)...
    (tmp_path / "prices").mkdir()
    pd.DataFrame({"date": pd.to_datetime(list(JAN)), "adj_close": list(JAN.values())}).to_parquet(
        tmp_path / "prices" / "AAA.parquet", index=False
    )
    mine = FakePriceProvider("my vendor", {"AAA": {"2020-01-02": 42.0}})

    df = load_prices(["aaa"], "2020-01-02", "2020-01-02", cache_dir=str(tmp_path), providers=[mine])

    assert df["adj_close"].tolist() == [42.0]  # my data, not another chain's cached prices
    assert len(mine.calls) == 1


def test_custom_chain_ignores_the_default_skiplist(tmp_path):
    # The default chain has already concluded AAA is unavailable everywhere it looks.
    (tmp_path / "unavailable_prices.json").write_text(json.dumps({"AAA": "2020-01-01T00:00:00+00:00"}))
    mine = FakePriceProvider("my vendor", {"AAA": JAN})

    df = load_prices(["aaa"], "2020-01-02", "2020-01-06", cache_dir=str(tmp_path), providers=[mine])

    assert len(df) == 3 and len(mine.calls) == 1
    # ...and a custom chain's misses never get written into it either.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        load_prices(["zzz"], "2020-01-02", "2020-01-06", cache_dir=str(tmp_path), providers=[mine])
    assert "ZZZ" not in json.loads((tmp_path / "unavailable_prices.json").read_text())


def test_custom_fundamentals_provider_ignores_the_default_skiplist(tmp_path):
    (tmp_path / "unavailable_fundamentals.json").write_text(json.dumps({"AAA": "2020-01-01T00:00:00+00:00"}))
    mine = FakeFundamentalsProvider(rows={"AAA": [{"report_date": "2020-01-15", "earnings": 1.0, "book_value": 10.0, "roe": 0.1}]})

    df = load_fundamentals(["aaa"], "2020-01-01", "2020-12-31", lag_days=0, cache_dir=str(tmp_path), provider=mine)

    assert df["earnings"].tolist() == [1.0]
    assert mine.calls == ["aaa"]

def test_a_legacy_cached_series_is_refreshed_by_a_later_provider_in_the_chain(tmp_path):
    # Cached before close/splits were stored; the first provider has nothing for
    # the ticker, so the chain must go on and let the second one refresh it.
    root = tmp_path / "providers" / "yfinance+Tiingo"  # this chain's cache namespace
    (root / "prices").mkdir(parents=True)
    pd.DataFrame({"date": pd.to_datetime(["2020-01-02"]), "adj_close": [5.0]}).to_parquet(
        root / "prices" / "GONE.parquet", index=False
    )
    first = FakePriceProvider("yfinance")
    second = FakePriceProvider("Tiingo", {"GONE": {"2020-01-02": 5.0, "2020-01-03": 5.1}})

    df = load_prices(["gone"], "2020-01-02", "2020-01-03", cache_dir=str(tmp_path), providers=[first, second])

    assert len(second.calls) == 1
    assert df["close"].tolist() == [5.0, 5.1]


def test_a_legacy_series_nobody_can_refresh_is_still_returned_for_returns(tmp_path):
    root = tmp_path / "providers" / "yfinance"
    (root / "prices").mkdir(parents=True)
    pd.DataFrame({"date": pd.to_datetime(["2020-01-02"]), "adj_close": [5.0]}).to_parquet(
        root / "prices" / "GONE.parquet", index=False
    )

    df = load_prices(["gone"], "2020-01-02", "2020-01-03", cache_dir=str(tmp_path), providers=[FakePriceProvider("yfinance")])

    assert df["adj_close"].tolist() == [5.0] and df["close"].isna().all()
