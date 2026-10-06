"""Shares outstanding: where the count comes from, and restating it across splits."""
import numpy as np
import pandas as pd
import pytest

from src.data.cache import PriceCache
from src.data.providers.xbrl import fundamentals_from_facts, shares_by_filing
from src.data.splits import ttm_fundamentals
from tests.fakes import FakePriceProvider


def _instant(rows):
    """XBRL instant facts: (end, filed, value, form) tuples."""
    return {"units": {"shares": [{"end": e, "filed": f, "val": v, "form": form} for e, f, v, form in rows]}}


def test_cover_page_count_is_preferred_and_the_balance_sheet_fills_gaps():
    dei = {"EntityCommonStockSharesOutstanding": _instant([("2020-07-20", "2020-07-31", 1000.0, "10-Q")])}
    us_gaap = {"CommonStockSharesOutstanding": _instant([
        ("2020-06-27", "2020-07-31", 990.0, "10-Q"),   # same filing: cover page wins
        ("2020-09-26", "2020-10-30", 3980.0, "10-K"),  # no cover-page count: balance sheet used
    ])}

    shares = shares_by_filing(dei, us_gaap).set_index("filed")["shares"]

    assert shares[pd.Timestamp("2020-07-31")] == 1000.0
    assert shares[pd.Timestamp("2020-10-30")] == 3980.0


def test_a_filing_with_no_share_count_gets_none_rather_than_an_approximation():
    eps_q = pd.DataFrame({"end": pd.to_datetime(["2020-06-27"]), "filed": pd.to_datetime(["2020-07-31"]), "val": [2.58]})
    eps_a = pd.DataFrame({"end": pd.Series(dtype="datetime64[ns]"), "filed": pd.Series(dtype="datetime64[ns]"), "val": pd.Series(dtype="float64")})

    facts = fundamentals_from_facts({}, eps_q, eps_a, dei={})

    assert np.isnan(facts["shares"].iloc[0])


@pytest.mark.parametrize("window_end", ["2020-12-31", "2020-08-15"])  # split inside / after the price window
def test_market_cap_is_unchanged_by_restating_shares_and_price_across_a_split(tmp_path, window_end):
    # 1,000 shares filed 2020-07-31 at an as-traded price of 400; a 4-for-1 on 2020-08-31.
    # The source quotes close on today's basis (100 for July), as Yahoo does.
    provider = FakePriceProvider(
        "src", {"AAA": {"2020-07-31": 100.0, "2020-08-31": 105.0, "2020-12-31": 120.0}}, splits={"AAA": {"2020-08-31": 4.0}}
    )
    prices = PriceCache(tmp_path).load(provider, "AAA", pd.Timestamp("2020-07-01"), pd.Timestamp(window_end))
    splits = prices.set_index("date")["split_ratio"]
    facts = pd.DataFrame({
        "report_date": pd.to_datetime(["2020-07-31"]), "period_end": pd.to_datetime(["2020-06-27"]),
        "period": ["quarter"], "eps": [2.0], "book_value": [np.nan], "roe": [np.nan], "shares": [1000.0],
    })

    restated = ttm_fundamentals(facts, splits)["shares_outstanding"].iloc[0]
    close = prices.set_index("date").loc["2020-07-31", "close"]

    assert restated * close == pytest.approx(1000.0 * 400.0)  # as-filed shares x as-traded price
