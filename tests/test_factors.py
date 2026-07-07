"""Sanity checks for Step 3: factors + cross-sectional standardization."""
import numpy as np
import pandas as pd
import pytest

from src.features.factors import momentum, quality, value
from src.features.transforms import combine_factors, zscore_cross_section

MONTH_ENDS = pd.date_range("2020-01-31", "2021-03-31", freq="ME")  # 15 months


def _long_prices(paths: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for ticker, series in paths.items():
        for dt, px in zip(MONTH_ENDS, series):
            rows.append({"date": dt, "ticker": ticker, "adj_close": px})
    return pd.DataFrame(rows)


def test_momentum_ranks_trend_correctly_and_starts_after_lookback():
    # AAA compounds up ~5%/month, BBB is flat, CCC drifts down ~3%/month.
    aaa = [100 * 1.05**i for i in range(len(MONTH_ENDS))]
    bbb = [100 * 1.005**i for i in range(len(MONTH_ENDS))]
    ccc = [100 * 0.97**i for i in range(len(MONTH_ENDS))]
    prices = _long_prices({"AAA": aaa, "BBB": bbb, "CCC": ccc})

    mom = momentum(prices, lookback_m=12, skip_m=1)

    # Needs 12 months of lookback + 1 skip month before the first value.
    assert mom.iloc[:12].isna().all().all()
    last = mom.iloc[-1]
    assert last["AAA"] > last["BBB"] > last["CCC"]


def test_momentum_skip_month_excludes_most_recent_return():
    # A one-month crash right at the end should not move 12-1 momentum,
    # since skip_m=1 excludes the most recent month from the window.
    steady = [100 * 1.02**i for i in range(len(MONTH_ENDS))]
    crashed = steady.copy()
    crashed[-1] = crashed[-2] * 0.5  # -50% in the final (skipped) month

    prices = _long_prices({"STEADY": steady, "CRASHED": crashed})
    mom = momentum(prices, lookback_m=12, skip_m=1)

    last = mom.iloc[-1]
    assert last["STEADY"] == pytest.approx(last["CRASHED"], rel=1e-9)


def test_value_earnings_yield_uses_latest_report_and_price():
    prices = _long_prices({"AAA": [100.0] * len(MONTH_ENDS), "BBB": [50.0] * len(MONTH_ENDS)})
    fundamentals = pd.DataFrame(
        [
            {"date": pd.Timestamp("2020-02-15"), "ticker": "AAA", "report_date": pd.Timestamp("2019-12-31"), "earnings": 4.0, "book_value": 20.0, "roe": 0.1},
            {"date": pd.Timestamp("2020-05-15"), "ticker": "AAA", "report_date": pd.Timestamp("2020-03-31"), "earnings": 5.0, "book_value": 21.0, "roe": 0.11},
            {"date": pd.Timestamp("2020-02-15"), "ticker": "BBB", "report_date": pd.Timestamp("2019-12-31"), "earnings": 2.5, "book_value": 10.0, "roe": 0.2},
        ]
    )

    ey = value(fundamentals, prices, metric="earnings_yield")

    assert ey.loc["2020-01-31", "AAA"] != ey.loc["2020-01-31", "AAA"]  # NaN before first report
    assert ey.loc["2020-02-29", "AAA"] == pytest.approx(4.0 / 100.0)
    assert ey.loc["2020-04-30", "AAA"] == pytest.approx(4.0 / 100.0)  # carried forward
    assert ey.loc["2020-06-30", "AAA"] == pytest.approx(5.0 / 100.0)  # new report picked up
    assert ey.loc["2020-02-29", "BBB"] == pytest.approx(2.5 / 50.0)

    with pytest.raises(NotImplementedError):
        value(fundamentals, prices, metric="book_to_price")


def test_quality_roe_carries_forward_point_in_time():
    fundamentals = pd.DataFrame(
        [
            {"date": pd.Timestamp("2020-02-15"), "ticker": "AAA", "report_date": pd.Timestamp("2019-12-31"), "earnings": 4.0, "book_value": 20.0, "roe": 0.10},
            {"date": pd.Timestamp("2020-05-15"), "ticker": "AAA", "report_date": pd.Timestamp("2020-03-31"), "earnings": 5.0, "book_value": 21.0, "roe": 0.15},
            {"date": pd.Timestamp("2020-08-15"), "ticker": "AAA", "report_date": pd.Timestamp("2020-06-30"), "earnings": 6.0, "book_value": 22.0, "roe": 0.20},
        ]
    )

    roe = quality(fundamentals, metric="roe")

    assert roe.loc["2020-02-29", "AAA"] == pytest.approx(0.10)
    assert roe.loc["2020-04-30", "AAA"] == pytest.approx(0.10)  # carried forward, no new report yet
    assert roe.loc["2020-05-31", "AAA"] == pytest.approx(0.15)  # new report picked up

    with pytest.raises(NotImplementedError):
        quality(fundamentals, metric="roa")


def test_zscore_cross_section_winsorizes_and_standardizes():
    row = pd.DataFrame({"AAA": [1.0], "BBB": [2.0], "CCC": [3.0], "DDD": [1000.0]}, index=[pd.Timestamp("2020-01-31")])

    z = zscore_cross_section(row, lower_q=0.01, upper_q=0.90)

    result = z.loc[pd.Timestamp("2020-01-31")]
    assert result.mean() == pytest.approx(0.0, abs=1e-9)
    assert result.std() == pytest.approx(1.0, abs=1e-9)
    # DDD was winsorized down, so it should no longer dwarf the others.
    assert result["DDD"] < 5.0
    assert result["AAA"] < result["BBB"] < result["CCC"] < result["DDD"]


def test_combine_factors_weighted_nan_aware_average():
    idx = [pd.Timestamp("2020-01-31")]
    momentum_z = pd.DataFrame({"AAA": [1.0], "BBB": [-1.0]}, index=idx)
    value_z = pd.DataFrame({"AAA": [np.nan], "BBB": [1.0]}, index=idx)  # AAA missing value score

    equal = combine_factors({"momentum": momentum_z, "value": value_z})
    assert equal.loc[idx[0], "AAA"] == pytest.approx(1.0)  # only momentum present -> its own score
    assert equal.loc[idx[0], "BBB"] == pytest.approx(0.0)  # (-1 + 1) / 2

    weighted = combine_factors({"momentum": momentum_z, "value": value_z}, weights={"momentum": 0.75, "value": 0.25})
    assert weighted.loc[idx[0], "BBB"] == pytest.approx(0.75 * -1.0 + 0.25 * 1.0)

    with pytest.raises(ValueError):
        combine_factors({"momentum": momentum_z}, weights={"bogus": 1.0})
