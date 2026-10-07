"""Low-volatility factor: orientation and the strictly-before window."""
import numpy as np
import pandas as pd
import pytest

from factor_backtester.features.registry import get_factor

DAYS = pd.bdate_range("2020-01-01", "2020-06-30")


def _prices(daily_moves: dict[str, float]) -> pd.DataFrame:
    """Each name alternates +move / -move daily, so its volatility is proportional to `move`."""
    signs = np.where(np.arange(len(DAYS)) % 2 == 0, 1.0, -1.0)
    rows = []
    for name, move in daily_moves.items():
        px = 100.0 * np.cumprod(1.0 + move * signs)
        rows += [{"date": d, "ticker": name, "adj_close": p, "close": p, "split_ratio": 1.0} for d, p in zip(DAYS, px)]
    return pd.DataFrame(rows)


def test_the_lowest_volatility_name_gets_the_highest_score():
    scores = get_factor("low_vol").compute(_prices({"CALM": 0.002, "MID": 0.01, "WILD": 0.04}), window_days=20)
    row = scores.loc["2020-05-31"]
    assert row.idxmax() == "CALM" and row.idxmin() == "WILD"


def test_the_window_ends_strictly_before_the_rebalance_date():
    prices = _prices({"AAA": 0.01, "BBB": 0.02})
    last_day = prices.loc[prices["date"] <= "2020-05-31", "date"].max()  # May's last trading day: the rebalance date
    day_before = prices.loc[prices["date"] < last_day, "date"].max()
    base = get_factor("low_vol").compute(prices, window_days=20).loc["2020-05-31", "AAA"]

    on_rebalance_day = prices.copy()
    on_rebalance_day.loc[(on_rebalance_day["ticker"] == "AAA") & (on_rebalance_day["date"] == last_day), "adj_close"] *= 1.5
    on_day_before = prices.copy()
    on_day_before.loc[(on_day_before["ticker"] == "AAA") & (on_day_before["date"] == day_before), "adj_close"] *= 1.5

    assert get_factor("low_vol").compute(on_rebalance_day, window_days=20).loc["2020-05-31", "AAA"] == pytest.approx(base)
    assert get_factor("low_vol").compute(on_day_before, window_days=20).loc["2020-05-31", "AAA"] != pytest.approx(base)


def test_a_name_without_a_full_window_is_nan():
    scores = get_factor("low_vol").compute(_prices({"AAA": 0.01}), window_days=63)
    assert np.isnan(scores.loc["2020-02-29", "AAA"])   # ~40 trading days in: not enough
    assert not np.isnan(scores.loc["2020-05-31", "AAA"])
