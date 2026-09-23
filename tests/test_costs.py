"""Flat vs. per-name (volatility-scaled) cost models."""
import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import apply_costs, build_costs, per_name_cost_bps
from src.backtest.engine import run_backtest

DAYS = pd.bdate_range("2019-09-02", "2020-03-31")  # > 63 trading days before the first rebalance
REBALANCE = pd.date_range("2020-01-31", periods=3, freq="ME")
FLAT = {"cost_model": "flat", "bps_per_trade": 8}
PER_NAME = {"cost_model": "per_name", "bps_per_trade": 8, "vol_lookback_days": 63}


def _prices(daily_move: dict[str, float], start_price: float = 100.0) -> pd.DataFrame:
    """Each name alternates +move / -move every day, so its daily-return volatility is
    exactly proportional to `move` — a name with 3x the move has exactly 3x the vol."""
    signs = np.where(np.arange(len(DAYS)) % 2 == 0, 1.0, -1.0)
    return pd.DataFrame(
        {name: start_price * np.cumprod(1.0 + move * signs) for name, move in daily_move.items()}, index=DAYS
    )


def _weights(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=REBALANCE[: len(rows)]).fillna(0.0)


def test_per_name_reduces_to_flat_when_every_name_has_the_same_volatility():
    prices = _prices({"A": 0.01, "B": 0.01, "C": 0.01})
    weights = _weights([{"A": 1.0, "B": -1.0}, {"B": 1.0, "C": -1.0}, {"A": 0.5, "C": 0.5, "B": -1.0}])

    per_name = build_costs(PER_NAME, prices, weights.index)

    assert per_name.to_numpy() == pytest.approx(8.0)
    pd.testing.assert_series_equal(apply_costs(weights, per_name), apply_costs(weights, 8))


def test_per_name_diverges_from_flat_and_charges_volatile_names_more():
    prices = _prices({"LOW": 0.005, "MID": 0.01, "HIGH": 0.03})  # median vol is MID's
    weights = _weights([{"HIGH": 1.0, "LOW": -1.0}])

    per_name = build_costs(PER_NAME, prices, weights.index)
    row = per_name.loc[REBALANCE[0]]

    assert row["MID"] == pytest.approx(8.0)    # the median name pays the flat rate
    assert row["HIGH"] == pytest.approx(24.0)  # 3x the vol, 3x the cost
    assert row["LOW"] == pytest.approx(4.0)
    # Opening the book: half of |dw| per name, each at its own cost.
    assert apply_costs(weights, per_name).iloc[0] == pytest.approx(0.5 * (24.0 + 4.0) / 1e4)
    assert apply_costs(weights, 8).iloc[0] == pytest.approx(0.5 * (8.0 + 8.0) / 1e4)


def test_config_switch_selects_the_model_end_to_end():
    prices = _prices({"LOW": 0.005, "MID": 0.01, "HIGH": 0.03})
    weights = _weights([{"HIGH": 1.0, "LOW": -1.0}, {"HIGH": 1.0, "LOW": -1.0}])
    forward_returns = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)

    assert build_costs(FLAT, prices, weights.index) == 8
    assert isinstance(build_costs(PER_NAME, prices, weights.index), pd.DataFrame)
    assert build_costs({"bps_per_trade": 8}, prices, weights.index) == 8  # flat is the default

    flat_net = run_backtest(weights, forward_returns, cost_bps=build_costs(FLAT, prices, weights.index))
    per_name_net = run_backtest(weights, forward_returns, cost_bps=build_costs(PER_NAME, prices, weights.index))
    assert per_name_net.iloc[0] != pytest.approx(flat_net.iloc[0])

    with pytest.raises(ValueError, match="cost_model"):
        build_costs({"cost_model": "square_root", "bps_per_trade": 8}, prices, weights.index)


def test_per_name_cost_uses_only_prices_up_to_the_rebalance_date():
    prices = _prices({"LOW": 0.005, "MID": 0.01, "HIGH": 0.03})
    before = per_name_cost_bps(prices, REBALANCE, base_bps=8)

    shocked = prices.copy()
    shocked.loc["2020-02-03":, "LOW"] *= np.linspace(1.0, 5.0, len(shocked.loc["2020-02-03":]))  # wild moves after Jan 31
    after = per_name_cost_bps(shocked, REBALANCE, base_bps=8)

    pd.testing.assert_series_equal(before.loc[REBALANCE[0]], after.loc[REBALANCE[0]])
    assert after.loc[REBALANCE[1], "LOW"] != pytest.approx(before.loc[REBALANCE[1], "LOW"])


def test_name_without_a_full_volatility_window_pays_the_flat_rate():
    prices = _prices({"LOW": 0.005, "MID": 0.01, "HIGH": 0.03})
    prices.loc[: "2020-01-02", "HIGH"] = np.nan  # listed ~20 trading days before the first rebalance

    costs = per_name_cost_bps(prices, REBALANCE, base_bps=8)

    assert costs.loc[REBALANCE[0], "HIGH"] == pytest.approx(8.0)


def test_apply_costs_refuses_to_let_a_trade_go_uncosted():
    weights = _weights([{"A": 1.0, "B": -1.0}])
    costs = pd.DataFrame({"A": [8.0], "B": [np.nan]}, index=weights.index)

    with pytest.raises(ValueError, match="no per-name cost"):
        apply_costs(weights, costs)
