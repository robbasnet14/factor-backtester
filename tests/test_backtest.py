"""Sanity checks for Steps 4-5: portfolio construction, costs, backtest engine."""
import warnings

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import apply_costs
from src.backtest.engine import run_backtest
from src.backtest.portfolio import decile_portfolios, tradable_on_rebalance

DATES = pd.date_range("2020-01-31", periods=3, freq="ME")
TICKERS = [f"T{i}" for i in range(10)]  # 10 names -> clean top/bottom decile of 1 each


def _scores_row(values):
    return pd.DataFrame([values], index=[DATES[0]], columns=TICKERS)


def test_decile_portfolios_long_short_is_dollar_neutral_and_ranked():
    scores = _scores_row(list(range(10)))  # T0 worst .. T9 best

    weights = decile_portfolios(scores, n_deciles=10, long_short=True)
    row = weights.loc[DATES[0]]

    assert row["T9"] == pytest.approx(1.0)   # top decile, long
    assert row["T0"] == pytest.approx(-1.0)  # bottom decile, short
    assert row.drop(["T0", "T9"]).eq(0.0).all()
    assert row.sum() == pytest.approx(0.0)   # dollar-neutral


def test_decile_portfolios_long_only():
    scores = _scores_row(list(range(10)))
    weights = decile_portfolios(scores, n_deciles=10, long_short=False)
    row = weights.loc[DATES[0]]

    assert row["T9"] == pytest.approx(1.0)
    assert row["T0"] == pytest.approx(0.0)  # no short leg
    assert row.sum() == pytest.approx(1.0)


def test_decile_portfolios_skips_dates_with_too_few_names():
    sparse = pd.DataFrame([{**{t: np.nan for t in TICKERS}, "T0": 1.0, "T1": 2.0}], index=[DATES[0]])[TICKERS]
    weights = decile_portfolios(sparse, n_deciles=10, long_short=True)
    assert (weights.loc[DATES[0]] == 0.0).all()


def test_apply_costs_charges_full_turnover_on_first_period_and_zero_when_unchanged():
    weights = pd.DataFrame(
        {"A": [1.0, 1.0, -1.0], "B": [-1.0, -1.0, 1.0]},
        index=DATES,
    )
    cost = apply_costs(weights, bps_per_trade=10)

    assert cost.iloc[0] == pytest.approx(0.5 * 2.0 * 10 / 1e4)  # from flat -> full book
    assert cost.iloc[1] == pytest.approx(0.0)  # unchanged weights, no turnover
    assert cost.iloc[2] == pytest.approx(0.5 * 4.0 * 10 / 1e4)  # A and B both flip sign


def test_run_backtest_uses_forward_returns_not_contemporaneous():
    weights = pd.DataFrame({"A": [1.0, 1.0]}, index=DATES[:2])
    # If the engine looked at contemporaneous returns it would pick up the
    # +100% move at t; forward_returns says the move actually earned is -50%.
    contemporaneous_returns = pd.DataFrame({"A": [1.0, 0.2]}, index=DATES[:2])
    forward_returns = pd.DataFrame({"A": [-0.5, 0.2]}, index=DATES[:2])

    net = run_backtest(weights, forward_returns, cost_bps=0)

    assert net.loc[DATES[0]] == pytest.approx(-0.5)
    assert net.loc[DATES[0]] != pytest.approx(contemporaneous_returns.loc[DATES[0], "A"])


def test_run_backtest_nets_out_costs():
    weights = pd.DataFrame({"A": [1.0], "B": [-1.0]}, index=[DATES[0]])
    forward_returns = pd.DataFrame({"A": [0.05], "B": [0.05]}, index=[DATES[0]])

    net = run_backtest(weights, forward_returns, cost_bps=50)
    gross = 1.0 * 0.05 + -1.0 * 0.05  # = 0.0
    expected_cost = 0.5 * 2.0 * 50 / 1e4  # full turnover from flat
    assert net.loc[DATES[0]] == pytest.approx(gross - expected_cost)


def test_run_backtest_warns_on_missing_return_for_held_position():
    weights = pd.DataFrame({"A": [1.0]}, index=[DATES[0]])
    forward_returns = pd.DataFrame({"A": [np.nan]}, index=[DATES[0]])

    with pytest.warns(UserWarning):
        net = run_backtest(weights, forward_returns, cost_bps=0)
    assert net.loc[DATES[0]] == pytest.approx(0.0 - apply_costs(weights, 0).loc[DATES[0]])


def test_run_backtest_exits_delisted_position_at_last_available_price():
    dates = DATES[:2]
    weights = pd.DataFrame({"A": [1.0, 0.0]}, index=dates)
    # A has no forward return at date0 (delisted before the next rebalance),
    # but `prices` still has its entry price and the last price it ever traded at.
    forward_returns = pd.DataFrame({"A": [np.nan, 0.1]}, index=dates)
    prices = pd.DataFrame({"A": [50.0, 40.0]}, index=dates)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a real exit price was found -> must NOT warn
        net = run_backtest(weights, forward_returns, cost_bps=0, prices=prices)

    assert net.loc[dates[0]] == pytest.approx(40.0 / 50.0 - 1.0)


def test_run_backtest_still_warns_when_no_exit_price_available_either():
    weights = pd.DataFrame({"A": [1.0]}, index=[DATES[0]])
    forward_returns = pd.DataFrame({"A": [np.nan]}, index=[DATES[0]])
    prices = pd.DataFrame({"A": [np.nan]}, index=[DATES[0]])  # never had a recorded price at all

    with pytest.warns(UserWarning):
        net = run_backtest(weights, forward_returns, cost_bps=0, prices=prices)
    assert net.loc[DATES[0]] == pytest.approx(0.0)  # documented 0% fallback, nothing to exit at


def test_run_backtest_without_prices_keeps_old_zero_fallback_behavior():
    # Backward compatible default: omitting `prices` behaves exactly as before this fix.
    weights = pd.DataFrame({"A": [1.0]}, index=[DATES[0]])
    forward_returns = pd.DataFrame({"A": [np.nan]}, index=[DATES[0]])

    with pytest.warns(UserWarning):
        net = run_backtest(weights, forward_returns, cost_bps=0)
    assert net.loc[DATES[0]] == pytest.approx(0.0)


def test_run_backtest_mixes_exit_priced_and_normal_positions_correctly():
    dates = DATES[:2]
    weights = pd.DataFrame({"A": [1.0, 0.0], "B": [1.0, 1.0]}, index=dates)
    # A delists after date0 (no forward return, but a real exit price exists);
    # B just trades normally throughout with real forward returns both periods.
    forward_returns = pd.DataFrame({"A": [np.nan, 0.0], "B": [0.05, 0.02]}, index=dates)
    prices = pd.DataFrame({"A": [50.0, 45.0], "B": [100.0, 105.0]}, index=dates)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # both positions resolve cleanly -> must NOT warn
        net = run_backtest(weights, forward_returns, cost_bps=0, prices=prices)

    # A exits at its last available price (45 vs entry 50); B earns its real 5% forward return.
    assert net.loc[dates[0]] == pytest.approx((45.0 / 50.0 - 1.0) + 0.05)


# --- Tradability: only names with a price on the rebalance date can be held ---

DAYS = pd.bdate_range("2020-01-01", "2020-03-31")  # DATES are this range's three month-ends


def _flat_daily_prices(names):
    return pd.DataFrame(100.0, index=DAYS, columns=names)


def test_tradable_on_rebalance_requires_a_price_on_the_months_last_trading_day():
    daily = _flat_daily_prices(["LIVE", "STALE", "NONE"])
    daily.loc["2020-01-29":"2020-01-31", "STALE"] = np.nan  # last January trade 3 days before month-end
    daily.loc["2020-01-01":"2020-01-31", "NONE"] = np.nan   # never traded in January

    tradable = tradable_on_rebalance(daily, pd.DatetimeIndex(["2020-01-31", "2020-04-30"]))

    assert tradable.loc["2020-01-31"].to_dict() == {"LIVE": True, "STALE": False, "NONE": False}
    assert not tradable.loc["2020-04-30"].any()  # no price data in that month at all


def test_name_with_no_entry_price_gets_no_weight_even_with_the_best_score():
    names = TICKERS + ["X"]
    daily = _flat_daily_prices(names)
    daily["X"] = np.nan  # delisted before January: nothing to buy it at on the rebalance date
    scores = pd.DataFrame([list(range(10)) + [99.0]], index=[DATES[0]], columns=names)  # X scores best

    weights = decile_portfolios(scores, n_deciles=10, long_short=True, tradable=tradable_on_rebalance(daily, scores.index))
    row = weights.loc[DATES[0]]

    assert row["X"] == 0.0
    # Deciles are formed from the 10 tradable names only, so T9 is the whole long leg.
    assert row["T9"] == pytest.approx(1.0)
    assert row["T0"] == pytest.approx(-1.0)


def test_name_with_no_exit_price_exits_at_last_traded_price_without_warning():
    daily = _flat_daily_prices(["GONE"])
    daily.loc["2020-02-01":, "GONE"] = np.nan  # traded on the January rebalance date, never again
    monthly = daily.resample("ME").last()
    forward_returns = monthly.pct_change(fill_method=None).shift(-1)
    weights = pd.DataFrame({"GONE": [1.0, 0.0]}, index=DATES[:2])

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # there IS a last traded price to exit at -> no warning
        net = run_backtest(weights, forward_returns, cost_bps=0, prices=monthly)

    # Last traded price == entry price: 0%. The actual delisting payout isn't in the data.
    assert net.loc[DATES[0]] == pytest.approx(0.0)


def test_name_that_delists_mid_holding_period_exits_at_last_trade_and_is_not_held_again():
    names = TICKERS + ["MID"]
    daily = _flat_daily_prices(names)
    daily.loc["2020-02-01":"2020-02-14", "MID"] = 80.0  # falls 20%, then stops trading mid-February
    daily.loc["2020-02-15":, "MID"] = np.nan
    # MID has the best score on both rebalance dates (e.g. forward-filled fundamentals keep it scoring).
    scores = pd.DataFrame([list(range(10)) + [99.0]] * 2, index=DATES[:2], columns=names)

    weights = decile_portfolios(scores, n_deciles=10, long_short=True, tradable=tradable_on_rebalance(daily, scores.index))
    monthly = daily.resample("ME").last()
    forward_returns = monthly.pct_change(fill_method=None).shift(-1)

    assert weights.loc[DATES[0], "MID"] > 0          # traded on January's rebalance date: held
    assert forward_returns.loc[DATES[0], "MID"] == pytest.approx(80.0 / 100.0 - 1.0)  # exit at last trade
    assert weights.loc[DATES[1], "MID"] == 0.0       # no price on February's rebalance date: not held
    assert weights.loc[DATES[1], "T9"] == pytest.approx(1.0)

