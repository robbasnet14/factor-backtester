"""Size factor: orientation, the share basis, and the public-float gate."""

import numpy as np
import pandas as pd
import pytest

from factor_backtester.features.registry import get_factor

MONTHS = pd.date_range("2020-01-31", periods=3, freq="ME")


def _prices(closes: dict[str, float], adj_scale: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": d, "ticker": t, "adj_close": c * adj_scale, "close": c, "split_ratio": 1.0}
            for t, c in closes.items()
            for d in MONTHS
        ]
    )


def _fundamentals(shares: dict[str, float], public_float: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2020-01-15"),
                "ticker": t,
                "report_date": pd.Timestamp("2019-12-31"),
                "earnings": 1.0,
                "book_value": 1.0,
                "roe": 0.1,
                "shares_outstanding": s,
                "public_float": public_float.get(t, np.nan),
            }
            for t, s in shares.items()
        ]
    )


def test_the_smallest_market_cap_gets_the_highest_score():
    shares = {"SMALL": 1e6, "MID": 1e7, "LARGE": 1e8}
    fund = _fundamentals(shares, {t: 0.8 * s * 50.0 for t, s in shares.items()})
    scores = get_factor("size").compute(fund, _prices({t: 50.0 for t in shares})).loc[MONTHS[1]]
    assert scores.idxmax() == "SMALL" and scores.idxmin() == "LARGE"


def test_market_cap_uses_close_not_the_dividend_adjusted_price():
    fund = _fundamentals({"AAA": 1e6}, {"AAA": 0.8 * 1e6 * 50.0})
    scores = get_factor("size").compute(fund, _prices({"AAA": 50.0}, adj_scale=0.5))
    assert scores.loc[MONTHS[1], "AAA"] == pytest.approx(-np.log(1e6 * 50.0))


def test_the_gate_keeps_crashes_and_insider_held_names_and_drops_data_errors():
    cap = 1e6 * 50.0  # every name: 1M shares at 50
    floats = {
        "OK": 0.8 * cap,  # cap 1.25x float
        "CRASHED": cap / 0.04,  # cap 0.04x float: a 96% fall since the float date (NKTR-like): kept
        "INSIDERS": cap / 22.0,  # cap 22x float: founder holdings outside SEC's float (NKE-like): kept
        "TOO_SMALL": cap / 0.001,  # cap 0.001x float: one class priced as another (BRK.B-like): dropped
        "TOO_LARGE": cap / 2000,  # cap 2,000x float: a count scaled by 1,000: dropped
    }
    names = list(floats) + ["NO_FLOAT"]
    scores = (
        get_factor("size")
        .compute(_fundamentals({t: 1e6 for t in names}, floats), _prices({t: 50.0 for t in names}))
        .loc[MONTHS[1]]
    )

    for kept in ("OK", "CRASHED", "INSIDERS"):
        assert not np.isnan(scores[kept]), kept
    for dropped in ("TOO_SMALL", "TOO_LARGE", "NO_FLOAT"):
        assert np.isnan(scores[dropped]), dropped


def test_the_band_is_configurable():
    cap = 1e6 * 50.0
    fund = _fundamentals({"AAA": 1e6}, {"AAA": cap / 0.04})
    assert np.isnan(get_factor("size").compute(fund, _prices({"AAA": 50.0}), float_floor=0.5).loc[MONTHS[1], "AAA"])
