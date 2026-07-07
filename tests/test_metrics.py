import numpy as np
import pandas as pd
import pytest

from src.analytics.metrics import (
    annualized_return,
    annualized_volatility,
    average_turnover,
    coverage_report,
    deflated_sharpe,
    hit_rate,
    max_drawdown,
    print_coverage_summary,
    sharpe,
)


def test_sharpe_positive():
    r = pd.Series([0.001] * 252)
    assert sharpe(r) > 0


def test_drawdown_nonpositive():
    r = pd.Series([0.01, -0.02, 0.005])
    assert max_drawdown(r) <= 0


def test_annualized_return_compounds_correctly():
    # 1% every month for 12 months should annualize to (1.01**12 - 1), not 12%.
    r = pd.Series([0.01] * 12)
    assert annualized_return(r, periods_per_year=12) == pytest.approx(1.01**12 - 1)


def test_annualized_volatility_scales_by_sqrt_periods():
    r = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01, -0.01])
    assert annualized_volatility(r, periods_per_year=12) == pytest.approx(r.std() * np.sqrt(12))


def test_hit_rate_counts_strictly_positive_periods():
    r = pd.Series([0.01, -0.01, 0.0, 0.02])
    assert hit_rate(r) == pytest.approx(0.5)  # two of four (0.01, 0.02) are > 0


def test_average_turnover_matches_manual_calc():
    weights = pd.DataFrame({"A": [1.0, 1.0, -1.0], "B": [-1.0, -1.0, 1.0]})
    # period 0: from flat -> turnover 1.0; period 1: unchanged -> 0.0; period 2: full flip -> 2.0
    assert average_turnover(weights) == pytest.approx((1.0 + 0.0 + 2.0) / 3)


def test_deflated_sharpe_returns_probability_and_penalizes_more_trials():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.01, 0.03, 120))

    dsr_1 = deflated_sharpe(r, n_trials=1)
    dsr_many = deflated_sharpe(r, n_trials=100)

    assert 0.0 <= dsr_1 <= 1.0
    assert 0.0 <= dsr_many <= 1.0
    # Trying more configurations raises the bar for "genuine skill", so the
    # deflated Sharpe for the same track record must not increase with n_trials.
    assert dsr_many <= dsr_1


def test_deflated_sharpe_rejects_invalid_n_trials():
    r = pd.Series([0.01, 0.02, -0.01, 0.03])
    with pytest.raises(ValueError):
        deflated_sharpe(r, n_trials=0)


def test_deflated_sharpe_nan_on_zero_variance():
    r = pd.Series([0.01, 0.01, 0.01])
    assert deflated_sharpe(r, n_trials=1) != deflated_sharpe(r, n_trials=1)  # NaN != NaN


def test_coverage_report_computes_composite_and_value_quality_fractions():
    dates = pd.to_datetime(["2020-01-31", "2020-02-29"])
    tickers = ["A", "B", "C", "D"]
    nan = np.nan

    composite = pd.DataFrame({"A": [1.0, 1.0], "B": [1.0, 1.0], "C": [1.0, nan], "D": [nan, nan]}, index=dates)
    value = pd.DataFrame({"A": [1.0, 1.0], "B": [1.0, nan], "C": [nan, nan], "D": [nan, nan]}, index=dates)
    quality = pd.DataFrame({"A": [1.0, 1.0], "B": [nan, 1.0], "C": [1.0, nan], "D": [nan, nan]}, index=dates)
    universe = pd.DataFrame(True, index=dates, columns=tickers)

    report = coverage_report(composite, {"value": value, "quality": quality}, universe)

    assert list(report.columns) == ["universe_size", "composite_coverage", "value_quality_coverage"]
    assert report["universe_size"].tolist() == [4, 4]
    assert report.loc[dates[0], "composite_coverage"] == pytest.approx(0.75)  # A, B, C valid
    assert report.loc[dates[1], "composite_coverage"] == pytest.approx(0.5)  # A, B valid
    # value AND quality both valid: only A, on both dates (B/C each miss one factor).
    assert report.loc[dates[0], "value_quality_coverage"] == pytest.approx(0.25)
    assert report.loc[dates[1], "value_quality_coverage"] == pytest.approx(0.25)


def test_coverage_report_denominator_is_universe_membership_not_all_columns():
    dates = pd.to_datetime(["2020-01-31"])
    composite = pd.DataFrame({"A": [1.0], "B": [1.0]}, index=dates)
    universe = pd.DataFrame({"A": [True], "B": [False]}, index=dates)  # B isn't actually in the universe

    report = coverage_report(composite, {}, universe)

    assert report.loc[dates[0], "universe_size"] == 1
    assert report.loc[dates[0], "composite_coverage"] == pytest.approx(1.0)  # only A counts, not B


def test_coverage_report_value_quality_is_nan_when_factors_not_supplied():
    dates = pd.to_datetime(["2020-01-31"])
    composite = pd.DataFrame({"A": [1.0]}, index=dates)
    universe = pd.DataFrame({"A": [True]}, index=dates)

    report = coverage_report(composite, {"momentum": composite}, universe)  # no value/quality passed

    assert report.loc[dates[0], "value_quality_coverage"] != report.loc[dates[0], "value_quality_coverage"]  # NaN


def test_print_coverage_summary_reports_min_and_mean(capsys):
    report = pd.DataFrame(
        {"universe_size": [4, 4], "composite_coverage": [0.75, 0.5], "value_quality_coverage": [0.25, 0.25]},
        index=pd.to_datetime(["2020-01-31", "2020-02-29"]),
    )

    print_coverage_summary(report)

    out = capsys.readouterr().out
    assert "composite score" in out
    assert "value & quality" in out
    assert "50.0%" in out  # min of composite_coverage
    assert "62.5%" in out  # mean of composite_coverage
