"""Restating EPS across stock splits (factor_backtester/data/splits.py)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factor_backtester.data.providers.xbrl import first_usable_eps, fundamentals_from_facts
from factor_backtester.data.splits import later_split_factor, ttm_fundamentals

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "sec_eps_aapl_nvda_2018_2022.json").read_text())


def _split_history(events: dict[str, float], start: str = "2017-01-01", end: str = "2023-12-31") -> pd.Series:
    days = pd.bdate_range(start, end)
    ratios = pd.Series(list(events.values()), index=pd.to_datetime(list(events)), dtype=float)
    return pd.Series(1.0, index=days).mul(ratios, fill_value=1.0)


def _facts(rows: list[tuple]) -> pd.DataFrame:
    """(report_date, period_end, period, eps) tuples -> as-filed facts."""
    df = pd.DataFrame(rows, columns=["report_date", "period_end", "period", "eps"])
    df[["report_date", "period_end"]] = df[["report_date", "period_end"]].apply(pd.to_datetime)
    return df.assign(book_value=np.nan, roe=np.nan)


def _ttm_from_sec_fixture(ticker: str, splits: pd.Series) -> pd.Series:
    quarterly, annual, _ = first_usable_eps(FIXTURE[ticker])
    ttm = ttm_fundamentals(fundamentals_from_facts(FIXTURE[ticker], quarterly, annual), splits)
    return ttm.set_index("report_date")["earnings"].dropna()


def test_split_factor_counts_only_splits_strictly_after_each_date():
    splits = _split_history({"2020-08-31": 4.0, "2021-07-20": 2.0})
    dates = pd.Series(pd.to_datetime(["2020-07-31", "2020-08-31", "2021-01-28", "2021-08-01"]))

    assert later_split_factor(splits, dates).tolist() == [8.0, 2.0, 2.0, 1.0]


def test_split_factor_is_unknown_before_the_history_starts_or_when_it_has_gaps():
    splits = _split_history({}, start="2019-01-01")
    assert np.isnan(later_split_factor(splits, pd.Series([pd.Timestamp("2018-12-31")]))[0])

    unknown = splits.copy()
    unknown.iloc[5] = np.nan  # a series cached before split data existed
    assert later_split_factor(unknown, pd.Series([pd.Timestamp("2020-01-01")])).isna().all()


def test_each_quarter_is_restated_only_by_splits_after_its_own_filing():
    # Q1-Q2 filed before a 4-for-1, Q3-Q4 after (already on the new basis as filed).
    facts = _facts([
        ("2020-05-01", "2020-03-31", "quarter", 4.0),
        ("2020-08-01", "2020-06-30", "quarter", 4.0),
        ("2020-11-01", "2020-09-30", "quarter", 1.0),
        ("2021-02-01", "2020-12-31", "quarter", 1.0),
    ])
    ttm = ttm_fundamentals(facts, _split_history({"2020-09-15": 4.0}))

    assert ttm["earnings"].iloc[-1] == pytest.approx(4.0)  # 1 + 1 + 1 + 1, not 4 + 4 + 1 + 1
    # Without the split in the history, nothing is restated (no look-ahead from later splits).
    assert ttm_fundamentals(facts, _split_history({}))["earnings"].iloc[-1] == pytest.approx(10.0)


def test_q4_is_derived_after_restating_not_before():
    # Q4 appears only inside the fiscal-year total, which was filed after a
    # 4-for-1 split while Q1 was filed before it (NVDA FY2022's shape).
    facts = _facts([
        ("2021-05-26", "2021-05-02", "quarter", 3.03),
        ("2021-08-20", "2021-08-01", "quarter", 0.94),
        ("2021-11-22", "2021-10-31", "quarter", 0.97),
        ("2022-03-18", "2022-01-30", "year", 3.85),
    ])
    ttm = ttm_fundamentals(facts, _split_history({"2021-07-20": 4.0}))

    assert ttm.set_index("report_date").loc["2022-03-18", "earnings"] == pytest.approx(3.85)
    # i.e. the derived Q4 is 3.85 - (3.03/4 + 0.94 + 0.97) = 1.1825, not -1.09


def test_with_no_splits_the_result_is_the_plain_trailing_sum():
    facts = _facts([(f"2020-{m:02d}-15", f"2020-{m - 1:02d}-28", "quarter", float(m)) for m in (2, 5, 8, 11)])
    ttm = ttm_fundamentals(facts, _split_history({}))

    assert ttm["earnings"].iloc[-1] == pytest.approx(2 + 5 + 8 + 11)
    assert ttm["earnings"].iloc[:3].isna().all()


def test_aapl_ttm_eps_is_smooth_across_its_2020_split():
    # Real SEC filings. As filed, TTM EPS "fell" 61% over the year after the
    # 2020-08-31 4-for-1, only because pre- and post-split quarters were summed.
    as_filed = _ttm_from_sec_fixture("AAPL", _split_history({}))
    restated = _ttm_from_sec_fixture("AAPL", _split_history({"2020-08-31": 4.0}))

    window = slice("2020-01-01", "2021-12-31")
    assert (as_filed[window].pct_change().dropna() < -0.15).any()     # the old artifact
    assert (restated[window].pct_change().dropna() > -0.10).all()     # gone after restating
    assert restated["2020-10-30"] == pytest.approx(3.26, abs=0.01)   # FY2020; the 10-K reports $3.28 for the year
    assert restated["2021-10-29"] == pytest.approx(5.61, abs=0.01)   # FY2021, matching the 10-K


def test_nvda_ttm_eps_is_smooth_across_its_2021_split():
    as_filed = _ttm_from_sec_fixture("NVDA", _split_history({}))
    restated = _ttm_from_sec_fixture("NVDA", _split_history({"2021-07-20": 4.0}))

    window = slice("2021-01-01", "2022-06-30")
    assert (as_filed[window].pct_change().dropna() < -0.15).any()
    assert (restated[window].pct_change().dropna() > -0.10).all()
    assert restated["2022-03-18"] == pytest.approx(3.85, abs=0.01)  # FY2022, matching the 10-K


def test_a_later_split_in_the_window_rescales_everything_by_exactly_its_ratio():
    # NVDA's 2024 10-for-1 is a pure change of units for every earlier figure.
    to_2023 = _ttm_from_sec_fixture("NVDA", _split_history({"2021-07-20": 4.0}, end="2023-12-31"))
    to_2024 = _ttm_from_sec_fixture("NVDA", _split_history({"2021-07-20": 4.0, "2024-06-10": 10.0}, end="2024-12-31"))

    pd.testing.assert_series_equal(to_2024, to_2023 / 10.0)
