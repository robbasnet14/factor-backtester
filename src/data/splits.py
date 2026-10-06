"""Restating per-share figures across stock splits.

A split changes the unit a per-share number is quoted in, not the company:
after a 4-for-1, last quarter's $4.00 of EPS is $1.00 per new share. SEC EPS
is filed on the share basis in effect on its filing date, while `close` from
`load_prices` is on the basis in effect at the end of the price window. To
compare them, each quarter's EPS is divided by every split dated after its
own filing date, up to the end of the window — exactly the splits already
applied to `close`. In any EPS/price ratio the adjustment cancels, so it
uses no information that wasn't public at the time.

Combining figures without this mixes bases. Summing four quarters into a
trailing twelve months made AAPL's TTM EPS appear to fall 61% in the year
after its 2020 4-for-1 split, while its earnings didn't; deriving a Q4 as
the fiscal-year total minus Q1-Q3 gave NVDA a loss of -$1.09 for the quarter
after its 2021 split, when it earned $1.18.
"""
import pandas as pd

from src.data.providers.xbrl import fill_missing_q4


def later_split_factor(splits: pd.Series, dates: pd.Series) -> pd.Series:
    """For each date in `dates`, the product of split ratios in `splits`
    (a date-indexed series of daily `split_ratio`, 1.0 on non-split days)
    dated strictly after it. NaN where it can't be known: `splits` is empty
    or has unknown (NaN) entries, or the date is before `splits` begins."""
    if splits.empty or splits.isna().any():
        return pd.Series(float("nan"), index=dates.index)
    splits = splits.sort_index()
    first_known = splits.index.min()
    events = splits[splits != 1.0]

    def factor(date: pd.Timestamp) -> float:
        if date < first_known:
            return float("nan")
        return float(events[events.index > date].prod())

    return dates.map(factor)


def ttm_fundamentals(facts: pd.DataFrame, splits: pd.Series) -> pd.DataFrame:
    """Turn as-filed EPS facts [report_date, period_end, period, eps,
    book_value, roe, shares] (`period` "quarter" or "year") into one row per
    filing date [report_date, earnings, book_value, roe, shares_outstanding],
    where `earnings` is trailing-twelve-month EPS and `shares_outstanding`
    the filing's share count, both restated onto the share basis at the end
    of `splits`: per-share figures are divided by the later splits, share
    counts multiplied by them, so market cap (shares x a `close` on the same
    basis) is unchanged by the restatement.

    Order matters: each fact is restated first, then any quarter reported
    only inside a fiscal-year total is derived (year minus the three prior
    quarters), then four quarters are summed. `earnings` is NaN where any
    figure it depends on can't be restated (see `later_split_factor`) rather
    than combined on mixed bases.
    """
    factor = later_split_factor(splits, facts["report_date"])
    shares = facts["shares"] if "shares" in facts else pd.Series(float("nan"), index=facts.index)
    facts = facts.assign(eps=facts["eps"] / factor, shares=shares * factor)
    as_xbrl = {"period_end": "end", "report_date": "filed", "eps": "val"}
    quarterly = facts[facts["period"] == "quarter"].rename(columns=as_xbrl)[["end", "filed", "val"]]
    annual = facts[facts["period"] == "year"].rename(columns=as_xbrl)[["end", "filed", "val"]]
    q = fill_missing_q4(quarterly.sort_values("end"), annual).rename(columns={v: k for k, v in as_xbrl.items()})

    # book_value / roe are per period end; a derived Q4 takes its fiscal year's.
    per_end = facts.sort_values("period")[["period_end", "book_value", "roe"]].drop_duplicates("period_end")
    q = q.merge(per_end, on="period_end", how="left").sort_values("period_end").reset_index(drop=True)
    q["earnings"] = q["eps"].rolling(4).sum()  # TTM = trailing 4 single-quarter values
    per_filing = facts[["report_date", "shares"]].dropna().drop_duplicates("report_date")
    q = q.merge(per_filing.rename(columns={"shares": "shares_outstanding"}), on="report_date", how="left")

    # A single filing can bundle multiple historical periods in one document
    # (e.g. a 10-K's multi-year "selected quarterly data" table), so several
    # periods can share the exact same `report_date`. Collapse each
    # report_date group to one row — the most recent underlying period — so
    # a single filing date maps to a single TTM figure, not several
    # contradictory ones (ties in `filed` are broken by the newest period,
    # which is what "most recent filing" means once dates are tied).
    q = q.drop_duplicates(subset="report_date", keep="last")
    columns = ["report_date", "earnings", "book_value", "roe", "shares_outstanding"]
    return q[columns].sort_values("report_date").reset_index(drop=True)
