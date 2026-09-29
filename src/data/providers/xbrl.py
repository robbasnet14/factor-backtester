"""Pure parsing of SEC EDGAR XBRL "company facts" into point-in-time
fundamentals — no I/O, so it can be tested directly against fixture JSON.

`earnings` is trailing-twelve-month (TTM) diluted EPS: the sum of the last 4
*single-quarter* `EarningsPerShareDiluted` values as of each filing, not
just the latest quarter. This matters because a single quarter's EPS is
noisy and seasonal (e.g. retailers' Q4) — TTM smooths that out and is the
conventional denominator for an earnings yield. `book_value` is the latest
`StockholdersEquity` (total, not per-share) and `roe` is TTM
`NetIncomeLoss` / that same `StockholdersEquity` snapshot.
"""
import pandas as pd

ACCEPTED_FORMS = ("10-Q", "10-K", "10-Q/A", "10-K/A")
_QUARTER_MIN_DAYS = 80
_QUARTER_MAX_DAYS = 100
_ANNUAL_MIN_DAYS = 340
_ANNUAL_MAX_DAYS = 380

# EPS concept fallback chain, tried in order — most filers use the first,
# but plenty of simpler capital structures only tag BasicAndDiluted or Basic.
EPS_CONCEPTS = ("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic")


def first_usable_eps(us_gaap: dict) -> tuple[pd.DataFrame | None, str | None]:
    """Try each concept in `EPS_CONCEPTS` in order, filling any missing Q4
    from the annual figure, and return the first that yields any quarterly
    observations at all.
    """
    for concept in EPS_CONCEPTS:
        quarterly, annual = extract_duration_facts(us_gaap.get(concept, {}))
        filled = fill_missing_q4(quarterly, annual)
        if not filled.empty:
            return filled, concept
    return None, None


def fundamentals_from_facts(us_gaap: dict, eps_q: pd.DataFrame) -> pd.DataFrame:
    """Combine quarterly EPS (from `first_usable_eps`) with net income and
    stockholders' equity into [report_date, earnings, book_value, roe]."""
    ni_quarterly, ni_annual = extract_duration_facts(us_gaap.get("NetIncomeLoss", {}))
    ni_q = fill_missing_q4(ni_quarterly, ni_annual)
    equity_q = extract_instant_facts(us_gaap.get("StockholdersEquity", {}))

    eps_q = eps_q.copy()
    eps_q["earnings"] = eps_q["val"].rolling(4).sum()  # TTM = trailing 4 single-quarter values
    ni_q["ttm_net_income"] = ni_q["val"].rolling(4).sum()

    result = eps_q[["end", "filed", "earnings"]].merge(ni_q[["end", "ttm_net_income"]], on="end", how="left")
    result = result.merge(equity_q[["end", "val"]].rename(columns={"val": "book_value"}), on="end", how="left")
    result["roe"] = result["ttm_net_income"] / result["book_value"]

    result = result.rename(columns={"filed": "report_date"})
    result["report_date"] = pd.to_datetime(result["report_date"])

    # A single filing can bundle multiple historical periods in one document
    # (e.g. a 10-K's multi-year "selected quarterly data" table), so several
    # `end` periods can share the exact same `report_date`. Collapse each
    # report_date group to one row — the most recent underlying period — so
    # a single filing date maps to a single TTM figure, not several
    # contradictory ones (ties in `filed` are broken by the newest `end`,
    # which is what "most recent filing" means once dates are tied).
    result = result.sort_values("end").drop_duplicates(subset="report_date", keep="last")

    return result[["report_date", "earnings", "book_value", "roe"]].sort_values("report_date").reset_index(drop=True)


def fill_missing_q4(quarterly: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    """Derive a missing Q4 as (annual FY total) - (the 3 quarters immediately
    before the fiscal year end), for any fiscal year where we have an annual
    figure but no matching discrete-quarter one. Some filers only tag a
    single 3-month figure for Q1-Q3 and report Q4 solely as part of the
    full-year 10-K total, which would otherwise leave a permanent gap in
    every trailing-twelve-month sum that includes that quarter.
    """
    if annual.empty:
        return quarterly
    existing_ends = set(quarterly["end"])
    derived_rows = []
    for _, arow in annual.iterrows():
        if arow["end"] in existing_ends:
            continue
        prior = quarterly[quarterly["end"] < arow["end"]].sort_values("end").tail(3)
        if len(prior) != 3 or (arow["end"] - prior["end"].min()).days > 400:
            continue  # not enough of, or too stale a, Q1-Q3 run to derive Q4 from
        derived_rows.append({"end": arow["end"], "filed": arow["filed"], "val": arow["val"] - prior["val"].sum()})

    if not derived_rows:
        return quarterly
    combined = pd.concat([quarterly, pd.DataFrame(derived_rows)], ignore_index=True)
    return combined.sort_values("end").reset_index(drop=True)


def extract_duration_facts(concept_facts: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a duration concept's (has a `start`, e.g. NetIncomeLoss, EPS)
    facts into (quarterly ~3-month, annual ~12-month) observations, each
    deduped to one row per period `end`. Only `ACCEPTED_FORMS` are kept.
    Periods that are neither ~quarterly nor ~annual (e.g. 6- or 9-month
    YTD) are dropped — mixing those into a "quarterly" sum would silently
    corrupt it.
    """
    quarterly_rows, annual_rows = [], []
    for unit_values in concept_facts.get("units", {}).values():
        for item in unit_values:
            if item.get("form") not in ACCEPTED_FORMS:
                continue
            end, filed, val, start = item.get("end"), item.get("filed"), item.get("val"), item.get("start")
            if end is None or filed is None or val is None or start is None:
                continue
            duration_days = (pd.Timestamp(end) - pd.Timestamp(start)).days
            row = {"end": pd.Timestamp(end), "filed": pd.Timestamp(filed), "val": float(val)}
            if _QUARTER_MIN_DAYS <= duration_days <= _QUARTER_MAX_DAYS:
                quarterly_rows.append(row)
            elif _ANNUAL_MIN_DAYS <= duration_days <= _ANNUAL_MAX_DAYS:
                annual_rows.append(row)
    return _dedupe_by_end(quarterly_rows), _dedupe_by_end(annual_rows)


def extract_instant_facts(concept_facts: dict) -> pd.DataFrame:
    """Flatten an instant concept's (no `start`, e.g. StockholdersEquity)
    facts into one point-in-time observation per period `end`.
    """
    rows = []
    for unit_values in concept_facts.get("units", {}).values():
        for item in unit_values:
            if item.get("form") not in ACCEPTED_FORMS:
                continue
            end, filed, val = item.get("end"), item.get("filed"), item.get("val")
            if end is None or filed is None or val is None:
                continue
            rows.append({"end": pd.Timestamp(end), "filed": pd.Timestamp(filed), "val": float(val)})
    return _dedupe_by_end(rows)


def _dedupe_by_end(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["end", "filed", "val"])
    # A period can be re-disclosed in a later filing (e.g. as a prior-year
    # comparative); keep the earliest `filed` date so `report_date` reflects
    # when a figure FIRST became public, not a later restatement.
    df = pd.DataFrame(rows).sort_values("filed").drop_duplicates(subset="end", keep="first")
    return df.sort_values("end").reset_index(drop=True)
