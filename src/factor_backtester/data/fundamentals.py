"""Point-in-time fundamentals: `load_fundamentals` asks one
`FundamentalsProvider` (by default SEC EDGAR) for each ticker and applies the
availability lag. Pass `provider=` to use a different source; see
`factor_backtester.data.providers.base` for the contract.
"""

import warnings
from pathlib import Path

import pandas as pd

from factor_backtester.data.cache import FundamentalsCache, Skiplist, custom_cache_root
from factor_backtester.data.partial import check_complete
from factor_backtester.data.providers.base import FundamentalsProvider
from factor_backtester.data.providers.sec_edgar import SecEdgarProvider
from factor_backtester.data.splits import ttm_fundamentals

_UNAVAILABLE_FUNDAMENTALS_FILENAME = "unavailable_fundamentals.json"


def default_fundamentals_provider(cache_dir: str) -> FundamentalsProvider:
    return SecEdgarProvider(Path(cache_dir) / "sec_company_tickers.json")


def load_fundamentals(
    tickers: list[str],
    start: str,
    end: str,
    lag_days: int,
    cache_dir: str = "data_cache",
    force_refresh: bool = False,
    *,
    splits: pd.DataFrame,
    provider: FundamentalsProvider | None = None,
    allow_partial: bool = False,
) -> pd.DataFrame:
    """Load quarterly fundamentals (TTM earnings, book value, ROE) for `tickers`.

    `splits` is the split history to restate EPS with: a long frame with
    [date, ticker, split_ratio], normally `load_prices` output over a window
    ending on the same date as the prices the EPS will be compared with, and
    starting early enough to cover the oldest quarter in any TTM (about two
    years before `start`). `earnings` is TTM EPS with each quarter restated
    onto the share basis at the end of that window (see `factor_backtester.data.splits`),
    so it's directly comparable with `close`; it's NaN where a quarter can't
    be restated, such as a ticker with no split history.

    By default pulls from SEC EDGAR's XBRL company-facts API (see
    `SecEdgarProvider`, which keeps SEC's ticker->CIK file at
    `cache_dir/sec_company_tickers.json`) and caches one Parquet file per
    ticker (unlagged) in `cache_dir/fundamentals/`. The provider logs, per
    ticker, whether SEC actually returned facts — so a ticker quietly coming
    back empty is always visible in the logs, not just a silently smaller
    output DataFrame. An empty or failed fetch is never cached, so a
    transient failure or a since-fixed extraction bug doesn't permanently
    stick a ticker at "no data."

    A ticker the provider has no data for (for SEC: no CIK, or no usable EPS
    concept) is a permanent kind of failure: with the default provider it's
    recorded in `cache_dir/unavailable_fundamentals.json` and skipped with no
    network call on future calls. Failures that raise (a network hiccup, SEC
    rate limiting) are NOT recorded there, since those are worth retrying
    next run regardless; they stop the load instead: once every ticker has
    been tried, `PartialDataError` names each one that failed, unless
    `allow_partial=True`, which continues without them (see
    `factor_backtester.data.partial`). Pass `force_refresh=True` to re-check every ticker —
    or just delete the file to reset it. An explicit `provider` is cached
    under `cache_dir/providers/<name>/` and doesn't use the skiplist.

    `report_date` is when each figure became public, so it has no look-ahead
    by construction. The `date` column is still `report_date + lag_days` for
    compatibility with the config's `fundamentals_lag_days` (and as a small
    extra buffer for filing-to-availability lag in whatever feed you'd use in
    production) — but `lag_days=0` would already be safe. The final date
    filter (`date` between `start` and `end`, inclusive) is applied on this
    already-lagged `date`, so a row is only ever included if its lagged
    availability date falls on or before the requested `end`.

    Tickers the provider has no data for are skipped with a warning rather
    than failing the whole batch (for SEC, see `SecEdgarProvider` on why
    some have none).
    """
    custom = provider is not None
    if provider is not None:
        root = custom_cache_root(cache_dir, (provider,))
    else:
        root = Path(cache_dir)
        provider = default_fundamentals_provider(cache_dir)

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    cache = FundamentalsCache(root)
    skiplist = Skiplist(None if custom else root / _UNAVAILABLE_FUNDAMENTALS_FILENAME)

    split_history = {str(t).upper(): g.set_index("date")["split_ratio"] for t, g in splits.groupby("ticker")}
    no_history = pd.Series(dtype="float64")

    frames = []
    failures = {}
    for ticker in tickers:
        upper = ticker.upper()
        if not force_refresh and upper in skiplist:
            continue
        try:
            raw = cache.load(provider, ticker)
        except Exception as e:
            # Transient (network error, rate limiting, etc.): not skiplisted,
            # and reported by check_complete below.
            failures[ticker] = f"{provider.name}: {e}"
            continue
        if raw.empty:
            # Permanent: the provider answered and has nothing — worth
            # remembering so we stop asking.
            skiplist.mark(upper)
            warnings.warn(f"Skipping fundamentals for {ticker}: {provider.name} has no data for it")
            continue

        ttm = ttm_fundamentals(raw, split_history.get(upper, no_history))
        frames.append(_lag_and_window(ttm, ticker, lag_days, start_ts, end_ts))
        skiplist.clear(upper)  # force_refresh may have proved a previously-unavailable ticker now works

    skiplist.save()

    if not frames:
        out = pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "report_date",
                "earnings",
                "book_value",
                "roe",
                "shares_outstanding",
                "public_float",
            ]
        )
    else:
        out = pd.concat(frames, ignore_index=True).sort_values(["ticker", "report_date"]).reset_index(drop=True)
    return check_complete(out, failures, allow_partial, "fundamentals")


def _lag_and_window(
    raw: pd.DataFrame, ticker: str, lag_days: int, start_ts: pd.Timestamp, end_ts: pd.Timestamp
) -> pd.DataFrame:
    raw = raw.copy()
    raw["date"] = raw["report_date"] + pd.Timedelta(days=lag_days)
    raw.insert(1, "ticker", ticker.upper())
    # `date` (report_date + lag_days) must fall on or before `end_ts` to be
    # included — this is what keeps a lagged fundamentals row from leaking
    # past the requested window.
    mask = (raw["date"] >= start_ts) & (raw["date"] <= end_ts)
    columns = ["date", "ticker", "report_date", "earnings", "book_value", "roe", "shares_outstanding", "public_float"]
    return raw.loc[mask, columns]
