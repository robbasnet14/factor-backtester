"""Price & fundamentals ingestion — free, keyless-first sources.

Prices come from Yahoo Finance via `yfinance` (auto-adjusted for splits and
dividends, no API key) for the common case. When Yahoo has nothing for a
ticker — a frequent problem for long-delisted or acquired names — we fall
back to Tiingo's Daily Prices API, which has much better historical
coverage of delisted tickers but requires a `TIINGO_KEY` env var. Because
the fallback only fires when yfinance comes up empty, most runs never touch
Tiingo or need a key at all.

Fundamentals come from SEC EDGAR's public XBRL "company facts" API (also
keyless, but SEC requires a descriptive `User-Agent` with contact info on
every request, and rate-limits abusive callers — see `_SEC_HEADERS`).

`earnings` is trailing-twelve-month (TTM) diluted EPS: the sum of the last 4
*single-quarter* `EarningsPerShareDiluted` values as of each filing, not
just the latest quarter. This matters because a single quarter's EPS is
noisy and seasonal (e.g. retailers' Q4) — TTM smooths that out and is the
conventional denominator for an earnings yield. `book_value` is the latest
`StockholdersEquity` (total, not per-share) and `roe` is TTM
`NetIncomeLoss` / that same `StockholdersEquity` snapshot.
"""
import json
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Callable

import pandas as pd
import requests
import yfinance as yf

_logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 30


def _cache_path(cache_dir: str, subdir: str, ticker: str) -> Path:
    d = Path(cache_dir) / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{ticker.upper()}.parquet"


def _empty_price_frame() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "adj_close": pd.Series(dtype="float64")})


# ---------------------------------------------------------------------------
# Persistent skiplists — tickers confirmed permanently unavailable don't get
# re-hit over the network on every future run. Delete the file (or pass
# force_refresh=True) to give a ticker another chance, e.g. after a real fix
# upstream (a new SEC filing, a Yahoo/Tiingo coverage change).
# ---------------------------------------------------------------------------


def _load_skiplist(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_skiplist(path: Path, skiplist: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(skiplist, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Prices — Yahoo Finance via yfinance, with a Tiingo fallback for coverage
# ---------------------------------------------------------------------------

_YFINANCE_MAX_RETRIES = 3
_YFINANCE_RETRY_BACKOFF_SECONDS = 1.0
_YFINANCE_POLITE_DELAY_SECONDS = 0.3

_TIINGO_BASE = "https://api.tiingo.com"
_TIINGO_MAX_RETRIES = 5
_TIINGO_RETRY_BASE_SECONDS = 60  # 429 backoff: 60s, 120s, 240s, ...
_TIINGO_POLITE_DELAY_SECONDS = 0.5

_UNAVAILABLE_PRICES_FILENAME = "unavailable_prices.json"


def _to_yahoo_symbol(ticker: str) -> str:
    """Yahoo uses `-` for share classes where most other sources use `.` (BRK.B -> BRK-B)."""
    return ticker.upper().replace(".", "-")


def load_prices(tickers: list[str], start: str, end: str, cache_dir: str, force_refresh: bool = False) -> pd.DataFrame:
    """Load daily adjusted-close prices for `tickers` between `start` and `end`.

    Tries Yahoo Finance (`yfinance`, `auto_adjust=True` so `Close` is already
    split/dividend-adjusted) first. If Yahoo has no data for a ticker, falls
    back to Tiingo's Daily Prices API (`TIINGO_KEY` env var required only
    when this fallback actually fires), which covers many names Yahoo has
    dropped entirely. Either source caches to the same per-ticker Parquet
    file in `cache_dir/prices/`, keyed by the ORIGINAL ticker spelling — a
    re-run reads from cache regardless of which source originally supplied
    it.

    A ticker with no data from EITHER source is recorded in
    `cache_dir/unavailable_prices.json` and, on future calls, skipped with no
    network call at all — re-fetching a name that's confirmed gone from both
    providers on every run wastes real time on a large universe. Pass
    `force_refresh=True` to re-check every ticker (a previously-unavailable
    one that now succeeds is removed from the skiplist; one that still fails
    stays on it) — or just delete the file to reset it entirely.

    Returns a long DataFrame with columns [date, ticker, adj_close], sorted
    by (date, ticker). Ends with a one-line log summarizing how many tickers
    came from each source, including how many were skipped via the skiplist.
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    skiplist_path = Path(cache_dir) / _UNAVAILABLE_PRICES_FILENAME
    skiplist = _load_skiplist(skiplist_path)
    skiplist_changed = False

    frames = []
    source_counts = {"yfinance": 0, "tiingo": 0, "unavailable": 0, "skiplisted": 0}

    for i, ticker in enumerate(tickers):
        upper = ticker.upper()
        if not force_refresh and upper in skiplist:
            source_counts["skiplisted"] += 1
            continue

        if i > 0:
            time.sleep(_YFINANCE_POLITE_DELAY_SECONDS)

        series, source = _load_price_series_with_fallback(ticker, start_ts, end_ts, cache_dir)
        if series is None:
            source_counts["unavailable"] += 1
            skiplist_changed = skiplist_changed or upper not in skiplist
            skiplist[upper] = pd.Timestamp.now("UTC").isoformat()
            continue

        if upper in skiplist:  # force_refresh proved a previously-unavailable ticker now works
            del skiplist[upper]
            skiplist_changed = True

        source_counts[source] += 1
        frames.append(series)

    if skiplist_changed:
        _save_skiplist(skiplist_path, skiplist)

    _logger.info(
        "load_prices summary: %d from yfinance, %d from Tiingo fallback, %d unavailable, %d skiplisted",
        source_counts["yfinance"],
        source_counts["tiingo"],
        source_counts["unavailable"],
        source_counts["skiplisted"],
    )

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "adj_close"])
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def _load_price_series_with_fallback(
    ticker: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp, cache_dir: str
) -> tuple[pd.DataFrame | None, str | None]:
    try:
        series = _load_one_price_series(
            ticker, start_ts, end_ts, cache_dir, lambda s, e: _fetch_yfinance_prices(_to_yahoo_symbol(ticker), s, e)
        )
    except Exception as e:
        warnings.warn(f"yfinance error for {ticker}: {e}; trying Tiingo fallback")
        series = pd.DataFrame(columns=["date", "ticker", "adj_close"])

    if not series.empty:
        return series, "yfinance"

    # Yahoo had nothing — fall back to Tiingo, which covers many long-delisted names.
    try:
        series = _load_one_price_series(
            ticker, start_ts, end_ts, cache_dir, lambda s, e: _fetch_tiingo_prices(ticker, s, e)
        )
    except Exception as e:
        warnings.warn(f"Skipping {ticker}: no data from yfinance, and Tiingo fallback failed ({e})")
        return None, None

    if series.empty:
        warnings.warn(f"No price data returned for {ticker} from yfinance or Tiingo (possibly delisted everywhere); skipping.")
        return None, None

    _logger.info("%s: served from Tiingo fallback (yfinance had no data)", ticker)
    return series, "tiingo"


def _load_one_price_series(
    ticker: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    cache_dir: str,
    fetch_fn: Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame],
) -> pd.DataFrame:
    path = _cache_path(cache_dir, "prices", ticker)
    cached = pd.read_parquet(path) if path.exists() else _empty_price_frame()

    if not cached.empty and cached["date"].min() <= start_ts and cached["date"].max() >= end_ts:
        merged = cached
    else:
        fetch_start = min(cached["date"].min(), start_ts) if not cached.empty else start_ts
        fetch_end = max(cached["date"].max(), end_ts) if not cached.empty else end_ts
        fetched = fetch_fn(fetch_start, fetch_end)
        merged = (
            pd.concat([cached, fetched], ignore_index=True)
            .drop_duplicates(subset="date")
            .sort_values("date")
            .reset_index(drop=True)
        )
        merged.to_parquet(path, index=False)

    sliced = merged[(merged["date"] >= start_ts) & (merged["date"] <= end_ts)].copy()
    sliced.insert(1, "ticker", ticker.upper())
    return sliced[["date", "ticker", "adj_close"]]


def _fetch_yfinance_prices(symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    df = None
    for attempt in range(_YFINANCE_MAX_RETRIES):
        try:
            df = yf.download(
                symbol,
                start=start_ts.date().isoformat(),
                end=(end_ts + pd.Timedelta(days=1)).date().isoformat(),  # yfinance's end is exclusive
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            break
        except Exception:
            if attempt == _YFINANCE_MAX_RETRIES - 1:
                raise
            time.sleep(_YFINANCE_RETRY_BACKOFF_SECONDS * (attempt + 1))

    if df is None or df.empty:
        return _empty_price_frame()

    if isinstance(df.columns, pd.MultiIndex):  # yfinance always uses (field, ticker) columns
        df.columns = df.columns.get_level_values(0)

    out = df[["Close"]].rename(columns={"Close": "adj_close"})
    out.index.name = "date"
    out = out.reset_index()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    return out[["date", "adj_close"]]


def _tiingo_headers() -> dict:
    key = os.environ.get("TIINGO_KEY")
    if not key:
        raise EnvironmentError(
            "TIINGO_KEY is not set — required for the Tiingo fallback used when yfinance has no "
            "data for a ticker. Sign up at https://www.tiingo.com and run: export TIINGO_KEY=your_key_here"
        )
    return {"Content-Type": "application/json", "Authorization": f"Token {key}"}


def _tiingo_get(url: str, params: dict) -> list:
    """GET a Tiingo URL. 404 (no data for this ticker) returns `[]` — a real,
    non-retried "nothing here." 429 (rate limited) waits with exponential
    backoff (60s, 120s, 240s, ...) and retries; any other error raises.
    """
    headers = _tiingo_headers()
    last_error = None
    for attempt in range(_TIINGO_MAX_RETRIES):
        resp = requests.get(url, headers=headers, params=params, timeout=_REQUEST_TIMEOUT)
        if resp.status_code == 404:
            return []
        if resp.status_code == 429:
            last_error = requests.HTTPError(f"429 rate limited fetching {url}")
            if attempt < _TIINGO_MAX_RETRIES - 1:
                backoff = _TIINGO_RETRY_BASE_SECONDS * (2**attempt)
                _logger.info("Tiingo rate limit hit; waiting %ds before retry %d/%d", backoff, attempt + 2, _TIINGO_MAX_RETRIES)
                time.sleep(backoff)
                continue
            raise last_error
        resp.raise_for_status()  # other 4xx/5xx fail immediately, no retry
        time.sleep(_TIINGO_POLITE_DELAY_SECONDS)
        return resp.json()
    raise last_error  # pragma: no cover — loop always returns or raises above


def _fetch_tiingo_prices(ticker: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    url = f"{_TIINGO_BASE}/tiingo/daily/{ticker}/prices"
    params = {
        "startDate": start_ts.date().isoformat(),
        "endDate": end_ts.date().isoformat(),
        "format": "json",
    }
    rows = _tiingo_get(url, params)
    if not rows:
        return _empty_price_frame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df = df.rename(columns={"adjClose": "adj_close"})
    return df[["date", "adj_close"]]


# ---------------------------------------------------------------------------
# Fundamentals — SEC EDGAR XBRL company facts
# ---------------------------------------------------------------------------

_SEC_HEADERS = {"User-Agent": "factor-backtester rob basnet basnetrob@gmail.com"}
_SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
_SEC_MAX_RETRIES = 3
_SEC_RETRY_BACKOFF_SECONDS = 1.0
_SEC_POLITE_DELAY_SECONDS = 0.2

_ACCEPTED_FORMS = ("10-Q", "10-K", "10-Q/A", "10-K/A")
_QUARTER_MIN_DAYS = 80
_QUARTER_MAX_DAYS = 100
_ANNUAL_MIN_DAYS = 340
_ANNUAL_MAX_DAYS = 380

# EPS concept fallback chain, tried in order — most filers use the first,
# but plenty of simpler capital structures only tag BasicAndDiluted or Basic.
_EPS_CONCEPTS = ("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic")

# SEC's own https://www.sec.gov/files/company_tickers.json omits some real,
# actively-filing tickers outright (WBA — Walgreens Boots Alliance, CIK
# 1618921 — isn't in the file at all as of this writing) and aliases others
# under a ticker SEC uses internally instead of the exchange ticker (MMC —
# Marsh & McLennan — is listed there under "MRSH", not "MMC"). Verified by
# direct lookup against SEC's own file and companyfacts API. Add entries
# here as more gaps like this turn up; there's no general fix since the gap
# is in SEC's source data, not in how we're matching it.
_CIK_OVERRIDES = {
    "MMC": 62709,
    "WBA": 1618921,
}

_UNAVAILABLE_FUNDAMENTALS_FILENAME = "unavailable_fundamentals.json"


def load_fundamentals(
    tickers: list[str],
    start: str,
    end: str,
    lag_days: int,
    cache_dir: str = "data_cache",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load quarterly fundamentals (TTM earnings, book value, ROE) for `tickers`.

    Pulls from SEC EDGAR's XBRL company-facts API and caches one Parquet
    file per ticker (unlagged) in `cache_dir/fundamentals/`, plus a shared
    ticker->CIK lookup cached at `cache_dir/sec_company_tickers.json`. Logs,
    per ticker, whether SEC actually returned facts — so a ticker quietly
    coming back empty is always visible in the logs, not just a silently
    smaller output DataFrame. A failed or empty fetch is never cached (see
    `_load_one_fundamentals_series`), so a transient failure or a since-fixed
    extraction bug doesn't permanently stick a ticker at "no data."

    A ticker with no SEC CIK or no usable EPS concept at all — a permanent,
    not transient, kind of failure — is recorded in
    `cache_dir/unavailable_fundamentals.json` and skipped with no network
    call on future calls. Transient failures (a network hiccup, SEC rate
    limiting) are NOT recorded there, since those are worth retrying next
    run regardless. Pass `force_refresh=True` to re-check every ticker
    (fixes itself if a ticker was skiplisted, then later gets a CIK/EPS
    concept it didn't have before) — or just delete the file to reset it.

    NOTE — one-time cache clear: this function used to cache empty/partial
    fundamentals fetches (and had a narrower EPS/CIK matching path), so a
    `cache_dir/fundamentals/<TICKER>.parquet` written by an older version of
    this code can be stale or wrong and won't self-heal — the cache-hit path
    always wins over re-fetching. If a ticker looks like it's missing data it
    shouldn't be, delete its file (or the whole `data_cache/fundamentals/`
    directory) once and re-run.

    Unlike a period-end-date proxy, SEC's `filed` date is exactly when a
    10-Q/10-K's numbers actually became public, so `report_date` here has no
    look-ahead by construction. The `date` column is still `report_date +
    lag_days` for compatibility with the config's `fundamentals_lag_days`
    (and as a small extra buffer for filing-to-availability lag in whatever
    feed you'd use in production) — but `lag_days=0` would already be safe.
    The final date filter (`date` between `start` and `end`, inclusive) is
    applied on this already-lagged `date`, so a row is only ever included if
    its lagged availability date falls on or before the requested `end`.

    Tickers with no SEC CIK, no filings, or missing concepts are skipped
    with a warning rather than failing the whole batch. Some tickers
    genuinely have no usable EPS concept in SEC's structured company-facts
    data at all — notably companies with multiple share classes (e.g. V,
    STZ), which often tag per-share figures through a company-specific XBRL
    extension that this API doesn't expose. That's a real upstream data
    gap, not a bug here; it shows up as a skip warning and as reduced
    coverage in `coverage_report`, not a silent wrong number.
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    ticker_to_cik = _load_sec_ticker_to_cik_map(cache_dir)

    skiplist_path = Path(cache_dir) / _UNAVAILABLE_FUNDAMENTALS_FILENAME
    skiplist = _load_skiplist(skiplist_path)
    skiplist_changed = False

    frames = []
    for ticker in tickers:
        upper = ticker.upper()
        if not force_refresh and upper in skiplist:
            continue
        try:
            frames.append(_load_one_fundamentals_series(ticker, start_ts, end_ts, lag_days, cache_dir, ticker_to_cik))
        except ValueError as e:
            # Permanent failure (no CIK, or no usable EPS concept in any of
            # _EPS_CONCEPTS) — worth remembering so we stop asking SEC.
            skiplist_changed = skiplist_changed or upper not in skiplist
            skiplist[upper] = pd.Timestamp.now("UTC").isoformat()
            warnings.warn(f"Skipping fundamentals for {ticker}: {e}")
            continue
        except Exception as e:
            # Transient (network error, rate limiting, etc.) — not skiplisted.
            warnings.warn(f"Skipping fundamentals for {ticker}: {e}")
            continue

        if upper in skiplist:  # force_refresh proved a previously-unavailable ticker now works
            del skiplist[upper]
            skiplist_changed = True

    if skiplist_changed:
        _save_skiplist(skiplist_path, skiplist)

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "report_date", "earnings", "book_value", "roe"])
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["ticker", "report_date"]).reset_index(drop=True)


def _load_one_fundamentals_series(
    ticker: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp, lag_days: int, cache_dir: str, ticker_to_cik: dict
) -> pd.DataFrame:
    path = _cache_path(cache_dir, "fundamentals", ticker)
    if path.exists():
        raw = pd.read_parquet(path)
    else:
        raw = _fetch_sec_fundamentals(ticker, ticker_to_cik)
        if not raw.empty:
            raw.to_parquet(path, index=False)
        # An empty/failed fetch is deliberately NOT cached — see the
        # one-time cache-clear note in `load_fundamentals`'s docstring. A
        # ticker that's temporarily uncovered (network hiccup, a since-fixed
        # extraction gap) gets a fresh shot on the next run instead of being
        # stuck at "no data" forever just because we wrote an empty file once.

    raw = raw.copy()
    raw["date"] = raw["report_date"] + pd.Timedelta(days=lag_days)
    raw.insert(1, "ticker", ticker.upper())
    # `date` (report_date + lag_days) must fall on or before `end_ts` to be
    # included — this is what keeps a lagged fundamentals row from leaking
    # past the requested window.
    mask = (raw["date"] >= start_ts) & (raw["date"] <= end_ts)
    return raw.loc[mask, ["date", "ticker", "report_date", "earnings", "book_value", "roe"]]


def _sec_get(url: str) -> dict:
    """GET a SEC EDGAR URL with the required User-Agent, retrying 429/5xx
    with backoff and a polite delay after every request that goes through.
    """
    last_error = None
    for attempt in range(_SEC_MAX_RETRIES):
        resp = requests.get(url, headers=_SEC_HEADERS, timeout=_REQUEST_TIMEOUT)
        if resp.status_code == 429 or resp.status_code >= 500:
            last_error = requests.HTTPError(f"{resp.status_code} error fetching {url}")
            if attempt < _SEC_MAX_RETRIES - 1:
                time.sleep(_SEC_RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue
            raise last_error
        resp.raise_for_status()  # other 4xx (e.g. 404) fail immediately, no retry
        time.sleep(_SEC_POLITE_DELAY_SECONDS)
        return resp.json()
    raise last_error  # pragma: no cover — loop always returns or raises above


def _load_sec_ticker_to_cik_map(cache_dir: str) -> dict:
    """Rebuild the ticker->CIK map from SEC's company_tickers.json: an exact,
    uppercase-ticker match against `{index: {cik_str, ticker, title}}`, plus
    `_CIK_OVERRIDES` for the handful of real tickers SEC's own file omits or
    aliases (see that dict's docstring comment for specifics).
    """
    path = Path(cache_dir) / "sec_company_tickers.json"
    if path.exists():
        raw = json.loads(path.read_text())
    else:
        raw = _sec_get(_SEC_TICKERS_URL)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw))
    mapping = {entry["ticker"].upper(): int(entry["cik_str"]) for entry in raw.values()}
    mapping.update(_CIK_OVERRIDES)
    return mapping


def debug_print_resolved_ciks(tickers: list[str] = ("AAPL", "V", "MMC", "WBA"), cache_dir: str = "data_cache") -> dict:
    """Manual sanity check: resolve and print CIKs for the given tickers.

    Run directly (`python -m src.data.loader`) to eyeball that ticker->CIK
    resolution is working for known-tricky names. Returns the resolved map
    so it's also usable from a test/REPL.
    """
    ticker_to_cik = _load_sec_ticker_to_cik_map(cache_dir)
    resolved = {t: _resolve_cik(t, ticker_to_cik) for t in tickers}
    for ticker, cik in resolved.items():
        print(f"{ticker}: {'CIK ' + str(cik) if cik is not None else 'UNRESOLVED'}")
    return resolved


def _resolve_cik(ticker: str, ticker_to_cik: dict) -> int | None:
    upper = ticker.upper()
    for candidate in (upper, upper.replace(".", "-"), upper.replace("-", ".")):
        if candidate in ticker_to_cik:
            return ticker_to_cik[candidate]
    return None


def _fetch_sec_fundamentals(ticker: str, ticker_to_cik: dict) -> pd.DataFrame:
    cik = _resolve_cik(ticker, ticker_to_cik)
    if cik is None:
        _logger.info("%s: no SEC CIK found", ticker)
        raise ValueError(f"no SEC CIK found for ticker {ticker!r}")

    facts = _sec_get(_SEC_FACTS_URL.format(cik=cik))
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    if not us_gaap:
        _logger.info("%s: SEC returned no us-gaap facts (CIK %d)", ticker, cik)
    else:
        _logger.info("%s: SEC returned %d us-gaap concepts (CIK %d)", ticker, len(us_gaap), cik)

    eps_q, eps_concept_used = _first_usable_eps(us_gaap)
    if eps_q is None:
        _logger.info(
            "%s: no usable EPS facts in any of %s (CIK %d) — often a multi-share-class company that "
            "tags EPS via a custom XBRL extension SEC's company-facts API doesn't expose",
            ticker,
            _EPS_CONCEPTS,
            cik,
        )
        raise ValueError(f"no usable EPS facts for {ticker!r} (CIK {cik}) in any of {_EPS_CONCEPTS}")
    _logger.info("%s: using %s for EPS (%d quarterly observations)", ticker, eps_concept_used, len(eps_q))

    ni_quarterly, ni_annual = _extract_duration_facts(us_gaap.get("NetIncomeLoss", {}))
    ni_q = _fill_missing_q4(ni_quarterly, ni_annual)
    equity_q = _extract_instant_facts(us_gaap.get("StockholdersEquity", {}))

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


def _first_usable_eps(us_gaap: dict) -> tuple[pd.DataFrame | None, str | None]:
    """Try each concept in `_EPS_CONCEPTS` in order, filling any missing Q4
    from the annual figure, and return the first that yields any quarterly
    observations at all.
    """
    for concept in _EPS_CONCEPTS:
        quarterly, annual = _extract_duration_facts(us_gaap.get(concept, {}))
        filled = _fill_missing_q4(quarterly, annual)
        if not filled.empty:
            return filled, concept
    return None, None


def _fill_missing_q4(quarterly: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
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


def _dedupe_by_end(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["end", "filed", "val"])
    # A period can be re-disclosed in a later filing (e.g. as a prior-year
    # comparative); keep the earliest `filed` date so `report_date` reflects
    # when a figure FIRST became public, not a later restatement.
    df = pd.DataFrame(rows).sort_values("filed").drop_duplicates(subset="end", keep="first")
    return df.sort_values("end").reset_index(drop=True)


def _extract_duration_facts(concept_facts: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a duration concept's (has a `start`, e.g. NetIncomeLoss, EPS)
    facts into (quarterly ~3-month, annual ~12-month) observations, each
    deduped to one row per period `end`. Only `_ACCEPTED_FORMS` are kept.
    Periods that are neither ~quarterly nor ~annual (e.g. 6- or 9-month
    YTD) are dropped — mixing those into a "quarterly" sum would silently
    corrupt it.
    """
    quarterly_rows, annual_rows = [], []
    for unit_values in concept_facts.get("units", {}).values():
        for item in unit_values:
            if item.get("form") not in _ACCEPTED_FORMS:
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


def _extract_instant_facts(concept_facts: dict) -> pd.DataFrame:
    """Flatten an instant concept's (no `start`, e.g. StockholdersEquity)
    facts into one point-in-time observation per period `end`.
    """
    rows = []
    for unit_values in concept_facts.get("units", {}).values():
        for item in unit_values:
            if item.get("form") not in _ACCEPTED_FORMS:
                continue
            end, filed, val = item.get("end"), item.get("filed"), item.get("val")
            if end is None or filed is None or val is None:
                continue
            rows.append({"end": pd.Timestamp(end), "filed": pd.Timestamp(filed), "val": float(val)})
    return _dedupe_by_end(rows)


if __name__ == "__main__":
    debug_print_resolved_ciks()
