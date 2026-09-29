"""Price & fundamentals loading — the data layer's public entry points.

`load_prices` asks an ordered chain of `PriceProvider`s for each ticker and
takes the first non-empty answer; by default that's Yahoo Finance (free, no
key) followed by Tiingo (better coverage of long-delisted names, needs
`TIINGO_KEY`). `load_fundamentals` asks one `FundamentalsProvider`, by
default SEC EDGAR. Both cache through `src.data.cache` and remember tickers
that are permanently unavailable, so reruns don't keep asking.

Pass your own `providers=` / `provider=` to use a data source the engine
doesn't ship with; see `src.data.providers.base` for the contract. A custom
chain gets its own cache namespace and doesn't use the default chain's
skiplist, since what the default sources lack says nothing about yours.
"""
import logging
import os
import re
import warnings
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.data.cache import FundamentalsCache, PriceCache, load_json, save_json
from src.data.providers.base import FundamentalsProvider, PriceProvider
from src.data.providers.sec_edgar import SecEdgarProvider
from src.data.providers.tiingo import TiingoProvider
from src.data.providers.yahoo import YahooProvider

_logger = logging.getLogger(__name__)

# Persistent skiplists — tickers confirmed permanently unavailable don't get
# re-hit over the network on every future run. Delete the file (or pass
# force_refresh=True) to give a ticker another chance, e.g. after a real fix
# upstream (a new SEC filing, a Yahoo/Tiingo coverage change).
_UNAVAILABLE_PRICES_FILENAME = "unavailable_prices.json"
_UNAVAILABLE_FUNDAMENTALS_FILENAME = "unavailable_fundamentals.json"


def default_price_providers() -> tuple[PriceProvider, ...]:
    """Yahoo Finance, then Tiingo if `TIINGO_KEY` is set. Without a key
    Tiingo can't answer for any ticker, so it's left out rather than failing
    on every name Yahoo doesn't have."""
    if os.environ.get("TIINGO_KEY"):
        return (YahooProvider(), TiingoProvider())
    _logger.info("TIINGO_KEY is not set; using yfinance only (no Tiingo fallback for delisted names)")
    return (YahooProvider(),)


def default_fundamentals_provider(cache_dir: str) -> FundamentalsProvider:
    return SecEdgarProvider(Path(cache_dir) / "sec_company_tickers.json")


def load_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: str,
    force_refresh: bool = False,
    *,
    providers: Sequence[PriceProvider] | None = None,
) -> pd.DataFrame:
    """Load daily adjusted-close prices for `tickers` between `start` and `end`.

    Each ticker goes to the providers in `providers` order (default: Yahoo
    Finance, then Tiingo when `TIINGO_KEY` is set) and the first non-empty
    answer wins. A provider that raises is warned about and the next one is
    tried. All providers
    share one per-ticker Parquet cache in `cache_dir/prices/`, keyed by the
    ORIGINAL ticker spelling — a re-run reads from cache regardless of which
    provider originally supplied it.

    With the default chain, a ticker every provider answered "no data" for
    is recorded in `cache_dir/unavailable_prices.json` and, on future calls,
    skipped with no network call at all — re-fetching a name that's confirmed gone from
    every provider on every run wastes real time on a large universe. Pass
    `force_refresh=True` to re-check every ticker (a previously-unavailable
    one that now succeeds is removed from the skiplist; one that still fails
    stays on it) — or just delete the file to reset it entirely. A ticker
    that came up empty only because a provider failed (network error, rate
    limit, missing key) is never recorded: that's "unknown", not
    "unavailable", and it's retried next run.

    The date range already requested for each cached ticker is recorded in
    `cache_dir/price_cache_ranges.json`, so a later call inside that range is
    served from cache even when the data itself starts later or ends earlier
    (a name listed or delisted mid-period, or a start date on a holiday).

    An explicit `providers` chain is cached under
    `cache_dir/providers/<chain>/` and doesn't use or write the skiplist.

    Returns a long DataFrame with columns [date, ticker, adj_close], sorted
    by (date, ticker). Ends with a one-line log summarizing how many tickers
    came from each provider, including how many were skipped via the skiplist.
    """
    custom = providers is not None
    chain = tuple(providers) if custom else default_price_providers()
    if not chain:
        raise ValueError("load_prices: providers must contain at least one PriceProvider")

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    root = _custom_cache_root(cache_dir, chain) if custom else Path(cache_dir)
    cache = PriceCache(root)
    skiplist_path = root / _UNAVAILABLE_PRICES_FILENAME
    skiplist = {} if custom else load_json(skiplist_path)
    skiplist_changed = False

    frames = []
    source_counts = {**{p.name: 0 for p in chain}, "unavailable": 0, "failed": 0, "skiplisted": 0}

    for ticker in tickers:
        upper = ticker.upper()
        if not force_refresh and upper in skiplist:
            source_counts["skiplisted"] += 1
            continue

        series, source = _first_provider_with_data(chain, cache, ticker, start_ts, end_ts)
        if series is None:
            source_counts[source] += 1
            if source == "unavailable" and not custom:
                skiplist_changed = skiplist_changed or upper not in skiplist
                skiplist[upper] = pd.Timestamp.now("UTC").isoformat()
            continue

        if upper in skiplist:  # force_refresh proved a previously-unavailable ticker now works
            del skiplist[upper]
            skiplist_changed = True

        source_counts[source] += 1
        frames.append(series)

    if skiplist_changed:
        save_json(skiplist_path, skiplist)
    cache.save()

    _logger.info(
        "load_prices summary: %s, %d unavailable, %d failed, %d skiplisted",
        ", ".join(f"{source_counts[p.name]} from {p.name}" for p in chain),
        source_counts["unavailable"],
        source_counts["failed"],
        source_counts["skiplisted"],
    )

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "adj_close"])
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def _first_provider_with_data(
    chain: tuple[PriceProvider, ...], cache: PriceCache, ticker: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp
) -> tuple[pd.DataFrame | None, str]:
    """Walk the chain in order. Returns (series, provider name) from the first
    provider with data; otherwise (None, "unavailable") if every provider
    answered "no data", or (None, "failed") if any of them raised."""
    failed = []
    for i, provider in enumerate(chain):
        try:
            series = cache.load(provider, ticker, start_ts, end_ts)
        except Exception as e:
            failed.append(provider.name)
            next_step = f"trying {chain[i + 1].name}" if i < len(chain) - 1 else "no provider left"
            warnings.warn(f"{provider.name} error for {ticker}: {e}; {next_step}")
            continue
        if not series.empty:
            if i > 0:
                _logger.info("%s: served from %s fallback (%s had no data)", ticker, provider.name, chain[i - 1].name)
            return series, provider.name

    if failed:
        warnings.warn(
            f"Skipping {ticker} this run: {' and '.join(failed)} failed and no other provider had data; "
            "not marking it unavailable, so it's retried next run."
        )
        return None, "failed"
    names = " or ".join(p.name for p in chain)
    warnings.warn(f"No price data returned for {ticker} from {names} (possibly delisted everywhere); skipping.")
    return None, "unavailable"


def load_fundamentals(
    tickers: list[str],
    start: str,
    end: str,
    lag_days: int,
    cache_dir: str = "data_cache",
    force_refresh: bool = False,
    *,
    provider: FundamentalsProvider | None = None,
) -> pd.DataFrame:
    """Load quarterly fundamentals (TTM earnings, book value, ROE) for `tickers`.

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
    next run regardless. Pass `force_refresh=True` to re-check every ticker —
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

    Tickers with no data are skipped with a warning rather than failing the
    whole batch (for SEC, see `SecEdgarProvider` on why some have none).
    """
    custom = provider is not None
    if custom:
        root = _custom_cache_root(cache_dir, (provider,))
    else:
        root = Path(cache_dir)
        provider = default_fundamentals_provider(cache_dir)

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    cache = FundamentalsCache(root)
    skiplist_path = root / _UNAVAILABLE_FUNDAMENTALS_FILENAME
    skiplist = {} if custom else load_json(skiplist_path)
    skiplist_changed = False

    frames = []
    for ticker in tickers:
        upper = ticker.upper()
        if not force_refresh and upper in skiplist:
            continue
        try:
            raw = cache.load(provider, ticker)
        except Exception as e:
            # Transient (network error, rate limiting, etc.) — not skiplisted.
            warnings.warn(f"Skipping fundamentals for {ticker}: {e}")
            continue
        if raw.empty:
            # Permanent: the provider answered and has nothing — worth
            # remembering so we stop asking.
            if not custom:
                skiplist_changed = skiplist_changed or upper not in skiplist
                skiplist[upper] = pd.Timestamp.now("UTC").isoformat()
            warnings.warn(f"Skipping fundamentals for {ticker}: {provider.name} has no data for it")
            continue

        frames.append(_lag_and_window(raw, ticker, lag_days, start_ts, end_ts))
        if upper in skiplist:  # force_refresh proved a previously-unavailable ticker now works
            del skiplist[upper]
            skiplist_changed = True

    if skiplist_changed:
        save_json(skiplist_path, skiplist)

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "report_date", "earnings", "book_value", "roe"])
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["ticker", "report_date"]).reset_index(drop=True)


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
    return raw.loc[mask, ["date", "ticker", "report_date", "earnings", "book_value", "roe"]]


def _custom_cache_root(cache_dir: str, providers: Sequence) -> Path:
    """Cache namespace for a caller-supplied chain, so its data never mixes
    with (or gets served from) the default chain's cache."""
    chain_id = "+".join(re.sub(r"[^A-Za-z0-9._-]+", "_", p.name) for p in providers)
    return Path(cache_dir) / "providers" / chain_id
