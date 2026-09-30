"""Daily prices: `load_prices` asks an ordered chain of `PriceProvider`s for
each ticker and takes the first non-empty answer. By default that's Yahoo
Finance (free, no key), then Tiingo (better coverage of long-delisted names)
when `TIINGO_KEY` is set. Pass `providers=` to use data sources the engine
doesn't ship with; see `src.data.providers.base` for the contract.
"""
import logging
import os
import warnings
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.data.cache import PriceCache, Skiplist, custom_cache_root
from src.data.providers.base import PriceProvider
from src.data.providers.tiingo import TiingoProvider
from src.data.providers.yahoo import YahooProvider

_logger = logging.getLogger(__name__)

_UNAVAILABLE_PRICES_FILENAME = "unavailable_prices.json"


def default_price_providers() -> tuple[PriceProvider, ...]:
    """Yahoo Finance, then Tiingo if `TIINGO_KEY` is set. Without a key
    Tiingo can't answer for any ticker, so it's left out rather than failing
    on every name Yahoo doesn't have."""
    if os.environ.get("TIINGO_KEY"):
        return (YahooProvider(), TiingoProvider())
    _logger.info("TIINGO_KEY is not set; using yfinance only (no Tiingo fallback for delisted names)")
    return (YahooProvider(),)


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
    root = custom_cache_root(cache_dir, chain) if custom else Path(cache_dir)
    cache = PriceCache(root)
    skiplist = Skiplist(None if custom else root / _UNAVAILABLE_PRICES_FILENAME)

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
            if source == "unavailable":
                skiplist.mark(upper)
            continue

        skiplist.clear(upper)  # force_refresh may have proved a previously-unavailable ticker now works

        source_counts[source] += 1
        frames.append(series)

    skiplist.save()
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
