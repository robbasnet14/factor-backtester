"""On-disk caching for the data layer.

Caching is orthogonal to where data comes from, so it wraps providers rather
than living inside them: any `PriceProvider` or `FundamentalsProvider` gets
the same Parquet cache without knowing it exists.
"""
import json
import re
import warnings
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.data.providers.base import FundamentalsProvider, PriceProvider, empty_prices

# Per-ticker date range already requested from a provider. A name listed after
# the start date, delisted before the end date, or a start date that falls on a
# market holiday means the cached data can never span the full request, so the
# data alone can't tell a complete cache from an incomplete one.
_PRICE_CACHE_RANGES_FILENAME = "price_cache_ranges.json"


def load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def custom_cache_root(cache_dir: str, providers: Sequence) -> Path:
    """Cache namespace for a caller-supplied chain, so its data never mixes
    with (or gets served from) the default chain's cache."""
    chain_id = "+".join(re.sub(r"[^A-Za-z0-9._-]+", "_", p.name) for p in providers)
    return Path(cache_dir) / "providers" / chain_id


class Skiplist:
    """Tickers confirmed permanently unavailable, persisted as JSON at `path`
    ({TICKER: when it was marked}) so they aren't re-asked over the network
    on every run. Delete the file (or pass force_refresh=True to the loader)
    to give a ticker another chance, e.g. after a real fix upstream (a new
    SEC filing, a Yahoo/Tiingo coverage change). `path=None` gives a
    disabled skiplist that never matches or records anything.
    """

    def __init__(self, path: Path | None):
        self._path = path
        self._entries = load_json(path) if path is not None else {}
        self._changed = False

    def __contains__(self, ticker: str) -> bool:
        return ticker.upper() in self._entries

    def mark(self, ticker: str) -> None:
        if self._path is None:
            return
        self._changed = self._changed or ticker.upper() not in self._entries
        self._entries[ticker.upper()] = pd.Timestamp.now("UTC").isoformat()

    def clear(self, ticker: str) -> None:
        if ticker.upper() in self._entries:
            del self._entries[ticker.upper()]
            self._changed = True

    def save(self) -> None:
        if self._changed:
            save_json(self._path, self._entries)
            self._changed = False


def _ticker_file(directory: Path, ticker: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{ticker.upper()}.parquet"


class PriceCache:
    """Daily prices under `root`: one Parquet file per ticker in `root/prices/`
    shared by every provider in a chain (keyed by the caller's ticker
    spelling, so a rerun reads the cache whichever provider filled it), plus
    `root/price_cache_ranges.json` recording the range already requested for
    each ticker, so a later request inside it is a cache hit even when the
    data itself starts later or ends earlier.

    Each file stores its split history (`split_ratio`) alongside the prices
    it was adjusted with, and the two are only ever replaced together: if a
    fresh download's splits differ from the cached file's (a new split since
    it was cached), the cached rows are discarded rather than merged, since
    every one of them is on the old share basis.

    A download only replaces cached rows if it covers everything cached. A
    partial answer that disagrees with the cache — often a delisted ticker
    the source has since reassigned to a different company, returning only
    the newcomer's recent history — is ignored with a warning and the cached
    series kept, since it's internally consistent and the answer isn't.
    """

    def __init__(self, root: Path):
        self.directory = root / "prices"
        self._ranges_path = root / _PRICE_CACHE_RANGES_FILENAME
        self.ranges = load_json(self._ranges_path)
        self._ranges_on_disk = dict(self.ranges)

    def save(self) -> None:
        if self.ranges != self._ranges_on_disk:
            save_json(self._ranges_path, self.ranges)
            self._ranges_on_disk = dict(self.ranges)

    def load(self, provider: PriceProvider, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """[date, ticker, adj_close, close, split_ratio] for `ticker` in
        [start, end], asking `provider` only when the cache doesn't already
        cover the range.

        `close` comes back on the share basis in effect at `end`: splits the
        source applied after `end` are undone, so the result doesn't depend
        on when the data was downloaded, and the `split_ratio` values inside
        the window are exactly what's needed to put per-share figures on the
        same basis. A series cached before `close` existed, that no provider
        can refresh, comes back with `close` and `split_ratio` missing (NaN)
        rather than guessed.
        """
        path = _ticker_file(self.directory, ticker)
        cached = pd.read_parquet(path) if path.exists() else empty_prices()
        legacy = not cached.empty and "close" not in cached.columns  # written before close/split_ratio existed
        key = ticker.upper()
        recorded = self.ranges.get(key)

        data_covers = not cached.empty and cached["date"].min() <= start and cached["date"].max() >= end
        range_covers = (
            not cached.empty
            and recorded is not None
            and pd.Timestamp(recorded[0]) <= start
            and pd.Timestamp(recorded[1]) >= end
        )

        if (data_covers or range_covers) and not legacy:
            merged = cached
        else:
            fetch_start = min(cached["date"].min(), start) if not cached.empty else start
            fetch_end = max(cached["date"].max(), end) if not cached.empty else end
            fetched = provider.fetch(ticker, fetch_start, fetch_end).sort_values("date").reset_index(drop=True)
            covers = not fetched.empty and (cached.empty or _covers(fetched, cached))
            if fetched.empty:
                merged = cached  # nothing new; a legacy file stays as it is, to be retried next run
            elif legacy and not covers:
                merged = cached  # can't check a partial answer against a file with no split history
            elif cached.empty or covers:
                merged = fetched  # the fresh download is the whole story, whatever its splits
            elif _split_events(cached, fetched["date"].min()) != _split_events(fetched, fetched["date"].min()):
                warnings.warn(
                    f"{provider.name} returned only {fetched['date'].min().date()}..{fetched['date'].max().date()} for "
                    f"{ticker} with a split history that disagrees with the cached series (possibly a reused "
                    "ticker); keeping the cached series."
                )
                merged = cached
            else:
                # Same splits, but the source no longer returns part of the cached
                # history: keep those older rows, and the fresh ones everywhere else.
                merged = (
                    pd.concat([cached, fetched], ignore_index=True)
                    .drop_duplicates(subset="date", keep="last")
                    .sort_values("date")
                    .reset_index(drop=True)
                )
            if merged is not cached:
                merged.to_parquet(path, index=False)
            # Record the range once the ticker has data, whether from this fetch or
            # the cache (e.g. a delisted name Tiingo supplied and Yahoo now returns
            # nothing for). With no data at all it goes on to the next provider /
            # skiplist as before. The end is capped at yesterday so a range reaching
            # into the future gets re-fetched once it has data.
            if not merged.empty:
                covered_end = min(fetch_end, pd.Timestamp.today().normalize() - pd.Timedelta(days=1))
                if recorded is not None:
                    fetch_start = min(fetch_start, pd.Timestamp(recorded[0]))
                    covered_end = max(covered_end, pd.Timestamp(recorded[1]))
                self.ranges[key] = [fetch_start.date().isoformat(), covered_end.date().isoformat()]

        if "close" not in merged.columns:
            merged = merged.assign(close=float("nan"), split_ratio=float("nan"))
        sliced = merged[(merged["date"] >= start) & (merged["date"] <= end)].copy()
        # Put `close` on the basis in effect at `end` by undoing later splits.
        sliced["close"] *= merged.loc[merged["date"] > end, "split_ratio"].prod()
        sliced.insert(1, "ticker", ticker.upper())
        return sliced[["date", "ticker", "adj_close", "close", "split_ratio"]]


def _split_events(prices: pd.DataFrame, since: pd.Timestamp) -> set:
    rows = prices[(prices["date"] >= since) & (prices["split_ratio"] != 1.0) & prices["split_ratio"].notna()]
    return set(zip(rows["date"], rows["split_ratio"]))


_COVER_SLACK = pd.Timedelta(days=7)  # a first/last trading day can differ a little between sources


def _covers(fetched: pd.DataFrame, cached: pd.DataFrame) -> bool:
    """Whether a fresh download spans every cached date (within a few days)."""
    return (
        fetched["date"].min() <= cached["date"].min() + _COVER_SLACK
        and fetched["date"].max() >= cached["date"].max() - _COVER_SLACK
    )


class FundamentalsCache:
    """One Parquet file per ticker in `root/fundamentals/`, holding the
    provider's full, unlagged history.

    NOTE — one-time cache clear: older versions of this code cached
    empty/partial fundamentals fetches (and had a narrower EPS/CIK matching
    path), so a `fundamentals/<TICKER>.parquet` written by them can be stale
    or wrong and won't self-heal — the cache-hit path always wins over
    re-fetching. If a ticker looks like it's missing data it shouldn't be,
    delete its file (or the whole `fundamentals/` directory) once and re-run.
    """

    def __init__(self, root: Path):
        self.directory = root / "fundamentals"

    def load(self, provider: FundamentalsProvider, ticker: str) -> pd.DataFrame:
        path = _ticker_file(self.directory, ticker)
        if path.exists():
            cached = pd.read_parquet(path)
            if "period" in cached.columns:
                return cached
            # Written before as-filed EPS facts were stored (it held a TTM sum
            # that can't be restated across splits), so fetch it again.
        raw = provider.fetch(ticker)
        if not raw.empty:
            raw.to_parquet(path, index=False)
        # An empty/failed fetch is deliberately NOT cached — see the
        # one-time cache-clear note above. A
        # ticker that's temporarily uncovered (network hiccup, a since-fixed
        # extraction gap) gets a fresh shot on the next run instead of being
        # stuck at "no data" forever just because we wrote an empty file once.
        return raw
