"""On-disk caching for the data layer.

Caching is orthogonal to where data comes from, so it wraps providers rather
than living inside them: any `PriceProvider` or `FundamentalsProvider` gets
the same Parquet cache without knowing it exists.
"""
import json
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
        """[date, ticker, adj_close] for `ticker` in [start, end], asking
        `provider` only when the cache doesn't already cover the range."""
        path = _ticker_file(self.directory, ticker)
        cached = pd.read_parquet(path) if path.exists() else empty_prices()
        key = ticker.upper()
        recorded = self.ranges.get(key)

        data_covers = not cached.empty and cached["date"].min() <= start and cached["date"].max() >= end
        range_covers = (
            not cached.empty
            and recorded is not None
            and pd.Timestamp(recorded[0]) <= start
            and pd.Timestamp(recorded[1]) >= end
        )

        if data_covers or range_covers:
            merged = cached
        else:
            fetch_start = min(cached["date"].min(), start) if not cached.empty else start
            fetch_end = max(cached["date"].max(), end) if not cached.empty else end
            fetched = provider.fetch(ticker, fetch_start, fetch_end)
            merged = (
                pd.concat([cached, fetched], ignore_index=True)
                .drop_duplicates(subset="date")
                .sort_values("date")
                .reset_index(drop=True)
            )
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

        sliced = merged[(merged["date"] >= start) & (merged["date"] <= end)].copy()
        sliced.insert(1, "ticker", ticker.upper())
        return sliced[["date", "ticker", "adj_close"]]


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
            return pd.read_parquet(path)
        raw = provider.fetch(ticker)
        if not raw.empty:
            raw.to_parquet(path, index=False)
        # An empty/failed fetch is deliberately NOT cached — see the
        # one-time cache-clear note above. A
        # ticker that's temporarily uncovered (network hiccup, a since-fixed
        # extraction gap) gets a fresh shot on the next run instead of being
        # stuck at "no data" forever just because we wrote an empty file once.
        return raw
