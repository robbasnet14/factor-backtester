"""Point-in-time, survivorship-bias-free equity universe.

For name="SP500", membership history comes from the public, free
fja05680/sp500 GitHub dataset ("S&P 500 Historical Components & Changes",
https://github.com/fja05680/sp500) — a community-maintained log of every
addition/removal to the index since 1996, including names that were later
delisted or acquired. Each row is a full membership snapshot as of that
date; a ticker suffix like "-201503" marks a *known future* removal date
and is stripped here, since a ticker's presence in a row already means it
was a member on that date — the actual removal is reflected by its absence
from the next logged snapshot.
"""
import re
from pathlib import Path

import pandas as pd
import requests

_SP500_SOURCE_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes.csv"
)
_DELISTING_SUFFIX = re.compile(r"-\d{6}$")
_REQUEST_TIMEOUT = 30


def build_universe(
    name: str,
    start: str,
    end: str,
    cache_dir: str = "data_cache",
    source_url: str = _SP500_SOURCE_URL,
) -> pd.DataFrame:
    """Return a boolean membership matrix: index=date, columns=ticker.

    True where `ticker` was a member of index `name` on that business day.
    Includes historically delisted/acquired tickers — no survivorship bias.

    Only name="SP500" is implemented (see module docstring for the data
    source). The change log is downloaded once and cached to
    `cache_dir/universe/`; delete that file to pick up upstream updates.
    `start`/`end` bound the returned business-day calendar — membership at
    `start` is taken from the most recent logged change on or before it, so
    the first row is correct even if `start` falls between two changes.
    """
    if name != "SP500":
        raise NotImplementedError(f"build_universe: universe {name!r} is not implemented (only 'SP500' is)")

    changes = _load_sp500_membership_changes(cache_dir, source_url)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    prior = changes.index[changes.index <= start_ts]
    window_start = prior.max() if len(prior) else changes.index.min()
    window = changes.loc[(changes.index >= window_start) & (changes.index <= end_ts)]
    if window.empty:
        window = changes.iloc[[0]]

    all_tickers = sorted(set().union(*window["tickers"]))
    snapshots = pd.DataFrame(False, index=window.index, columns=all_tickers)
    for dt, tickers in window["tickers"].items():
        snapshots.loc[dt, list(tickers)] = True

    calendar = pd.bdate_range(start_ts, end_ts)
    combined = snapshots.reindex(snapshots.index.union(calendar)).ffill().fillna(False)
    return combined.reindex(calendar).astype(bool)


def _load_sp500_membership_changes(cache_dir: str, source_url: str) -> pd.DataFrame:
    """Return the raw change log: index=change_date, column 'tickers' =
    frozenset of tickers active from that date until the next change.
    """
    cache_path = Path(cache_dir) / "universe" / "sp500_membership_changes.csv"
    if cache_path.exists():
        raw = pd.read_csv(cache_path)
    else:
        resp = requests.get(source_url, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(resp.content)
        raw = pd.read_csv(cache_path)

    raw["date"] = pd.to_datetime(raw["date"])
    raw["tickers"] = raw["tickers"].apply(_parse_ticker_list)
    return raw.set_index("date").sort_index()


def _parse_ticker_list(cell: str) -> frozenset:
    if not isinstance(cell, str) or not cell.strip():
        return frozenset()
    return frozenset(_DELISTING_SUFFIX.sub("", t.strip()) for t in cell.split(",") if t.strip())
