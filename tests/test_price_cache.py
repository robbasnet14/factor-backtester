"""PriceCache: split-aware refreshes and the share basis of `close`."""

import pandas as pd
import pytest

from factor_backtester.data.cache import PriceCache
from tests.fakes import FakePriceProvider

T = pd.Timestamp


def _write_cached(tmp_path, ticker: str, rows: dict[str, float], splits: dict[str, float] | None = None, legacy=False):
    (tmp_path / "prices").mkdir(exist_ok=True)
    df = pd.DataFrame({"date": pd.to_datetime(list(rows)), "adj_close": list(rows.values())})
    if not legacy:
        df["close"] = df["adj_close"]
        df["split_ratio"] = df["date"].map({T(d): r for d, r in (splits or {}).items()}).fillna(1.0)
    df.to_parquet(tmp_path / "prices" / f"{ticker}.parquet", index=False)


def test_a_new_split_replaces_the_cached_series_instead_of_appending_to_it(tmp_path):
    # Cached before a 2-for-1 on 2020-01-15, so every cached close is on the old basis.
    _write_cached(tmp_path, "AAA", {"2020-01-02": 100.0, "2020-01-03": 101.0, "2020-01-06": 102.0})
    # The source has since re-based its whole history for the split, and no longer
    # returns the first two days at all.
    provider = FakePriceProvider(
        "src",
        {"AAA": {"2020-01-06": 51.0, "2020-01-14": 52.0, "2020-01-15": 105.0 / 2, "2020-01-16": 53.0}},
        splits={"AAA": {"2020-01-15": 2.0}},
    )

    df = PriceCache(tmp_path).load(provider, "AAA", T("2020-01-01"), T("2020-01-16"))

    assert provider.calls[0][1] == T("2020-01-01")  # asked for the full span, not just the new dates
    assert T("2020-01-02") not in set(df["date"])  # old-basis rows discarded, not mixed in
    assert df["close"].max() < 60  # everything on the new basis
    assert df.loc[df["date"] == T("2020-01-06"), "close"].item() == pytest.approx(51.0)


def test_with_the_same_splits_older_cached_history_the_source_dropped_is_kept(tmp_path):
    # The source no longer returns December (more than a few days' slack), same splits.
    _write_cached(tmp_path, "AAA", {"2019-12-02": 100.0, "2019-12-03": 101.0, "2020-01-06": 102.0})
    provider = FakePriceProvider("src", {"AAA": {"2020-01-06": 102.0, "2020-01-07": 103.0}})

    df = PriceCache(tmp_path).load(provider, "AAA", T("2019-12-01"), T("2020-01-07"))

    assert df["close"].tolist() == [100.0, 101.0, 102.0, 103.0]


def test_close_is_returned_on_the_basis_in_effect_at_end(tmp_path):
    # The source applied a 4-for-1 dated after `end`; undo it for [start, end].
    provider = FakePriceProvider(
        "src",
        {"AAA": {"2020-01-02": 25.0, "2020-01-03": 26.0, "2020-06-01": 30.0}},
        splits={"AAA": {"2020-06-01": 4.0}},
    )

    df = PriceCache(tmp_path).load(provider, "AAA", T("2020-01-01"), T("2020-01-31"))

    assert df["close"].tolist() == [100.0, 104.0]  # as traded in January
    assert df["adj_close"].tolist() == [25.0, 26.0]  # returns series untouched
    assert (df["split_ratio"] == 1.0).all()  # no split inside the window


def test_a_legacy_cache_file_is_refreshed_into_the_new_format(tmp_path):
    _write_cached(tmp_path, "AAA", {"2020-01-02": 100.0}, legacy=True)
    provider = FakePriceProvider("src", {"AAA": {"2020-01-02": 100.0, "2020-01-03": 101.0}})

    df = PriceCache(tmp_path).load(provider, "AAA", T("2020-01-01"), T("2020-01-03"))

    assert len(provider.calls) == 1
    assert df["close"].tolist() == [100.0, 101.0]
    assert "close" in pd.read_parquet(tmp_path / "prices" / "AAA.parquet").columns


def test_a_legacy_file_no_source_can_refresh_comes_back_with_unknown_close(tmp_path):
    _write_cached(tmp_path, "GONE", {"2020-01-02": 5.0, "2020-01-03": 5.1}, legacy=True)

    df = PriceCache(tmp_path).load(FakePriceProvider("src"), "GONE", T("2020-01-01"), T("2020-01-03"))

    assert df["adj_close"].tolist() == [5.0, 5.1]  # returns still usable
    assert df["close"].isna().all() and df["split_ratio"].isna().all()  # not guessed
    assert "close" not in pd.read_parquet(tmp_path / "prices" / "GONE.parquet").columns  # retried next run


def test_a_partial_answer_with_different_splits_never_replaces_the_cached_series(tmp_path):
    # A delisted ticker the source has since reassigned: it returns only the
    # newcomer's recent history, with a split the cached company never had.
    _write_cached(tmp_path, "OLD", {"2015-01-02": 40.0, "2015-01-05": 41.0, "2016-06-30": 45.0})
    provider = FakePriceProvider(
        "src", {"OLD": {"2026-08-14": 184.0, "2026-08-17": 190.0}}, splits={"OLD": {"2026-08-17": 2.793}}
    )

    with pytest.warns(UserWarning, match="reused ticker"):
        df = PriceCache(tmp_path).load(provider, "OLD", T("2015-01-01"), T("2016-12-31"))

    assert df["close"].tolist() == [40.0, 41.0, 45.0]  # untouched, and not rebased by the foreign split
    assert len(pd.read_parquet(tmp_path / "prices" / "OLD.parquet")) == 3


def test_a_partial_answer_does_not_overwrite_a_legacy_file(tmp_path):
    _write_cached(tmp_path, "OLD", {"2015-01-02": 40.0, "2016-06-30": 45.0}, legacy=True)
    provider = FakePriceProvider("src", {"OLD": {"2026-08-14": 184.0}})

    df = PriceCache(tmp_path).load(provider, "OLD", T("2015-01-01"), T("2016-12-31"))

    assert df["adj_close"].tolist() == [40.0, 45.0] and df["close"].isna().all()  # still legacy, for a later provider
    assert "close" not in pd.read_parquet(tmp_path / "prices" / "OLD.parquet").columns
