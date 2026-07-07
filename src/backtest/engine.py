"""Backtest loop: weights + returns -> portfolio return series."""
import logging
import warnings

import pandas as pd

from src.backtest.costs import apply_costs

_logger = logging.getLogger(__name__)


def run_backtest(
    weights: pd.DataFrame,
    forward_returns: pd.DataFrame,
    cost_bps: float,
    prices: pd.DataFrame | None = None,
) -> pd.Series:
    """Combine target weights with forward returns, net of costs.

    No look-ahead: `forward_returns.loc[t]` must already be the return
    earned FROM rebalance date `t` TO the next one (e.g.
    `monthly_price.pct_change().shift(-1)`), never the return that produced
    the score `weights.loc[t]` was decided on. Pairing weights with
    contemporaneous (unshifted) returns here would silently reintroduce
    look-ahead bias.

    A held position (nonzero weight) can have a NaN forward return when its
    ticker stops appearing in the price data mid-holding-period — a
    delisting, acquisition, or bankruptcy. Silently treating that as a flat
    0% return is a well-known backtesting bias: it erases whatever gain or
    loss a real investor would actually have taken. If `prices` (a wide
    date x ticker panel of raw price LEVELS, same date/ticker convention as
    `weights`) is supplied, a missing forward return for a held position is
    instead filled by exiting at that ticker's LAST available price anywhere
    in `prices` — i.e. "sell at the last price we ever saw" rather than
    "assume nothing happened."

    Documented limitation: if `prices` isn't supplied, or a ticker has no
    recorded price at all to exit at, that cell still falls back to a 0%
    assumption with a warning. There is nothing to exit at in that case, and
    the true delisting outcome (a merger payout, a bankruptcy wipeout, or
    anything between) is genuinely unknown from price data alone — this is
    not modeled, and the warning is how that limitation stays visible
    instead of silently understating risk.

    Returns a periodic net-of-cost P&L series indexed by rebalance date `t`,
    representing the return earned by the position established at `t`.
    """
    index = weights.index.intersection(forward_returns.index)
    columns = weights.columns.intersection(forward_returns.columns)
    w = weights.loc[index, columns]
    r = forward_returns.loc[index, columns].copy()

    missing_before = r.isna() & (w != 0)
    n_missing_before = int(missing_before.to_numpy().sum())

    if prices is not None and n_missing_before:
        exit_returns = _delisting_exit_returns(missing_before, prices, index, columns)
        r = r.where(~missing_before, exit_returns)

    still_missing = r.isna() & (w != 0)
    n_still_missing = int(still_missing.to_numpy().sum())
    n_exited = n_missing_before - n_still_missing
    if n_exited:
        _logger.info("%d held position(s) exited at their last available price instead of assuming 0%%", n_exited)
    if n_still_missing:
        warnings.warn(
            f"{n_still_missing} held position(s) had no forward return AND no exit price available "
            "(e.g. a ticker that never appears again in the price data) — treated as a flat 0% return "
            "for that period. This is a documented limitation: the true delisting outcome (merger "
            "payout, bankruptcy wipeout, etc.) is unknown from price data alone and is not modeled."
        )

    gross_pnl = (w.fillna(0.0) * r.fillna(0.0)).sum(axis=1)
    cost = apply_costs(w.fillna(0.0), cost_bps)
    return (gross_pnl - cost).rename("net_return")


def _delisting_exit_returns(
    missing_return: pd.DataFrame, prices: pd.DataFrame, index: pd.Index, columns: pd.Index
) -> pd.DataFrame:
    """For each (date, ticker) flagged True in `missing_return`, the return
    from that ticker's price at `date` to the last non-null price it has
    ANYWHERE in `prices` (forward-filled through the end of the panel).
    Cells with no entry price or no exit price at all come back NaN, and the
    caller's 0%-fallback + warning still applies to those.
    """
    aligned_prices = prices.reindex(index=index, columns=columns)
    last_known_price = aligned_prices.ffill().iloc[-1]  # per-ticker, most recent price seen anywhere in the panel
    entry_price = aligned_prices.where(missing_return)  # only the cells that actually need an exit return
    return entry_price.rdiv(last_known_price) - 1.0
