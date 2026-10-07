"""Transaction cost model."""
import pandas as pd

COST_MODELS = ("flat", "per_name")


def turnover(weights: pd.DataFrame) -> pd.Series:
    """Per-period turnover: 0.5 * sum(|w_t - w_(t-1)|).

    w_(-1) is treated as all-zero (cash), so the first period correctly
    reflects the turnover of putting on the initial book rather than being
    skipped.
    """
    prev = weights.shift(1).fillna(0.0)
    return 0.5 * (weights - prev).abs().sum(axis=1)


def apply_costs(weights: pd.DataFrame, bps_per_trade: float | pd.DataFrame) -> pd.Series:
    """Per-period transaction cost.

    With a scalar `bps_per_trade` (the flat model): turnover_t * bps / 1e4.
    With a (date x ticker) frame of per-name costs in bps: each name's share
    of turnover, 0.5 * |w_t - w_(t-1)|, is charged at that name's own cost on
    that date — the same turnover convention, so a frame filled with a single
    value gives exactly the flat result. Raises if a traded cell has no cost,
    rather than letting it trade for free.
    """
    if not isinstance(bps_per_trade, pd.DataFrame):
        return turnover(weights) * bps_per_trade / 1e4

    traded = 0.5 * (weights - weights.shift(1).fillna(0.0)).abs()
    bps = bps_per_trade.reindex(index=weights.index, columns=weights.columns)
    uncosted = (traded > 0) & bps.isna()
    if uncosted.to_numpy().any():
        raise ValueError(f"apply_costs: {int(uncosted.to_numpy().sum())} traded cell(s) have no per-name cost")
    return (traded * bps.fillna(0.0)).sum(axis=1) / 1e4


def per_name_cost_bps(
    daily_prices: pd.DataFrame, rebalance_dates: pd.Index, base_bps: float, vol_lookback_days: int = 63
) -> pd.DataFrame:
    """Volatility-scaled per-name costs: base_bps * sigma_i,t / median_j(sigma_j,t).

    sigma is the standard deviation of daily returns over the
    `vol_lookback_days` trading days ending on each month's last trading day
    (prices up to the rebalance date only — no look-ahead). The median name
    on each date pays exactly `base_bps`, so the overall cost level stays
    anchored to the flat model and only its distribution across names
    changes. A name without a full lookback window (e.g. a recent listing)
    pays `base_bps`.

    This is a proxy: it assumes trading cost rises with volatility. Spreads
    are driven mainly by liquidity, which this data doesn't have (no
    volume), so a liquid high-volatility mega-cap is overcharged and an
    illiquid low-volatility name undercharged. It also normalizes within
    each date, so a market-wide crisis doesn't raise costs across the board.
    """
    returns = daily_prices.pct_change(fill_method=None)
    vol = returns.rolling(vol_lookback_days, min_periods=vol_lookback_days).std()

    trading_days = daily_prices.index.to_series()
    last_day = trading_days.groupby(trading_days.dt.to_period("M")).max()
    vol_on_rebalance = vol.loc[last_day.to_numpy()]
    vol_on_rebalance.index = last_day.index  # monthly periods

    relative = vol_on_rebalance.div(vol_on_rebalance.median(axis=1), axis=0)
    rebalance_dates = pd.DatetimeIndex(rebalance_dates)
    relative = relative.reindex(rebalance_dates.to_period("M"))
    relative.index = rebalance_dates
    return (base_bps * relative).fillna(base_bps)


def build_costs(
    cost_cfg: dict, daily_prices: pd.DataFrame, rebalance_dates: pd.Index
) -> float | pd.DataFrame:
    """Cost input for `run_backtest` from the `costs` section of config.yaml:
    the flat bps (cost_model: flat, the default) or a per-name cost frame
    (cost_model: per_name)."""
    model = cost_cfg.get("cost_model", "flat")
    if model == "flat":
        return cost_cfg["bps_per_trade"]
    if model == "per_name":
        return per_name_cost_bps(
            daily_prices, rebalance_dates, cost_cfg["bps_per_trade"], cost_cfg.get("vol_lookback_days", 63)
        )
    raise ValueError(f"costs.cost_model must be one of {COST_MODELS}, got {model!r}")
