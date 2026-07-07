"""Transaction cost model."""
import pandas as pd


def turnover(weights: pd.DataFrame) -> pd.Series:
    """Per-period turnover: 0.5 * sum(|w_t - w_(t-1)|).

    w_(-1) is treated as all-zero (cash), so the first period correctly
    reflects the turnover of putting on the initial book rather than being
    skipped.
    """
    prev = weights.shift(1).fillna(0.0)
    return 0.5 * (weights - prev).abs().sum(axis=1)


def apply_costs(weights: pd.DataFrame, bps_per_trade: float) -> pd.Series:
    """Per-period transaction cost: turnover_t * bps_per_trade / 1e4."""
    return turnover(weights) * bps_per_trade / 1e4
