"""Low volatility: lower trailing volatility scores higher.

Volatility is the standard deviation of daily total returns (`adj_close`)
over the `window_days` trading days ending the trading day *before* each
month's last trading day, so the rebalance day's own return is never in the
window. A name without a full window is NaN. The score is minus that
volatility: low volatility is the attractive end (see the registry's sign
convention).
"""
import pandas as pd

from factor_backtester.features.factors import _pivot_prices_wide
from factor_backtester.features.registry import register_factor


@register_factor("low_vol", inputs=("prices",))
def compute(prices: pd.DataFrame, window_days: int = 63) -> pd.DataFrame:
    returns = _pivot_prices_wide(prices).pct_change(fill_method=None)
    volatility = returns.rolling(window_days, min_periods=window_days).std()
    ending_the_day_before = volatility.shift(1)
    return -ending_the_day_before.resample("ME").last()
