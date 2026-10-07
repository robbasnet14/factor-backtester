"""12-1 momentum: higher trailing return (skipping the latest month) scores higher."""

import pandas as pd

from factor_backtester.features.factors import momentum
from factor_backtester.features.registry import register_factor


@register_factor("momentum", inputs=("prices",))
def compute(prices: pd.DataFrame, lookback_months: int = 12, skip_months: int = 1) -> pd.DataFrame:
    return momentum(prices, lookback_months, skip_months)
