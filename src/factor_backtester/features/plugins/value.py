"""Value: higher earnings yield (cheaper per unit of earnings) scores higher."""
import pandas as pd

from factor_backtester.features.factors import value
from factor_backtester.features.registry import register_factor


@register_factor("value", inputs=("fundamentals", "prices"))
def compute(fundamentals: pd.DataFrame, prices: pd.DataFrame, metric: str = "earnings_yield") -> pd.DataFrame:
    return value(fundamentals, prices, metric)
