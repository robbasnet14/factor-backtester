"""Quality: higher return on equity scores higher."""

import pandas as pd

from factor_backtester.features.factors import quality
from factor_backtester.features.registry import register_factor


@register_factor("quality", inputs=("fundamentals",))
def compute(fundamentals: pd.DataFrame, metric: str = "roe") -> pd.DataFrame:
    return quality(fundamentals, metric)
