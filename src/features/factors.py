"""Cross-sectional factor computation. STEP 3."""
import pandas as pd

def momentum(prices: pd.DataFrame, lookback_m: int = 12, skip_m: int = 1) -> pd.DataFrame:
    """12-1 momentum: return over the past `lookback_m` months, skipping the last `skip_m`."""
    raise NotImplementedError("Step 3: implement momentum")

def value(fundamentals: pd.DataFrame, metric: str = "earnings_yield") -> pd.DataFrame:
    raise NotImplementedError("Step 3: implement value factor")

def quality(fundamentals: pd.DataFrame, metric: str = "roe") -> pd.DataFrame:
    raise NotImplementedError("Step 3: implement quality factor")
