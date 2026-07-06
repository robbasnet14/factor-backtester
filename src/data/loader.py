"""Price & fundamentals ingestion. STEP 2.

Fill these in during Step 2. Return tidy DataFrames indexed by (date, ticker).
"""
import pandas as pd

def load_prices(tickers: list[str], start: str, end: str, cache_dir: str) -> pd.DataFrame:
    """Return adjusted daily prices: index=date, columns=ticker (or long format).
    TODO: pull from provider, adjust for splits/dividends, cache to parquet.
    """
    raise NotImplementedError("Step 2: implement price loading + caching")

def load_fundamentals(tickers: list[str], start: str, end: str, lag_days: int) -> pd.DataFrame:
    """Return point-in-time fundamentals, lagged by `lag_days` to avoid look-ahead.
    TODO: earnings, book value, ROE; align to report/announcement dates.
    """
    raise NotImplementedError("Step 2: implement fundamentals loading + reporting lag")
