"""Backtest loop: weights + returns -> portfolio return series. STEP 4-5."""
import pandas as pd

def run_backtest(weights: pd.DataFrame, forward_returns: pd.DataFrame, cost_bps: float) -> pd.Series:
    """Combine target weights with forward returns, net of costs. Return daily/periodic P&L."""
    raise NotImplementedError("Step 4-5: implement backtest loop")
