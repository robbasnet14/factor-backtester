"""Performance analytics. STEP 6."""
import numpy as np
import pandas as pd

def sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    return float(np.sqrt(periods_per_year) * r.mean() / r.std()) if r.std() else float("nan")

def max_drawdown(returns: pd.Series) -> float:
    curve = (1 + returns.fillna(0)).cumprod()
    return float((curve / curve.cummax() - 1).min())

def deflated_sharpe(returns: pd.Series, n_trials: int) -> float:
    """Deflated Sharpe ratio (Bailey & Lopez de Prado) — adjusts for multiple testing.
    TODO: implement full formula using skew, kurtosis, and n_trials.
    """
    raise NotImplementedError("Step 6: implement deflated Sharpe")
