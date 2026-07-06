import pandas as pd
from src.analytics.metrics import sharpe, max_drawdown

def test_sharpe_positive():
    r = pd.Series([0.001] * 252)
    assert sharpe(r) > 0

def test_drawdown_nonpositive():
    r = pd.Series([0.01, -0.02, 0.005])
    assert max_drawdown(r) <= 0
