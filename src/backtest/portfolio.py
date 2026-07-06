"""Portfolio construction from factor scores. STEP 4."""
import pandas as pd

def decile_portfolios(scores: pd.DataFrame, n_deciles: int = 10, long_short: bool = True) -> pd.DataFrame:
    """Return target weights: long top decile, short bottom decile (if long_short)."""
    raise NotImplementedError("Step 4: implement decile portfolio weights")
