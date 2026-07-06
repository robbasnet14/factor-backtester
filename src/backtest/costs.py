"""Transaction cost model. STEP 5."""
import pandas as pd

def apply_costs(weights: pd.DataFrame, bps_per_trade: float) -> pd.Series:
    """Given a weight path, compute per-period cost = turnover * bps. Returns a cost series."""
    raise NotImplementedError("Step 5: implement cost + turnover")
