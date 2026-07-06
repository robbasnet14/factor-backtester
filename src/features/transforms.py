"""Cross-sectional standardization. STEP 3."""
import pandas as pd

def zscore_cross_section(factor: pd.DataFrame) -> pd.DataFrame:
    """Standardize each date's cross-section to mean 0, std 1 (winsorize first)."""
    raise NotImplementedError("Step 3: implement cross-sectional z-score")

def combine_factors(factors: dict[str, pd.DataFrame], weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Equal-weight (or weighted) composite of standardized factors."""
    raise NotImplementedError("Step 3: implement factor combination")
