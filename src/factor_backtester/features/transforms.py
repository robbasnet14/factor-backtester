"""Cross-sectional standardization and factor combination."""
import numpy as np
import pandas as pd


def zscore_cross_section(factor: pd.DataFrame, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.DataFrame:
    """Standardize each date's cross-section to mean 0, std 1 (winsorize first).

    Winsorizing at `lower_q`/`upper_q` per row (date) before standardizing
    keeps a handful of extreme outliers from dominating the composite score.
    Rows with fewer than 2 non-NaN values, or with zero cross-sectional
    variance, come back as NaN — there's nothing to standardize against.
    """
    lower = factor.quantile(lower_q, axis=1)
    upper = factor.quantile(upper_q, axis=1)
    winsorized = factor.clip(lower=lower, upper=upper, axis=0)

    mean = winsorized.mean(axis=1)
    std = winsorized.std(axis=1).replace(0, np.nan)
    return winsorized.sub(mean, axis=0).div(std, axis=0)


def combine_factors(factors: dict[str, pd.DataFrame], weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Weighted composite of standardized factors, averaging over whichever
    factors are non-NaN for a given (date, ticker) rather than propagating
    NaN whenever any single factor is missing.

    `weights` defaults to equal weight across `factors`. Weights are
    re-normalized per cell by the weights of the factors actually present,
    so a cell where one of three factors is missing still gets a sensible
    two-factor average instead of going NaN or silently under-weighting.
    """
    if not factors:
        raise ValueError("combine_factors: no factors provided")
    if weights is None:
        weights = {name: 1.0 / len(factors) for name in factors}
    unknown = set(weights) - set(factors)
    if unknown:
        raise ValueError(f"combine_factors: weights reference unknown factors: {unknown}")

    all_index = sorted(set().union(*(f.index for f in factors.values())))
    all_columns = sorted(set().union(*(f.columns for f in factors.values())))

    weighted_sum = pd.DataFrame(0.0, index=all_index, columns=all_columns)
    weight_total = pd.DataFrame(0.0, index=all_index, columns=all_columns)

    for name, w in weights.items():
        aligned = factors[name].reindex(index=all_index, columns=all_columns)
        present = aligned.notna()
        weighted_sum += aligned.fillna(0.0) * w
        weight_total += present * w

    return weighted_sum / weight_total.replace(0.0, np.nan)
