"""Performance analytics."""
import numpy as np
import pandas as pd
from scipy import stats

from src.backtest.costs import turnover as _turnover

_EULER_MASCHERONI = 0.5772156649015329


def sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    return float(np.sqrt(periods_per_year) * r.mean() / r.std()) if r.std() else float("nan")


def max_drawdown(returns: pd.Series) -> float:
    curve = (1 + returns.fillna(0)).cumprod()
    return float((curve / curve.cummax() - 1).min())


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Geometric (compounded) annualized return."""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    growth = (1 + r).prod()
    return float(growth ** (periods_per_year / len(r)) - 1)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    return float(r.std() * np.sqrt(periods_per_year)) if len(r) else float("nan")


def hit_rate(returns: pd.Series) -> float:
    """Fraction of periods with a strictly positive return."""
    r = returns.dropna()
    return float((r > 0).mean()) if len(r) else float("nan")


def average_turnover(weights: pd.DataFrame) -> float:
    """Mean per-period turnover (0.5 * sum|delta weight|), independent of cost rate."""
    t = _turnover(weights)
    return float(t.mean()) if len(t) else float("nan")


def deflated_sharpe(returns: pd.Series, n_trials: int) -> float:
    """Deflated Sharpe ratio (Bailey & Lopez de Prado, 2014).

    Answers: given that `n_trials` factor/parameter configurations were
    tried before landing on this one, what's the probability the *true*
    Sharpe ratio is actually positive — after accounting for (a) the
    selection bias of picking the best of `n_trials` tries, and (b) the
    extra estimation noise from non-normal (skewed, fat-tailed) returns?

    Returns a probability in [0, 1]; conventionally >= 0.95 is treated as
    "likely genuine skill, not noise you got to cherry-pick."

    Implementation notes / simplifications:
    - Operates on the return series' own per-period Sharpe ratio (not
      annualized) and its own sample size T — annualizing first would
      distort the significance test.
    - The paper's full method needs the *realized* Sharpe ratio of all
      `n_trials` trials to estimate the variance of the max. Lacking that
      (the usual case — most trials' results aren't kept around), this uses
      the standard practitioner approximation: the standard error of this
      strategy's own Sharpe ratio (Mertens 2002 / Bailey & Lopez de Prado
      2012) stands in for the cross-trial variance.
    """
    if n_trials < 1:
        raise ValueError("deflated_sharpe: n_trials must be >= 1")

    r = returns.dropna()
    t = len(r)
    if t < 2 or r.std() == 0:
        return float("nan")

    sr = r.mean() / r.std()
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)  # Pearson kurtosis: normal = 3

    variance_term = max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12)
    sigma_sr = np.sqrt(variance_term / (t - 1))

    if n_trials > 1:
        sr_benchmark = sigma_sr * (
            (1 - _EULER_MASCHERONI) * stats.norm.ppf(1 - 1 / n_trials)
            + _EULER_MASCHERONI * stats.norm.ppf(1 - 1 / (n_trials * np.e))
        )
    else:
        sr_benchmark = 0.0

    z = (sr - sr_benchmark) / sigma_sr
    return float(stats.norm.cdf(z))


def coverage_report(
    composite: pd.DataFrame,
    factor_frames: dict[str, pd.DataFrame],
    universe: pd.DataFrame,
) -> pd.DataFrame:
    """Per-rebalance-date data coverage against the active universe.

    For each date in `composite`'s index, computes what fraction of that
    date's active universe (from `universe`, a date x ticker boolean
    membership matrix — see `build_universe`) has (a) a valid (non-NaN)
    composite score, and (b) valid `value` AND `quality` factor values
    specifically (if both are present in `factor_frames`).

    (b) exists because momentum only needs price history, which is reliably
    available, while value/quality depend on SEC fundamentals coverage,
    which — as the CIK-resolution and EPS-concept gaps in `load_fundamentals`
    show — is meaningfully less complete. A composite score can look fully
    covered while quietly being carried almost entirely by momentum; this
    number is what tells you that's happening.

    Returns a DataFrame indexed by `composite`'s dates with columns
    [universe_size, composite_coverage, value_quality_coverage] — the full
    per-date detail, for a caller that wants to print a min/mean summary,
    save it, or plot it.
    """
    active = universe.reindex(index=composite.index, method="ffill")
    active = active.reindex(columns=composite.columns, fill_value=False)
    universe_size = active.sum(axis=1)
    safe_denominator = universe_size.replace(0, np.nan)

    composite_valid = composite.notna() & active
    composite_coverage = composite_valid.sum(axis=1) / safe_denominator

    value, quality = factor_frames.get("value"), factor_frames.get("quality")
    if value is not None and quality is not None:
        value_aligned = value.reindex(index=composite.index, columns=composite.columns)
        quality_aligned = quality.reindex(index=composite.index, columns=composite.columns)
        vq_valid = value_aligned.notna() & quality_aligned.notna() & active
        value_quality_coverage = vq_valid.sum(axis=1) / safe_denominator
    else:
        value_quality_coverage = pd.Series(float("nan"), index=composite.index)

    return pd.DataFrame(
        {
            "universe_size": universe_size,
            "composite_coverage": composite_coverage,
            "value_quality_coverage": value_quality_coverage,
        }
    )


def print_coverage_summary(report: pd.DataFrame) -> None:
    """Print the min/mean of a `coverage_report` DataFrame's coverage columns."""
    print("\nCoverage report (fraction of active universe with valid data, per rebalance date):")
    for col, label in [("composite_coverage", "composite score"), ("value_quality_coverage", "value & quality")]:
        series = report[col].dropna()
        if series.empty:
            print(f"  {label:<16} -> no data")
            continue
        print(f"  {label:<16} -> min {series.min():.1%}, mean {series.mean():.1%}")
