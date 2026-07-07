"""Entry point: wire everything together. Grows as you complete each step."""
import argparse
from pathlib import Path

import pandas as pd

from src.analytics.metrics import (
    annualized_return,
    annualized_volatility,
    average_turnover,
    coverage_report,
    deflated_sharpe,
    hit_rate,
    max_drawdown,
    print_coverage_summary,
    sharpe,
)
from src.analytics.plots import plot_equity_curve
from src.backtest.engine import run_backtest
from src.backtest.portfolio import decile_portfolios
from src.backtest.validation import walk_forward_backtest
from src.data.loader import load_fundamentals, load_prices
from src.data.universe import build_universe
from src.features.factors import momentum, quality, value
from src.features.transforms import combine_factors, zscore_cross_section
from src.utils.config import load_config

PERIODS_PER_YEAR = 12  # monthly rebalance


def print_summary(title: str, returns: pd.Series, weights: pd.DataFrame, n_trials: int):
    weights_realized = weights.reindex(returns.index)
    summary = {
        "Annualized return": f"{annualized_return(returns, PERIODS_PER_YEAR):.2%}",
        "Annualized volatility": f"{annualized_volatility(returns, PERIODS_PER_YEAR):.2%}",
        "Sharpe ratio": f"{sharpe(returns, PERIODS_PER_YEAR):.2f}",
        "Max drawdown": f"{max_drawdown(returns):.2%}",
        "Hit rate": f"{hit_rate(returns):.2%}",
        "Avg monthly turnover": f"{average_turnover(weights_realized):.2%}",
        f"Deflated Sharpe (n_trials={n_trials})": f"{deflated_sharpe(returns, n_trials):.3f}",
    }
    label_width = max(len(k) for k in summary)
    print(f"\n{title} ({len(returns)} periods)")
    print("-" * (label_width + 12))
    for label, value_str in summary.items():
        print(f"{label:<{label_width}}  {value_str:>10}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    uni_cfg, data_cfg = cfg["universe"], cfg["data"]
    factor_cfg, port_cfg, cost_cfg = cfg["factors"], cfg["portfolio"], cfg["costs"]
    start, end, cache_dir = uni_cfg["start_date"], uni_cfg["end_date"], data_cfg["cache_dir"]

    print("Loaded config:", uni_cfg["name"], start, "->", end)

    universe = build_universe(uni_cfg["name"], start, end, cache_dir=cache_dir)
    tickers = sorted(universe.columns[universe.any(axis=0)])
    print(f"Universe: {len(tickers)} distinct tickers over the period")

    prices = load_prices(tickers, start, end, cache_dir=cache_dir)
    fundamentals = load_fundamentals(tickers, start, end, lag_days=data_cfg["fundamentals_lag_days"], cache_dir=cache_dir)

    factor_frames = {}
    if factor_cfg["momentum"]["enabled"]:
        mom_cfg = factor_cfg["momentum"]
        factor_frames["momentum"] = zscore_cross_section(
            momentum(prices, mom_cfg["lookback_months"], mom_cfg["skip_months"])
        )
    if factor_cfg["value"]["enabled"]:
        factor_frames["value"] = zscore_cross_section(value(fundamentals, prices, factor_cfg["value"]["metric"]))
    if factor_cfg["quality"]["enabled"]:
        factor_frames["quality"] = zscore_cross_section(quality(fundamentals, factor_cfg["quality"]["metric"]))
    if not factor_frames:
        raise ValueError("No factors enabled in config.yaml")

    composite = combine_factors(factor_frames)

    # Restrict scores to the point-in-time universe: a ticker only gets a
    # score on dates it was actually a member of the index (no survivorship bias).
    monthly_membership = universe.reindex(composite.index, method="ffill")
    monthly_membership = monthly_membership.reindex(columns=composite.columns, fill_value=False)
    composite = composite.where(monthly_membership)

    weights = decile_portfolios(composite, n_deciles=port_cfg["n_deciles"], long_short=port_cfg["long_short"])

    # Forward monthly returns: the return earned FROM each rebalance date TO
    # the next one, so weights decided at t never see the return that produced them.
    monthly_prices = prices.pivot(index="date", columns="ticker", values="adj_close").resample("ME").last()
    forward_returns = monthly_prices.pct_change().shift(-1)
    # The final rebalance date has no realized forward return yet (data just
    # ends) — drop it rather than let the engine score it as a 0% period,
    # which would charge turnover cost for a position with no offsetting P&L.
    forward_returns = forward_returns.dropna(how="all")

    # Passing raw price levels lets the engine exit a delisted/acquired name
    # at its last available price instead of silently assuming it earned 0%.
    net_returns = run_backtest(
        weights, forward_returns, cost_bps=cost_cfg["bps_per_trade"], prices=monthly_prices
    ).dropna()

    out_path = Path("outputs") / "net_returns.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    net_returns.to_csv(out_path, header=True)
    print(f"Backtest complete: {len(net_returns)} periods -> {out_path}")

    n_trials = cfg.get("validation", {}).get("n_trials", 1)
    print_summary("Full-period summary (IN-SAMPLE — reference only, not the headline number)", net_returns, weights, n_trials)

    report = coverage_report(composite, factor_frames, universe)
    report.to_csv(Path("outputs") / "coverage_report.csv")
    print_coverage_summary(report)

    wf_cfg = cfg.get("validation", {}).get("walk_forward", {})
    oos_returns, folds = walk_forward_backtest(
        weights,
        forward_returns,
        cost_bps=cost_cfg["bps_per_trade"],
        start=start,
        end=end,
        initial_train_months=wf_cfg.get("initial_train_months", 60),
        test_months=wf_cfg.get("test_months", 12),
        embargo_months=wf_cfg.get("embargo_months", 1),
        prices=monthly_prices,
    )
    print(f"\nWalk-forward folds: {len(folds)}")
    for i, fold in enumerate(folds):
        print(
            f"  fold {i}: train [{fold['train_start'].date()} -> {fold['train_end'].date()}], "
            f"test [{fold['test_start'].date()} -> {fold['test_end'].date()})"
        )

    oos_path = Path("outputs") / "oos_net_returns.csv"
    oos_returns.to_csv(oos_path, header=True)
    print_summary("Walk-forward OUT-OF-SAMPLE summary (THE headline number)", oos_returns, weights, n_trials)

    benchmark_ticker = cfg["benchmark"]
    benchmark_prices = load_prices([benchmark_ticker], start, end, cache_dir=cache_dir)
    benchmark_monthly = benchmark_prices.pivot(index="date", columns="ticker", values="adj_close")[benchmark_ticker]
    benchmark_forward_returns = benchmark_monthly.resample("ME").last().pct_change().shift(-1)

    plot_equity_curve(net_returns, benchmark=benchmark_forward_returns, path="outputs/equity_curve.png")
    plot_equity_curve(oos_returns, benchmark=benchmark_forward_returns, path="outputs/equity_curve_oos.png")
    print("\nSaved outputs/equity_curve.png (full period) and outputs/equity_curve_oos.png (walk-forward OOS)")


if __name__ == "__main__":
    main()
