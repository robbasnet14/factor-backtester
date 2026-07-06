# Factor Backtester

A point-in-time, multi-factor equity backtesting framework in Python.
Implements momentum, value, and quality factors with walk-forward out-of-sample
validation, transaction-cost modeling, and turnover-adjusted performance analytics
benchmarked against SPY.

> Author: Rob Basnet — independent quantitative research project (2026)

## What it does
- Builds a **survivorship-bias-free**, point-in-time equity universe
- Computes cross-sectional factors (momentum / value / quality) and standardizes them
- Constructs long–short decile portfolios and rebalances on a schedule
- Applies **realistic transaction costs** and measures turnover
- Reports Sharpe, drawdown, and a **deflated Sharpe ratio** (guards against overfitting)
- Validates walk-forward, out-of-sample only

## Project layout
```
src/
  data/       ingestion, cleaning, point-in-time universe
  features/   factor computation + cross-sectional transforms
  backtest/   portfolio construction, cost model, engine loop
  analytics/  performance metrics + plots
  utils/      config loader
scripts/      run_backtest.py (entry point)
notebooks/    exploration + results
tests/        unit tests
config.yaml   all parameters (universe, dates, costs, factors)
```

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_backtest.py --config config.yaml
```

## Status
Building step by step — see the Build Playbook. Current: Step 1 (scaffold) complete.

## Results
_(equity curve + headline metrics go here once Step 6 is done)_
