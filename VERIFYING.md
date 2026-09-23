# Verifying this yourself

There's no frontend here — no web page, no UI. This is a **backend-only, command-line
pipeline**: you run one Python script, it does its work (network calls, math, file I/O),
and it writes results to files. "Testing it yourself" means two different things, and
they're verified two different ways:

1. **Does the code do what it claims?** → run the automated test suite.
2. **Does the actual pipeline run end-to-end and produce the numbers in this README?** →
   run the script for real and inspect its output.

Here's both, step by step.

## 0. One-time setup

```bash
cd factor-backtester
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

No API keys required for the default path (Yahoo Finance + SEC EDGAR are both free and
keyless). Skip straight to step 1.

## 1. Run the test suite (fast, offline, ~5 seconds)

```bash
python -m pytest tests/ -v
```

What you're checking: all 65 tests pass. This proves the individual pieces — the momentum
formula, the point-in-time lag on fundamentals, the cost model, the walk-forward fold
logic, the delisting-exit handling, and so on — behave correctly in isolation, using
synthetic data. It does **not** by itself prove real market data flows through cleanly;
that's step 2.

If you want to confirm the tests really are network-free (no silent dependency on your
internet connection), you can disconnect and re-run — everything except one clearly-named
integration test (`test_load_fundamentals_real_aapl_returns_nonempty_2019_2020`) should
still pass.

## 2. Run the real pipeline

```bash
python scripts/run_backtest.py --config config.yaml
```

This is the actual "does it work" test. What happens, in order, and what to watch for:

- **Universe build**: fetches the point-in-time S&P 500 membership history (a public
  GitHub CSV) and prints the ticker count. Should be several hundred.
- **Price + fundamentals download**: hits Yahoo Finance, SEC EDGAR, and occasionally
  Tiingo, for every ticker that's ever been in the universe over the configured date
  range. **This is slow the first time** (the full 2010–2024 S&P 500 universe can take
  a while — expect real wall-clock minutes, not seconds) because it's making one network
  call per ticker with polite rate-limit delays baked in. You'll see per-ticker warnings
  scroll by for names that are delisted or have gaps in fundamentals coverage — that's
  expected, not a failure.
- **Factor computation + backtest**: fast, all local computation once data is loaded.
- **Console output**: a full-period ("in-sample, reference only") summary table, a
  coverage report (composite / value-and-quality coverage per date), a list of the
  walk-forward fold boundaries, and the out-of-sample ("headline") summary table —
  this is where the numbers in this README's Results table come from.
- **Files written to `outputs/`**: `net_returns.csv`, `oos_net_returns.csv`,
  `coverage_report.csv`, `equity_curve.png`, `equity_curve_oos.png`.

**Sanity checks after it finishes:**

- `outputs/equity_curve_oos.png` exists and opens — it should show two lines (strategy vs.
  SPY) starting at $1 and diverging over time.
- The printed "Walk-forward OUT-OF-SAMPLE summary" numbers roughly match the Results table
  above (they won't be bit-for-bit identical if your data cache differs slightly, e.g. a
  ticker that's delisted since I last ran this, but they should be in the same
  ballpark — Sharpe near 0, not suddenly 2.0).
- `outputs/coverage_report.csv` — spot check that `composite_coverage` is high (~90%) and
  `value_quality_coverage` is meaningfully lower (~70%), matching the caveat above about
  fundamentals gaps.

## 3. Confirm the caching actually works

```bash
time python scripts/run_backtest.py --config config.yaml
```

Run it a second time and time it. The first run is slow (network-bound); this second run
should be dramatically faster (seconds, not minutes) because every price/fundamentals call
now hits the local Parquet cache in `data_cache/` instead of the network. If the second run
is just as slow as the first, something's wrong with the caching — worth flagging.

## 4. (Optional) Poke at one piece directly

If you want to verify a specific claim without running the whole pipeline, you can call
the library functions directly from a Python shell:

```python
from src.data.loader import load_prices, load_fundamentals
from src.data.universe import build_universe

# Real AAPL prices, no mocking:
load_prices(["AAPL"], "2020-01-01", "2020-01-15", "data_cache")

# Real SEC fundamentals — TTM EPS, book value, ROE, all point-in-time:
load_fundamentals(["AAPL"], "2019-01-01", "2020-06-30", lag_days=90, cache_dir="data_cache")

# Confirm Lehman Brothers shows up as a member and then disappears in Sept 2008:
u = build_universe("SP500", "2008-06-01", "2008-10-01", cache_dir="data_cache")
u["LEHMQ"]
```

That last one is the single sanity check I trust most: if a backtester quietly drops
bankrupt companies instead of holding them through the crash, its returns are fiction.
Seeing Lehman present-then-gone at exactly the right date is what convinced me this one
isn't doing that.
