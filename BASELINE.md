# Baseline

Measurements of the repo before any structural refactoring, so later changes can be
compared against real numbers instead of estimates. Measured 2026-09-29 at commit
`f327daf` (source code identical to `150389d`; the only later changes are to README.md).

## Environment

| | |
|---|---|
| Machine | Apple M2, 8 cores (4 performance + 4 efficiency), 8 GB RAM, macOS 15.7.3 |
| Python | 3.12.9 (local venv) |
| pandas / numpy | 3.0.3 / 2.5.1 |
| pytest / pytest-cov / coverage | 9.1.1 / 7.1.0 / 7.16.2 |
| CI (for reference) | Python 3.11, pandas 2.2.3 and 3.0.6 (matrix) |

## Timings

Wall-clock via `time.perf_counter()` around a subprocess, one untimed warm-up run first,
then 3 timed runs. Warm cache: every price and fundamentals series already in `data_cache/`.

| What | Run 1 | Run 2 | Run 3 | Mean | Std dev |
|---|---|---|---|---|---|
| Full backtest, `python scripts/run_backtest.py --config config.yaml` | 6.12 s | 5.73 s | 5.67 s | **5.84 s** | 0.24 s |
| Test suite, CI selection (75 tests) | 2.15 s | 2.03 s | 2.11 s | **2.10 s** | 0.06 s |

Both include interpreter startup and imports, since that's what a user waits for.

## Tests

- **76 tests** collected; CI runs **75** (the live SEC EDGAR integration test is deselected).
- All 75 pass; suite runtime above.

## Coverage

`pytest --cov=src --cov=scripts`, CI test selection.

| Module | Statements | Missed | Coverage |
|---|---|---|---|
| src/analytics/metrics.py | 66 | 3 | 95% |
| src/analytics/plots.py | 33 | 0 | 100% |
| src/backtest/costs.py | 33 | 0 | 100% |
| src/backtest/engine.py | 32 | 0 | 100% |
| src/backtest/portfolio.py | 24 | 0 | 100% |
| src/backtest/validation.py | 29 | 3 | 90% |
| src/data/loader.py | 341 | 22 | 94% |
| src/data/universe.py | 40 | 4 | 90% |
| src/features/factors.py | 25 | 0 | 100% |
| src/features/transforms.py | 27 | 1 | 96% |
| src/utils/config.py | 5 | 5 | 0% |
| scripts/run_backtest.py | 101 | 101 | 0% |
| **src/ total** | **655** | **38** | **94.2%** |
| **src/ + scripts/ total** | **756** | **139** | **82%** |

(Empty `__init__.py` files omitted.) `scripts/run_backtest.py` and `src/utils/config.py`
are at 0% because no test runs the entry point.

## Line counts

`find src scripts -name "*.py" | xargs wc -l | sort -n` (empty `__init__.py` files omitted):

| Lines | File |
|---|---|
| 7 | src/utils/config.py |
| 53 | src/features/transforms.py |
| 60 | src/analytics/plots.py |
| 76 | src/features/factors.py |
| 77 | src/backtest/portfolio.py |
| 86 | src/backtest/costs.py |
| 90 | src/data/universe.py |
| 111 | src/backtest/validation.py |
| 120 | src/backtest/engine.py |
| 156 | src/analytics/metrics.py |
| 194 | scripts/run_backtest.py |
| 704 | src/data/loader.py |
| **1,735** | **total** |

`src/data/loader.py` is 41% of the code.

## Backtest output fingerprint

The refactors that follow are meant to be purely structural, so a run must reproduce these
files byte for byte. Two consecutive runs on the warm cache produced identical files.

| File | SHA-256 |
|---|---|
| outputs/net_returns.csv | `ea5202ada89f8ae807a505242a769a5a1bfc620409dacb64221895c15846d62e` |
| outputs/oos_net_returns.csv | `24f1cb6d87568e7081ee2ee1ed092e073c5d5c14140a96acf813b1e983cf4c21` |
| outputs/coverage_report.csv | `926675e5c01ba122ac5e99f4d3ace7835b0715147ab39e864a9f04202555953a` |

Headline (walk-forward out-of-sample, flat 8 bps): annualized return −1.59%, Sharpe 0.02,
max drawdown −50.38%, deflated Sharpe 0.521, average monthly turnover 39.13%.

## Reproducing these numbers

```bash
# timings (warm cache): 1 warm-up + 3 timed runs each
python - <<'PY'
import statistics, subprocess, time
DESELECT = "tests/test_data.py::test_load_fundamentals_real_aapl_returns_nonempty_2019_2020"
for label, cmd in [
    ("backtest", ["python", "scripts/run_backtest.py", "--config", "config.yaml"]),
    ("tests", ["python", "-m", "pytest", "tests/", "-q", "-p", "no:cacheprovider", "--deselect", DESELECT]),
]:
    subprocess.run(cmd, capture_output=True, check=True)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        subprocess.run(cmd, capture_output=True, check=True)
        times.append(time.perf_counter() - t0)
    print(label, [round(t, 2) for t in times], "mean", round(statistics.mean(times), 2))
PY

# coverage
python -m pytest tests/ --cov=src --cov=scripts --cov-report=term \
  --deselect tests/test_data.py::test_load_fundamentals_real_aapl_returns_nonempty_2019_2020

# output fingerprint (after a run)
shasum -a 256 outputs/net_returns.csv outputs/oos_net_returns.csv outputs/coverage_report.csv
```
