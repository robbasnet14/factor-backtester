# factor-backtester

[![tests](https://github.com/robbasnet14/factor-backtester/actions/workflows/tests.yml/badge.svg)](https://github.com/robbasnet14/factor-backtester/actions/workflows/tests.yml)

A Python library and command-line tool for testing **stock-ranking rules** on historical
data without fooling yourself.

You write a rule as a plain function: given prices and company financials, score every
stock each month, higher meaning "I'd rather own this one". The engine does the rest. It
buys the top-scoring tenth of the market and bets against (sells short) the bottom tenth,
re-ranks every month, charges trading costs, and reports how that would have performed.
The hard part, and most of the code, is making that performance number trustworthy.
That means four things:

- **No look-ahead.** A score on a given date only uses information public on that date.
  Company financials count from when they were filed (plus a buffer, 90 days by default),
  not from the end of the quarter they describe.
- **No survivorship bias.** The stock universe is the S&P 500 as it actually was on each
  date, including companies that later went bankrupt or were acquired. Testing only on
  today's survivors flatters any strategy.
- **Costs and tradability.** Every trade pays a cost. A stock that didn't trade on the
  rebalance date can't be bought or sold that day.
- **Out-of-sample headline.** The headline is measured *walk-forward*: on yearly blocks after
  an initial five-year window, each preceded by a one-month gap, not on the whole period
  at once. The built-in factors are fixed formulas with nothing fitted, so here this
  mainly guards against one lucky full-period number. It's also where fitted parameters
  (factor weights, lookbacks) would plug in without leaking into the years they're
  scored on.

In finance terms, the scoring rules are *factors*, the portfolio is a long/short
decile portfolio rebalanced monthly, and the scores are cross-sectionally z-scored and
averaged into a composite. You don't need those terms to use it.

## Install

Python 3.11 or later.

```bash
git clone https://github.com/robbasnet14/factor-backtester
cd factor-backtester
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

That installs the `factor_backtester` package (the distribution is named
`factor-backtester`) and a `factor-backtest` command.

## Run the example study

```bash
factor-backtest run --config config.yaml
```

`config.yaml` describes one backtest: the stock universe and dates (S&P 500, 2010–2024),
which factors run, portfolio construction, trading costs and the walk-forward setup.
Prices come from Yahoo Finance and company financials from SEC EDGAR, both free with no
API key. A [Tiingo](https://www.tiingo.com) key in `TIINGO_KEY` is optional and only used
as a price fallback for a few delisted companies.

The first run downloads and caches everything (slow); later runs read the cache and take
well under a minute. Results go to `outputs/`: the monthly return series (`net_returns.csv`
in-sample, `oos_net_returns.csv` walk-forward), a coverage report, and equity-curve
charts. The paths in a config are relative to the config file, so the command reads and
writes the same places from any directory.

```bash
factor-backtest factors                     # the registered factors, their inputs and parameters
factor-backtest run --config config.yaml --allow-partial
```

If a data source *fails* for some tickers (a network error, a rate limit), as opposed to
having no data for them, the run stops before backtesting and names them. Everything
that did load is cached, so running again only fetches those. `--allow-partial`
continues without them and prints the missing tickers before and after the results.
That's opt-in because it matters: dropping just MSFT moves the example study's
out-of-sample Sharpe from −0.16 to −0.15.

### With Docker

```bash
docker build -t factor-backtester .
docker run --rm \
  -v "$(pwd)/data_cache:/app/data_cache" \
  -v "$(pwd)/outputs:/app/outputs" \
  factor-backtester
```

The `data_cache` mount keeps downloaded data on your machine between runs, and `outputs`
is where the results land. Add `-e TIINGO_KEY` to pass a Tiingo key through.

## Write your own factor

A factor is one Python file in a directory of your own. Here's a one-month *reversal*
factor, which bets that last month's biggest losers bounce back. It's an illustration,
not a recommendation. From the repository root:

```bash
mkdir -p my_factors
cat > my_factors/reversal.py <<'EOF'
import pandas as pd

from factor_backtester.features.registry import register_factor


@register_factor("reversal", inputs=("prices",))
def compute(prices: pd.DataFrame, lookback_months: int = 1) -> pd.DataFrame:
    # One column per ticker, one row per month-end.
    monthly = prices.pivot(index="date", columns="ticker", values="adj_close").resample("ME").last()
    past_return = monthly.pct_change(lookback_months, fill_method=None)
    return -past_return  # higher score = more attractive, so the biggest losers score highest
EOF
```

Then point `config.yaml` at the directory and switch the factor on:

```yaml
plugin_dirs: [my_factors]          # relative to config.yaml

factors:
  reversal: {enabled: true, lookback_months: 1}
  momentum: {enabled: true, lookback_months: 12, skip_months: 1}
  # ...the rest as before
```

```bash
factor-backtest factors --config config.yaml   # reversal is now listed
factor-backtest run --config config.yaml
```

Nothing else in the engine needs to know the factor exists. The contract:

- **Inputs, by name.** Declare which ones you need in `inputs=`:
  - `prices`: one row per ticker per trading day, with columns `date`, `ticker`,
    `adj_close` (adjusted for splits and dividends, so use it for returns), `close`
    (adjusted for splits only, so use it next to per-share accounting figures) and
    `split_ratio`.
  - `fundamentals`: one row per ticker per filing, with columns `date` (when the figures
    may be used), `ticker`, `report_date`, `earnings` (trailing-twelve-month earnings per share),
    `book_value`, `roe`, `shares_outstanding` and `public_float`.
- **Parameters.** Every other keyword argument is a parameter, set from the factor's line
  in the config. `enabled` is the engine's switch, not a parameter.
- **Output.** A DataFrame indexed by month-end date, one column per ticker. NaN means
  "no score this month", and that stock is left out.
- **Sign convention: higher means more attractive.** The engine standardizes each
  factor month by month, averages them, buys the top tenth and shorts the bottom tenth.
  If your raw quantity is the other way round (lower volatility is better, say), negate
  it inside the factor.

A name that clashes with an existing factor fails when the file is loaded, so a plugin
can't silently replace a built-in one. The built-in factors are written the same way, in
[`src/factor_backtester/features/plugins/`](src/factor_backtester/features/plugins/);
`low_vol` and `size` are good second examples.

### From Python

For a notebook, or to embed the engine, register the factor in code and call the pipeline
directly. `run` prints the same report and returns the results:

```python
import pandas as pd

from factor_backtester.features.registry import register_factor
from factor_backtester.pipeline import run
from factor_backtester.utils.config import load_config


@register_factor("reversal", inputs=("prices",))
def reversal(prices: pd.DataFrame, lookback_months: int = 1) -> pd.DataFrame:
    monthly = prices.pivot(index="date", columns="ticker", values="adj_close").resample("ME").last()
    return -monthly.pct_change(lookback_months, fill_method=None)


cfg = load_config("config.yaml")
cfg["factors"]["reversal"] = {"enabled": True, "lookback_months": 1}
result = run(cfg)
print(result.oos_returns.tail())  # walk-forward monthly returns, net of costs
```

`result` also carries `net_returns` (in-sample), `weights` (the portfolio on each
rebalance date) and `failed_tickers`.

## Architecture

```
src/factor_backtester/
  data/        loading, caching and the point-in-time universe
    providers/   Yahoo, Tiingo and SEC EDGAR, behind two Protocols
  features/    factor formulas, standardization, the registry
    plugins/     the built-in factors, one file each
  backtest/    portfolio construction, cost models, engine, walk-forward
  analytics/   metrics (Sharpe, deflated Sharpe, drawdown, turnover), coverage, charts
  pipeline.py  the run a config describes, end to end
  cli.py       the factor-backtest command
```

**Data sources are Protocols.** A `PriceProvider` is any object with a `name` and
`fetch(ticker, start, end)`, and a `FundamentalsProvider` has `fetch(ticker)`
([`providers/base.py`](src/factor_backtester/data/providers/base.py) documents the
columns). `load_prices(..., providers=[...])` takes an ordered chain and uses the first
provider with data for each ticker; `load_fundamentals(..., provider=...)` takes one. So
a paid vendor, a CSV dump or an internal feed plugs in without touching the engine. The
contract has one rule that everything else depends on:

- **No data is a normal answer.** Return an empty DataFrame, and the ticker is skipped.
  With the built-in sources it's also remembered as unavailable, so it isn't asked again.
- **Failing to answer must raise.** Raise on a network error or exhausted retries, and
  the run stops and reports the ticker (see `--allow-partial` above).

Mixing the two up makes results depend on the network: a throttled request that comes
back empty looks like a company with no data. yfinance does exactly that by default, so
the Yahoo provider asks it to raise instead. A test per provider fails the network
connection underneath it and checks that `fetch` raises.

**Factors are found, not wired in.** `@register_factor` adds a function to a registry.
The built-in plugins are discovered by scanning their package, and yours by scanning the
config's `plugin_dirs`. The registry checks that every factor a config names exists,
before any data is loaded, so a typo fails in seconds instead of being skipped. It's a
plain decorator rather than setuptools entry points: entry points pay off when separately
installed packages contribute factors, and a plugin directory covers your own.

**Caching wraps providers instead of living in them.** Providers only fetch. The cache
layer ([`data/cache.py`](src/factor_backtester/data/cache.py)) sits between them and the
loader and keeps one Parquet file per ticker. So every provider, including yours, gets
the same caching without implementing it, and the hard parts live in one place:

- remembering which date range was already requested, so a stock listed mid-period isn't
  re-downloaded on every run;
- replacing a cached series when the source reports a new stock split, because every
  cached price is then on the old per-share basis;
- refusing a download that covers less than the cache holds, which is usually a delisted
  ticker the source has reassigned to a different company.

A custom provider chain gets its own cache namespace, so one source's data is never
served as another's.

## An example study run with this engine

`config.yaml` as shipped: S&P 500, 2010–2024, momentum, value and quality blended
equally, monthly rebalance, 8 bps cost per trade. Walk-forward, 10 yearly folds:

| Out-of-sample, after costs | |
|---|---|
| Annualized return | −5.0% |
| Sharpe ratio (return per unit of risk) | −0.16 |
| Deflated Sharpe (probability the true Sharpe is above zero) | 0.31 |
| Max drawdown (worst peak-to-trough fall) | −56.9% |
| Avg monthly turnover | 42.3% |

**There's no edge here, and costs aren't the reason.** With costs set to zero, the
out-of-sample Sharpe is still −0.14 (return −4.6% a year). The blended signal itself
doesn't separate future winners from losers in large US stocks over this period.
In-sample it's no better (−0.18). The deflated Sharpe of 0.31 is no evidence of an edge
in either direction. It's computed with `n_trials: 1`, as if this were the only
configuration ever tried; counting the variants tried along the way would only lower it. This backtest can't say *why*: factor decay, crowding and a naive
equal-weight blend are all candidates. That's a legitimate result. The point of building
the engine carefully was to get an answer worth trusting, and a version that looked
great would more likely have meant a bug.

Two corrections got the study here, and earlier versions of this README reported better
numbers before them:

1. **Untradable holdings.** The engine could hold a delisted stock in months it never
   traded. Fixing that moved the out-of-sample return from −1.4% to −1.6%.
2. **A look-ahead leak in the value factor**, worth about 0.17 of Sharpe. Earnings per
   share as filed were divided by prices adjusted for *later* stock splits, so every stock
   that would go on to split, which is mostly stocks that went on to rise, looked cheap in
   advance (NVDA by 40× in 2015). Earnings are now restated across later splits and
   compared with split-adjusted prices.

   | Walk-forward out-of-sample | Return | Sharpe |
   |---|---|---|
   | Before the fix | −1.6% | 0.02 |
   | Same code on re-downloaded data (revisions in the source data only) | −1.8% | 0.01 |
   | Dividend part of the fix only | −3.3% | −0.07 |
   | Split part of the fix only | −4.4% | −0.13 |
   | Both (current) | −5.0% | −0.16 |

![Out-of-sample equity curve vs SPY](outputs/equity_curve_oos.png)

Every run prints the same portfolios under three cost models (none, flat 8 bps, and a
volatility-scaled per-stock cost), so the effect of the cost assumption is shown rather
than asserted.

**A follow-up study** uses this engine's linear blend as a baseline:
[ml-vs-linear-factors](https://github.com/robbasnet14/ml-vs-linear-factors) asks whether
a pre-registered, walk-forward, cost-aware machine-learning model can beat it. Short
answer: no. That repo vendors a snapshot of this engine as of commit `31b92b4`, before
the tradability, cost-model and value-factor fixes, so its baseline numbers differ from
the ones here. This repo is the maintained version.

## Some honest caveats

- **Value coverage is ~71%.** Fundamentals are missing for some renamed/delisted tickers,
  and a few multi-share-class names (e.g. Visa) tag EPS in a custom way SEC's API doesn't
  expose. Momentum/quality coverage is higher (~91% composite).
- **Delisting payouts aren't modeled.** A name that stops trading while held is exited at
  its last traded price (36 position-months); 4 of those never traded again after entry, so
  they're booked at 0%. What shareholders actually received — a takeover premium or a
  bankruptcy loss — isn't in free price data (it needs something like CRSP delisting
  returns), so the direction of this bias is unclear, but it touches very few of the
  ~15,900 position-months.
- **Six names have no split history, so no value score.** BBBY, BK, CSRA, EQR, HOT and
  NFX are cached from before split data was stored, and neither Yahoo nor Tiingo returns
  their history any more. They keep their cached prices for returns and momentum, but
  their EPS can't be restated, so they get no earnings yield (248 name-months) rather
  than one on a guessed share basis.
- **Size covers 70.2% of universe name-months, and its exclusions are mostly data
  errors.** Market cap (SEC shares outstanding x split-adjusted close) is set to NaN
  unless it lies between 0.01x and 100x the company's latest reported public float.
  That gate drops 1,219 name-months across 61 names (524 below, 695 above, including
  359 where the reported float is zero); another 223 name-months across 10 names have a
  reported share count of zero, and 3,190 across 280 names have no float to check
  against (2,198 of them in the first 15 months, before the first 10-K in the window).
  The bounds come from where errors and genuine values separate in this data, not from
  a rule: real values reach 0.04x (a 96% fall since the float date) and 22x (founder
  holdings outside SEC's float definition), while errors are orders of magnitude off
  (counts scaled by 1,000 or 1,000,000, BRK.B's Class A count priced at the Class B
  price). It can't catch smaller errors (ZTS sits at ~0.07x), and when the two figures
  disagree it can't tell which is wrong: where the float is the bad one (HST, WAT), a
  correct market cap is dropped. Multi-class companies are the structural limit: one
  price series per ticker can't give sum(shares x price) across classes.
- **Ticker symbols aren't permanent identifiers.** Sources reassign delisted symbols to
  new companies: re-downloading turned up several whose symbol now returns a different
  company's recent history, and the previous cache held a different security under COL.
  The cache refuses a download that doesn't cover what it already holds, which catches
  the partial-history case, but only a permanent security identifier (like CRSP's PERMNO)
  would guarantee a symbol means the same company throughout.
- **Costs are a flat 8 bps per trade in the headline** — a reasonable stand-in, not the
  truth; real costs vary by name and size.
- **The per-name cost model is a proxy for the wrong variable.** It scales cost with each
  name's trailing 63-day volatility (the median name pays 8 bps), because volatility is the
  only cost-relevant signal in the data. But spreads are driven mainly by liquidity, not
  volatility: a high-volatility mega-cap is still cheap to trade and a low-volatility thin
  name isn't. There's no volume data here, so it can't be modeled properly. It also scales
  within each date, so a market-wide crisis doesn't raise costs across the board. And the
  scaling is linear (a name with 3× the median volatility pays 3× the cost), which is an
  assumption rather than an estimate: if spreads rise less than proportionally with
  volatility, as is often found, it overstates costs for the most volatile names.
- **The "embargo" in the walk-forward is a settling gap, not an ML-style leakage guard,**
  because the factors are fixed formulas with nothing trained. There's a note in
  `validation.py`.
- Early-year (2010) coverage is lower because more of that era's names were later
  renamed or delisted.

## Tests

```bash
pip install -e ".[dev]"
python -m pytest
```

153 tests, all network-mocked except one opt-in live SEC integration check. They cover the
easy-to-get-wrong stuff: momentum's skip-month, the point-in-time fundamentals lag, the
delisted-name universe, never holding a name on a date it didn't trade, EPS restated across
splits (checked against AAPL's and NVDA's real filings), turnover cost math, the
yfinance→Tiingo fallback, every provider raising when its network connection fails, a
failed download stopping the run instead of quietly dropping the name, plugin directories,
and the one I care about most: a test proving the engine trades on *forward* returns,
never contemporaneous ones. CI also runs ruff and mypy.

CI runs the suite on both pandas 2.2 and 3.x, since `pyproject.toml` allows either and
they differ in ways that matter here: on 2.x, `pct_change` forward-fills a missing price
by default, which silently suppressed the engine's missing-price warning. One test checks
that the warning still fires.

For a step-by-step walkthrough of checking the tests and the real pipeline yourself, see
[VERIFYING.md](VERIFYING.md).

## Still to do

- A liquidity-based cost model (e.g. square-root impact on dollar volume), which needs
  volume data the loader doesn't store yet.
