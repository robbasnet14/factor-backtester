"""The data layer's public entry points, kept here so existing imports keep
working: `load_prices` lives in `factor_backtester.data.prices` and `load_fundamentals` in
`factor_backtester.data.fundamentals`. Both cache through `factor_backtester.data.cache` and remember
tickers that are permanently unavailable, so reruns don't keep asking. A
caller-supplied `providers=` / `provider=` gets its own cache namespace and
doesn't use the default sources' skiplist, since what they lack says
nothing about yours.
"""

from factor_backtester.data.fundamentals import default_fundamentals_provider, load_fundamentals
from factor_backtester.data.prices import default_price_providers, load_prices

__all__ = ["default_fundamentals_provider", "default_price_providers", "load_fundamentals", "load_prices"]
