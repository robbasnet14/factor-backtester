"""The data layer's public entry points, kept here so existing imports keep
working: `load_prices` lives in `src.data.prices` and `load_fundamentals` in
`src.data.fundamentals`. Both cache through `src.data.cache` and remember
tickers that are permanently unavailable, so reruns don't keep asking. A
caller-supplied `providers=` / `provider=` gets its own cache namespace and
doesn't use the default sources' skiplist, since what they lack says
nothing about yours.
"""
from src.data.fundamentals import default_fundamentals_provider, load_fundamentals
from src.data.prices import default_price_providers, load_prices

__all__ = ["default_fundamentals_provider", "default_price_providers", "load_fundamentals", "load_prices"]
