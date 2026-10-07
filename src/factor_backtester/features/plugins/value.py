"""Value: higher earnings yield (cheaper per unit of earnings) scores higher."""
from factor_backtester.features.factors import value
from factor_backtester.features.registry import register_factor


@register_factor("value", inputs=("fundamentals", "prices"))
def compute(fundamentals, prices, metric="earnings_yield"):
    return value(fundamentals, prices, metric)
