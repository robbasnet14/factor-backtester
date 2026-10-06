"""Value: higher earnings yield (cheaper per unit of earnings) scores higher."""
from src.features.factors import value
from src.features.registry import register_factor


@register_factor("value", inputs=("fundamentals", "prices"))
def compute(fundamentals, prices, metric="earnings_yield"):
    return value(fundamentals, prices, metric)
