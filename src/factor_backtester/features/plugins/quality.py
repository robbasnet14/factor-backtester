"""Quality: higher return on equity scores higher."""
from factor_backtester.features.factors import quality
from factor_backtester.features.registry import register_factor


@register_factor("quality", inputs=("fundamentals",))
def compute(fundamentals, metric="roe"):
    return quality(fundamentals, metric)
