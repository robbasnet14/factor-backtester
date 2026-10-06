"""12-1 momentum: higher trailing return (skipping the latest month) scores higher."""
from src.features.factors import momentum
from src.features.registry import register_factor


@register_factor("momentum", inputs=("prices",))
def compute(prices, lookback_months=12, skip_months=1):
    return momentum(prices, lookback_months, skip_months)
