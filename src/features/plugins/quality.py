"""Quality: higher return on equity scores higher."""
from src.features.factors import quality
from src.features.registry import register_factor


@register_factor("quality", inputs=("fundamentals",))
def compute(fundamentals, metric="roe"):
    return quality(fundamentals, metric)
