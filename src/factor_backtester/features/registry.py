"""Factor plugin registry.

A factor is a function decorated with `@register_factor(name, inputs=...)` in
a module under `factor_backtester/features/plugins/`. Every module in that package is
imported by `discover()`, so adding a factor means adding one file there:
nothing else in the engine refers to it by name. `config.yaml`'s `factors:`
section decides which registered factors run.

Contract for a factor function:

- It takes the inputs it declares, by name: `prices` (long frame from
  `load_prices`: date, ticker, adj_close, close, split_ratio) and/or
  `fundamentals` (long frame from `load_fundamentals`), plus keyword
  parameters, which come from its `config.yaml` section (every key except
  `enabled`).
- It returns a wide panel, index = month-end date, columns = ticker.
- **Sign convention: higher means more attractive.** The engine z-scores
  each factor, averages them, and goes long the top decile, so a factor
  must be oriented so that the names you'd want to own score highest. If
  the raw quantity is the other way round (low volatility, small size),
  invert it inside the factor; don't leave it to the composite.

Registration is a plain decorator into a dict rather than setuptools entry
points: entry points only pay off once separately installed packages
contribute factors, and nothing installs this one yet. If that changes, an
entry-point group can feed the same `register_factor` without changing any
factor.
"""
import importlib
import pkgutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import pandas as pd

from factor_backtester.features.transforms import zscore_cross_section

KNOWN_INPUTS = ("prices", "fundamentals")
PLUGIN_PACKAGE = "factor_backtester.features.plugins"


@dataclass(frozen=True)
class Factor:
    name: str
    compute: Callable[..., pd.DataFrame]
    inputs: tuple[str, ...]


_REGISTRY: dict[str, Factor] = {}


def register_factor(
    name: str, *, inputs: tuple[str, ...]
) -> Callable[[Callable[..., pd.DataFrame]], Callable[..., pd.DataFrame]]:
    """Decorator registering `fn` as factor `name`, needing `inputs`.
    Unknown inputs or a duplicate name fail here, at import time, rather
    than on the first backtest that uses the factor."""
    inputs = tuple(inputs)
    if not inputs:
        raise ValueError(f"factor {name!r} must declare at least one input from {KNOWN_INPUTS}")
    unknown = [i for i in inputs if i not in KNOWN_INPUTS]
    if unknown:
        raise ValueError(f"factor {name!r} declares unknown input(s) {unknown}; known inputs are {KNOWN_INPUTS}")
    if name in _REGISTRY:
        raise ValueError(f"factor {name!r} is already registered (by {_REGISTRY[name].compute.__module__})")

    def decorator(fn: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
        _REGISTRY[name] = Factor(name=name, compute=fn, inputs=inputs)
        return fn

    return decorator


def discover() -> None:
    """Import every module in the plugin package, registering its factors.
    Safe to call repeatedly: modules already imported aren't re-run."""
    package = importlib.import_module(PLUGIN_PACKAGE)
    for module in pkgutil.iter_modules(package.__path__):
        if not module.name.startswith("_"):
            importlib.import_module(f"{PLUGIN_PACKAGE}.{module.name}")


def registered_factors() -> list[str]:
    discover()
    return sorted(_REGISTRY)


def get_factor(name: str) -> Factor:
    discover()
    if name not in _REGISTRY:
        raise KeyError(f"factor {name!r} is not registered; registered factors: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def compute_factors(factor_cfg: Mapping[str, Mapping], inputs: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Compute and cross-sectionally z-score every enabled factor in
    `factor_cfg` (config.yaml's `factors:` section), in config order.

    Every factor named in the config must be registered, enabled or not, so
    a typo fails loudly instead of quietly dropping a factor.
    """
    unknown = [name for name in factor_cfg if name not in registered_factors()]
    if unknown:
        raise KeyError(f"config.yaml names factor(s) {unknown} that aren't registered; registered factors: {registered_factors()}")

    frames = {}
    for name, cfg in factor_cfg.items():
        if not cfg.get("enabled", False):
            continue
        factor = get_factor(name)
        params = {k: v for k, v in cfg.items() if k != "enabled"}
        raw = factor.compute(**{i: inputs[i] for i in factor.inputs}, **params)
        frames[name] = zscore_cross_section(raw)
    if not frames:
        raise ValueError("No factors enabled in config.yaml")
    return frames
