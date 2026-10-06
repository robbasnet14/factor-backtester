"""The factor plugin registry: discovery, lookup, validation, and config wiring."""
import sys
import uuid
from pathlib import Path

import pandas as pd
import pytest

from src.features import registry
from src.features.factors import momentum, value
from src.features.registry import compute_factors, get_factor, register_factor, registered_factors
from src.features.transforms import zscore_cross_section

PLUGINS_DIR = Path(registry.__file__).parent / "plugins"
MONTHS = pd.date_range("2019-01-31", periods=18, freq="ME")
TICKERS = ["A", "B", "C", "D"]


def _prices():
    rows = []
    for i, t in enumerate(TICKERS):
        for j, d in enumerate(MONTHS):
            px = 100.0 * (1 + 0.01 * (i + 1)) ** j
            rows.append({"date": d, "ticker": t, "adj_close": px, "close": px, "split_ratio": 1.0})
    return pd.DataFrame(rows)


def _fundamentals():
    return pd.DataFrame(
        [{"date": MONTHS[3], "ticker": t, "report_date": MONTHS[1], "earnings": 1.0 + i,
          "book_value": 10.0, "roe": 0.1 * (i + 1), "shares_outstanding": 1e6} for i, t in enumerate(TICKERS)]
    )


def test_the_built_in_factors_are_registered_with_their_inputs():
    assert {"momentum", "value", "quality"} <= set(registered_factors())
    assert get_factor("momentum").inputs == ("prices",)
    assert set(get_factor("value").inputs) == {"fundamentals", "prices"}
    assert get_factor("quality").inputs == ("fundamentals",)


def test_a_module_dropped_into_the_plugin_package_is_registered_with_nothing_else_referencing_it():
    name = f"tmp_factor_{uuid.uuid4().hex[:8]}"
    module_file = PLUGINS_DIR / f"{name}.py"
    module_file.write_text(
        "from src.features.registry import register_factor\n\n\n"
        f"@register_factor({name!r}, inputs=('prices',))\n"
        "def compute(prices):\n"
        "    return prices.pivot(index='date', columns='ticker', values='close')\n"
    )
    try:
        assert name in registered_factors()  # discovery alone found it
        assert get_factor(name).inputs == ("prices",)
    finally:
        module_file.unlink()
        sys.modules.pop(f"{registry.PLUGIN_PACKAGE}.{name}", None)
        registry._REGISTRY.pop(name, None)


def test_looking_up_an_unregistered_factor_names_the_registered_ones():
    with pytest.raises(KeyError, match="'size_typo' is not registered.*momentum"):
        get_factor("size_typo")


def test_a_config_naming_an_unregistered_factor_fails_loudly_even_if_disabled():
    cfg = {"momentum": {"enabled": True}, "valeu": {"enabled": False, "metric": "earnings_yield"}}
    with pytest.raises(KeyError, match=r"\['valeu'\] that aren't registered"):
        compute_factors(cfg, {"prices": _prices(), "fundamentals": _fundamentals()})


def test_registration_rejects_unknown_inputs_no_inputs_and_duplicate_names():
    with pytest.raises(ValueError, match="unknown input"):
        register_factor("bad_input", inputs=("prices", "volume"))
    with pytest.raises(ValueError, match="at least one input"):
        register_factor("no_input", inputs=())
    with pytest.raises(ValueError, match="already registered"):
        register_factor("momentum", inputs=("prices",))


def test_compute_factors_matches_calling_the_factor_functions_directly():
    prices, fundamentals = _prices(), _fundamentals()
    cfg = {
        "momentum": {"enabled": True, "lookback_months": 12, "skip_months": 1},
        "value": {"enabled": True, "metric": "earnings_yield"},
        "quality": {"enabled": False, "metric": "roe"},
    }

    frames = compute_factors(cfg, {"prices": prices, "fundamentals": fundamentals})

    assert list(frames) == ["momentum", "value"]  # config order, disabled skipped
    pd.testing.assert_frame_equal(frames["momentum"], zscore_cross_section(momentum(prices, 12, 1)))
    pd.testing.assert_frame_equal(frames["value"], zscore_cross_section(value(fundamentals, prices, "earnings_yield")))


def test_with_nothing_enabled_compute_factors_refuses():
    with pytest.raises(ValueError, match="No factors enabled"):
        compute_factors({"quality": {"enabled": False, "metric": "roe"}}, {"fundamentals": _fundamentals()})
