"""Factors from a plugin directory outside the package: a config lists the
directory, a file in it registers a factor, and the CLI and the registry's
config wiring pick it up without the package knowing its name."""

import sys
import uuid
from pathlib import Path

import pandas as pd
import pytest

from factor_backtester import cli
from factor_backtester.features import registry
from factor_backtester.features.registry import compute_factors, get_factor, load_plugin_dirs, registered_factors
from factor_backtester.utils.config import load_config
from tests.test_registry import _fundamentals, _prices

PLUGIN = """\
import pandas as pd

from factor_backtester.features.registry import register_factor


@register_factor({name!r}, inputs=("prices",))
def compute(prices: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
    close = prices.pivot(index="date", columns="ticker", values="close")
    return scale * close.resample("ME").last()
"""


@pytest.fixture
def plugin_dir(tmp_path):
    """A directory holding one plugin file that registers a uniquely named
    factor; unregistered and unimported again afterwards."""
    name = f"my_factor_{uuid.uuid4().hex[:8]}"
    directory = tmp_path / "my_factors"
    directory.mkdir()
    (directory / f"{name}.py").write_text(PLUGIN.format(name=name))
    (directory / "_helpers.py").write_text("raise RuntimeError('files starting with _ are not plugins')\n")
    yield directory, name
    registry._REGISTRY.pop(name, None)
    for module in [m for m in sys.modules if m.startswith("_factor_backtester_plugin_")]:
        del sys.modules[module]


def test_a_file_in_a_plugin_directory_registers_its_factor(plugin_dir):
    directory, name = plugin_dir

    assert load_plugin_dirs([directory]) == [name]

    assert name in registered_factors()
    assert get_factor(name).inputs == ("prices",)


def test_loading_a_plugin_directory_again_is_harmless(plugin_dir):
    directory, _ = plugin_dir
    load_plugin_dirs([directory])

    assert load_plugin_dirs([directory]) == []  # nothing new, and no "already registered" error


def test_a_missing_plugin_directory_is_an_error_naming_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope"):
        load_plugin_dirs([tmp_path / "nope"])


def test_a_plugin_cannot_silently_replace_a_built_in_factor(tmp_path):
    (tmp_path / "shadow.py").write_text(PLUGIN.format(name="momentum"))

    with pytest.raises(ValueError, match="'momentum' is already registered"):
        load_plugin_dirs([tmp_path])
    assert get_factor("momentum").compute.__module__ == "factor_backtester.features.plugins.momentum"


def test_plugin_dirs_in_a_config_are_relative_to_the_config_file(tmp_path, monkeypatch):
    (tmp_path / "study").mkdir()
    config = tmp_path / "study" / "config.yaml"
    config.write_text('data: {cache_dir: data_cache}\nplugin_dirs: ["my_factors", "/abs/elsewhere"]\n')
    monkeypatch.chdir(tmp_path)

    assert load_config(config)["plugin_dirs"] == [str(tmp_path.resolve() / "study" / "my_factors"), "/abs/elsewhere"]


def test_the_cli_lists_a_config_s_plugin_factors_and_a_config_can_enable_them(plugin_dir, capsys):
    directory, name = plugin_dir
    config = directory.parent / "config.yaml"
    config.write_text(
        "data: {cache_dir: data_cache}\n"
        "plugin_dirs: [my_factors]\n"
        f"factors:\n  momentum: {{enabled: true}}\n  {name}: {{enabled: true, scale: 2.0}}\n"
    )

    assert cli.main(["factors", "--config", str(config)]) == 0
    listing = capsys.readouterr().out
    assert f"{name}" in listing and "params: scale=1.0" in listing

    cfg = load_config(config)
    frames = compute_factors(cfg["factors"], {"prices": _prices(), "fundamentals": _fundamentals()})
    assert list(frames) == ["momentum", name]
    assert isinstance(frames[name], pd.DataFrame) and frames[name].notna().any().any()


def test_without_its_plugin_directory_a_config_naming_the_factor_fails(plugin_dir, tmp_path):
    _, name = plugin_dir  # written, but never loaded

    with pytest.raises(KeyError, match=name):
        compute_factors({name: {"enabled": True}}, {"prices": _prices()})


def test_the_pipeline_loads_plugin_dirs_before_any_data(tmp_path):
    from factor_backtester import pipeline

    cfg = {"plugin_dirs": [str(tmp_path / "missing")], "factors": {}, "output_dir": str(tmp_path)}

    with pytest.raises(FileNotFoundError, match="missing"):
        pipeline.run(cfg)  # no universe or data config at all: it never got that far
    assert not Path(tmp_path / "net_returns.csv").exists()
