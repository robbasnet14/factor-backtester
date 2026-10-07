"""The `factor-backtest` command and config loading: paths in a config are
relative to the config file, wherever the command runs from."""
from pathlib import Path

import pytest

from factor_backtester import cli, pipeline
from factor_backtester.data.partial import PartialDataError
from factor_backtester.utils.config import load_config

MINIMAL = """\
universe: {{name: SP500, start_date: "2020-01-01", end_date: "2020-12-31"}}
data: {{cache_dir: {cache_dir}, fundamentals_lag_days: 90}}
{output}"""


def _write_config(directory: Path, cache_dir: str = "data_cache", output: str = "") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.yaml"
    path.write_text(MINIMAL.format(cache_dir=cache_dir, output=output))
    return path


def test_relative_paths_resolve_against_the_config_file_not_the_working_directory(tmp_path, monkeypatch):
    config = _write_config(tmp_path / "study", output='output_dir: "results/run1"')
    elsewhere = tmp_path / "somewhere" / "else"
    elsewhere.mkdir(parents=True)
    monkeypatch.chdir(elsewhere)

    cfg = load_config(Path("../../study/config.yaml"))  # relative to the working directory, like a CLI argument

    assert cfg["data"]["cache_dir"] == str((tmp_path / "study" / "data_cache").resolve())
    assert cfg["output_dir"] == str((tmp_path / "study" / "results" / "run1").resolve())
    assert load_config(config) == cfg  # the same config, however it's named


def test_output_dir_defaults_to_outputs_beside_the_config(tmp_path):
    cfg = load_config(_write_config(tmp_path))

    assert cfg["output_dir"] == str(tmp_path.resolve() / "outputs")


def test_absolute_and_home_relative_paths_are_used_as_given(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    absolute = tmp_path / "shared_cache"

    cfg = load_config(_write_config(tmp_path / "study", cache_dir=f'"{absolute}"', output='output_dir: "~/out"'))

    assert cfg["data"]["cache_dir"] == str(absolute)
    assert cfg["output_dir"] == str(tmp_path / "home" / "out")


def test_run_passes_the_resolved_config_and_the_partial_data_flag(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(pipeline, "run", lambda cfg, allow_partial: seen.update(cfg=cfg, allow_partial=allow_partial))
    config = _write_config(tmp_path)

    assert cli.main(["run", "--config", str(config), "--allow-partial"]) == 0

    assert seen["allow_partial"] is True
    assert seen["cfg"]["data"]["cache_dir"] == str(tmp_path.resolve() / "data_cache")


def test_run_stops_with_exit_code_1_and_names_the_failures(tmp_path, monkeypatch, capsys):
    def fails(cfg, allow_partial):
        raise PartialDataError("price", {"MSFT": "yfinance: rate limited"})

    monkeypatch.setattr(pipeline, "run", fails)

    assert cli.main(["run", "--config", str(_write_config(tmp_path))]) == 1
    assert "MSFT: yfinance: rate limited" in capsys.readouterr().err


def test_factors_lists_each_registered_factor_with_its_inputs_and_parameters(capsys):
    assert cli.main(["factors"]) == 0

    lines = {line.split()[0]: line for line in capsys.readouterr().out.splitlines()}
    assert {"momentum", "value", "quality", "low_vol", "size"} <= set(lines)
    assert "inputs: prices" in lines["momentum"] and "lookback_months=12" in lines["momentum"]
    assert "inputs: fundamentals, prices" in lines["size"] and "float_floor=0.01" in lines["size"]


def test_a_subcommand_is_required():
    with pytest.raises(SystemExit):
        cli.main([])
