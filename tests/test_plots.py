import pandas as pd

from src.analytics.plots import plot_equity_curve


def test_plot_equity_curve_saves_png(tmp_path):
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    returns = pd.Series([0.02, -0.01, 0.03, 0.01, -0.02, 0.015], index=dates)
    benchmark = pd.Series([0.01, 0.0, 0.02, 0.01, -0.01, 0.005], index=dates)

    out_path = tmp_path / "equity_curve.png"
    plot_equity_curve(returns, benchmark=benchmark, path=str(out_path))

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_equity_curve_without_benchmark(tmp_path):
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    returns = pd.Series([0.01, 0.02, -0.01, 0.03], index=dates)

    out_path = tmp_path / "equity_curve.png"
    plot_equity_curve(returns, path=str(out_path))

    assert out_path.exists()
