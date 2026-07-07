"""Charts."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_SURFACE = "#fcfcfb"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_STRATEGY_COLOR = "#2a78d6"   # categorical slot 1 (blue)
_BENCHMARK_COLOR = "#1baf7a"  # categorical slot 2 (aqua)


def plot_equity_curve(returns: pd.Series, benchmark: pd.Series | None = None, path: str = "outputs/equity_curve.png"):
    """Plot cumulative growth of $1 for `returns`, optionally overlaid with
    `benchmark` (e.g. SPY) on the same periods, and save to `path`.

    Both series are period returns (not already cumulative) sharing a
    comparable calendar; each is compounded independently from $1 so the two
    lines are directly comparable regardless of gaps in one series.
    """
    r = returns.dropna()
    equity = (1 + r).cumprod()

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)

    ax.plot(equity.index, equity.values, color=_STRATEGY_COLOR, linewidth=2, solid_capstyle="round", label="Strategy")

    if benchmark is not None:
        b = benchmark.reindex(r.index).dropna()
        bench_equity = (1 + b).cumprod()
        ax.plot(
            bench_equity.index,
            bench_equity.values,
            color=_BENCHMARK_COLOR,
            linewidth=2,
            linestyle="--",
            dash_capstyle="round",
            label="SPY",
        )

    ax.set_title("Cumulative Growth of $1", color=_INK_PRIMARY, fontsize=14, fontweight="bold", loc="left")
    ax.set_ylabel("Portfolio value ($)", color=_INK_SECONDARY)
    ax.tick_params(colors=_INK_MUTED)

    ax.grid(True, axis="y", color=_GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(_GRIDLINE)

    legend = ax.legend(frameon=False, loc="upper left", labelcolor=_INK_SECONDARY)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_SURFACE)
    plt.close(fig)
