"""The `factor-backtest` command.

    factor-backtest run --config config.yaml [--allow-partial]
    factor-backtest factors

`run` runs the backtest a config describes (see `pipeline.run`); paths in
the config are relative to the config file. `factors` lists the registered
factors, the inputs each one takes, and its parameters with their defaults,
which are the keys its `factors:` section in a config can set.
"""

import argparse
import inspect
import sys
from importlib.metadata import version

from factor_backtester.data.partial import PartialDataError
from factor_backtester.features.registry import get_factor, registered_factors
from factor_backtester.utils.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="factor-backtest", description="Walk-forward backtests of equity factors.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {version('factor-backtester')}")
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="run the backtest a config file describes")
    run_parser.add_argument("--config", default="config.yaml", help="path to the config file (default: config.yaml)")
    run_parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="if a data source fails for some tickers, continue without them instead of stopping",
    )
    run_parser.set_defaults(handler=_run)

    factors_parser = commands.add_parser("factors", help="list the registered factors and their inputs")
    factors_parser.set_defaults(handler=_factors)

    args = parser.parse_args(argv)
    return args.handler(args)


def _run(args: argparse.Namespace) -> int:
    from factor_backtester.pipeline import run  # imports matplotlib; `factors` doesn't need it

    try:
        run(load_config(args.config), allow_partial=args.allow_partial)
    except PartialDataError as e:
        print(f"\nStopped before running the backtest: {e}", file=sys.stderr)
        return 1
    return 0


def _factors(args: argparse.Namespace) -> int:
    for name in registered_factors():
        factor = get_factor(name)
        params = [
            p.name if p.default is inspect.Parameter.empty else f"{p.name}={p.default!r}"
            for p in inspect.signature(factor.compute).parameters.values()
            if p.name not in factor.inputs
        ]
        print(f"{name:<10} inputs: {', '.join(factor.inputs):<22} params: {', '.join(params) or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
