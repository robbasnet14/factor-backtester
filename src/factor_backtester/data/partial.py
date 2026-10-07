"""What the loaders do when a source fails for some tickers.

A provider answering "no data" for a ticker is an answer: the ticker is
left out (and skiplisted). A provider *failing* (a network error, a rate
limit) is not: leaving the ticker out would make the run's universe depend
on how the network behaved that day, with nothing in the results to show
it. So by default a load in which any ticker failed raises
`PartialDataError` once every ticker has been tried, naming each failure.
Everything that did load is cached by then, so a re-run only asks again for
the failures.

`allow_partial=True` continues without the failed tickers instead, with a
warning, and records them in the returned frame's
`attrs["failed_tickers"]` (ticker -> error) so a caller can report them.
"""
import warnings

import pandas as pd


class PartialDataError(RuntimeError):
    """A data source failed for some tickers (as opposed to having no data
    for them). `failures` maps each ticker to the error."""

    def __init__(self, what: str, failures: dict[str, str]):
        self.failures = dict(failures)
        listing = "\n".join(f"  {ticker}: {error}" for ticker, error in sorted(self.failures.items()))
        super().__init__(
            f"{len(self.failures)} ticker(s) couldn't be loaded because the {what} source failed, "
            f"not because it has no data for them:\n{listing}\n"
            "Re-run to retry them (nothing was marked unavailable, and everything that did load is cached), "
            "or pass allow_partial=True (--allow-partial) to continue without them."
        )


def check_complete(out: pd.DataFrame, failures: dict[str, str], allow_partial: bool, what: str) -> pd.DataFrame:
    """Raise `PartialDataError` if anything failed, unless `allow_partial`;
    then record the failures on `out` and return it."""
    if failures and not allow_partial:
        raise PartialDataError(what, failures)
    if failures:
        warnings.warn(
            f"PARTIAL DATA: continuing without {len(failures)} ticker(s) whose {what} failed to load: "
            f"{', '.join(sorted(failures))}"
        )
    out.attrs["failed_tickers"] = dict(failures)
    return out
