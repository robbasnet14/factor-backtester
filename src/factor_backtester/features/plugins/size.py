"""Size: smaller market cap scores higher.

Market cap is shares outstanding (as of the latest filing, restated across
later splits) times the month-end `close` (split-adjusted, not
dividend-adjusted, so both are per the same share). The score is minus its
log: small is the attractive end (see the registry's sign convention).

Data-quality gate: market cap is checked against an independent reported
figure, the company's latest public float (dei:EntityPublicFloat, the 10-K
cover page's market value of non-affiliate shares), and set to NaN unless

    float_floor <= market cap / public float <= float_ceiling

(default 0.01 to 100), or if no public float has been reported to check
against. The bounds come from the observed distribution of that ratio over
the S&P 500, 2010-2024, not from a rule: values two or more orders of
magnitude off on either side were data errors (share counts of zero,
counts or floats scaled by 1,000 or 1,000,000, one share class's count
priced at another: BRK.B's Class A count times the Class B price gives
~$0.1B against a ~$150B float), while genuine values reached 0.04 (NKTR
after a 96% fall since its float was measured) and x22 (NKE, where SEC's
non-affiliate definition excludes the founder's holdings). A tighter floor
drops stocks for having crashed, which biases size the way survivorship
does: at 0.5 it removed over 1,000 genuine name-months, concentrated in
March 2020 and the 2015 energy crash.

What the gate can't do: catch errors smaller than two orders of magnitude
(ZTS sits at ~0.07 for three months), or tell which side is wrong when the
two figures disagree. Where the float is the mis-scaled one (HST, WAT), a
correct market cap is dropped. That is the cost of a cross-check.
"""

from typing import cast

import numpy as np
import pandas as pd

from factor_backtester.features.factors import _fundamentals_metric_to_monthly, _pivot_prices_wide
from factor_backtester.features.registry import register_factor


@register_factor("size", inputs=("fundamentals", "prices"))
def compute(
    fundamentals: pd.DataFrame, prices: pd.DataFrame, float_floor: float = 0.01, float_ceiling: float = 100.0
) -> pd.DataFrame:
    close = _pivot_prices_wide(prices, "close").resample("ME").last()
    shares = _fundamentals_metric_to_monthly(fundamentals, "shares_outstanding", close.index)
    public_float = _fundamentals_metric_to_monthly(fundamentals, "public_float", close.index)
    shares, close = shares.align(close, join="outer")
    public_float = public_float.reindex(index=close.index, columns=close.columns)

    market_cap = shares * close
    ratio = market_cap / public_float
    plausible = (market_cap > 0) & (ratio >= float_floor) & (ratio <= float_ceiling)
    # np.log on a DataFrame returns a DataFrame (pandas implements numpy's ufuncs); the stubs say ndarray.
    return cast(pd.DataFrame, -np.log(market_cap.where(plausible)))
