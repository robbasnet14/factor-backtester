"""Fundamentals from SEC EDGAR's public XBRL "company facts" API.

Keyless, but SEC requires a descriptive `User-Agent` with contact info on
every request and rate-limits abusive callers (see `_SEC_HEADERS`). SEC's
`filed` date is exactly when a 10-Q/10-K's numbers became public, so the
`report_date` this returns has no look-ahead by construction.
"""
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from factor_backtester.data.providers.base import empty_fundamentals
from factor_backtester.data.providers.xbrl import EPS_CONCEPTS, first_usable_eps, fundamentals_from_facts

_logger = logging.getLogger(__name__)

_SEC_HEADERS = {"User-Agent": "factor-backtester rob basnet basnetrob@gmail.com"}
_SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
_REQUEST_TIMEOUT = 30
_MAX_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 1.0
_POLITE_DELAY_SECONDS = 0.2

# SEC's own https://www.sec.gov/files/company_tickers.json omits some real,
# actively-filing tickers outright (WBA — Walgreens Boots Alliance, CIK
# 1618921 — isn't in the file at all as of this writing) and aliases others
# under a ticker SEC uses internally instead of the exchange ticker (MMC —
# Marsh & McLennan — is listed there under "MRSH", not "MMC"). Verified by
# direct lookup against SEC's own file and companyfacts API. Add entries
# here as more gaps like this turn up; there's no general fix since the gap
# is in SEC's source data, not in how we're matching it.
CIK_OVERRIDES = {
    "MMC": 62709,
    "WBA": 1618921,
}


class SecEdgarProvider:
    """EPS, book value, ROE and shares outstanding per filing, from SEC company facts.

    A ticker SEC has no CIK for, or with no usable EPS concept, is an empty
    answer; network errors raise. Some tickers genuinely have no usable EPS
    concept in SEC's structured company-facts data at all — notably
    companies with multiple share classes (e.g. V, STZ), which often tag
    per-share figures through a company-specific XBRL extension that this
    API doesn't expose. That's a real upstream data gap, not a bug here; it
    shows up as a skip warning and as reduced coverage in `coverage_report`,
    not a silent wrong number.

    `cik_map_path` is where SEC's ticker->CIK file is kept. It's reference
    data for this provider rather than a cache of results, so it lives here
    instead of in the cache layer; it's loaded on first use.
    """

    name = "SEC EDGAR"

    def __init__(self, cik_map_path: str | Path):
        self.cik_map_path = Path(cik_map_path)
        self._ticker_to_cik: dict | None = None

    def resolve_cik(self, ticker: str) -> int | None:
        if self._ticker_to_cik is None:
            self._ticker_to_cik = load_ticker_to_cik_map(self.cik_map_path)
        return resolve_cik(ticker, self._ticker_to_cik)

    def fetch(self, ticker: str) -> pd.DataFrame:
        cik = self.resolve_cik(ticker)
        if cik is None:
            _logger.info("%s: no SEC CIK found", ticker)
            return empty_fundamentals()

        facts = sec_get(_SEC_FACTS_URL.format(cik=cik))
        us_gaap = facts.get("facts", {}).get("us-gaap", {})
        if not us_gaap:
            _logger.info("%s: SEC returned no us-gaap facts (CIK %d)", ticker, cik)
        else:
            _logger.info("%s: SEC returned %d us-gaap concepts (CIK %d)", ticker, len(us_gaap), cik)

        eps_q, eps_annual, eps_concept_used = first_usable_eps(us_gaap)
        if eps_q is None:
            _logger.info(
                "%s: no usable EPS facts in any of %s (CIK %d) — often a multi-share-class company that "
                "tags EPS via a custom XBRL extension SEC's company-facts API doesn't expose",
                ticker,
                EPS_CONCEPTS,
                cik,
            )
            return empty_fundamentals()
        _logger.info("%s: using %s for EPS (%d quarterly observations)", ticker, eps_concept_used, len(eps_q))
        return fundamentals_from_facts(us_gaap, eps_q, eps_annual, facts.get("facts", {}).get("dei", {}))


def sec_get(url: str) -> dict:
    """GET a SEC EDGAR URL with the required User-Agent, retrying 429/5xx
    with backoff and a polite delay after every request that goes through.
    """
    last_error = None
    for attempt in range(_MAX_RETRIES):
        resp = requests.get(url, headers=_SEC_HEADERS, timeout=_REQUEST_TIMEOUT)
        if resp.status_code == 429 or resp.status_code >= 500:
            last_error = requests.HTTPError(f"{resp.status_code} error fetching {url}")
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue
            raise last_error
        resp.raise_for_status()  # other 4xx (e.g. 404) fail immediately, no retry
        time.sleep(_POLITE_DELAY_SECONDS)
        return resp.json()
    raise last_error  # pragma: no cover — loop always returns or raises above


def load_ticker_to_cik_map(path: str | Path) -> dict:
    """Build the ticker->CIK map from SEC's company_tickers.json (downloaded
    to `path` on first use): an exact, uppercase-ticker match against
    `{index: {cik_str, ticker, title}}`, plus `CIK_OVERRIDES` for the handful
    of real tickers SEC's own file omits or aliases.
    """
    path = Path(path)
    if path.exists():
        raw = json.loads(path.read_text())
    else:
        raw = sec_get(_SEC_TICKERS_URL)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw))
    mapping = {entry["ticker"].upper(): int(entry["cik_str"]) for entry in raw.values()}
    mapping.update(CIK_OVERRIDES)
    return mapping


def resolve_cik(ticker: str, ticker_to_cik: dict) -> int | None:
    upper = ticker.upper()
    for candidate in (upper, upper.replace(".", "-"), upper.replace("-", ".")):
        if candidate in ticker_to_cik:
            return ticker_to_cik[candidate]
    return None


def debug_print_resolved_ciks(tickers: list[str] = ("AAPL", "V", "MMC", "WBA"), cache_dir: str = "data_cache") -> dict:
    """Manual sanity check: resolve and print CIKs for the given tickers.

    Run directly (`python -m factor_backtester.data.providers.sec_edgar`) to eyeball that
    ticker->CIK resolution is working for known-tricky names. Returns the
    resolved map so it's also usable from a test/REPL.
    """
    provider = SecEdgarProvider(Path(cache_dir) / "sec_company_tickers.json")
    resolved = {t: provider.resolve_cik(t) for t in tickers}
    for ticker, cik in resolved.items():
        print(f"{ticker}: {'CIK ' + str(cik) if cik is not None else 'UNRESOLVED'}")
    return resolved


if __name__ == "__main__":
    debug_print_resolved_ciks()
