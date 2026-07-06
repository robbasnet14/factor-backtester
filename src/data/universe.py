"""Point-in-time, survivorship-bias-free universe. STEP 2."""
import pandas as pd

def build_universe(name: str, start: str, end: str) -> pd.DataFrame:
    """Return a boolean membership matrix: index=date, columns=ticker, True if in universe.
    MUST include delisted names historically (no survivorship bias).
    TODO: load index constituent history (or top-N by dollar volume) per date.
    """
    raise NotImplementedError("Step 2: implement point-in-time universe")
