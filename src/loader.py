import os
from pathlib import Path

import pandas as pd


DEFAULT_DATA_URL = (
    "https://raw.githubusercontent.com/backjhello/LoanApproval/main/"
    "data/processed/customer_features.csv"
)


def load_customer_features() -> pd.DataFrame:
    """Load the main dataset used across dashboard pages.

    Priority:
    1) Local checked-in CSV under data/processed
    2) Remote fallback URL (and cache it locally when possible)
    """
    base = Path(__file__).resolve().parents[1]
    local_path = base / "data" / "processed" / "customer_features.csv"

    if local_path.exists():
        return pd.read_csv(local_path)

    fallback_url = os.getenv("LOANAPPROVAL_DATA_URL", DEFAULT_DATA_URL)

    try:
        df = pd.read_csv(fallback_url)
    except Exception as exc:
        raise FileNotFoundError(
            f"Dataset not found at {local_path} and fallback download failed "
            f"from {fallback_url}."
        ) from exc

    # Best-effort local cache for subsequent loads.
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(local_path, index=False)
    except OSError:
        pass

    return df
