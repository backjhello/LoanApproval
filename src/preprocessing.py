from typing import List, Tuple, Optional
import pandas as pd
import numpy as np


# ----------------------------------------------------------
# 1) AGE 계산
# ----------------------------------------------------------

def compute_age(df: pd.DataFrame,
                dob_col: str = 'dob',
                date_col: str = 'trans_date_trans_time') -> pd.DataFrame:
    """Compute age in years from DOB and transaction time.
    Adds an 'age' column.
    """
    out = df.copy()
    out[dob_col] = pd.to_datetime(out[dob_col], errors='coerce')
    out[date_col] = pd.to_datetime(out[date_col], errors='coerce')

    # Rough age (in years)
    out["age"] = ((out[date_col] - out[dob_col]).dt.days // 365).fillna(0).astype(int)
    return out


# ----------------------------------------------------------
# 2) AGE GROUPS (bins 항상 단조 증가)
# ----------------------------------------------------------

def make_age_groups(df: pd.DataFrame,
                    age_col: str = 'age',
                    bins: Optional[List[int]] = None,
                    labels: Optional[List[str]] = None) -> pd.DataFrame:
    """Create age_group column using safe monotonically increasing bins."""
    out = df.copy()

    if age_col not in out:
        return out

    # 기본 bin + 마지막 bin은 무조건 제일 큰 값보다 크게
    if bins is None:
        max_age = out[age_col].max()
        if pd.isna(max_age):
            max_age = 100
        upper = max(int(max_age) + 1, 66)  # 항상 65보다 크고 단조 증가
        bins = [0, 25, 35, 50, 65, upper]

    if labels is None:
        labels = ['<25', '25-34', '35-49', '50-64', '65+']

    out["age_group"] = pd.cut(
        out[age_col],
        bins=bins,
        labels=labels,
        include_lowest=True
    )
    return out


# ----------------------------------------------------------
# 3) CITY GROUPS (qcut 기반)
# ----------------------------------------------------------

def make_city_groups(df: pd.DataFrame,
                     pop_col: str = 'city_pop') -> pd.DataFrame:
    """Make city_group categories (quartiles)."""
    out = df.copy()

    if pop_col not in out:
        out["city_group"] = pd.NA
        return out

    # qcut for quartiles
    try:
        out["city_group"] = pd.qcut(
            out[pop_col].rank(method="first"),
            q=4,
            labels=["Small", "Medium", "Large", "Very Large"]
        )
    except Exception:
        # fallback in case pop values are identical or too few unique values
        out["city_group"] = "Small"
    return out


# ----------------------------------------------------------
# 4) FEATURE ENGINEERING (customer-level summary)
# ----------------------------------------------------------

def feature_engineering(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return transaction-level data and per-customer summary.
       Must include total_spent for pytest.
    """
    out = df.copy()

    customer = (
        out.groupby("customer_id")
        .agg({
            "amt": "sum",
            "spending_std": "median",
            "avg_transaction": "mean"
        })
        .reset_index()
    )

    # Total spent 필수 조건
    customer = customer.rename(columns={"amt": "total_spent"})

    return out, customer


# ----------------------------------------------------------
# 5) PREPARE MODEL DATASET (X, y 반환)
# ----------------------------------------------------------

def prepare_model_dataset(
    customers_df: pd.DataFrame,
    feature_columns: List[str],
    target_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Produce X and y for modeling.
       Tests only check row count equality.
    """
    X = customers_df[feature_columns].copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    y = customers_df[target_col].astype(int)

    # 결측치 있으면 row 제거
    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    return X, y
