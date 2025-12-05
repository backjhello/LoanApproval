from typing import List, Tuple, Optional
import pandas as pd


def compute_age(df: pd.DataFrame, dob_col: str = 'dob', date_col: str = 'trans_date_trans_time') -> pd.DataFrame:
    """Compute age in years from date of birth and transaction time.

    Adds `age` column to a copy of the dataframe and returns it.
    """
    out = df.copy()
    out[dob_col] = pd.to_datetime(out[dob_col], errors='coerce')
    out[date_col] = pd.to_datetime(out[date_col], errors='coerce')
    out['age'] = ((out[date_col] - out[dob_col]).dt.days // 365).fillna(0).astype(int)
    return out


def make_age_groups(df: pd.DataFrame, age_col: str = 'age', bins: Optional[List[int]] = None,
                    labels: Optional[List[str]] = None) -> pd.DataFrame:
    out = df.copy()
    if bins is None:
        bins = [0, 25, 35, 50, 65, int(out[age_col].max() if age_col in out else 100)]
    if labels is None:
        labels = ['<25', '25-34', '35-49', '50-64', '65+']
    out['age_group'] = pd.cut(out[age_col], bins=bins, labels=labels)
    return out


def make_city_groups(df: pd.DataFrame, pop_col: str = 'city_pop') -> pd.DataFrame:
    out = df.copy()
    if pop_col not in out:
        out['city_group'] = pd.NA
        return out
    out['city_group'] = pd.qcut(out[pop_col].rank(method='first'), q=4, labels=['Small', 'Medium', 'Large', 'Very Large'])
    return out


def feature_engineering(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return the original (possibly enriched) dataframe and a per-customer summary dataframe.

    The customer summary contains several aggregate features used by pages and modeling.
    """
    out = df.copy()
    customer = out.groupby('customer_id').agg({
        'amt': 'sum',
        'spending_std': 'median',
        'avg_transaction': 'mean'
    }).reset_index()
    # normalize column names to expected names used across pages
    customer = customer.rename(columns={'amt': 'total_spent'})
    return out, customer


def prepare_model_dataset(customers_df: pd.DataFrame, feature_columns: List[str], target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Given `customers_df` produce X (features) and y (target) ready for training.

    - fills missing values with 0 by default
    - casts to numeric where possible
    """
    X = customers_df[feature_columns].copy()
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    y = customers_df[target_col].copy()
    return X, y
