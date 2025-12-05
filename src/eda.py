from typing import Tuple
import pandas as pd
from scipy.stats import f_oneway


def anova_by_group(df: pd.DataFrame, group_col: str, value_col: str) -> Tuple[float, float]:
    """Run a one-way ANOVA test for `value_col` across groups defined by `group_col`.

    Returns (F-statistic, p-value).
    """
    groups = [sub[value_col].dropna() for _, sub in df.groupby(group_col)]
    if len(groups) < 2:
        raise ValueError("Need at least two groups to run ANOVA")
    f, p = f_oneway(*groups)
    return f, p


def describe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a concise description table with dtype and missing counts for each column."""
    return pd.DataFrame({
        'dtype': df.dtypes.astype(str),
        'n_missing': df.isna().sum(),
        'n_unique': df.nunique(dropna=False)
    })


def correlation_matrix(df: pd.DataFrame, numeric_only: bool = True) -> pd.DataFrame:
    """Return the correlation matrix for numeric columns (default) or all columns if requested."""
    return df.corr() if numeric_only else df.corr(method='pearson')


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a series/dataframe summarizing missing values per column (sorted descending)."""
    s = df.isna().sum()
    return s[s > 0].sort_values(ascending=False).to_frame('n_missing')

