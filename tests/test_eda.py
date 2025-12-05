import pandas as pd
from src.eda import describe_df, missing_summary, anova_by_group


def test_describe_and_missing():
    df = pd.DataFrame({'a': [1, 2, None], 'b': ['x', 'y', 'z']})
    desc = describe_df(df)
    assert 'dtype' in desc.columns and 'n_missing' in desc.columns

    miss = missing_summary(df)
    assert 'n_missing' in miss.columns


def test_anova():
    df = pd.DataFrame({'group': ['A', 'A', 'B', 'B'], 'val': [1.0, 1.2, 2.1, 2.3]})
    f, p = anova_by_group(df, 'group', 'val')
    assert p >= 0.0 and p <= 1.0
