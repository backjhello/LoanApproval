import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_distribution(df, col):
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(df[col], kde=True, ax=ax, color="#1f77b4")
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    return fig

def plot_spending_by_age(df):
    fig, ax = plt.subplots(figsize=(8,5))
    sns.boxplot(data=df, x='age_group', y='spending_std', ax=ax)
    ax.set_title("Spending Volatility by Age Group")
    return fig

def plot_city_spending(df):
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(data=df, x='city_group', y='total_spent', ax=ax)
    ax.set_title("Total Spending by City Population Group")
    return fig


def plot_missingness(df):
    """Return a bar plot showing number of missing values per column (descending)."""
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=miss.values, y=miss.index, ax=ax, palette='rocket')
    ax.set_xlabel('Missing count')
    ax.set_title('Missing values by column')
    return fig


def plot_correlation_heatmap(df, cols=None):
    """Return a correlation heatmap figure for specified cols (or all numeric cols)."""
    if cols is not None:
        data = df[cols]
    else:
        data = df.select_dtypes(include=[np.number])
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', center=0, ax=ax)
    ax.set_title('Correlation matrix')
    return fig


def plot_feature_importance(model, feature_names=None, top_n: int = 10):
    """Return a bar plot of feature importances from a fitted tree-based model.

    If feature_names is None, uses range indices.
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError('Model has no feature_importances_ attribute')
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    names = feature_names if feature_names is not None else [f'feat_{i}' for i in range(len(importances))]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=importances[idx], y=[names[i] for i in idx], ax=ax, palette='mako')
    ax.set_xlabel('Importance')
    ax.set_title('Feature importances')
    return fig

