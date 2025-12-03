import matplotlib.pyplot as plt
import seaborn as sns

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

