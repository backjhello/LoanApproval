import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.loader import load_customer_features

# Load data
df = load_customer_features()

# 🎨 Global seaborn theme for prettier charts
sns.set_theme(
    style="whitegrid",
    rc={
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": "lightgray",
        "grid.color": "lightgray",
        "axes.labelsize": 12,
        "axes.titlesize": 17,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11
    }
)

# ================================================
#                    TABS
# ================================================
tab1, tab2, tab3 = st.tabs(["📊 Basic EDA", "👥 Age Analysis", "🌆 City Analysis"])


# ======================================================
#                    TAB 1 — BASIC EDA
# ======================================================
with tab1:
    st.header("📊 Basic Exploratory Data Analysis")

    st.subheader("Preview of Dataset")
    st.dataframe(df.head())

    st.subheader("Summary Statistics")
    st.dataframe(df.describe().T)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(df[['total_spent','avg_transaction','transaction_count','spending_std']].corr(),
                annot=True, cmap="Blues", ax=ax)
    ax.set_title("Correlation Between Key Variables")
    st.pyplot(fig)

    st.markdown("""
    ### Key Observations
    - **total_spent** is moderately correlated with **transaction_count** and **avg_transaction**.  
    - **spending_std** shows weaker correlation, meaning volatility is not directly tied to purchase amounts.  
    - These relationships provide a baseline understanding before deeper grouping analysis.  
    """)



# ======================================================
#                    TAB 2 — AGE ANALYSIS
# ======================================================
with tab2:
    st.header("👥 Deep Analysis: Age Groups")

    # Create age groups
    bins = [0, 25, 35, 45, 55, 65, 120]
    labels = ['<25', '25–34', '35–44', '45–54', '55–64', '65+']
    df['age_group'] = pd.cut(df['avg_age'], bins=bins, labels=labels, right=False)

    # ---------- Spending Ratio ----------
    st.subheader("📌 Average Spending Ratio by Age Group")

    age_spend = df.groupby('age_group')[['luxury','necessity','wellbeing','misc']].mean().reset_index()
    st.dataframe(age_spend)

    fig, ax = plt.subplots(figsize=(8,5))
    palette = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

    age_spend.set_index("age_group").plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=palette,
        edgecolor="none"
    )
    ax.set_title("Average Spending Ratio by Age Group", pad=20)
    sns.despine()
    st.pyplot(fig)

    st.markdown("""
    ### Insights  
    - Customers **under 25** show higher luxury and miscellaneous spending, indicating more impulsive patterns.  
    - Ages **25–64** demonstrate stable, necessity-dominant spending behavior.  
    - The **65+** group shows increasing variability again, possibly due to irregular expenses.  
    """)

    # ---------- Volatility ----------
    st.subheader("📌 Spending Volatility (spending_std) by Age Group")

    age_std = df.groupby('age_group')['spending_std'].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    sns.barplot(data=age_std, x="age_group", y="spending_std", palette="Blues", ax=axes[0])
    axes[0].set_title("Average Spending Std by Age Group")

    sns.boxplot(data=df, x="age_group", y="spending_std", palette="coolwarm", ax=axes[1])
    axes[1].set_title("Distribution of Spending Std by Age Group")

    st.pyplot(fig)

    st.markdown("""
    ### Volatility Insights  
    - The **<25** group shows the highest volatility with many outliers.  
    - **25–64** groups show stable and predictable spending behavior.  
    - **65+** customers experience increased volatility again, potentially reflecting lifestyle or medical-related variability.  
    """)

    # ---------- Total Spending & Avg Transaction ----------
    st.subheader("📌 Total Spending & Avg Transaction by Age Group")

    age_spend2 = df.groupby('age_group')[['total_spent','avg_transaction']].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    sns.barplot(data=age_spend2, x="age_group", y="total_spent", palette="Purples", ax=axes[0])
    axes[0].set_title("Average Total Spending")

    sns.barplot(data=age_spend2, x="age_group", y="avg_transaction", palette="Greens", ax=axes[1])
    axes[1].set_title("Average Transaction Amount")

    st.pyplot(fig)

    st.markdown("""
    ### Spending Insights  
    - Ages **25–44** record the highest total spending overall.  
    - The **45–64** group spends less, possibly due to conservative financial habits.  
    - The **<25** and **65+** groups show lower total spending but higher volatility.  
    """)




# ======================================================
#                    TAB 3 — CITY ANALYSIS
# ======================================================
with tab3:
    st.header("🌆 Deep Analysis: City Population Groups")

    bins = [0, 20000, 50000, 200000, 500000, float("inf")]
    labels = ['Very Small (<20k)', 'Small (20k–50k)', 'Medium (50k–200k)', 'Large (200k–500k)', 'Metro (500k+)']
    df['city_group'] = pd.cut(df['avg_city_pop'], bins=bins, labels=labels, right=False)

    # ---------- Spending Ratio ----------
    st.subheader("📌 Average Spending Ratio by City Group")

    city_spend = df.groupby('city_group')[['luxury','necessity','wellbeing','misc']].mean().reset_index()
    st.dataframe(city_spend)

    fig, ax = plt.subplots(figsize=(8,5))
    city_spend.set_index("city_group").plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=["#4c72b0", "#dd8452", "#55a868", "#c44e52"],
        edgecolor="none"
    )
    ax.set_title("Average Spending Ratio by City Group", pad=20)
    st.pyplot(fig)

    st.markdown("""
    ### Insights  
    - Spending distribution remains relatively stable across city sizes.  
    - **Metro** areas show slightly higher necessity spending.  
    - Medium and large cities display a lower share of wellbeing spending.  
    """)

    # ---------- Volatility ----------
    st.subheader("📌 Spending Volatility by City Group")

    city_std = df.groupby('city_group')['spending_std'].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    sns.barplot(data=city_std, x="city_group", y="spending_std", palette="viridis", ax=axes[0])
    axes[0].set_title("Average Spending Std by City Group")

    sns.boxplot(data=df, x="city_group", y="spending_std", palette="coolwarm", ax=axes[1])
    axes[1].set_title("Distribution of Spending Std by City Group")

    st.pyplot(fig)

    st.markdown("""
    ### Volatility Insights  
    - **Very Small (<20k)** cities show the most stable and predictable spending.  
    - **Metro** populations show increased volatility, possibly due to lifestyle diversity.  
    - Small and medium cities fall in the mid-range of volatility.  
    """)

    # ---------- Total Spending & Avg Transaction ----------
    st.subheader("📌 Total Spending & Avg Transaction by City Group")

    city_spend2 = df.groupby('city_group')[['total_spent','avg_transaction']].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    sns.barplot(data=city_spend2, x="city_group", y="total_spent", palette="Purples", ax=axes[0])
    axes[0].set_title("Average Total Spending")

    sns.barplot(data=city_spend2, x="city_group", y="avg_transaction", palette="Oranges", ax=axes[1])
    axes[1].set_title("Average Transaction Amount")

    st.pyplot(fig)

    st.markdown("""
    ### Spending Insights  
    - **Medium** and **Small** cities have the highest total spending.  
    - **Metro** areas spend less in total, which may reflect frequent low-value purchases.  
    """)

    # ---------- ANOVA ----------
    st.subheader("📌 Statistical Tests (ANOVA)")

    from scipy.stats import f_oneway

    groups_total = [g["total_spent"].dropna() for _, g in df.groupby("city_group")]
    F, p = f_oneway(*groups_total)
    st.write(f"**ANOVA for Total Spending:** F = {F:.3f}, p = {p:.3e}")

    groups_avg = [g["avg_transaction"].dropna() for _, g in df.groupby("city_group")]
    F2, p2 = f_oneway(*groups_avg)
    st.write(f"**ANOVA for Avg Transaction:** F = {F2:.3f}, p = {p2:.3e}")

    st.markdown("""
    ### Interpretation  
    - There is **no significant difference** in total spending across city groups (p > 0.05).  
    - There **is** a statistically significant difference in average transaction amounts (p < 0.05).  
    """)
