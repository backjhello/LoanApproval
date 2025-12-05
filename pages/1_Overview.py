import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.loader import load_customer_features

st.set_page_config(page_title="Dataset Overview", layout="wide")
st.title("📄 Dataset Overview")

df = load_customer_features()
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔁 Pipeline", 
    "📌 Metrics", 
    "🧁 Spending Composition",
    "📈 Distributions",
    "🔗 Correlations",
    "📚 Feature Description"
])

# ---------------------------------------------------------
# 1. PIPELINE TAB
# ---------------------------------------------------------
with tab1:
    st.subheader("🔁 Full Project Data Pipeline")

    st.markdown("""
    Below is the full transformation path from **1.3M raw transactions**  
    → **customer_features dataset** → **job clustering** → **modeling dataset**.

    ```
    Raw Transactions (1.3M rows)
        └─ groupby(cc_num)
           └─ Compute total_spent, avg_transaction, transaction_count, spending_std
              └─ Pivot 14 categories → 4 spending types
                 └─ Merge into customer_features
                    └─ Job-level aggregation (494 jobs)
                       └─ KMeans clustering on job_features
                          └─ Merge job_cluster back to customer_features
                             └─ Modeling (Risk Score + Loan Approval)
    ```
    """)

    st.info("This page displays the *final engineered dataset* used for clustering and modeling.")


# ---------------------------------------------------------
# 2. METRICS TAB
# ---------------------------------------------------------
with tab2:
    st.subheader("📌 Key Metrics Dashboard")

    # ======================================================
    # 1️⃣ Basic Customer Overview
    # ======================================================
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Total Spending (All Customers)", f"${df['total_spent'].sum():,.0f}")
    c3.metric("Avg Spending per Customer", f"${df['total_spent'].mean():.2f}")
    c4.metric("Median Spending", f"${df['total_spent'].median():.2f}")

    # ======================================================
    # 2️⃣ Behavior & Risk Indicators
    # ======================================================
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Avg Transaction Count", f"{df['transaction_count'].mean():.1f}")
    c6.metric("Avg Transaction Amount", f"${df['avg_transaction'].mean():.2f}")
    c7.metric("Avg Spending Volatility (std)", f"{df['spending_std'].mean():.3f}")
    c8.metric("High-Volatility Customers", f"{(df['spending_std'] > df['spending_std'].median()).sum():,}")

    # ======================================================
    # 3️⃣ Spending Composition KPI
    # ======================================================
    st.subheader("🧁 Spending Category Ratios")

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Luxury Ratio", f"{df['luxury'].mean():.1%}")
    c10.metric("Necessity Ratio", f"{df['necessity'].mean():.1%}")
    c11.metric("Wellbeing Ratio", f"{df['wellbeing'].mean():.1%}")
    c12.metric("Misc Ratio", f"{df['misc'].mean():.1%}")

    # ======================================================
    # 4️⃣ Cluster-Level Metrics (job cluster)
    # ======================================================
    if "cluster" in df.columns:
        st.subheader("🔮 Segment (Cluster) Overview")

        cluster_summary = df.groupby("cluster").agg({
            "total_spent":"mean",
            "transaction_count":"mean",
            "spending_std":"mean"
        }).rename(columns={
            "total_spent":"Avg Spend",
            "transaction_count":"Avg Tx Count",
            "spending_std":"Avg Volatility"
        })

        st.dataframe(cluster_summary.style.format({
            "Avg Spend": "${:.1f}",
            "Avg Tx Count": "{:.1f}",
            "Avg Volatility": "{:.3f}"
        }))
    else:
        st.info("Cluster information not found. If you merge job_features → customer_features, this table will appear.")

    # ======================================================
    # 5️⃣ Auto Insights
    # ======================================================
    st.subheader("📝 Automated Insights")

    high_spenders = df[df['total_spent'] > df['total_spent'].quantile(0.9)]
    high_risk = df[df['spending_std'] > df['spending_std'].quantile(0.9)]

    st.markdown(f"""
    ### Key Findings
    - **Top 10% spenders:** {len(high_spenders):,} customers  
    - **Top 10% volatility customers (risk-prone):** {len(high_risk):,} customers  
    - Customers spend **{df['luxury'].mean():.1%}** of their money on luxury categories  
    - Essential spending accounts for **{df['necessity'].mean():.1%}**  
    - Cluster 0 tends to spend the most (if cluster info exists)
    """)


# ---------------------------------------------------------
# 3. SPENDING COMPOSITION
# ---------------------------------------------------------
with tab3:
    st.subheader("🧁 Spending Composition (Enhanced Donut Chart)")

    spend_cols = ['luxury', 'necessity', 'wellbeing', 'misc']
    total = df[spend_cols].mean()

    colors = ["#6A5ACD", "#00BFA6", "#FF6F61", "#FFC300"]  # 보라/민트/코랄/옐로

    fig, ax = plt.subplots(figsize=(6,6))
    wedges, texts, autotexts = ax.pie(
        total,
        labels=total.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        pctdistance=0.75,
        textprops={'color':'#333', 'fontsize':11}
    )

    # Donut hole
    centre = plt.Circle((0,0), 0.55, fc='white')
    fig.gca().add_artist(centre)

    # Center Text
    ax.text(0, 0, "Avg\nSpend", ha='center', va='center', fontsize=14, weight='bold')

    st.pyplot(fig)

    st.caption("Average proportion of spending across all customers.")


# ---------------------------------------------------------
# 4. DISTRIBUTIONS TAB
# ---------------------------------------------------------
with tab4:
    st.subheader("📈 Key Variable Distributions")

    var = st.selectbox(
        "Select a variable to explore:",
        ['total_spent', 'avg_transaction', 'transaction_count', 'spending_std']
    )

    fig, ax = plt.subplots()
    sns.histplot(df[var], kde=True, ax=ax)
    ax.set_title(f"Distribution of {var}")
    st.pyplot(fig)


# ---------------------------------------------------------
# 5. CORRELATION TAB
# ---------------------------------------------------------
with tab5:
    st.subheader("🔗 Correlation Snapshot")

    corr = df[['total_spent','avg_transaction','transaction_count','spending_std']].corr()

    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
    st.pyplot(fig)


# ---------------------------------------------------------
# 6. FEATURE DESCRIPTION TAB
# ---------------------------------------------------------
with tab6:
    st.subheader("📚 Full Feature Description (Customer Dataset)")

    feature_info = {
        # 1) Spending summary
        "total_spent": "Total spending amount for the customer across all transactions.",
        "avg_transaction": "Average amount per transaction.",
        "transaction_count": "Total number of transactions made by the customer.",
        "spending_std": "Volatility of spending patterns (higher = more unpredictable).",

        # 2) Spending category ratios
        "luxury": "Proportion of spending in luxury/leisure categories (entertainment, shopping, travel).",
        "necessity": "Proportion of essential spending (grocery, gas, personal care).",
        "wellbeing": "Proportion of wellbeing-related spending (health, home, family).",
        "misc": "Proportion of miscellaneous/unclassified spending.",

        # 3) Customer demographic signals
        "avg_age": "Average customer age (calculated from raw transaction age fields).",
        "avg_city_pop": "Average population of cities where the customer made purchases.",
        "category_diversity": "Number of unique merchant categories used by the customer.",
        "fraud_rate": "Historical fraud rate (mean of is_fraud for customer's transactions).",

        # 4) Job-related info
        "job": "Customer's job category (string label).",
        "cluster": "Job-level spending pattern cluster (K-Means based on job_features)."
    }

    info_df = pd.DataFrame.from_dict(feature_info, orient='index', columns=['Description'])
    st.dataframe(info_df)