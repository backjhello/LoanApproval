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
    st.subheader("📌 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Avg Total Spent", f"${df['total_spent'].mean():.2f}")
    col3.metric("Avg Transaction Count", f"{df['transaction_count'].mean():.1f}")
    col4.metric("Avg Spending Variability (std)", f"{df['spending_std'].mean():.2f}")


# ---------------------------------------------------------
# 3. SPENDING COMPOSITION
# ---------------------------------------------------------
with tab3:
    st.subheader("🧁 Overall Spending Composition")

    spend_cols = ['luxury', 'necessity', 'wellbeing', 'misc']
    total = df[spend_cols].mean()

    fig, ax = plt.subplots()
    ax.pie(total, labels=total.index, autopct='%1.1f%%', startangle=90)
    ax.add_artist(plt.Circle((0,0), 0.6, color='white'))
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
    st.subheader("📚 Feature Description")

    feature_info = {
        "total_spent": "Total spending amount per customer.",
        "avg_transaction": "Average amount spent per transaction.",
        "transaction_count": "Total number of purchases.",
        "spending_std": "Spending volatility.",
        "luxury": "Spending proportion in luxury categories.",
        "necessity": "Essential spending categories.",
        "wellbeing": "Health, home, and family-related spending.",
        "misc": "Unclassified or irregular spending."
    }

    info_df = pd.DataFrame.from_dict(feature_info, orient='index', columns=['Description'])
    st.dataframe(info_df)