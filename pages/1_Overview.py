import streamlit as st
import pandas as pd
import plotly.express as px

from src.loader import load_df, load_customer_features

st.set_page_config(page_title="Overview", page_icon="📊")

# Load data
df = load_df()
customer = load_customer_features()

st.title("📊 Dataset Overview")
st.write("This page provides a high-level summary of the transaction dataset and the engineered customer dataset.")

# ================================
# SECTION 1 — HIGH LEVEL METRICS
# ================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Transactions", f"{len(df):,}")

with col2:
    st.metric("Unique Customers", f"{df['cc_num'].nunique():,}")

with col3:
    avg_spend = df['amount'].mean()
    st.metric("Avg. Transaction Amount", f"${avg_spend:,.2f}")

with col4:
    fraud_rate = df['is_fraud'].mean() * 100
    st.metric("Fraud Rate", f"{fraud_rate:.2f}%")

st.divider()

# ================================
# SECTION 2 — CATEGORY DISTRIBUTION
# ================================

st.subheader("🛍️ Transaction Category Distribution")

category_counts = df['category'].value_counts().reset_index()
category_counts.columns = ["category", "count"]

fig_cat = px.bar(
    category_counts,
    x="category",
    y="count",
    color="category",
    title="Number of Transactions per Category",
)
st.plotly_chart(fig_cat, use_container_width=True)

st.divider()

# ================================
# SECTION 3 — SPENDING DISTRIBUTION
# ================================

st.subheader("💸 Distribution of Transaction Amounts")

fig_amount = px.histogram(
    df,
    x="amount",
    nbins=50,
    title="Transaction Amount Distribution",
)
st.plotly_chart(fig_amount, use_container_width=True)

st.divider()

# ================================
# SECTION 4 — CUSTOMER FEATURES SUMMARY
# ================================

st.subheader("👥 Customer Feature Summary")

colA, colB = st.columns(2)

with colA:
    st.metric("Avg Total Spending per Customer", f"${customer['total_spent'].mean():,.2f}")

with colB:
    st.metric("Avg # Transactions per Customer", f"{customer['transaction_count'].mean():,.2f}")

st.write("### Spending Category Breakdown (Mean)")

category_cols = ["luxury", "misc", "necessity", "wellbeing"]
mean_categories = customer[category_cols].mean().reset_index()
mean_categories.columns = ["category", "avg"]

fig_cat2 = px.bar(
    mean_categories,
    x="category",
    y="avg",
    color="category",
    title="Average Category Spending per Customer",
)
st.plotly_chart(fig_cat2, use_container_width=True)

st.divider()

# ================================
# SECTION 5 — CLUSTER DISTRIBUTION
# ================================

st.subheader("🔍 Customer Segment Distribution (Clusters)")

cluster_counts = customer["cluster"].value_counts().reset_index()
cluster_counts.columns = ["cluster", "count"]

fig_cluster = px.pie(
    cluster_counts,
    names="cluster",
    values="count",
    title="Cluster Breakdown",
)
st.plotly_chart(fig_cluster, use_container_width=True)
