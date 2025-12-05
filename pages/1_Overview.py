import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.loader import load_customer_features

st.title("📄 Project Overview")

df = load_customer_features()

# --- High-level Metrics ---
st.subheader("📌 Key Dataset Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{len(df):,}")
col2.metric("Avg. Total Spending", f"${df['total_spent'].mean():.2f}")
col3.metric("Avg. Transaction Value", f"${df['avg_transaction'].mean():.2f}")
col4.metric("Avg. Transaction Count", f"{df['transaction_count'].mean():.1f}")

st.write("---")

# --- Preview & Description ---
st.subheader("📊 Dataset Preview")
st.write("""
The dataset contains aggregated customer features engineered from raw transaction logs.  
These features summarize a customer’s spending behavior, category preferences, and behavioral variability.
""")
st.dataframe(df.head())

st.write("---")

# --- Summary Statistics ---
st.subheader("📈 Summary Statistics")
st.dataframe(df.describe().T)

st.write("---")

# --- Missing Value Heatmap ---
st.subheader("🧩 Missing Value Overview")
fig, ax = plt.subplots(figsize=(10,4))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
ax.set_title("Missing Value Heatmap")
st.pyplot(fig)

st.write("""
Even though the dataset is mostly complete, this visualization helps confirm data quality  
before proceeding with analysis such as EDA, clustering, and credit-limit modeling.
""")