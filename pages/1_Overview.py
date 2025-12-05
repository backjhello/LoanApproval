import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.loader import load_customer_features

st.title("📄 Dataset Overview")

df = load_customer_features()

# ----------------------------
# 1. High-level Metrics
# ----------------------------
st.subheader("📌 Key Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{len(df):,}")
col2.metric("Avg Total Spent", f"${df['total_spent'].mean():.2f}")
col3.metric("Avg Transaction Count", f"{df['transaction_count'].mean():.1f}")
col4.metric("Avg Spending Variability (std)", f"{df['spending_std'].mean():.2f}")

st.write("---")


# ----------------------------
# 2. Spending Composition Donut
# ----------------------------
st.subheader("🧁 Overall Spending Composition")

spend_cols = ['luxury', 'necessity', 'wellbeing', 'misc']
total = df[spend_cols].mean()

fig, ax = plt.subplots()
ax.pie(total, labels=total.index, autopct='%1.1f%%', startangle=90)
ax.add_artist(plt.Circle((0,0), 0.6, color='white'))
st.pyplot(fig)

st.write("This chart illustrates the average distribution of spending across all customers.")


st.write("---")


# ----------------------------
# 3. Distribution Overview
# ----------------------------
st.subheader("📊 Key Variable Distributions")

var = st.selectbox(
    "Select a variable to explore:",
    ['total_spent', 'avg_transaction', 'transaction_count', 'spending_std']
)

fig, ax = plt.subplots()
sns.histplot(df[var], kde=True, ax=ax)
ax.set_title(f"Distribution of {var}")
st.pyplot(fig)

st.write("---")


# ----------------------------
# 4. Correlation Snapshot
# ----------------------------
st.subheader("🔗 Correlation Snapshot")

corr = df[['total_spent','avg_transaction','transaction_count','spending_std']].corr()

fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
st.pyplot(fig)

st.write("---")


# ----------------------------
# 5. Feature Description Table
# ----------------------------
st.subheader("📚 Feature Description")

feature_info = {
    "total_spent": "Total spending amount of the customer.",
    "avg_transaction": "Average price per transaction.",
    "transaction_count": "Total number of purchases.",
    "spending_std": "Variability in spending behavior.",
    "luxury": "Proportion of spending in luxury category.",
    "necessity": "Proportion of necessity spending.",
    "wellbeing": "Proportion of wellbeing-related spending.",
    "misc": "Proportion of miscellaneous spending."
}

info_df = pd.DataFrame.from_dict(feature_info, orient='index', columns=['Description'])
st.dataframe(info_df)

st.write("---")


# ----------------------------
# 6. Insights
# ----------------------------
st.subheader("📝 High-Level Insights")

st.markdown("""
- Customers tend to maintain stable spending patterns, with moderate variability.
- Luxury and necessity categories represent the majority of customer spending.
- Total spending shows moderate correlation with transaction count and average transaction value.
- These patterns provide strong foundations for clustering and credit-limit modeling.
""")
