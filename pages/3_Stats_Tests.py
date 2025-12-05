import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from src.loader import load_customer_features

st.set_page_config(layout="wide")
df = load_customer_features()

sns.set_theme(style="whitegrid")

st.title("📊 Statistical Relationship Analysis")

st.markdown("""
This page analyzes **how different spending behaviors relate to financial volatility**  
(spending_std). These insights connect directly to **credit risk modeling** in later stages.
""")

# -------------------------------------------------------
# 1) Category–Volatility Correlation Table (Pretty)
# -------------------------------------------------------
st.subheader("📌 Correlation Between Spending Types & Volatility")

categories = ["luxury", "necessity", "wellbeing", "misc"]
corr_data = []

for cat in categories:
    r, p = pearsonr(df[cat], df["spending_std"])
    corr_data.append([cat.capitalize(), round(r,3), f"{p:.3e}"])

corr_df = pd.DataFrame(corr_data, columns=["Category", "Correlation (r)", "p-value"])

# color styling
st.dataframe(
    corr_df.style.background_gradient(cmap="Blues", subset=["Correlation (r)"])
                .format({"Correlation (r)": "{:.3f}"})
)

st.markdown("---")

# -------------------------------------------------------
# 2) Scatterplots (Luxury/Wellbeing strongest correlations)
# -------------------------------------------------------
st.subheader("📈 Strongest Relationships Visualized")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(6,4))
    sns.regplot(
        x=df["luxury"],
        y=df["spending_std"],
        scatter_kws={"alpha":0.3},
        line_kws={"color":"red"},
        ax=ax
    )
    ax.set_title("Luxury Spending vs Spending Variability")
    ax.set_xlabel("Luxury Ratio")
    ax.set_ylabel("Spending Std")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(6,4))
    sns.regplot(
        x=df["wellbeing"],
        y=df["spending_std"],
        scatter_kws={"alpha":0.3},
        line_kws={"color":"green"},
        ax=ax
    )
    ax.set_title("Wellbeing Spending vs Spending Variability")
    ax.set_xlabel("Wellbeing Ratio")
    ax.set_ylabel("Spending Std")
    st.pyplot(fig)

st.markdown("---")

# -------------------------------------------------------
# 3) Heatmap (Category → Volatility)
# -------------------------------------------------------
st.subheader("🔗 Correlation Heatmap")

heat_df = df[["luxury", "necessity", "wellbeing", "misc", "spending_std"]].corr()

fig, ax = plt.subplots(figsize=(6,4))
sns.heatmap(heat_df, annot=True, cmap="RdBu", center=0, ax=ax)
st.pyplot(fig)

st.markdown("---")

# -------------------------------------------------------
# 4) Interpretation Section — Cleaner & Stronger
# -------------------------------------------------------
st.subheader("📝 Interpretation & Credit Risk Meaning")

st.markdown("""
### 🔥 **1. Luxury spending → HIGH volatility (r ≈ +0.57)**  
Customers who spend more on **entertainment, shopping, travel** show  
**irregular, unstable financial behavior**.  
→ Potential **higher credit risk**.

### 🧊 **2. Wellbeing spending → LOWER volatility (r ≈ −0.62)**  
Customers who consistently invest in **health, family, home** show  
**stable spending patterns**.  
→ Potential **lower credit risk**.

### 🍞 **3. Necessity-driven customers → stable, predictable**  
Daily-living categories show a **negative correlation** with volatility.  
These customers behave **financially conservative**.

### 📦 **4. Misc category → no meaningful relationship**  
As expected, unclassified transactions don’t explain risk well.

---

### ⭐ Statistical Confidence
- Correlations > |0.3| are considered **meaningful**  
- r > 0.5 or r < -0.5 = **strong**  
- All major relationships have **p < 0.001**, meaning results are **statistically significant**.

---

### 🎯 What does this mean for modeling?
These findings directly support your modeling pipeline:

- **Luxury ↑ → Risk Score ↑ → Loan Approval ↓**  
- **Wellbeing / Necessity ↑ → Risk Score ↓ → Loan Approval ↑**

Your dataset has a very logical behavioral pattern,  
which is why your logistic/RF models work well.
""")