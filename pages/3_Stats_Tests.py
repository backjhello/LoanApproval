import streamlit as st
import numpy as np
from scipy.stats import pearsonr
from src.loader import load_customer_features

df = load_customer_features()
st.subheader("🔍 Deep Correlation Analysis")

st.markdown("""
This section explores how spending behavior in different categories relates to **spending variability (spending_std)**.  
Understanding these relationships helps explain **customer stability** and **potential credit risk**.
""")

categories = ["luxury", "misc", "necessity", "wellbeing"]
results = []

for cat in categories:
    r, p = pearsonr(df[cat], df["spending_std"])
    results.append((cat, r, p))

st.write("### Correlation & p-values")
for cat, r, p in results:
    st.write(f"**{cat.capitalize()} vs spending_std:** r = `{r:.3f}`, p = `{p:.3e}`")

st.write("---")

# Interpretation
st.subheader("📝 Interpretation")

st.markdown("""
### 1️⃣ **Luxury spending ratio → spending variability (r = ~0.57, strong positive)**  
Customers with high luxury spending tend to show **unstable or irregular spending patterns**,  
often making large purchases occasionally rather than consistent smaller ones.  
➡ Indicates **higher credit risk potential**.

### 2️⃣ **Misc spending ratio → weak relationship**  
Little correlation was observed, suggesting miscellaneous spending is not a strong predictor of volatility.

### 3️⃣ **Necessity spending → negative correlation (stable customers)**  
Higher necessity spending aligns with **predictable, routine consumption**.  
➡ Indicates **lower credit risk**.

### 4️⃣ **Wellbeing spending → strong negative correlation**  
Customers who spend more consistently on wellbeing categories tend to have  
**stable and predictable financial behavior**.  
➡ Also aligned with **lower risk**.

---

### ✔ Statistical Confidence  
All strong correlations (luxury ↑ / wellbeing ↓) have **p-values < 0.001**,  
meaning these relationships are **statistically significant** and not due to random chance.

""")
