import streamlit as st
from src.loader import load_df       # ← FIXED
from src.eda import anova_by_group

df = load_df()                       # ← FIXED

st.title("📈 Statistical Significance Tests")

value = st.selectbox("Value to test", ["total_spent", "avg_transaction"])
group = st.selectbox("Group variable", ["age_group", "city_group"])

f, p = anova_by_group(df, group, value)

st.write(f"### ANOVA Results for {value} across {group}")
st.write(f"**F-statistic:** {f:.4f}")
st.write(f"**p-value:** {p:.3e}")

if p < 0.05:
    st.success("There is a statistically significant difference across groups.")
else:
    st.info("No statistically significant difference found.")

