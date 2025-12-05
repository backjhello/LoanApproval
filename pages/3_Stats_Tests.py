import streamlit as st
from scipy.stats import f_oneway
import pandas as pd
from src.loader import load_customer_features

st.title("📊 Statistical Tests")
df = load_customer_features()

groups = df['age_group'].unique()

value = st.selectbox("Variable", ['total_spent','avg_transaction'])
st.write("Comparing across age groups...")

data = [df[df['age_group']==g][value] for g in groups]

f, p = f_oneway(*data)

st.write(f"**F-statistic:** {f:.4f}")
st.write(f"**p-value:** {p:.4f}")

if p < 0.05:
    st.success("Significant differences exist between age groups.")
else:
    st.info("No significant difference detected.")

