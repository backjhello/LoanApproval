import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from src.loader import load_customer_features

st.title("📄 Overview")
df = load_customer_features()

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Summary Statistics")
st.dataframe(df.describe())

st.subheader("Missing Value Heatmap")
fig, ax = plt.subplots()
sns.heatmap(df.isnull(), cbar=False, ax=ax)
st.pyplot(fig)
