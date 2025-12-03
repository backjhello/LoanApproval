import streamlit as st
import pandas as pd
from src.visualization import plot_distribution

st.title("📈 Exploratory Analysis")

df = pd.read_csv("data/processed/cleaned_data.csv")

st.write("### Distributions")
selected_col = st.selectbox("Select column", df.columns)
plot_distribution(df, selected_col)
