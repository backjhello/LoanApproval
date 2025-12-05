import streamlit as st
from src.loader import load_processed_data

st.title("📊 Dataset Overview")

# Use the central loader so filename and caching are consistent across pages
df = load_processed_data()

st.write("### Data Preview")
st.dataframe(df.head())

st.write("### Summary Statistics")
st.write(df.describe())
