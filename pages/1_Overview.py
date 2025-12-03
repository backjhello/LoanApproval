import streamlit as st
import pandas as pd

st.title("📊 Dataset Overview")

df = pd.read_csv("data/processed/cleaned_data.csv")
st.write("### Data Preview")
st.dataframe(df.head())

st.write("### Summary Statistics")
st.write(df.describe())
