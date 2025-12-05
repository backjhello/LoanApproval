import pandas as pd
import os
import streamlit as st

# src 폴더 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# data/processed 경로
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")

@st.cache_data
def load_df():
    path = os.path.join(DATA_DIR, "df.csv")
    return pd.read_csv(path)

@st.cache_data
def load_customer_features():
    path = os.path.join(DATA_DIR, "customer_features.csv")
    return pd.read_csv(path)
