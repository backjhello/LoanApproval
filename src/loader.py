import pandas as pd
import streamlit as st

@st.cache_data
def load_raw_data():
    return pd.read_csv("credit_card_transactions.csv")

@st.cache_data
def load_processed_data():
    df = pd.read_csv("data/processed/transactions_cleaned.csv")
    return df
