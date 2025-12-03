import streamlit as st
from src.loader import load_processed_data
from src.viz import (
    plot_distribution,
    plot_spending_by_age,
    plot_city_spending
)

df = load_processed_data()

st.title("🔍 Exploratory Data Analysis")

st.subheader("1. Transaction Amount Distribution")
col = st.selectbox("Choose variable", ["amt", "spending_std", "avg_transaction"])
st.pyplot(plot_distribution(df, col))

st.subheader("2. Spending Behavior by Age Group")
st.pyplot(plot_spending_by_age(df))

st.subheader("3. Spending by City Population Group")
st.pyplot(plot_city_spending(df))

