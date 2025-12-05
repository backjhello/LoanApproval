import streamlit as st
from src.loader import load_processed_data
from src.eda import describe_df, missing_summary
from src.viz import (
    plot_distribution,
    plot_spending_by_age,
    plot_city_spending,
    plot_missingness,
    plot_correlation_heatmap,
)

df = load_processed_data()

st.title("🔍 Exploratory Data Analysis")

st.subheader("Dataset summary")
st.write(describe_df(df))

st.subheader("Missing values")
st.pyplot(plot_missingness(df))

st.subheader("Correlation matrix (numeric)")
num_cols = list(df.select_dtypes(include=['number']).columns)
sel_cols = st.multiselect("Columns to include (empty = all numeric)", options=num_cols, default=None)
st.pyplot(plot_correlation_heatmap(df, cols=sel_cols))

st.subheader("Distribution explorer")
col = st.selectbox("Choose variable", num_cols, index=0 if num_cols else None)
if col:
    st.pyplot(plot_distribution(df, col))

if 'age_group' in df.columns:
    st.subheader("Spending behavior by age_group")
    st.pyplot(plot_spending_by_age(df))

if 'city_group' in df.columns:
    st.subheader("Spending by city population group")
    st.pyplot(plot_city_spending(df))

