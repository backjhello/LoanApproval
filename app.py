import streamlit as st
from src.loader import load_customer_features

# Page Config
st.set_page_config(
    page_title="Customer Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load Data
df = load_customer_features()

# Title Section
st.markdown("<h1 style='text-align: center;'>📊 Customer Behavior Analysis Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Understanding customer spending habits, segmentations, and credit-limit modeling</h4>", unsafe_allow_html=True)

st.write("---")

# Overview Metrics
st.subheader("✨ Quick Snapshot")

col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", f"{len(df):,}")
col2.metric("Avg Total Spending", f"${df['total_spent'].mean():.2f}")
col3.metric("Avg Transaction Count", f"{df['transaction_count'].mean():.1f}")

st.write("---")

# Project Description
st.subheader("📘 About This Dashboard")
st.markdown("""
This dashboard provides an interactive exploration of customer spending behavior, segmentation patterns, 
and credit limit approval modeling. It was designed to help analysts quickly understand customer profiles, 
identify meaningful spending trends, and evaluate approval predictions using machine learning models.

**Key capabilities include:**
- 📈 Exploratory Data Analysis (EDA)
- 👥 Behavioral Segmentation & Clustering
- 🧪 Statistical Tests across groups
- 🤖 Credit Limit Prediction Model using ML
""")

st.write("---")

# Navigation Buttons
st.subheader("🚀 Jump to a Section")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📄 Overview"):
        st.switch_page("pages/1_Overview.py")
with col2:
    if st.button("📊 EDA"):
        st.switch_page("pages/2_EDA.py")
with col3:
    if st.button("🧪 Stats Tests"):
        st.switch_page("pages/3_Stats_Tests.py")

col4, col5 = st.columns(2)

with col4:
    if st.button("🤖 Clustering"):
        st.switch_page("pages/4_Clustering.py")
with col5:
    if st.button("💳 Credit Limit Model"):
        st.switch_page("pages/5_Credit_Limit_Model.py")

st.write("---")

# Footer
st.markdown("""
<div style='text-align: center; color: gray; margin-top: 30px;'>
    Created as part of the ATLAS STAT Project • UIUC  
</div>
""", unsafe_allow_html=True)