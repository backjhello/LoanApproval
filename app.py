import streamlit as st
from src.loader import load_processed_data

st.set_page_config(
    page_title="Credit Transaction Analytics",
    page_icon="📈",
    layout="wide"
)

# Load CSS safely on Windows (bypass cp949 decoding issues)
with open("assets/styles.css", "rb") as f:
    css = f.read().decode("utf-8", errors="ignore")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

st.title("📈 Credit Transaction Analytics Dashboard")
st.markdown("""
This dashboard summarizes **transaction patterns**,  
**consumer behavioral clusters**,  
**statistical significance testing**, and **machine learning modeling**  
from the FA25 STAT team project.
""")

st.sidebar.header("Navigation")
st.sidebar.markdown("Use the pages in the sidebar to explore the analysis.")

