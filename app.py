import streamlit as st
from src.loader import load_processed_data

st.set_page_config(
    page_title="Credit Transaction Analytics",
    page_icon="📈",
    layout="wide"
)

# Load CSS
with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("📈 Credit Transaction Analytics Dashboard")
st.markdown("""
This dashboard summarizes **transaction patterns**,  
**consumer behavioral clusters**,  
**statistical significance testing**, and **machine learning modeling**  
from the FA25 STAT team project.
""")

st.sidebar.header("Navigation")
st.sidebar.markdown("Use the pages in the sidebar to explore the analysis.")

