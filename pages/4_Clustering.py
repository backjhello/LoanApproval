import streamlit as st
from src.loader import load_df
from src.cluster import run_clustering
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🧭 Customer Clustering")

df = load_df()

st.markdown("This page shows a simple KMeans-based clustering of customers using a few engineered features.")

st.sidebar.header("Clustering options")
n_clusters = st.sidebar.slider("Number of clusters", min_value=2, max_value=8, value=4)

available = [c for c in ['avg_transaction', 'spending_std', 'total_spent'] if c in df.columns]
features = st.sidebar.multiselect("Features to use", options=available, default=available)

if not features:
    st.warning("No features selected for clustering. Choose at least one sidebar feature.")
else:
    with st.spinner("Computing clusters..."):
        clustered = run_clustering(df, feature_cols=features, n_clusters=n_clusters)

    st.subheader("Cluster scatter (2D embedding)")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=clustered.dropna(subset=['cluster', 'cluster_x', 'cluster_y']),
                    x='cluster_x', y='cluster_y', hue='cluster', palette='tab10', ax=ax)
    ax.set_xlabel('Embedding X')
    ax.set_ylabel('Embedding Y')
    st.pyplot(fig)

    st.subheader("Cluster summary")
    st.write(clustered.groupby('cluster')[features].median())

    st.subheader("Sample customers by cluster")
    st.dataframe(clustered.dropna(subset=['cluster']).groupby('cluster').head(5).reset_index(drop=True))
