import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from mpl_toolkits.mplot3d import Axes3D

from src.loader import load_customer_features


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Advanced Customer Clustering", layout="wide")
st.title("🔮 Advanced Customer Segmentation Dashboard")

sns.set_theme(style="whitegrid")


# ============================================================
# LOAD DATA
# ============================================================
df = load_customer_features()

cols = ['total_spent', 'avg_transaction', 'transaction_count', 'spending_std']
X_base = df[cols].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_base)

st.markdown("""
### 🔍 Overview
This dashboard provides a **complete segmentation analysis** of customers based on:
- Spending amount  
- Purchase frequency  
- Volatility of behavior  
""")

# ============================================================
# SELECT NUMBER OF CLUSTERS
# ============================================================
st.sidebar.header("⚙️ Clustering Settings")
k = st.sidebar.slider("Number of Clusters (k)", 2, 8, 3)


# ============================================================
# ELBOW METHOD (OPTIONAL VIEW)
# ============================================================
with st.expander("📉 Elbow Method (Determine Best k)"):
    inertias = []
    K_range = range(2, 10)

    for kk in K_range:
        km = KMeans(n_clusters=kk, random_state=42)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(K_range, inertias, marker="o")
    ax.set_title("Elbow Plot")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    st.pyplot(fig)

# ============================================================
# FIT FINAL K-MEANS
# ============================================================
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_scaled)
df["cluster"] = labels

sil_score = silhouette_score(X_scaled, labels)

st.subheader("📌 Clustering Summary")
st.write(f"**Silhouette Score:** {sil_score:.3f}")
st.write(df["cluster"].value_counts())

# ============================================================
# PCA VISUALIZATION (2D)
# ============================================================
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)

st.subheader("🎨 PCA Visualization (2D Scatter Plot)")
fig, ax = plt.subplots(figsize=(9, 6))

scatter = ax.scatter(
    components[:, 0],
    components[:, 1],
    c=labels,
    cmap="viridis",
    alpha=0.8
)

plt.colorbar(scatter)
ax.set_title("Customer Segments (PCA 2D)", pad=15)
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
st.pyplot(fig)


# ============================================================
# PCA VISUALIZATION (3D)
# ============================================================
pca3 = PCA(n_components=3)
comp3 = pca3.fit_transform(X_scaled)

with st.expander("💫 PCA 3D Visualization"):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(comp3[:,0], comp3[:,1], comp3[:,2],
                   c=labels, cmap="viridis", alpha=0.8)
    fig.colorbar(p)
    ax.set_title("Customer Segments (PCA 3D)")
    st.pyplot(fig)


# ============================================================
# PCA LOADINGS
# ============================================================
st.subheader("📌 PCA Feature Contribution")
loadings = pd.DataFrame(
    pca.components_,
    columns=cols,
    index=["PCA1", "PCA2"]
)
st.dataframe(loadings)


# ============================================================
# CLUSTER PROFILES (MEAN)
# ============================================================
st.subheader("📊 Cluster Profiles — Mean Values")
cluster_means = df.groupby("cluster")[cols].mean()
st.dataframe(cluster_means)

# Heatmap visualization
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(cluster_means, cmap="viridis", annot=True, fmt=".1f")
ax.set_title("Cluster Profile Heatmap — Means")
st.pyplot(fig)


# ============================================================
# RADAR CHART FOR CLUSTER PROFILES
# ============================================================
import numpy as np

with st.expander("🎯 Radar Chart (Cluster Characteristics)"):

    def radar_chart(mean_df):
        categories = list(mean_df.columns)
        N = len(categories)

        fig, axs = plt.subplots(1, k, subplot_kw=dict(polar=True), figsize=(15, 4))

        if k == 1:
            axs = [axs]

        for i, cluster_id in enumerate(mean_df.index):
            values = mean_df.loc[cluster_id].values
            values = np.append(values, values[0])  # close loop

            angles = np.linspace(0, 2*np.pi, N, endpoint=False)
            angles = np.append(angles, angles[0])

            axs[i].plot(angles, values, linewidth=2)
            axs[i].fill(angles, values, alpha=0.25)
            axs[i].set_title(f"Cluster {cluster_id}")
            axs[i].set_xticks(angles[:-1])
            axs[i].set_xticklabels(categories)

        st.pyplot(fig)

    radar_chart(cluster_means)


# ============================================================
# AUTO-GENERATED CLUSTER SUMMARIES (NLG)
# ============================================================
st.subheader("🧠 AI-Like Cluster Descriptions")

def describe_cluster(row):
    desc = []

    if row["total_spent"] > cluster_means["total_spent"].mean():
        desc.append("high total spending")
    else:
        desc.append("low or moderate total spending")

    if row["transaction_count"] > cluster_means["transaction_count"].mean():
        desc.append("frequent purchases")
    else:
        desc.append("infrequent purchases")

    if row["spending_std"] > cluster_means["spending_std"].mean():
        desc.append("volatile or unstable spending behavior")
    else:
        desc.append("stable and predictable spending behavior")

    return "Customers in this cluster tend to have " + ", ".join(desc) + "."

summary_text = {
    cluster_id: describe_cluster(vals)
    for cluster_id, vals in cluster_means.iterrows()
}

for cluster_id, text in summary_text.items():
    st.markdown(f"""
    ### **Cluster {cluster_id} — Summary**
    {text}
    """)


# ============================================================
# END
# ============================================================
st.success("✨ Advanced clustering dashboard loaded successfully!")