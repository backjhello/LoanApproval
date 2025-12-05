import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.loader import load_customer_features

st.set_page_config(page_title="Customer Clustering", layout="wide")
st.title("🔮 Customer Segmentation (K-Means Clustering)")

df = load_customer_features()

sns.set_theme(
    style="whitegrid",
    rc={
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": "lightgray",
        "grid.color": "lightgray",
        "axes.labelsize": 12,
        "axes.titlesize": 17,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11
    }
)

st.markdown("""
### Overview  
This clustering analysis groups customers based on their **spending behavior**, **frequency**, and **volatility**.  
We use **K-Means** with PCA to visualize customer segments.
""")

# --------------------------------------------------------
# 1️⃣ Feature selection
# --------------------------------------------------------
cols = ['total_spent', 'avg_transaction', 'transaction_count', 'spending_std']
X = df[cols].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------------
# 2️⃣ K-Means Clustering
# --------------------------------------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)

df.loc[X.index, 'cluster'] = labels

st.subheader("📌 Cluster Assignment Summary")
st.write(df['cluster'].value_counts())

# --------------------------------------------------------
# 3️⃣ PCA for visualization
# --------------------------------------------------------
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(8,6))
scatter = ax.scatter(
    components[:,0],
    components[:,1],
    c=df.loc[X.index, 'cluster'],
    cmap="viridis",
    alpha=0.8
)

plt.colorbar(scatter)
ax.set_title("K-Means Clusters (PCA Projection)", pad=15)
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
sns.despine()
st.pyplot(fig)

st.markdown("""
### **Interpretation of PCA Axes**
- **PCA1 (x-axis)**  
  Represents scale of spending behavior:  
  → Right = high spending / frequent purchases  
  → Left = low spending / infrequent purchases  

- **PCA2 (y-axis)**  
  Represents spending volatility:  
  → Up = unstable, large spending swings  
  → Down = stable, predictable spending patterns  
""")

# --------------------------------------------------------
# 4️⃣ PCA Loadings (feature contributions)
# --------------------------------------------------------
loadings = pd.DataFrame(
    pca.components_,
    columns=cols,
    index=["PCA1", "PCA2"]
)

st.subheader("📌 PCA Feature Contribution (Loadings)")
st.dataframe(loadings)

# --------------------------------------------------------
# 5️⃣ Cluster Profiles (Mean & Std)
# --------------------------------------------------------
st.subheader("📌 Cluster Profiles (Mean Values)")
cluster_means = df.groupby("cluster")[cols].mean()
st.dataframe(cluster_means)

st.markdown("""
### Interpretation (Mean)
- **Cluster 0** → High total spending, high frequency, moderate volatility  
- **Cluster 1** → Moderate spending, fewer transactions, stable behavior  
- **Cluster 2** → Very low total spending, extremely high volatility (risky consumers)  
""")

st.subheader("📌 Cluster Profiles (Standard Deviation)")
cluster_stds = df.groupby("cluster")[cols].std()
st.dataframe(cluster_stds)

st.markdown("""
### Interpretation (Std)
- High STD means spending is inconsistent, unpredictable  
- Cluster 2 shows extremely high volatility = **risk-prone customers**  
""")
