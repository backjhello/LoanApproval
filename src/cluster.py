import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    import umap
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False


def run_clustering(df: pd.DataFrame, feature_cols=None, n_clusters: int = 4, random_state: int = 42):
    """Run a simple clustering pipeline.

    Returns a copy of `df` with two new columns added: `cluster` and `cluster_x`, `cluster_y`
    (embedding coordinates). `feature_cols` selects columns used for clustering; if None,
    tries a reasonable default set present in the repo's data.
    """
    # Pick default features commonly present in processed data
    defaults = ['avg_transaction', 'spending_std', 'total_spent']
    if feature_cols is None:
        feature_cols = [c for c in defaults if c in df.columns]

    if not feature_cols:
        raise ValueError("No feature columns available for clustering")

    X = df[feature_cols].dropna()
    if X.empty:
        raise ValueError("Feature slice is empty after dropping NA")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(Xs)

    # Create a small embedding for visualization: prefer UMAP if available
    if _HAS_UMAP:
        reducer = umap.UMAP(random_state=random_state)
        emb = reducer.fit_transform(Xs)
    else:
        # Lightweight fallback using the cluster centers projection
        # create 2-dimensional pseudo-embedding from first two PCA-like components
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=random_state)
        emb = pca.fit_transform(Xs)

    out = df.copy()
    # Align labels and embeddings back to original index (X may have dropped NA rows)
    out.loc[X.index, 'cluster'] = labels
    out.loc[X.index, 'cluster_x'] = emb[:, 0]
    out.loc[X.index, 'cluster_y'] = emb[:, 1]

    return out
