import pandas as pd
from src.cluster import run_clustering


def test_run_clustering_basic():
    # create a tiny dataset
    df = pd.DataFrame({
        'avg_transaction': [10, 12, 50, 55, 60, 62],
        'spending_std': [1, 0.8, 5, 4.5, 6, 5.5],
        'total_spent': [100, 120, 500, 550, 600, 620]
    })

    out = run_clustering(df, feature_cols=['avg_transaction', 'spending_std', 'total_spent'], n_clusters=2)
    # should have cluster column and embedding
    assert 'cluster' in out.columns
    assert 'cluster_x' in out.columns and 'cluster_y' in out.columns
    # clusters for rows must not be null
    assert out['cluster'].notna().all()
