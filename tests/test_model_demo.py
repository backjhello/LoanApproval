import pandas as pd
from src.model import train_demo_model


def test_train_demo_model_basic():
    # small synthetic dataset
    X = pd.DataFrame({
        'avg_transaction': [10, 15, 20, 25, 30, 35],
        'spending_std': [1, 0.5, 0.6, 0.8, 1.2, 1.4],
        'amt': [100, 120, 200, 250, 300, 320]
    })
    y = (X['amt'] > 150).astype(int)

    model = train_demo_model(X, y)
    # should support predict_proba
    probs = model.predict_proba(X)
    assert probs.shape[0] == X.shape[0]
    # probabilities for class 1 should be between 0 and 1
    assert (probs[:, 1] >= 0).all() and (probs[:, 1] <= 1).all()
