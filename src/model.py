import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model


def train_demo_model(X_train, y_train, n_estimators=20, max_depth=5, random_state=42):
    """Train a lightweight demo RandomForest used by the Streamlit demo UI.

    This avoids long-running training inside the UI while still providing a usable model.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight='balanced',
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model

def save_model(model, path="models/rf_model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model(path="models/rf_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate a trained model on X_test/y_test and return a dict of metrics."""
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0))
    }
    # add ROC AUC when the model supports probabilities and y_test has both classes
    try:
        if hasattr(model, 'predict_proba') and len(set(y_test)) > 1:
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics['roc_auc'] = float(roc_auc_score(y_test, y_prob))
    except Exception:
        pass
    return metrics


def get_feature_importances(model, feature_names=None, top_n=10) -> list:
    """Return a sorted list of (feature_name, importance) tuples for top_n features."""
    if not hasattr(model, 'feature_importances_'):
        raise ValueError('Model does not expose feature_importances_')
    import numpy as np
    imps = model.feature_importances_
    idx = np.argsort(imps)[::-1][:top_n]
    names = feature_names if feature_names is not None else [f'feat_{i}' for i in range(len(imps))]
    return [(names[i], float(imps[i])) for i in idx]


