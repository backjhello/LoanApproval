import pickle
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model

def save_model(model, path="models/rf_model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model(path="models/rf_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


