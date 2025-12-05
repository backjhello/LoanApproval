import pandas as pd
import os

def load_customer_features():
    base = os.path.dirname(os.path.dirname(__file__))  # src/..
    path = os.path.join(base, "data", "processed", "customer_features.csv")
    return pd.read_csv(path)

