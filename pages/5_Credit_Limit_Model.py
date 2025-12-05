import streamlit as st
from src.loader import load_df
from src.preprocessing import feature_engineering
from src.model import load_model, save_model, train_demo_model, evaluate_model, get_feature_importances
from src.viz import plot_feature_importance
import pandas as pd
from sklearn.model_selection import train_test_split

st.title("🤖 Credit Limit Modeling (Demo)")

st.markdown("This page demonstrates loading a saved model or training a small demo model from the processed data."
            " For safety, demo training uses a small RandomForest so it finishes quickly in the UI.")

df = load_df()
df, customers = feature_engineering(df)

# Determine candidate feature and target columns
if 'amt' in customers.columns:
    feat_total = 'amt'
elif 'total_spent' in customers.columns:
    feat_total = 'total_spent'
else:
    feat_total = None

features = [c for c in ['avg_transaction', 'spending_std', feat_total] if c and c in customers.columns]

if not features:
    st.error("Not enough customer features to train demo model (missing avg_transaction/spending_std/amt).")
else:
    # Create synthetic target: high spender or not
    customers = customers.copy()
    target_name = 'high_spender'
    customers[target_name] = (customers[features[0]] > customers[features[0]].median()).astype(int)

    st.write("### Dataset preview")
    st.dataframe(customers.head())

    model = None
    try:
        model = load_model()
        st.success("Loaded saved model from models/rf_model.pkl")
    except Exception:
        st.info("No saved model found — you can train a lightweight demo model below.")

    if model is None:
        if st.button("Train demo model (small)"):
            X = customers[features].fillna(0)
            y = customers[target_name]
            # train/test split just to run a small training cycle
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            with st.spinner("Training demo model — should be quick..."):
                model = train_demo_model(X_train, y_train)
            # evaluate
            metrics = evaluate_model(model, X_test, y_test)
            st.metric("Demo model accuracy", f"{metrics.get('accuracy', 0):.3f}")
            st.json(metrics)
            st.success("Demo model trained in-memory")
            # Offer to save
            if st.checkbox("Save demo model to models/rf_model.pkl (overwrite if exists)"):
                save_model(model)
                st.success("Model saved to models/rf_model.pkl")

    # If we have a model and test split available, offer evaluation and importance
    if model is not None and 'X_test' in locals():
        st.subheader("Model evaluation")
        metrics = evaluate_model(model, X_test, y_test)
        st.write(metrics)
        try:
            fig = plot_feature_importance(model, feature_names=features)
            st.pyplot(fig)
        except Exception:
            st.write("Model does not expose feature importances")

    if model is not None:
        st.subheader("Make a prediction")
        idx = st.selectbox("Choose a customer (by index) to see model prediction", options=list(customers.index[:50]))
        sample = customers.loc[[idx], features].fillna(0)
        try:
            prob = model.predict_proba(sample)[0, 1]
        except Exception:
            prob = None
        st.write("Customer features:")
        st.write(sample.T)
        if prob is not None:
            st.write(f"Predicted probability of 'high_spender' = {prob:.3f}")
            st.write("Model allows you to determine which customers are most likely to be high spenders — in a real deployment you'd train on a labeled target.")
        else:
            st.warning("Model cannot provide predict_proba. The loaded object may not be a classifier or is incompatible.")
