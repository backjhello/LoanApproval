import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

from src.loader import load_customer_features
df = load_customer_features()

st.title("💳 Credit Limit Prediction Model")
st.write("Analyze customer behavior to estimate approval likelihood, risk score, and key drivers.")

sns.set_theme(style="whitegrid")

# ---------------------------------------
# TABS
# ---------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📌 Overview",
    "🤖 Model Training",
    "🌟 Feature Importance",
    "📊 Risk Visualizations"
])

# =====================================================================
# TAB 1 — OVERVIEW
# =====================================================================
with tab1:
    st.header("📌 Step 1: Dataset Overview & Preprocessing")

    st.write("""
    We use customer behavioral features to create a **pseudo credit approval model**.  
    First, numeric features are normalized using MinMax scaling.
    """)

    # Columns for normalization
    norm_cols = [
        "luxury", "necessity", "wellbeing", "misc",
        "spending_std", "total_spent", "avg_transaction", "transaction_count"
    ]

    scaler = MinMaxScaler()
    df[[c + "_norm" for c in norm_cols]] = scaler.fit_transform(df[norm_cols])

    st.subheader("Normalized Features Sample")
    st.dataframe(df[[c + "_norm" for c in norm_cols]].head())

    st.subheader("📌 Step 2: Create Pseudo Risk Score + Approval Label")
    df["risk_score"] = (
        0.40 * df["luxury_norm"] -
        0.20 * df["necessity_norm"] -
        0.15 * df["wellbeing_norm"] +
        0.25 * df["spending_std_norm"]
    )

    thr = df["risk_score"].median()
    df["loan_approved"] = (df["risk_score"] < thr).astype(int)

    st.write(f"Approval Rate: **{df['loan_approved'].mean():.3f}**")

    st.dataframe(df[["risk_score", "loan_approved"]].head())


# =====================================================================
# TAB 2 — MODEL TRAINING
# =====================================================================
with tab2:
    st.header("🤖 Step 3: Train Models (Logistic Regression & Random Forest)")

    feature_cols = [
        "luxury_norm", "necessity_norm", "wellbeing_norm", "misc_norm",
        "spending_std_norm", "total_spent_norm",
        "avg_transaction_norm", "transaction_count_norm"
    ]

    X = df[feature_cols]
    y = df["loan_approved"]

    std_scaler = StandardScaler()
    Xs = std_scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        Xs, y, test_size=0.2, random_state=42, stratify=y
    )

    # Logistic Regression
    logit = LogisticRegression(max_iter=2000, class_weight="balanced")
    logit.fit(X_train, y_train)
    p1 = logit.predict_proba(X_test)[:, 1]
    auc_logit = roc_auc_score(y_test, p1)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    rf.fit(X, y)
    p2 = rf.predict_proba(X)[:, 1]
    auc_rf = roc_auc_score(y, p2)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📘 Logistic Regression Results")
        st.write(f"**AUC = {auc_logit:.3f}**")
        st.text(classification_report(y_test, (p1 >= 0.5).astype(int)))

    with col2:
        st.subheader("🌲 Random Forest Results")
        st.write(f"**AUC = {auc_rf:.3f}**")


# =====================================================================
# TAB 3 — FEATURE IMPORTANCE
# =====================================================================
with tab3:
    st.header("🌟 Step 4: Feature Importance (Random Forest)")

    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()

    fig, ax = plt.subplots(figsize=(7, 5))
    importances.plot(kind="barh", color="skyblue", ax=ax)
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    st.pyplot(fig)

    st.subheader("Top Important Features")
    st.dataframe(importances.sort_values(ascending=False).head(5))


# =====================================================================
# TAB 4 — RISK VISUALIZATIONS
# =====================================================================
with tab4:
    st.header("📊 Step 5: Visualize Risk Score Patterns")

    # Scatterplot
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.regplot(
        x=df["spending_std_norm"],
        y=df["risk_score"],
        scatter_kws={"alpha": 0.3},
        order=2,
        ax=ax
    )
    ax.set_title("Spending Std vs Risk Score")
    st.pyplot(fig)

    # KDE comparison
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.kdeplot(
        data=df,
        x="spending_std",
        hue="loan_approved",
        fill=True,
        ax=ax
    )
    ax.set_title("Spending Std Distribution by Loan Approval")
    st.pyplot(fig)

    st.markdown("""
    ### Interpretation
    - Customers with **higher volatility (spending_std)** are **less likely to be approved**.  
    - Luxury-heavy spending increases risk_score → lowers approval probability.  
    - Both models identify consistent top predictors, mainly spending volatility and luxury ratio.  
    """)