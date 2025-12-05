import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

sns.set_theme(style="whitegrid")

# -------------------------------------------------------------
# 📌 Load Data
# -------------------------------------------------------------
@st.cache_data
def load_customer_features():
    path = "data/processed/customer_features.csv"
    return pd.read_csv(path)

df = load_customer_features()


# =============================================================
# ============   STREAMLIT PAGE STRUCTURE (TABS)   ============
# =============================================================

st.title("💳 Credit Limit Modeling (Realistic Non-Linear Version)")

tab1, tab2, tab3, tab4 = st.tabs([
    "📌 Step 1: Create Risk Score",
    "📊 Step 2: Train/Test Split",
    "🤖 Step 3: Train Models",
    "📈 Step 4: Feature Importance & Insights"
])

# -------------------------------------------------------------
# 📌 TAB 1 — CREATE RISK SCORE + LABELS
# -------------------------------------------------------------
with tab1:
    st.header("📌 Step 1: Create Non-Linear Risk Score & pseudo-labels")

    df1 = df.copy()

    # Variables used
    cols = ['total_spent','avg_transaction','transaction_count','spending_std',
            'luxury','misc','necessity','wellbeing']
    
    # Normalize (0–1)
    for c in cols:
        r = df1[c].max() - df1[c].min()
        df1[c + "_n"] = (df1[c] - df1[c].min()) / (r if r != 0 else 1)

    st.markdown("### 🔧 Normalized Features Preview")
    st.dataframe(df1[[c+"_n" for c in cols]].head())

    # ---------------------------------------------------------
    # Realistic Non-linear + Interaction Risk Score
    # ---------------------------------------------------------
    Z = df1  # simplified alias

    risk = (
        0.35 * (Z['spending_std_n']**2) +
        0.25 * (Z['luxury_n']**2) +
        0.20 * (Z['misc_n'] * Z['spending_std_n']) +
        0.12 * (Z['wellbeing_n'] * Z['necessity_n']) +
        0.10 * (Z['total_spent_n'] * Z['transaction_count_n']) +
        0.10 * (Z['avg_transaction_n'] * (Z['necessity_n'] - Z['luxury_n']))
    )

    # Add noise (realistic behavior)
    rng = np.random.default_rng(42)
    risk = risk + rng.normal(0, 0.03, size=len(risk))

    df1['risk_score'] = risk

    # Label creation
    thr = np.median(risk)
    df1['loan_approved'] = (df1['risk_score'] < thr).astype(int)

    st.markdown("### 📊 Risk Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df1['risk_score'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    st.success("Pseudo labels created successfully!")

# -------------------------------------------------------------
# 📊 TAB 2 — TRAIN/TEST SPLIT
# -------------------------------------------------------------
with tab2:
    st.header("📊 Step 2: Train/Test Split + Scaling")

    # prepare X, y
    feature_cols = [c+"_n" for c in cols]  # normalized versions
    X = df1[feature_cols]
    y = df1['loan_approved']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # scaler
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.markdown("### ✔ Data Prepared")
    st.write("Train size:", X_train.shape)
    st.write("Test size:", X_test.shape)

# -------------------------------------------------------------
# 🤖 TAB 3 — TRAIN MODELS
# -------------------------------------------------------------
with tab3:
    st.header("🤖 Step 3: Train Logistic Regression & Random Forest")

    # -------- Logistic Regression --------
    logit = LogisticRegression(max_iter=3000, class_weight='balanced')
    logit.fit(X_train_scaled, y_train)
    p1 = logit.predict_proba(X_test_scaled)[:,1]

    auc_log = roc_auc_score(y_test, p1)

    st.subheader("📘 Logistic Regression Results")
    st.write(f"**AUC = {auc_log:.3f}**")

    report_log = classification_report(y_test, (p1>0.5).astype(int), output_dict=True)
    st.dataframe(pd.DataFrame(report_log).T)

    # -------- Random Forest --------
    rf = RandomForestClassifier(
        n_estimators=400, class_weight='balanced', random_state=42
    )
    rf.fit(X_train, y_train)
    p2 = rf.predict_proba(X_test)[:,1]

    auc_rf = roc_auc_score(y_test, p2)

    st.subheader("🌲 Random Forest Results")
    st.write(f"**AUC = {auc_rf:.3f}**")

    report_rf = classification_report(y_test, (p2>0.5).astype(int), output_dict=True)
    st.dataframe(pd.DataFrame(report_rf).T)

# -------------------------------------------------------------
# 📈 TAB 4 — FEATURE IMPORTANCE + INSIGHTS
# -------------------------------------------------------------
with tab4:
    st.header("📈 Step 4: Feature Importance & Interpretation")

    # Feature importance
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()

    fig, ax = plt.subplots(figsize=(8,6))
    importances.plot(kind='barh', ax=ax, color="skyblue")
    ax.set_title("🔍 Feature Importance (Random Forest)")
    st.pyplot(fig)

    # Scatter: spending_std vs risk
    st.markdown("### 📌 Relationship: Spending Std vs Risk Score")
    fig, ax = plt.subplots()
    sns.regplot(x=df1['spending_std'], y=df1['risk_score'], scatter_kws={'alpha':0.3}, ax=ax)
    st.pyplot(fig)

    st.markdown("""
    ### **Interpretation (English)**

    - The **non-linear risk score** captures complex spending patterns more realistically.
    - Variables such as **luxury spending**, **spending volatility**, and **transaction frequency**  
      contribute strongly to the model’s decisions.
    - Random Forest performs worse than in the linear pseudo-label version because  
      the underlying rule now includes **quadratic + interaction + noise**, which prevents models  
      from perfectly reverse-engineering the rule.
    """)

