import streamlit as st
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report


# ======================================
# 0) LOAD DATA
# ======================================
@st.cache_data
def load_customer_features():
    base = os.path.dirname(os.path.dirname(__file__))  
    path = os.path.join(base, "data", "processed", "customer_features.csv")
    return pd.read_csv(path)

df = load_customer_features()

st.title("📊 Credit Risk Modeling Dashboard")
st.write("""
This page reconstructs the realistic ML pipeline used in the project.  
It includes risk score modeling, Logistic/RF classifiers, segmentation analysis,  
and an interactive risk-weight adjustment tool.
""")


# -------------------------------------------------------
# Tab Layout: EDA | Modeling | Segmentation
# -------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📈 EDA", "🤖 ML Modeling", "🧩 Segmentation Models"])



# ======================================================
# TAB 1 — BASIC EDA
# ======================================================
with tab1:
    st.header("📈 Exploratory Data Analysis")

    st.subheader("Dataset Overview")
    st.dataframe(df.head())

    st.subheader("Distribution of Key Financial Variables")
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    sns.histplot(df['total_spent'], kde=True, ax=axs[0,0])
    sns.histplot(df['avg_transaction'], kde=True, ax=axs[0,1])
    sns.histplot(df['transaction_count'], kde=True, ax=axs[1,0])
    sns.histplot(df['spending_std'], kde=True, ax=axs[1,1])
    st.pyplot(fig)

    st.markdown("""
    ### Key Observations
    - Spending varies widely across customers.
    - `spending_std` captures volatility in purchasing behavior.
    - These variables become important signals for risk modeling.
    """)



# ======================================================
# TAB 2 — MAIN ML MODELING (LOGIT + RF + SLIDER)
# ======================================================
with tab2:
    st.header("🤖 Machine Learning: Risk Score + Credit Approval Model")

    # ----------------------------
    # 1) Raw Feature Set
    # ----------------------------
    cols = [
        'total_spent', 'avg_transaction', 'transaction_count',
        'spending_std', 'luxury', 'misc', 'necessity', 'wellbeing'
    ]
    X0 = df[cols].copy()

    # ----------------------------
    # 2) Normalize Columns
    # ----------------------------
    for c in cols:
        r = X0[c].max() - X0[c].min()
        X0[c + "_n"] = (X0[c] - X0[c].min()) / (r if r != 0 else 1)


    # ----------------------------
    # ⭐ INTERACTIVE SLIDER
    # ----------------------------
    st.subheader("Adjust Risk Model Weights")

    w_std2      = st.slider("Weight: spending_std²", 0.0, 1.0, 0.35)
    w_luxstd    = st.slider("Weight: luxury × spending_std", 0.0, 1.0, 0.25)
    w_miscstd   = st.slider("Weight: misc × spending_std", 0.0, 1.0, 0.10)
    w_necwb     = st.slider("Weight: necessity × wellbeing (negative)", -1.0, 0.0, -0.20)
    w_ticketmix = st.slider("Weight: ticket_mix (negative)", -1.0, 0.0, -0.10)


    # ----------------------------
    # 3) Nonlinear features (Z)
    # ----------------------------
    Z = pd.DataFrame(index=X0.index)
    Z['std2']      = X0['spending_std_n']**2
    Z['lux2']      = X0['luxury_n']**2
    Z['nec2']      = X0['necessity_n']**2
    Z['wb2']       = X0['wellbeing_n']**2
    Z['lux_std']   = X0['luxury_n'] * X0['spending_std_n']
    Z['misc_std']  = X0['misc_n'] * X0['spending_std_n']
    Z['nec_wb']    = X0['necessity_n'] * X0['wellbeing_n']
    Z['size_freq'] = X0['total_spent_n'] * X0['transaction_count_n']
    Z['ticket_mix']= X0['avg_transaction_n'] * (X0['necessity_n'] - X0['luxury_n'])


    # ----------------------------
    # 4) Risk Score Calculation (Interactive!)
    # ----------------------------
    rng = np.random.default_rng(42)

    risk = (
        w_std2 * Z['std2'] +
        w_luxstd * Z['lux_std'] +
        w_miscstd * Z['misc_std'] +
        w_necwb * Z['nec_wb'] +
        w_ticketmix * Z['ticket_mix']
    ) + rng.normal(0, 0.03, size=len(Z))

    df['risk_score'] = risk
    df['loan_approved'] = (risk < np.median(risk)).astype(int)


    # Show distribution
    st.subheader("Risk Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['risk_score'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)


    # ----------------------------
    # 5) Train ML Models
    # ----------------------------
    X = X0[[c+'_n' for c in cols]].join(Z)
    y = df['loan_approved']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler(with_mean=False)
    Xs_tr = scaler.fit_transform(X_train)
    Xs_te = scaler.transform(X_test)

    logit = LogisticRegression(max_iter=2000, class_weight='balanced')
    logit.fit(Xs_tr, y_train)
    p1 = logit.predict_proba(Xs_te)[:,1]

    rf = RandomForestClassifier(n_estimators=400, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    p2 = rf.predict_proba(X_test)[:,1]


    # ----------------------------
    # 6) Performance Output
    # ----------------------------
    st.subheader("Model Performance")

    st.write(f"**Logistic Regression AUC:** {roc_auc_score(y_test, p1):.3f}")
    st.text(classification_report(y_test, (p1 >= 0.5).astype(int)))

    st.write(f"**Random Forest AUC:** {roc_auc_score(y_test, p2):.3f}")
    st.text(classification_report(y_test, (p2 >= 0.5).astype(int)))


    # ----------------------------
    # 7) Feature Importance (RF)
    # ----------------------------
    st.subheader("Feature Importance (Random Forest)")

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot(kind="barh", ax=ax)
    st.pyplot(fig)



# ======================================================
# TAB 3 — SEGMENTATION MODEL COMPARISON
# ======================================================
with tab3:
    st.header("🧩 Segmentation Modeling (Cluster / Age / City)")

    st.write("""
    Compare ML performance across different customer groups.  
    **cluster** = job-based consumption behavioral groups (already included in dataset)  
    """)

    # Create age_group & city_group if needed
    bins_age = [0,25,35,45,55,65,120]
    labels_age = ['<25','25–34','35–44','45–54','55–64','65+']
    df['age_group'] = pd.cut(df['avg_age'], bins=bins_age, labels=labels_age, right=False)

    bins_city = [0,20000,50000,200000,500000,float("inf")]
    labels_city = ['<20k','20k–50k','50k–200k','200k–500k','500k+']
    df['city_group'] = pd.cut(df['avg_city_pop'], bins_city, labels=labels_city, right=False)

    # Select segmentation
    seg_choice = st.selectbox("Choose segmentation variable:", ["cluster", "age_group", "city_group"])

    # Table of groups
    st.write("Group Value Counts:")
    st.write(df[seg_choice].value_counts())

    # Loop over groups and model
    results = []

    for grp, subset in df.groupby(seg_choice):
        if len(subset) < 30:
            continue

        # 🔥 0/1 클래스 모두 존재해야 학습 가능
        if subset['loan_approved'].nunique() < 2:
            st.warning(f"Group '{grp}' skipped — only one class present.")
            continue
        
        X_seg = X.loc[subset.index]
        y_seg = subset['loan_approved']

        X_train, X_test, y_train, y_test = train_test_split(
            X_seg, y_seg, test_size=0.25, random_state=42, stratify=y_seg
        )

        rf2 = RandomForestClassifier(n_estimators=200, random_state=42)
        rf2.fit(X_train, y_train)
        p = rf2.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, p)

        results.append((grp, auc))

    st.subheader("Segmentation AUC Comparison")
    df_seg = pd.DataFrame(results, columns=["Group", "AUC"]).sort_values("AUC", ascending=False)
    st.dataframe(df_seg)

    # Visualization
    fig, ax = plt.subplots()
    sns.barplot(x="Group", y="AUC", data=df_seg, palette="viridis", ax=ax)
    plt.xticks(rotation=30)
    st.pyplot(fig)


st.success("All modeling, segmentation, and interactive tools are fully loaded! 🎉")