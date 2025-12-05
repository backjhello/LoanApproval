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

from src.loader import load_customer_features

st.title("🤖 Machine Learning Models & Risk Analysis")

df = load_customer_features()

tab1, tab2, tab3 = st.tabs([
    "🔹 Model 1 (Simple Linear Risk Score)",
    "🔹 Model 2 (Nonlinear / Interaction Risk Score)",
    "🔹 PCA & Feature Importance"
])

with tab1:
    st.header("📌 Model 1: Simple Risk Score → Loan Approval Prediction")

    features = [
        'total_spent', 'avg_transaction', 'transaction_count', 'spending_std',
        'luxury', 'misc', 'necessity', 'wellbeing'
    ]
    df1 = df.copy()

    # normalize
    for col in features:
        r = df1[col].max() - df1[col].min()
        df1[col + "_norm"] = (df1[col] - df1[col].min()) / (r if r != 0 else 1)

    # simple linear risk
    df1['risk_score'] = (
        0.3 * df1['spending_std_norm'] +
        0.25 * df1['luxury_norm'] +
        0.10 * df1['misc_norm'] -
        0.20 * df1['necessity_norm'] -
        0.15 * df1['wellbeing_norm']
    )

    threshold = df1['risk_score'].median()
    df1['loan_approved'] = np.where(df1['risk_score'] < threshold, 1, 0)

    st.subheader("📊 Risk Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df1['risk_score'], kde=True, ax=ax)
    st.pyplot(fig)

    st.write(f"📌 **Approval Rate:** {df1['loan_approved'].mean().round(3)}")

    # model training
    X = df1[[c for c in df1.columns if c.endswith("_norm")]]
    y = df1['loan_approved']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logit = LogisticRegression(max_iter=2000, class_weight='balanced')
    logit.fit(X_train_scaled, y_train)
    p_log = logit.predict_proba(X_test_scaled)[:, 1]
    auc_log = roc_auc_score(y_test, p_log)

    st.write(f"### 🔹 Logistic Regression AUC: **{auc_log:.3f}**")
    st.text(classification_report(y_test, (p_log > 0.5).astype(int)))

    rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    p_rf = rf.predict_proba(X_test)[:, 1]
    auc_rf = roc_auc_score(y_test, p_rf)

    st.write(f"### 🔹 Random Forest AUC: **{auc_rf:.3f}**")
    st.text(classification_report(y_test, (p_rf > 0.5).astype(int)))

    st.markdown("""
    ### 📝 Interpretation
    - Model 1 learns extremely simple linear patterns → 너무 잘 학습함 (AUC ~1.0)
    - 실제 고객 행동보다 훨씬 단순한 구조라 **과적합된 pseudo-label을 그대로 재현한 것**
    - 즉, **현실성은 부족하지만, 위험 점수 구조를 모델이 그대로 따라간다는 것**을 보여줌
    """)

with tab2:
    st.header("📌 Model 2: Nonlinear + Interaction Features")

    df2 = df.copy()
    cols = ['total_spent','avg_transaction','transaction_count','spending_std',
            'luxury','misc','necessity','wellbeing']

    # normalize
    X0 = df2[cols].copy()
    for c in cols:
        r = X0[c].max() - X0[c].min()
        X0[c + "_n"] = (X0[c] - X0[c].min()) / (r if r != 0 else 1)

    Z = pd.DataFrame(index=df2.index)
    Z["std2"] = X0['spending_std_n'] ** 2
    Z["lux2"] = X0['luxury_n'] ** 2
    Z["wb2"] = X0['wellbeing_n'] ** 2
    Z["lux_std"] = X0['luxury_n'] * X0['spending_std_n']
    Z["nec_wb"] = X0['necessity_n'] * X0['wellbeing_n']
    Z["size_freq"] = X0['total_spent_n'] * X0['transaction_count_n']
    Z["ticket_mix"] = X0['avg_transaction_n'] * (X0['necessity_n'] - X0['luxury_n'])

    # nonlinear risk score
    risk = (
        0.35 * Z['std2'] +
        0.25 * Z['lux2'] +
        0.20 * Z['misc_std'] if 'misc_std' in Z else 0 +
        0.12 * Z['wb2'] +
        0.10 * Z['ticket_mix']
    )

    risk += np.random.normal(0, 0.03, len(risk))
    thr = np.median(risk)
    y = (risk < thr).astype(int)

    df2['risk_score2'] = risk
    df2['loan_approved2'] = y

    st.subheader("📊 New Nonlinear Risk Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(risk, kde=True, ax=ax)
    st.pyplot(fig)

    # model training
    X = pd.concat([X0[[c for c in X0 if c.endswith("_n")]], Z], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler(with_mean=False)
    Xs_tr = scaler.fit_transform(X_train)
    Xs_te = scaler.transform(X_test)

    # logistic
    logit = LogisticRegression(max_iter=2000, class_weight='balanced')
    logit.fit(Xs_tr, y_train)
    p1 = logit.predict_proba(Xs_te)[:, 1]
    auc1 = roc_auc_score(y_test, p1)

    st.write(f"### 🔹 Logistic Regression AUC: **{auc1:.3f}**")
    st.text(classification_report(y_test, (p1 > 0.5).astype(int)))

    # RF
    rf = RandomForestClassifier(n_estimators=400, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    p2 = rf.predict_proba(X_test)[:, 1]
    auc2 = roc_auc_score(y_test, p2)

    st.write(f"### 🔹 Random Forest AUC: **{auc2:.3f}**")
    st.text(classification_report(y_test, (p2 > 0.5).astype(int)))

    st.markdown("""
    ### 📝 Interpretation
    - More complex features → 모델이 완벽하게 재현 못함 → AUC가 0.7대
    - 현실적 소비 패턴은 단순 선형 규칙보다 훨씬 복잡하다는 것을 의미
    - 즉 **Model2는 현실 행동을 더 비슷하게 반영한 pseudo-label 구조**
    """)
    
with tab3:
    st.header("📌 PCA & Feature Importance")

    cols = ['total_spent','avg_transaction','transaction_count','spending_std']
    X = df[cols].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)

    st.subheader("🎨 PCA Projection with KMeans Clusters")
    fig, ax = plt.subplots(figsize=(7,5))
    scatter = ax.scatter(components[:,0], components[:,1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    st.pyplot(fig)

    st.subheader("📌 Cluster Group Statistics")
    df_cluster = df.copy()
    df_cluster['cluster'] = labels
    st.dataframe(df_cluster.groupby('cluster')[cols].mean())

    st.markdown("""
    ### Interpretation
    - **Cluster 0:** Heavy spenders + frequent transactions  
    - **Cluster 1:** Medium spending  
    - **Cluster 2:** Low-volume but high-variance users  
    """)
