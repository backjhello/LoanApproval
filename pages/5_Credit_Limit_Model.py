import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.loader import load_customer_features

st.title("💳 Credit Limit Approval Model")

df = load_customer_features()
df = df.copy()

cols = ['total_spent','avg_transaction','transaction_count','spending_std','luxury','misc','necessity','wellbeing']
X0 = df[cols].copy()

# normalization
for c in cols:
    r = X0[c].max() - X0[c].min()
    X0[c+'_n'] = (X0[c] - X0[c].min()) / (r if r!=0 else 1)

# nonlinear & interactions
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

risk = (
    0.45*Z['std2'] +
    0.15*Z['lux_std'] +
    0.10*Z['misc_std'] -
    0.20*Z['nec_wb'] -
    0.10*Z['ticket_mix']
)

thr = np.median(risk)
y = (risk < thr).astype(int)

X = X0[[c+'_n' for c in cols]].join(Z)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

scaler = StandardScaler(with_mean=False)
Xs_tr = scaler.fit_transform(X_train)
Xs_te = scaler.transform(X_test)

# logistic regression
logit = LogisticRegression(max_iter=2000, class_weight='balanced')
logit.fit(Xs_tr, y_train)
p1 = logit.predict_proba(Xs_te)[:,1]

# random forest
rf = RandomForestClassifier(n_estimators=400, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
p2 = rf.predict_proba(X_test)[:,1]

st.subheader("📌 Model Performance")
st.write("### Logistic Regression AUC:", round(roc_auc_score(y_test, p1), 3))
st.write("### Random Forest AUC:", round(roc_auc_score(y_test, p2), 3))

st.write("### Logistic Regression Classification Report")
st.text(classification_report(y_test, (p1>=0.5).astype(int)))

st.write("### Random Forest Classification Report")
st.text(classification_report(y_test, (p2>=0.5).astype(int)))
