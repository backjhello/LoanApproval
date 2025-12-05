import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from src.loader import load_customer_features

st.title("📊 Customer Credit Risk — Tab 1: Overall Analysis")

# ----------------------------------------------
# 1. Load Data
# ----------------------------------------------
df = load_customer_features()   # << 너의 loader 함수

st.subheader("📁 Preview of Customer Summary Data")
st.dataframe(df.head())

# ----------------------------------------------
# 2. Create Pseudo Labels (loan_approved)
# ----------------------------------------------
st.subheader("🔧 Creating Pseudo Labels")

df['risk_score'] = (
    0.4 * df['luxury_norm'] +
    0.3 * df['wellbeing_norm'] -
    0.2 * df['necessity_norm'] +
    0.1 * df['spending_std_norm']
)

threshold = df['risk_score'].median()
df['loan_approved'] = (df['risk_score'] > threshold).astype(int)

st.write(f"**Pseudo label distribution (1 = approved): {df['loan_approved'].mean():.3f}**")

# ----------------------------------------------
# 3. Train/Test Split + Modeling
# ----------------------------------------------
st.subheader("🤖 Logistic Regression & Random Forest Models")

features = [
    'luxury_norm','misc_norm','necessity_norm','wellbeing_norm',
    'spending_std_norm','total_spent_norm','avg_transaction_norm',
    'transaction_count_norm'
]

X = df[features]
y = df['loan_approved']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale only for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ---------------- Logistic Regression ----------------
logit = LogisticRegression(max_iter=1000)
logit.fit(X_train_scaled, y_train)
logit_pred = logit.predict(X_test_scaled)
logit_auc = roc_auc_score(y_test, logit_pred)

st.write("### 📌 Logistic Regression")
st.write(f"**AUC:** {logit_auc:.3f}")
st.text(classification_report(y_test, logit_pred))

# ---------------- Random Forest ----------------
rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_auc = roc_auc_score(y_test, rf_pred)

st.write("### 🌲 Random Forest")
st.write(f"**AUC:** {rf_auc:.3f}")
st.text(classification_report(y_test, rf_pred))

# ----------------------------------------------
# 4. Feature Importance Plot
# ----------------------------------------------
st.subheader("📈 Feature Importance (Random Forest)")

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)

plt.figure(figsize=(8,5))
importances.plot(kind='barh', color='skyblue')
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.grid(axis='x', linestyle='--', alpha=0.4)
st.pyplot()

st.write("### 🔝 Top Predictive Features")
st.write(importances.sort_values(ascending=False).head(10))

# ----------------------------------------------
# 5. Relationship Between Spending Std and Risk Score
# ----------------------------------------------
st.subheader("📉 Spending Std vs Risk Score")

plt.figure(figsize=(7,4))
sns.scatterplot(x=df['spending_std'], y=df['risk_score'], alpha=0.4)
sns.regplot(x=df['spending_std'], y=df['risk_score'], scatter=False, color='red')
plt.title("Relationship between Spending Std and Risk Score")
plt.xlabel("Spending Standard Deviation")
plt.ylabel("Risk Score")
st.pyplot()

st.markdown("""
**Interpretation:**  
- Higher spending variability → higher risk score  
- Customers with unstable spending patterns are viewed as higher credit risk  
""")

# ----------------------------------------------
# 6. KDE Plot: Spending Std by Loan Approval
# ----------------------------------------------
st.subheader("📊 Spending Std Distribution by Loan Approval")

plt.figure(figsize=(7,4))
sns.kdeplot(x=df['spending_std'], hue=df['loan_approved'], fill=True)
plt.title("Spending Std Distribution by Loan Approval")
plt.xlabel("Spending Standard Deviation")
plt.ylabel("Density")
st.pyplot()

st.markdown("""
**Interpretation:**  
- Approved customers (1) generally have lower spending variability  
- Declined customers (0) show a heavier right tail (higher instability)  
""")

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("👥 Tab 2: Age & City Based Spending Analysis")

df = load_customer_features()

# ----------------------------------------------
# 1. Age Group Summary
# ----------------------------------------------
st.subheader("📌 Spending Ratios by Age Group")

age_summary = df.groupby('age_group')[['luxury','necessity','wellbeing','misc']].mean().reset_index()
st.dataframe(age_summary)

plt.figure(figsize=(10,5))
age_summary.set_index('age_group').plot(kind='bar')
plt.title("Spending Category Ratios by Age Group")
plt.ylabel("Average Ratio")
plt.xticks(rotation=0)
st.pyplot()

# ----------------------------------------------
# 2. Age Group Spending Std
# ----------------------------------------------
st.subheader("📈 Spending Std by Age Group")

age_std = df.groupby('age_group')['spending_std'].mean().reset_index()
st.dataframe(age_std)

plt.figure(figsize=(7,4))
sns.barplot(data=age_std, x='age_group', y='spending_std')
plt.title("Average Spending Std by Age Group")
st.pyplot()

# ----------------------------------------------
# 3. City Group Summary
# ----------------------------------------------
st.subheader("🌆 Spending Ratios by City Size")

city_summary = df.groupby('city_group')[['luxury','necessity','wellbeing','misc']].mean().reset_index()
st.dataframe(city_summary)

plt.figure(figsize=(10,5))
city_summary.set_index('city_group').plot(kind='bar')
plt.title("Spending Category Ratios by City Group")
plt.ylabel("Average Ratio")
plt.xticks(rotation=0)
st.pyplot()

# ----------------------------------------------
# 4. Spending Std by City Group
# ----------------------------------------------
st.subheader("📉 Spending Std by City Group")

city_std = df.groupby('city_group')['spending_std'].mean().reset_index()
st.dataframe(city_std)

plt.figure(figsize=(7,4))
sns.barplot(data=city_std, x='city_group', y='spending_std')
plt.title("Average Spending Std by City Size")
plt.xticks(rotation=30)
st.pyplot()
