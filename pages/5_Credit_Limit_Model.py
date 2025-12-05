import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

# =========================================================
# 0. 스타일 세팅
# =========================================================
sns.set_theme(
    style="whitegrid",
    rc={
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": "lightgray",
        "grid.color": "lightgray",
        "axes.labelsize": 12,
        "axes.titlesize": 17,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11
    }
)

# =========================================================
# 1. 데이터 로더
# =========================================================
@st.cache_data
def load_customer_features():
    """
    data/processed/customer_features.csv 불러오는 함수.
    """
    base = os.path.dirname(os.path.dirname(__file__))  # 프로젝트 루트
    path = os.path.join(base, "data", "processed", "customer_features.csv")
    df = pd.read_csv(path)
    return df


# =========================================================
# 2. 피처 엔지니어링 + 리스크 스코어 + 라벨 생성 함수
# =========================================================
def build_model_data(
    df_raw: pd.DataFrame,
    w_std2: float,
    w_lux_std: float,
    w_misc_std: float,
    w_nec_wb: float,
    w_ticket_mix: float,
    noise_std: float = 0.03,
):
    """
    - 정규화된 X0_n 생성
    - 비선형/상호작용 Z 피처 생성
    - risk_score 계산 (가중치 + 노이즈)
    - 중앙값 기준 loan_approved 라벨 생성
    - X (모델 입력)와 y 반환
    """
    df = df_raw.copy()

    cols = [
        "total_spent",
        "avg_transaction",
        "transaction_count",
        "spending_std",
        "luxury",
        "misc",
        "necessity",
        "wellbeing",
    ]

    # --- 0~1 정규화 ---
    X0 = df[cols].copy()
    for c in cols:
        r = X0[c].max() - X0[c].min()
        X0[c + "_n"] = (X0[c] - X0[c].min()) / (r if r != 0 else 1)

    # --- 비선형 / 상호작용 피처 Z ---
    Z = pd.DataFrame(index=X0.index)
    Z["std2"] = X0["spending_std_n"] ** 2
    Z["lux2"] = X0["luxury_n"] ** 2
    Z["nec2"] = X0["necessity_n"] ** 2
    Z["wb2"] = X0["wellbeing_n"] ** 2
    Z["lux_std"] = X0["luxury_n"] * X0["spending_std_n"]
    Z["misc_std"] = X0["misc_n"] * X0["spending_std_n"]
    Z["nec_wb"] = X0["necessity_n"] * X0["wellbeing_n"]
    Z["size_freq"] = X0["total_spent_n"] * X0["transaction_count_n"]
    Z["ticket_mix"] = X0["avg_transaction_n"] * (X0["necessity_n"] - X0["luxury_n"])

    # --- risk_score 계산 ---
    rng = np.random.default_rng(42)
    risk = (
        w_std2 * Z["std2"]
        + w_lux_std * Z["lux_std"]
        + w_misc_std * Z["misc_std"]
        + w_nec_wb * Z["nec_wb"]
        + w_ticket_mix * Z["ticket_mix"]
    )
    if noise_std > 0:
        risk = risk + rng.normal(0, noise_std, size=len(risk))

    df["risk_score"] = risk

    # --- 중앙값 기준 0/1 라벨 ---
    thr = np.median(df["risk_score"])
    df["loan_approved"] = (df["risk_score"] < thr).astype(int)

    # --- 모델용 X, y ---
    X = X0[[c + "_n" for c in cols]].join(Z)
    y = df["loan_approved"]

    return df, X, y, X0, Z


# =========================================================
# 3. 페이지 제목 & 데이터 로드
# =========================================================
df_raw = load_customer_features()

st.title("💳 Credit Risk & Limit Modeling Dashboard")
st.caption("Using processed `customer_features` to simulate a realistic credit risk model.")

# =========================================================
# 4. 사이드바: Risk Score 가중치 슬라이더
# =========================================================
st.sidebar.header("⚙️ Risk Score Settings")

w_std2 = st.sidebar.slider("Weight: Variance (std²)", 0.0, 0.6, 0.35, 0.01)
w_lux_std = st.sidebar.slider("Weight: Luxury × Std", 0.0, 0.6, 0.25, 0.01)
w_misc_std = st.sidebar.slider("Weight: Misc × Std", 0.0, 0.3, 0.10, 0.01)
w_nec_wb = st.sidebar.slider(
    "Weight: Necessity × Wellbeing (negative)", -0.4, 0.0, -0.20, 0.01
)
w_ticket_mix = st.sidebar.slider(
    "Weight: Ticket Mix (negative)", -0.4, 0.0, -0.10, 0.01
)
noise_std = st.sidebar.slider("Noise level (avoid overfitting)", 0.0, 0.10, 0.03, 0.01)

st.sidebar.markdown("---")
st.sidebar.write("**Current risk formula (simplified):**")
st.sidebar.latex(
    r"""
    \text{risk} =
    w_{\text{std2}}\cdot \text{std2}
    + w_{\text{lux\_std}}\cdot \text{lux\_std}
    + w_{\text{misc\_std}}\cdot \text{misc\_std}
    + w_{\text{nec\_wb}}\cdot \text{nec\_wb}
    + w_{\text{ticket\_mix}}\cdot \text{ticket\_mix}
    """
)

# 슬라이더 값으로 실제 데이터 구성
df, X, y, X0, Z = build_model_data(
    df_raw,
    w_std2=w_std2,
    w_lux_std=w_lux_std,
    w_misc_std=w_misc_std,
    w_nec_wb=w_nec_wb,
    w_ticket_mix=w_ticket_mix,
    noise_std=noise_std,
)

# =========================================================
# 5. 모델 학습 (한 번만 실행해서 탭에서 같이 씀)
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

scaler = StandardScaler(with_mean=False)
Xs_tr = scaler.fit_transform(X_train)
Xs_te = scaler.transform(X_test)

logit = LogisticRegression(max_iter=2000, class_weight="balanced")
logit.fit(Xs_tr, y_train)
p1 = logit.predict_proba(Xs_te)[:, 1]

rf = RandomForestClassifier(
    n_estimators=400, class_weight="balanced", random_state=42
)
rf.fit(X_train, y_train)
p2 = rf.predict_proba(X_test)[:, 1]

auc_logit = roc_auc_score(y_test, p1)
auc_rf = roc_auc_score(y_test, p2)

# =========================================================
# 6. 탭 구조
# =========================================================
tab_eda, tab_model = st.tabs(["📊 EDA & Data Overview", "🤖 Modeling & Segmentation"])

# ---------------------------------------------------------
# TAB 1: EDA
# ---------------------------------------------------------
with tab_eda:
    st.header("📊 1. EDA & Data Overview")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Number of customers", len(df))
    with col_b:
        st.metric("Avg total spending", f"${df['total_spent'].mean():,.0f}")
    with col_c:
        st.metric("Average age", f"{df['avg_age'].mean():.1f} yrs")

    st.subheader("Sample of Processed Customer Features")
    st.caption("Includes engineered variables like category ratios and fraud_rate.")
    st.dataframe(df.head())

    # --- Spending_std & fraud_rate distribution ---
    st.subheader("Spending Volatility & Fraud Rate")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(df["spending_std"], bins=30, kde=True, ax=ax)
        ax.set_title("Distribution of Spending Std")
        ax.set_xlabel("spending_std")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.histplot(df["fraud_rate"], bins=30, kde=True, ax=ax)
        ax.set_title("Distribution of Fraud Rate (per customer)")
        ax.set_xlabel("fraud_rate")
        st.pyplot(fig)

    # --- Category ratio heatmap ---
    st.subheader("Correlation: Spending Ratios & Volatility")

    corr_cols = ["luxury", "necessity", "wellbeing", "misc", "spending_std"]
    corr = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Correlation among Spending Ratios & Std")
    st.pyplot(fig)

    st.markdown(
        """
        **Takeaways**
        - Higher **luxury spending ratio** tends to be positively related to volatility (`spending_std`).  
        - More **necessity / wellbeing**-oriented portfolios usually show lower volatility.  
        - These patterns motivate why we use `std²`, `lux_std`, `nec_wb`, etc. in the credit risk formula.
        """
    )

# ---------------------------------------------------------
# TAB 2: MODELING & SEGMENTATION
# ---------------------------------------------------------
with tab_model:
    st.header("🤖 2. Modeling & Risk-Based Segmentation")

    # -----------------------------
    # 2.1 Risk score & label overview
    # -----------------------------
    st.subheader("2.1 Risk Score & Loan Approval Labels")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Median risk score (threshold)", f"{df['risk_score'].median():.3f}")
    with col2:
        st.metric("Approval rate (loan_approved=1)", f"{df['loan_approved'].mean():.3f}")
    with col3:
        st.metric("Weight on std²", f"{w_std2:.2f}")

    fig, ax = plt.subplots()
    sns.histplot(df["risk_score"], bins=30, kde=True, ax=ax)
    ax.set_title("Distribution of Risk Score")
    st.pyplot(fig)

    # -----------------------------
    # 2.2 Model performance
    # -----------------------------
    st.subheader("2.2 Model Performance (Logistic vs Random Forest)")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Logistic Regression (with scaling)**")
        st.write(f"**AUC:** {auc_logit:.3f}")
        st.text(
            classification_report(
                y_test, (p1 >= 0.5).astype(int), digits=3
            )
        )

    with c2:
        st.markdown("**Random Forest (non-linear baseline)**")
        st.write(f"**AUC:** {auc_rf:.3f}")
        st.text(
            classification_report(
                y_test, (p2 >= 0.5).astype(int), digits=3
            )
        )

    # -----------------------------
    # 2.3 Feature importance (RF)
    # -----------------------------
    st.subheader("2.3 Feature Importance (Random Forest)")

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    importances.tail(15).plot(kind="barh", ax=ax)
    ax.set_title("Top 15 Features Driving Loan Approval")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    st.pyplot(fig)

    st.markdown(
        """
        **Interpretation (example)**  
        - Strong positive importance for `luxury_n` / `lux_std` → customers with **volatile luxury spending** are treated as riskier.  
        - Higher weight on `std2` means **unstable spending behavior** is heavily penalized.  
        - Negative interaction terms (`nec_wb`, `ticket_mix`) reward **stable, necessity-focused spending**.
        """
    )

    # -----------------------------
    # 2.4 spending_std vs risk_score 관계
    # -----------------------------
    st.subheader("2.4 Spending Std vs Risk Score & Approval")

    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots()
        sns.scatterplot(
            x=df["spending_std"],
            y=df["risk_score"],
            alpha=0.35,
            ax=ax,
        )
        sns.regplot(
            x=df["spending_std"],
            y=df["risk_score"],
            scatter=False,
            order=2,
            color="red",
            ax=ax,
        )
        ax.set_title("Spending Std vs Risk Score")
        ax.set_xlabel("Spending Std (spending_std)")
        ax.set_ylabel("Risk Score")
        st.pyplot(fig)

    with col_b:
        fig, ax = plt.subplots()
        sns.kdeplot(
            x=df["spending_std"],
            hue=df["loan_approved"],
            fill=True,
            common_norm=False,
            alpha=0.5,
            ax=ax,
        )
        ax.set_title("Spending Std Distribution by Loan Approval")
        ax.set_xlabel("Spending Std")
        st.pyplot(fig)

    st.markdown(
        """
        - 오른쪽 꼬리가 긴 high-std 구간에서 **거절(0)** 비율이 더 높은 패턴을 확인할 수 있습니다.  
        - 이는 우리가 만든 risk formula가 실제로 **spending_std를 중요한 리스크 신호로 사용한다**는 걸 보여줍니다.
        """
    )

    # -----------------------------
    # 2.5 Segmentation: cluster / age_group / job
    # -----------------------------
    st.subheader("2.5 Segment-Level Risk & Approval Comparison")

    available_group_vars = [
        g for g in ["cluster", "age_group", "city_group", "job"] if g in df.columns
    ]

    if not available_group_vars:
        st.warning(
            "No segmentation variables (`cluster`, `age_group`, `city_group`, `job`) found in the dataframe."
        )
    else:
        group_var = st.selectbox(
            "Choose a grouping variable:",
            options=available_group_vars,
            index=available_group_vars.index("cluster")
            if "cluster" in available_group_vars
            else 0,
        )

        st.write(f"Grouping by **{group_var}**")

        # job이 너무 많으면 상위 N개만
        if group_var == "job":
            g = (
                df.groupby("job")
                .agg(
                    n=("loan_approved", "size"),
                    approval_rate=("loan_approved", "mean"),
                    avg_risk=("risk_score", "mean"),
                    avg_total_spent=("total_spent", "mean"),
                )
                .reset_index()
            )
            g = g.sort_values("n", ascending=False).head(15)
        else:
            g = (
                df.groupby(group_var)
                .agg(
                    n=("loan_approved", "size"),
                    approval_rate=("loan_approved", "mean"),
                    avg_risk=("risk_score", "mean"),
                    avg_total_spent=("total_spent", "mean"),
                )
                .reset_index()
            )

        st.dataframe(g)

        # Barplot: approval rate by group
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(
            data=g,
            x=group_var,
            y="approval_rate",
            ax=ax,
        )
        ax.set_title(f"Approval Rate by {group_var}")
        ax.set_ylabel("Approval Rate")
        ax.set_xlabel(group_var)
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

        # Barplot: avg risk by group
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(
            data=g,
            x=group_var,
            y="avg_risk",
            ax=ax,
        )
        ax.set_title(f"Average Risk Score by {group_var}")
        ax.set_ylabel("Average Risk Score")
        ax.set_xlabel(group_var)
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

        st.markdown(
            """
            **How to read this section**
            - Groups with **higher average risk score** and **lower approval rate** are treated as riskier segments.  
            - Comparing `cluster`, `age_group`, and `job` helps explain **which customer segments are driving the model decisions**.
            """
        )

    st.success("✨ Modeling, feature importance, and segmentation analysis are all up-to-date with your current risk weights!")
