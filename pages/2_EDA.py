import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.loader import load_customer_features

st.title("📊 Exploratory Data Analysis")
df = load_customer_features()

# 나이대 그룹 생성
bins = [0, 25, 35, 45, 55, 65, 120]
labels = ['<25', '25-34', '35-44', '45-54', '55-64', '65+']
df['age_group'] = pd.cut(df['avg_age'], bins=bins, labels=labels, right=False)

# 평균 소비 비율 계산
age_spend = df.groupby('age_group')[['luxury','necessity','wellbeing','misc']].mean().reset_index()

st.subheader("Age Group Spending Ratio Table")
st.dataframe(age_spend)

fig, ax = plt.subplots(figsize=(7,5))
age_spend.set_index('age_group').plot(kind='bar', stacked=True, ax=ax)
ax.set_title("Average Spending Ratio by Age Group")
st.pyplot(fig)
