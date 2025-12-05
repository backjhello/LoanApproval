import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from src.loader import load_customer_features

st.title("🤖 Clustering")

df = load_customer_features()
X = df[['total_spent','avg_transaction','transaction_count','spending_std']]

k = st.slider("Number of clusters", 2, 8, 4)

km = KMeans(n_clusters=k, random_state=42)
df['cluster'] = km.fit_predict(X)

st.subheader("Cluster Counts")
st.write(df['cluster'].value_counts())

fig, ax = plt.subplots()
scatter = ax.scatter(df['total_spent'], df['avg_transaction'], c=df['cluster'], cmap='viridis')
ax.set_xlabel("Total Spent")
ax.set_ylabel("Avg Transaction")
st.pyplot(fig)
