import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

model = pickle.load(open('output_ccg.sav', 'rb'))

df = pd.read_excel("output_cc.xlsx")
features = ['Purchases', 'Balance']
X = df[features]

st.title('Clustering Credit Card')

numClusters = st.slider("Select Number of Clusters", min_value=1, max_value=10, value=3)

model = KMeans(n_clusters=numClusters)
clusters = model.fit_predict(X)

fig, ax = plt.subplots()
scatter = ax.scatter(X['Purchases'], X['Balance'], c=clusters, cmap='viridis')
ax.set_xlabel('Purchases')
ax.set_ylabel('Balance')

legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)

st.pyplot(fig)
