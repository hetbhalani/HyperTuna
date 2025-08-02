import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import streamlit as st

st.title("DBSCAN Clustering")

st.sidebar.header("Tuning Parameters")

df = pd.read_csv('./Datasets/happy_smiley.csv')

X = df[['x', 'y']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

eps = st.sidebar.slider("Eps", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
min_samples = st.sidebar.slider("Min Samples", min_value=2, max_value=20, value=5, step=1)

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
clusters = dbscan.fit_predict(X_scaled)

n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = list(clusters).count(-1)

st.markdown(f"<h4>Clusters Found: {n_clusters}</h4>", unsafe_allow_html=True)
st.markdown(f"<h4>Noise Points: {n_noise}</h4>", unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(10, 8))

unique_clusters = np.unique(clusters)
bright_colors = ['#FF1493', '#00FF00', '#FF4500', '#1E90FF', '#FFD700', 
                 '#FF69B4', '#00CED1', '#32CD32', '#FF6347', '#9370DB',
                 '#FF8C00', '#00FA9A', '#DC143C', '#4169E1', '#ADFF2F']


#again IDK
for i, cluster in enumerate(unique_clusters):
    if cluster == -1:
        mask = clusters == cluster
        ax.scatter(X_scaled[mask, 0], X_scaled[mask, 1], 
                  c='black', marker='x', s=50, alpha=0.7, label='Noise')
    else:
        mask = clusters == cluster
        color = bright_colors[i % len(bright_colors)] 
        ax.scatter(X_scaled[mask, 0], X_scaled[mask, 1], 
                  c=color, marker='o', s=50, alpha=0.8, label=f'Cluster {cluster}')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title(f'DBSCAN Clustering (eps={eps}, min_samples={min_samples})')
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)

slidebar_rmv = """
    <style>
    [data-testid="stSidebarNav"] {
        display: none;
    }
    [data-testid="stSidebarHeader"]{
            display:none;
    }
    </style>
"""
st.markdown(slidebar_rmv, unsafe_allow_html=True)