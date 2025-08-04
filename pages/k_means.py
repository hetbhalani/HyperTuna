import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.title("KMeans Clustering")

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

st.sidebar.header("KMeans Hyperparameters")

n_clusters = st.sidebar.slider("Number of Clusters (K)", 1, 10, 3, step=1)
init_method = st.sidebar.selectbox("Initialization Method", ["k-means++", "random"])
n_init = st.sidebar.slider("Number of Initializations", 1, 20, 10, step=1)
max_iter = st.sidebar.slider("Max Iterations", 100, 1000, 300, step=50)
algorithm = st.sidebar.selectbox("Algorithm", ["lloyd", "elkan"])

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

kmeans = KMeans(
    n_clusters=n_clusters,
    init=init_method,
    n_init=n_init,
    max_iter=max_iter,
    algorithm=algorithm,
    random_state=42
)
y_kmeans = kmeans.fit_predict(X)

X["Cluster"] = y_kmeans

pca = PCA(n_components=2)
components = pca.fit_transform(X.drop("Cluster", axis=1))
X["PCA1"] = components[:, 0]
X["PCA2"] = components[:, 1]

st.subheader("Cluster Visualization")
fig, ax = plt.subplots()
sns.scatterplot(data=X, x="PCA1", y="PCA2", hue="Cluster", palette="Set2", s=100, ax=ax)
ax.set_title("KMeans Clusters")
st.pyplot(fig)