import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix

st.title("KNN Classification Tuning")

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

st.sidebar.header("KNN Hyperparameters")

n_neighbors = st.sidebar.slider("Number of Neighbors (K)", 1, 20, 5, step=1)
weights = st.sidebar.selectbox("Weights", ["uniform", "distance"])
algorithm = st.sidebar.selectbox("Algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
leaf_size = st.sidebar.slider("Leaf Size", 10, 100, 30, step=5)
p = st.sidebar.selectbox("Distance Metric (p)", [1, 2], format_func=lambda x: "Manhattan (1)" if x==1 else "Euclidean (2)")

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

knn = KNeighborsClassifier(
    n_neighbors=n_neighbors,
    weights=weights,
    algorithm=algorithm,
    leaf_size=leaf_size,
    p=p
)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.markdown(f"<h3>Accuracy: {acc:.4f}</h3>", unsafe_allow_html=True)

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=iris.target_names, yticklabels=iris.target_names, ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
st.pyplot(fig_cm)

pca = PCA(n_components=2)
components = pca.fit_transform(X)
df_plot = pd.DataFrame(components, columns=["PCA1", "PCA2"])
df_plot["Target"] = iris.target

st.subheader("Iris Data in 2D")
fig_pca, ax_pca = plt.subplots()
sns.scatterplot(data=df_plot, x="PCA1", y="PCA2", hue="Target", palette="Set2", s=100, ax=ax_pca)
ax_pca.set_title("True Labels Visualized with")
st.pyplot(fig_pca)
