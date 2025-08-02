import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler


st.title("SVM Classifier Tuning")

st.sidebar.header("Hyperparameters")
kernel = st.sidebar.selectbox("Kernel", options=["linear", "poly", "rbf", "sigmoid"])
C = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0)
gamma = st.sidebar.selectbox("Gamma", options=["scale", "auto"])

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")

st.sidebar.subheader("Select Features for Visualization")
feature_x = st.sidebar.selectbox("X-axis feature", X.columns, index=2)
feature_y = st.sidebar.selectbox("Y-axis feature", X.columns, index=1)

X_all = X
X_vis = X[[feature_x, feature_y]]

X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.3, random_state=42)
X_vis_train, X_vis_test, _, _ = train_test_split(X_vis, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
st.markdown(f"<h3>Accuracy: {acc:.4f}</h3>", unsafe_allow_html=True)

st.subheader("Decision Boundary (2D)")

scaler_vis = StandardScaler()
X_vis_train_scaled = scaler_vis.fit_transform(X_vis_train)
X_vis_test_scaled = scaler_vis.transform(X_vis_test)

model_vis = SVC(kernel=kernel, C=C, gamma=gamma)
model_vis.fit(X_vis_train_scaled, y_train)

x_min, x_max = X_vis_train_scaled[:, 0].min() - 1, X_vis_train_scaled[:, 0].max() + 1
y_min, y_max = X_vis_train_scaled[:, 1].min() - 1, X_vis_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = model_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig_db, ax_db = plt.subplots(figsize=(10, 6))
contour = ax_db.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.Set1)  # soft background
scatter = ax_db.scatter(X_vis_train_scaled[:, 0], X_vis_train_scaled[:, 1], 
                        c=y_train, cmap=plt.cm.Set1, edgecolor='k', s=60)


ax_db.set_xlabel(feature_x)
ax_db.set_ylabel(feature_y)
ax_db.set_title("SVM Decision Regions with Colored Points")
handles, _ = scatter.legend_elements()
labels = iris.target_names
ax_db.legend(handles, labels, title="Classes", loc="upper right")

st.pyplot(fig_db)
