from sklearn.datasets import load_breast_cancer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import numpy as np
import streamlit as st

st.title("Decision Tree Tuning")

st.sidebar.header("Tuning Parameters")
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

data = load_breast_cancer()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['target'] = data.target

X_all = df.drop('target', axis=1) 
y = df['target']

X_plot = df[["mean radius", "mean texture"]] 

X_all_train, X_all_test, y_train, y_test = train_test_split(X_all, y, test_size=0.3, random_state=9)
X_plot_train, X_plot_test, _, _ = train_test_split(X_plot, y, test_size=0.3, random_state=9)

max_depth = st.sidebar.number_input("Max Depth", 1, 10, value=5, step=1)
min_samples_split = st.sidebar.slider("Min Samples Split", min_value=2, max_value=20, value=2, step=1)
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", min_value=1, max_value=20, value=1, step=1)
criterion = st.sidebar.selectbox("Criterion", options=["gini", "entropy", "log_loss"], index=0)
splitter = st.sidebar.selectbox("Splitter", options=["best", "random"], index=0)
max_features = st.sidebar.selectbox(
    "Max Features", 
    options=["sqrt", "log2", None] + list(range(1, 31))
)

model_all_features = DecisionTreeClassifier(
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    criterion=criterion,
    splitter=splitter,
    max_features=max_features,
    random_state=42
)

model_all_features.fit(X_all_train, y_train)
y_pred_all = model_all_features.predict(X_all_test)
acc_all_features = accuracy_score(y_test, y_pred_all)

model_plot = DecisionTreeClassifier(
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    criterion=criterion,
    splitter=splitter,
    max_features=max_features,
    random_state=42
)

model_plot.fit(X_plot_train, y_train)
y_pred_plot = model_plot.predict(X_plot_test)
acc_plot_features = accuracy_score(y_test, y_pred_plot)

st.markdown(f"<h3>Accuracy: {acc_all_features:.4f}</h3>", unsafe_allow_html=True)

x_min, x_max = X_plot.iloc[:, 0].min() - 1, X_plot.iloc[:, 0].max() + 1
y_min, y_max = X_plot.iloc[:, 1].min() - 1, X_plot.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = model_plot.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(10, 8))
ax.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.coolwarm)
ax.scatter(X_plot_test.iloc[:, 0], X_plot_test.iloc[:, 1], c=y_test, cmap=plt.cm.coolwarm, marker='s', edgecolors='k', s=50, label='Test')
ax.set_xlabel('Mean Radius')
ax.set_ylabel('Mean Texture')
ax.set_title('Decision Tree Decision Boundary(2 features for visualization)')
ax.legend()

st.pyplot(fig)

st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

fig2, ax2 = plt.subplots(figsize=(16, 10))
plot_tree(model_plot, filled=True, feature_names=["Mean Radius", "Mean Texture"], 
          class_names=["Malignant", "Benign"], rounded=True, fontsize=12, ax=ax2)
ax2.set_title("Decision Tree Structure (2 Features)")

st.pyplot(fig2)

