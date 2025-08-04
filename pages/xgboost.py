import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("XGBoost Tuning")

slidebar_rmv = """
    <style>
    [data-testid="stSidebarNav"] {
        display: none;
    }
    [data-testid="stSidebarHeader"] {
        display: none;
    }
    </style>
"""
st.markdown(slidebar_rmv, unsafe_allow_html=True)

st.sidebar.header("XGBoost Tuning")

n_estimators = st.sidebar.slider("n_estimators (Number of Trees)", 10, 1000, 100, step=10)
max_depth = st.sidebar.slider("max_depth (Tree Depth)", 1, 50, 6, step=1)
learning_rate = st.sidebar.slider("learning_rate", 0.001, 1.0, 0.1, step=0.01)
subsample = st.sidebar.slider("subsample (Row Sample Ratio)", 0.1, 1.0, 1.0, step=0.05)
colsample_bytree = st.sidebar.slider("colsample_bytree (Feature Sample Ratio)", 0.1, 1.0, 1.0, step=0.05)
gamma = st.sidebar.slider("gamma (Min Loss Reduction for Split)", 0.0, 10.0, 0.0, step=0.1)

df = pd.read_csv("./Datasets/random_forest_data.csv")  

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = XGBClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    learning_rate=learning_rate,
    subsample=subsample,
    colsample_bytree=colsample_bytree,
    gamma=gamma,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.markdown(f"<h3>Accuracy: {acc:.4f}</h3>", unsafe_allow_html=True)

conf_matrix = pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"])
st.subheader("Heatmap: Actual vs Predicted")
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
st.pyplot(fig)

st.subheader("Feature Importances")
importances = pd.Series(model.feature_importances_, index=X.columns)
fig1, ax1 = plt.subplots(figsize=(12, 6))
sns.barplot(x=importances, y=importances.index, palette="coolwarm", ax=ax1)
ax1.set_title("XGBoost Feature Importances")
st.pyplot(fig1)
