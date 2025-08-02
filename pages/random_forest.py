import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("Random Forest Tuning")

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

st.sidebar.header("Hyperparameter Tuning")
n_estimators = st.sidebar.slider("Number of Trees", 10, 500, 100, step=10)
max_depth = st.sidebar.slider("Max Depth", 1, 50, 10, step=1)
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 20, 1)
max_features = st.sidebar.selectbox("Max Features", options=["sqrt", "log2", None])
bootstrap = st.sidebar.checkbox("Bootstrap", value=True)

df = pd.read_csv("../Datasets/random_forest_data.csv")  

X = df.iloc[:, :-1]
y = df.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    max_features=max_features,
    bootstrap=bootstrap,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.markdown(f"<h3>Accuracy: {acc:.4f}</h3>", unsafe_allow_html=True)

conf_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
st.subheader("Heatmap of Actual vs Predicted")
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
st.pyplot(fig)


st.subheader("Feature Importances")
importances = pd.Series(model.feature_importances_, index=X.columns)
fig1, ax1 = plt.subplots(figsize=(12, 6))
sns.barplot(x=importances, y=importances.index, palette="coolwarm", ax=ax1)
ax1.set_title("Feature Importance")
st.pyplot(fig1)
