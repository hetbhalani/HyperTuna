import streamlit as st

st.set_page_config(page_title = "ML Hyperparameter Tuning", layout="centered")

st.title("ML Hyperparameter Tuning")
st.write("Select a model")

model = st.selectbox("choose a model", ["Linear Regression", "Logistic Regression","Random Forest","Decision Tree"])

if st.button("Tune"):
    if model == "Linear Regression":
        st.switch_page("pages/linear_reg.py")
    elif model == "Random Forest":
        st.switch_page("pages/random_forest.py")
    elif model == "Decision Tree":
        st.switch_page("pages/decision_tree.py")