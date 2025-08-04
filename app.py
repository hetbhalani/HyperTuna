import streamlit as st

st.set_page_config(page_title="ML Hyperparameter Tuning", layout="centered")

slidebar_rmv = """
    <style>
    [data-testid="stSidebar"] {
        display: none;
    }
    [data-testid="stSidebarNav"] {
        display: none;
    }
    [data-testid="stSidebarHeader"]{
        display: none;
    }
    
    /* Custom CSS to make selectbox wider */
    .stSelectbox > div > div {
        width: 100% !important;
    }
    
    /* Alternative: Make the selectbox container wider */
    div[data-baseweb="select"] {
        width: 100% !important;
    }
    </style>
"""
st.markdown(slidebar_rmv, unsafe_allow_html=True)

st.title("🎯 ML Hyperparameter Tuning")
st.write("Choose a machine learning model below to start tuning its hyperparameters with interactive controls.")

col1, col2, col3 = st.columns([0.5, 3, 0.5])  

with col2:
    st.markdown("#### Select Model")
    model = st.selectbox(
        "",
        ["Select a model...","Random Forest", "Decision Tree", "DBSCAN","SVM","XGBoost"],
        key="model_selector"
    )
    
    st.markdown("")
    
    if st.button("Start Tuning", type="primary", use_container_width=True):
        if model == "Random Forest":
            st.switch_page("pages/random_forest.py")
        elif model == "Decision Tree":
            st.switch_page("pages/decision_tree.py")
        elif model == "DBSCAN":
            st.switch_page("pages/dbscan.py")
        elif model == "SVM":
            st.switch_page("pages/svm.py")
        elif model == "XGBoost":
            st.switch_page("pages/xgboost.py")
        else:
            st.error("Please select a model first!")

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🎛️ Interactive Controls**")
    st.write("Adjust parameters with sliders and see results instantly")

with col2:
    st.markdown("**📊 Real-time Visualization**")
    st.write("Watch your model performance change as you tune")

with col3:
    st.markdown("**⚡ Quick Results**")
    st.write("Get immediate feedback on your parameter choices")

st.markdown("")
st.markdown("")
st.markdown("*Built with Streamlit*", help="Simple and effective ML model tuning")