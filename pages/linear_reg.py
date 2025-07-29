import streamlit as st
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

st.title("Linear Regression Tuning")

st.sidebar.header("Tuning Parameters")
fit_intercept = st.sidebar.checkbox("Fit Intercept", value=True)
normalize = st.sidebar.checkbox("Normalize (deprecated, for demo only)", value=False)

df = pd.read_csv("./Datasets/linear_reg_data.csv")
X = df.iloc[:,:1]
y = df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression(fit_intercept=fit_intercept)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
st.write(f"**Mean Squared Error:** {mse:.2f}")

fig, ax = plt.subplots()
ax.scatter(X_test, y_test, color='blue', label='Actual')
ax.plot(X_test, y_pred, color='red', label='Predicted')
ax.set_title("Linear Regression Fit")
ax.legend()
st.pyplot(fig)
