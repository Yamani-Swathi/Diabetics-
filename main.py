import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

# Streamlit setup
st.set_page_config(page_title="Diabetes SHAP Explainer", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return X, y

# Load data
X, y = load_data()

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Streamlit app
st.title("🧠 Diabetes Prediction with SHAP Explainable AI")
st.markdown("Enter patient details below to predict **Diabetic / Not Diabetic** and explain with SHAP.")

# Input sliders for each feature
input_values = {}
for col in X.columns:
    min_val = float(X[col].min())
    max_val = float(X[col].max())
    mean_val = float(X[col].mean())
    input_values[col] = st.slider(col, min_val, max_val, mean_val)

# Convert inputs to DataFrame
input_df = pd.DataFrame([input_values])

# Predict and Explain
if st.button("🔍 Predict and Explain"):
    prediction = model.predict(input_df)[0]

    # ✅ Classification logic (based on regression score)
    if prediction > 140:
        st.error(f"🔴 Likely Diabetic (Score: {prediction:.2f})")
    else:
        st.success(f"🟢 Likely Not Diabetic (Score: {prediction:.2f})")

    st.subheader("📊 SHAP Feature Contribution (Waterfall Plot)")

    # SHAP Explanation
    explainer = shap.Explainer(model, X)
    shap_values = explainer(input_df)

    # ✅ Do NOT change this (graph is correct)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
