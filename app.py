import streamlit as st

st.title("Car Price Predictor")

model_choice = st.selectbox("Choose a model", ["Linear", "XGBoost"])
price_range = st.slider("Set price range", 0, 100000)

if st.button("Predict"):
    st.write(f"Running prediction with {model_choice} model for price range up to {price_range}")

import shap
import xgboost


