# 🚗 Car Price Prediction App — Streamlit + Diagnostics
# ============================================================

# Imports
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from streamlit_shap import st_shap

# Load Trained Pipeline
# ============================================================
try:
    pipeline = joblib.load("best_random_forest_pipeline.pkl")
    feature_list = joblib.load("model_features.pkl")
except FileNotFoundError:
    st.error("❌ Model file not found. Please check your path or retrain the model.")
    st.stop()

# 🏷️ App Title
# ============================================================
st.title("🚗 Car Price Prediction App")

# 📋 Sidebar Inputs
# ============================================================
st.sidebar.header("Enter Car Details")

engine = st.sidebar.number_input("Engine (cc)", min_value=500, max_value=5000, value=1500)
max_power = st.sidebar.number_input("Max Power (bhp)", min_value=20, max_value=500, value=100)
vehicle_age = st.sidebar.slider("Vehicle Age (years)", 0, 30, value=5)
fuel_type = st.sidebar.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
transmission_type = st.sidebar.selectbox("Transmission", ['Manual', 'Automatic'])
seller_type = st.sidebar.selectbox("Seller Type", ['Dealer', 'Individual', 'Trustmark Dealer'])
brand = st.sidebar.selectbox("Brand", ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'BMW', 'Audi'])

# Prediction Trigger
# ============================================================
if st.sidebar.button("Predict Price"):

    # ✅ Prepare input
    input_dict = {
        'engine': engine,
        'max_power': max_power,
        'vehicle_age': vehicle_age,
        'fuel_type': fuel_type,
        'transmission_type': transmission_type,
        'seller_type': seller_type,
        'brand': brand
    }
    input_df = pd.DataFrame([input_dict])

    # ✅ Align input with training features
    try:
        input_aligned = input_df[feature_list]
    except KeyError as e:
        st.error(f"❌ Input alignment failed: {e}")
        st.stop()

    # ✅ Predict log price and convert to actual price
    try:
        log_price = pipeline.predict(input_aligned)[0]
        predicted_price = np.expm1(log_price)
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

    # abs for Prediction & Diagnostics
    # ============================================================
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "💎 SHAP Audit", "📊 Global Summary"])

    # Prediction Output
    with tab1:
        st.subheader("Estimated Price")
        st.success(f"Your {input_dict['brand']} is valued at **₹ {predicted_price:,.0f}**")

    # SHAP Force Plot (Local)
    with tab2:
        st.subheader("🔍 Feature Impact (SHAP)")
        try:
            transformed_input = pipeline.named_steps["preprocessor"].transform(input_aligned)
            model = pipeline.named_steps["model"]
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(transformed_input)

            st_shap(shap.force_plot(
                explainer.expected_value,
                shap_values[0],
                features=input_aligned,
                matplotlib=False
            ))
        except Exception as e:
            st.warning(f"SHAP force plot failed: {e}")

    # Summary Plot (Global)
    with tab3:
        st.subheader("Global Feature Importance")
        try:
            st.image("shap_summary.png", caption="SHAP Summary Plot", use_column_width=True)
        except Exception as e:
            st.warning(f"SHAP summary plot unavailable: {e}")
