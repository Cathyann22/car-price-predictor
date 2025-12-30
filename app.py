
 # 🚗 STREAMLIT APP

import streamlit as st
import joblib
import pandas as pd

# 🚗 Modern Title with Styling
st.set_page_config(page_title="Car Price Prediction 🚗", page_icon="🚘", layout="centered")
st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>Car Price Prediction App 🚗</h1>",
    unsafe_allow_html=True
)
st.markdown("<p style='text-align: center;'>Interactive app powered by Gradient Boosting — with Luxury Car Mode!</p>", unsafe_allow_html=True)

# ✅ Load the trained pipeline
model = joblib.load("models/gradient_boosting_pipeline.pkl")

# 🔰 Sidebar for Inputs
st.sidebar.header("🔧 Input Features")
st.sidebar.write("Adjust sliders to configure car specifications:")

max_power = st.sidebar.slider("Max Power (bhp)", 40, 250, 120)
engine = st.sidebar.slider("Engine Size (cc)", 800, 4000, 1600)
mileage = st.sidebar.slider("Mileage (km/l)", 5, 25, 15)
vehicle_age = st.sidebar.slider("Vehicle Age (years)", 0, 20, 5)

# 🔰 Luxury Car Mode toggle
luxury_mode = st.sidebar.checkbox("Enable Luxury Car Mode 🚘")

# ✅ Adjust inputs if Luxury Car Mode is enabled
if luxury_mode:
    st.sidebar.info("Luxury Car Mode activated: boosting performance features for premium predictions.")
    max_power = max_power * 1.5   # simulate higher performance
    engine = engine * 1.3         # larger engine size
    mileage = mileage * 0.8       # luxury cars often less fuel-efficient

# 🔰 Main Section
st.markdown("---")
st.subheader("📊 Prediction Results")

if st.button("Predict Selling Price"):
    input_data = pd.DataFrame({
        'max_power': [max_power],
        'engine': [engine],
        'mileage': [mileage],
        'vehicle_age': [vehicle_age]
    })
    prediction = model.predict(input_data)
    st.success(f"✅ Predicted Selling Price: R{prediction[0]:,.2f}")

    # 🔰 Extra: Show input summary
    st.write("### 🔍 Input Summary")
    st.dataframe(input_data.style.highlight_max(axis=0))

st.markdown(
    "<div style='background-color:#2E86C1;padding:15px;border-radius:10px'>"
    "<h2 style='color:white;text-align:center;'>🚗 Car Price Prediction Dashboard</h2>"
    "</div>",
    unsafe_allow_html=True
)


# 🚗 Car Price Prediction App — Streamlit + Diagnostics


# Imports

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from streamlit_shap import st_shap

# Load Trained Pipeline

try:
    pipeline = joblib.load("best_random_forest_pipeline.pkl")
    feature_list = joblib.load("model_features.pkl")
except FileNotFoundError:
    st.error("❌ Model file not found. Please check your path or retrain the model.")
    st.stop()

# 🏷️ App Title

st.title("🚗 Car Price Prediction App")

# 📋 Sidebar Inputs
st.sidebar.header("Enter Car Details")

engine = st.sidebar.number_input("Engine (cc)", min_value=500, max_value=5000, value=1500)
max_power = st.sidebar.number_input("Max Power (bhp)", min_value=20, max_value=500, value=100)
vehicle_age = st.sidebar.slider("Vehicle Age (years)", 0, 30, value=5)
fuel_type = st.sidebar.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
transmission_type = st.sidebar.selectbox("Transmission", ['Manual', 'Automatic'])
seller_type = st.sidebar.selectbox("Seller Type", ['Dealer', 'Individual', 'Trustmark Dealer'])
brand = st.sidebar.selectbox("Brand", ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'BMW', 'Audi'])

# Prediction Trigger

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

