# 🚗 STREAMLIT APP — Gradient Boosting Model

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
try:
    model = joblib.load("models/gradient_boosting_pipeline.pkl")
except FileNotFoundError:
    st.error("❌ Model file not found. Please check your path or retrain the model.")
    st.stop()

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
    try:
        prediction = model.predict(input_data)
        st.success(f"✅ Predicted Selling Price: R{prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

    # 🔰 Extra: Show input summary
    st.write("### 🔍 Input Summary")
    st.dataframe(input_data.style.highlight_max(axis=0))

# 🚗 Footer Banner
st.markdown(
    "<div style='background-color:#2E86C1;padding:15px;border-radius:10px'>"
    "<h2 style='color:white;text-align:center;'>🚗 Car Price Prediction Dashboard</h2>"
    "</div>",
    unsafe_allow_html=True
)

