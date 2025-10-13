# CATHY MAHUMANE=STREAMLIT FILE!!!!

# CAR PRICE PREDICTION DIAGNOSTIC MODEL


# ============================================TITLE: CAR PREDICTION APP=====================

# ➡︎ SUMMARY:

#  ➡︎ Purpose

# This is a Streamlit-based web app that allows users to input car details and receive an estimated market price using a trained machine learning pipeline (car_price_pipeline.pkl).

# ➡︎ Core Functionality

# Model Loading: Loads a pre-trained pipeline using joblib.load("car_price_pipeline.pkl"), which includes preprocessing and an XGBoost regressor.

# ➡︎ User Input: Collects car features via interactive widgets:

# Brand: Maruti, Hyundai, Honda, Toyota, Volkswagen, Ford, Mahindra, Tata, BMW, Audi, Mercedes-Benz
# Year: Slider from 2000 to 2025
# Mileage: Numeric input
# Fuel Type: Petrol, Diesel, CNG, LPG, Electric
# Transmission: Manual, Automatic

# ➡︎ Prediction: Uses the pipeline to predict car price based on user input.

# ➡︎ Luxury Mode: If enabled and the predicted price exceeds ₹1,000,000, the app displays SHAP diagnostics to explain feature contributions.

# ➡︎ Output
# Displays the predicted price in a formatted success message.

# ➡︎  It renders a SHAP waterfall plot for interpretability.

# ➡︎ PIPELINE========

#  import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load the trained pipeline (includes preprocessing + model)
pipeline = joblib.load("car_price_pipeline.pkl")

# Configure Streamlit page
st.set_page_config(page_title="Car Price Predictor", layout="wide")

st.title("🚗 Car Price Prediction App")

st.markdown("Use the sidebar to enter car details and estimate its market price.")

# Sidebar: Input widgets for user data
with st.sidebar:
    st.header(" Input Features")

    # Dropdown for car brand — customized from dataset
    brand = st.selectbox("Brand", [
        "Maruti", "Hyundai", "Honda", "Toyota", "Volkswagen",
        "Ford", "Mahindra", "Tata", "BMW", "Audi", "Mercedes-Benz"
    ])

    # Slider for car year
    year = st.slider("Year", min_value=2000, max_value=2025, value=2020)

    # Numeric input for mileage
    mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000, value=30000)

    # Dropdown for fuel type
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])

    # Dropdown for transmission type
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

    # Checkbox to enable SHAP diagnostics for luxury cars
    luxury_mode = st.checkbox(" Show SHAP diagnostics for luxury cars")

# Prepare input data as a single-row DataFrame
input_df = pd.DataFrame([{
    "Brand": brand,
    "Year": year,
    "Mileage": mileage,
    "Fuel_Type": fuel_type,
    "Transmission": transmission
}])

#  Prediction trigger
if st.button("Predict Price"):
    # Run prediction using the pipeline
    predicted_price = pipeline.predict(input_df)[0]
    st.success(f"Estimated Price: ₹{int(predicted_price):,}")

    # SHAP diagnostics for luxury cars (price > ₹1,000,000)
    if luxury_mode and predicted_price > 1_000_000:
        st.subheader(" SHAP Explanation")

        # Extract model and preprocessor from pipeline
        model = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocessor"]

        # Transform input for SHAP
        transformed_input = preprocessor.transform(input_df)

        # Create SHAP explainer and compute values
        explainer = shap.Explainer(model, transformed_input)
        shap_values = explainer(transformed_input)

        # Plot SHAP waterfall chart
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)



#==========================================#===========================================






