# CATHY MAHUMANE=STREAMLIT FILE!!!!

# CAR PRICE PREDICTION DIAGNOSTIC MODEL


# ============================================TITLE: CAR PREDICTION APP=====================

#  ✅  Purpose

# This is a Streamlit-based web app that allows users to input car details and receive an estimated market price using a trained machine learning pipeline (car_price_pipeline.pkl).

#✅ Model Integration

# Model Loading: Loads a pre-trained pipeline using joblib.load("car_price_pipeline.pkl").
# Loads a full pipeline with preprocessing and an XGBoost regressor:Core Functionality
 pipeline = joblib.load("car_price_pipeline.pkl")

#✅ User Input Widgets
# Sidebar collects:
# Brand: 
# Maruti, Hyundai, Honda, Toyota, Volkswagen, Ford, Mahindra, Tata, 
# BMW, Audi, Mercedes-Benz 
# Year: 2000–2025
# Mileage: 0–500,000 km
# Fuel Type: Petrol, Diesel, CNG, LPG, Electric
# Transmission: Manual, Automatic
# Luxury Mode: Checkbox to enable SHAP diagnostics

#✅ Prediction Logic
# On button click:
# Constructs a DataFrame from user input
# Predicts price using the pipeline
# Displays formatted result:
	st.success(f"Estimated Price: ₹{int(predicted_price):,}")
	
#✅ SHAP Diagnostics
# If luxury mode is enabled and price > ₹1,000,000:
# Extracts model and preprocessor
# Computes SHAP values
# Displays a waterfall plot for interpretability

#✅ Deployment Confirmation
# Includes a minimal test block at the end to confirm successful deployment:
st.title("🚗 Minimal Deployment Test")
st.write("If you're seeing this, your app deployed successfully!")

#✅ Output
# Estimated price in rupees
# SHAP waterfall plot for luxury cars
# Sample dataframe for visual confirmation

# ✅  User Input: Collects car features via interactive widgets:

# Brand: Maruti, Hyundai, Honda, Toyota, Volkswagen, Ford, Mahindra, Tata, BMW, Audi, Mercedes-Benz
# Year: Slider from 2000 to 2025
# Mileage: Numeric input
# Fuel Type: Petrol, Diesel, CNG, LPG, Electric
# Transmission: Manual, Automatic

# ✅ Prediction: Uses the pipeline to predict car price based on user input.
#  Luxury Mode: If enabled and the predicted price exceeds ₹1,000,000, the app displays SHAP diagnostics to explain feature contributions.

# ✅ Output
# Displays the predicted price in a formatted success message.
#  It renders a SHAP waterfall plot for interpretability.

 # ✅ PIPELINE

#  Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import tempfile
from weasyprint import HTML

# ✅  Load the trained pipeline (includes preprocessing + model)
pipeline = joblib.load("car_price_pipeline.pkl")

# ✅ Configure Streamlit page
st.set_page_config(page_title="Car Price Predictor", layout="wide")
st.title("🚗 Car Price Prediction App")
st.markdown("Use the sidebar to enter car details and estimate its market price.")

# ✅ Sidebar: Input widgets for user data
with st.sidebar:
    st.header("🔧 Input Features")

    brand = st.selectbox("Brand", [
        "Maruti", "Hyundai", "Honda", "Toyota", "Volkswagen",
        "Ford", "Mahindra", "Tata", "BMW", "Audi", "Mercedes-Benz"
    ])
    year = st.slider("Year", min_value=2000, max_value=2025, value=2020)
    mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000, value=30000)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    luxury_mode = st.checkbox("💎 Show SHAP diagnostics for luxury cars")

# ✅  Prepare input data
input_df = pd.DataFrame([{
    "Brand": brand,
    "Year": year,
    "Mileage": mileage,
    "Fuel_Type": fuel_type,
    "Transmission": transmission
}])

# ✅  Create tabs for Prediction and Export
tab1, tab2 = st.tabs(["📈 Prediction", "📄 Export Report"])

# ✅ Tab 1: Prediction and SHAP diagnostics
with tab1:
    st.subheader("📈 Predict Car Price")
    if st.button("Predict Price"):
        predicted_price = pipeline.predict(input_df)[0]
        st.success(f"Estimated Price: ₹{int(predicted_price):,}")

        # SHAP diagnostics for luxury cars
        if luxury_mode and predicted_price > 1_000_000:
            st.subheader("🔍 SHAP Explanation")
            model = pipeline.named_steps["model"]
            preprocessor = pipeline.named_steps["preprocessor"]
            transformed_input = preprocessor.transform(input_df)
            explainer = shap.Explainer(model, transformed_input)
            shap_values = explainer(transformed_input)
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            st.pyplot(fig)

# ✅ Tab 2: PDF Export
with tab2:
    st.subheader("📄 Generate PDF Report")
    if st.button("🖨️ Export Prediction as PDF"):
        html_content = f"""
        <h1>Car Price Prediction Report</h1>
        <p><strong>Brand:</strong> {brand}</p>
        <p><strong>Year:</strong> {year}</p>
        <p><strong>Mileage:</strong> {mileage} km</p>
        <p><strong>Fuel Type:</strong> {fuel_type}</p>
        <p><strong>Transmission:</strong> {transmission}</p>
        <p><strong>Estimated Price:</strong> ₹{int(predicted_price):,}</p>
        """

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            HTML(string=html_content).write_pdf(tmp_file.name)
            with open(tmp_file.name, "rb") as f:
                st.download_button(
                    label="📥 Download PDF",
                    data=f.read(),
                    file_name="car_price_report.pdf",
                    mime="application/pdf"
                )

# ✅ Deployment confirmation block

st.title("🚗 Minimal Deployment Test")
st.write("If you're seeing this, your app deployed successfully!")

# ✅  Sample dataframe for visual confirmation
df = pd.DataFrame({
    "Brand": ["Toyota", "BMW", "Audi"],
    "Price": [500000, 1200000, 1500000]
})
st.dataframe(df)




#==========================================#===========================================






#==========================================#===========================================









