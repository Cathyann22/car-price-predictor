import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# --- Load model and optional preprocessor ---
model = joblib.load("car_price_model.pkl")
# Uncomment if you have a preprocessor
# preprocessor = joblib.load("preprocessor.pkl")

# --- Define brand encoding ---
brand_options = ["Toyota", "BMW", "Mercedes", "Hyundai"]
brand_encoding = {brand: idx for idx, brand in enumerate(brand_options)}

def main():
    st.set_page_config(page_title="Car Price Predictor", layout="centered")
    st.title("🚗 Car Price Prediction Dashboard")
    st.write("Estimate your car's resale value with luxury mode and diagnostics.")

    # --- Input Section ---
    st.subheader("🔧 Input Car Details")
    selected_brand = st.selectbox("Select Car Brand", brand_options)
    brand_encoded = brand_encoding[selected_brand]

    year = st.slider("Year of Manufacture", 2000, 2025, 2015)
    mileage = st.number_input("Mileage (in km)", min_value=0)
    luxury_mode = st.selectbox("Luxury Mode", ["Yes", "No"])
    luxury_flag = 1 if luxury_mode == "Yes" else 0

    # --- Prediction Trigger ---
    if st.button("🔍 Predict Price"):
        # Create input DataFrame
        sample_df = pd.DataFrame({
            "brand": [brand_encoded],
            "year": [year],
            "mileage": [mileage],
            "luxury_mode": [luxury_flag]
        })

        # Optional: Apply preprocessing
        # processed_df = preprocessor.transform(sample_df)
        # prediction_input = processed_df
        prediction_input = sample_df  # if no preprocessor

        # --- Run Prediction ---
        predicted_price = model.predict(prediction_input)[0]
        st.metric("💰 Estimated Price", f"₹ {predicted_price:,.0f}")

        # --- SHAP Diagnostics ---
        st.subheader("📊 Feature Importance (SHAP)")
        explainer = shap.Explainer(model.predict, prediction_input)
        shap_values = explainer(prediction_input)

        fig, ax = plt.subplots()
        shap.plots.bar(shap_values, show=False)
        st.pyplot(fig)

        # --- Error Segmentation Placeholder ---
        st.subheader("🧪 Error Segmentation")
        st.write("This section will show error clusters once test data is integrated.")

if __name__ == "__main__":
    main()


