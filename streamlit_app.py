import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# -------------------------------
# Load model and SHAP explainer
# -------------------------------
@st.cache_resource
def load_model_and_explainer():
    # Load trained model
    with open("car_price_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Create a masker using dummy input (shape must match model input)
    dummy_input = np.zeros((1, 4))  # 4 features: year, mileage, engine_size, luxury_mode
    masker = shap.maskers.Independent(dummy_input)

    # Create SHAP explainer using model and masker
    explainer = shap.Explainer(model, masker)
    return model, explainer

model, explainer = load_model_and_explainer()

# -------------------------------
# App Title and Description
# -------------------------------
st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Car Price Predictor with SHAP Insights")
st.markdown("Estimate car prices and understand **why** with SHAP interpretability and luxury diagnostics.")

# -------------------------------
# User Inputs
# -------------------------------
st.header("🔧 Input Car Details")
year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, value=2015)
mileage = st.number_input("Mileage (in km)", min_value=0, max_value=500000, value=60000)
engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=8.0, value=2.0)
luxury_mode = st.checkbox("Luxury Mode")

# -------------------------------
# Prepare Input DataFrame
# -------------------------------
input_df = pd.DataFrame({
    "year": [year],
    "mileage": [mileage],
    "engine_size": [engine_size],
    "luxury_mode": [int(luxury_mode)]
})

# -------------------------------
# Prediction and Diagnostics
# -------------------------------
if st.button("🔮 Predict Price"):
    try:
        # Run prediction
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Estimated Price: **${prediction:,.2f}**")

        # Luxury diagnostics
        st.subheader("💎 Luxury Mode Diagnostics")
        if luxury_mode:
            st.info("Luxury mode is **enabled** — premium pricing applied.")
        else:
            st.warning("Luxury mode is **disabled** — base pricing used.")

        # SHAP explanation
        st.subheader("🔍 Feature Contributions (SHAP)")
        shap_values = explainer(input_df)

        # Display SHAP values numerically
        for name, value in zip(input_df.columns, shap_values.values[0]):
            st.write(f"**{name}**: {value:+.2f}")

        # SHAP waterfall plot
        st.subheader("📊 SHAP Waterfall Plot")
        try:
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)
        except Exception as plot_err:
            st.warning(f"SHAP plot could not be rendered: {plot_err}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")








