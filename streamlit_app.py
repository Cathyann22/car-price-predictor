import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

pipeline = joblib.load("random_forest_best_pipeline.joblib")
feature_list = joblib.load("model_features.pkl")

st.set_page_config(page_title="Car Price Predictor", layout="wide")
st.title("🚗 Car Price Prediction Dashboard")
st.markdown("Upload car details to predict selling price and audit model decisions using SHAP.")

with st.sidebar.form("car_input_form"):
    st.subheader("Enter Car Details")
    input_data = {}
    for feature in feature_list:
        if feature in ['brand', 'model', 'seller_type', 'fuel_type', 'transmission_type']:
            input_data[feature] = st.text_input(f"{feature.title()}", "")
        elif feature == 'seats':
            input_data[feature] = st.slider("Seats", 2, 10, 5)
        else:
            input_data[feature] = st.number_input(f"{feature.title()}", value=0.0)
    submitted = st.form_submit_button("Predict Price")

if submitted:
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_list, fill_value=np.nan)
    input_df[['brand', 'model', 'seller_type', 'fuel_type', 'transmission_type']] = input_df[
        ['brand', 'model', 'seller_type', 'fuel_type', 'transmission_type']
    ].astype(str)

    log_price = pipeline.predict(input_df)[0]
    price = np.expm1(log_price)
    st.success(f"💰 Predicted Selling Price: ₹{price:,.0f}")

    # SHAP audit
    explainer = shap.TreeExplainer(pipeline.named_steps['model'])
    transformed_input = pipeline.named_steps['preprocessor'].transform(input_df)
    if hasattr(transformed_input, 'toarray'):
        transformed_input = transformed_input.toarray()

    shap_values = explainer.shap_values(transformed_input)
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    shap_expl = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=transformed_input[0],
        feature_names=feature_names
    )

    st.subheader("🔍 SHAP Waterfall Audit")
    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots._waterfall.waterfall_legacy(
        shap_expl.base_values,
        shap_expl.values,
        shap_expl.data,
        feature_names=shap_expl.feature_names,
        show=False
    )
    st.pyplot(fig)

fpdf

generate_pdf(price, input_data)
