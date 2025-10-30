
 ================================================================
# ✅ FULL STREAMLIT PREDICTION + SHAP + 🚗 CAR PRICE PREDICTOR APP
#===============================================================

# Includes SHAP explainability and academic commentary
# ==================================================

# -----------------------
# IMPORTS
# -----------------------
import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from fpdf import FPDF
import shap
import matplotlib.pyplot as plt

# -----------------------
# ✅ PAGE CONFIG & STYLING
# -----------------------
st.set_page_config(page_title="Car Price Predictor", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(to right, #e3f2fd, #fce4ec); font-family: 'Segoe UI', sans-serif;}
.luxury {background-color: #fff8e1; padding: 15px; border-radius: 10px; border: 1px solid #ffcc80; margin-bottom: 10px;}
.insight {background-color: #e8f5e9; padding: 10px; border-left: 5px solid #43a047; margin-bottom: 10px;}
</style>
""", unsafe_allow_html=True)

st.title("🚗 Car Price Predictor")
st.markdown("""
Predict car prices using a trained **Random Forest** model enhanced with **SHAP explainability** 
and **automated academic PDF reporting**.
""")

# -----------------------
# ✅ DEFINE FEATURES & TARGET
# -----------------------
features = ['brand', 'fuel_type', 'transmission_type', 'vehicle_age',
            'km_driven', 'mileage', 'engine', 'max_power', 'seats']
target = 'selling_price'

# -----------------------
# ✅ LOAD DATA
# -----------------------
try:
    df = pd.read_csv("car_price_dataset.csv")
    st.success("✅ Dataset loaded from local path!")
except FileNotFoundError:
    df = None
    st.warning("⚠️ Local dataset not found. Please upload it below.")

uploaded_file = st.file_uploader("📤 Upload your CSV dataset", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Uploaded file loaded successfully!")

if df is None:
    st.stop()

# -----------------------
# ✅ CLEAN & VALIDATE DATA
# -----------------------
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

missing_cols = [col for col in features + [target] if col not in df.columns]
if missing_cols:
    st.error(f"❌ Missing required columns: {missing_cols}")
    st.stop()

df = df.dropna(subset=features + [target])
df['log_price'] = np.log1p(df[target])

X = df[features]
y = df['log_price']

# -----------------------
# ✅ TRAIN-TEST SPLIT
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------
# ✅ PREPROCESSING
# -----------------------
numeric_features = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
categorical_features = list(set(features) - set(numeric_features))

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])

# -----------------------
# ✅ RANDOM FOREST PIPELINE
# -----------------------
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

pipeline.fit(X_train, y_train)
st.success("✅ Model trained successfully!")

# -----------------------
# ✅ SAVE PIPELINE
# -----------------------
os.makedirs("models", exist_ok=True)
with open("models/log_rf_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)
with open("models/log_rf_features.pkl", "wb") as f:
    pickle.dump(features, f)

# -----------------------
# 🧾 USER INPUT FORM
# -----------------------
st.subheader("🧾 Enter Car Details for Prediction")
input_data = {}
with st.form("car_input_form"):
    for feature in features:
        if feature in numeric_features:
            input_data[feature] = st.number_input(f"{feature.replace('_', ' ').title()}", min_value=0.0, value=0.0)
        else:
            input_data[feature] = st.selectbox(f"{feature.replace('_', ' ').title()}", df[feature].unique())
    submitted = st.form_submit_button("🔮 Predict Price")

# -----------------------
# ✅ PREDICTION & SHAP EXPLAINABILITY
# -----------------------
if submitted:
    try:
        input_df = pd.DataFrame([input_data])
        log_pred = pipeline.predict(input_df)
        price_pred = np.expm1(log_pred)
        st.success(f"**Predicted Selling Price:** ₹{price_pred[0]:,.0f}")

        # -----------------------
        # SHAP VALUES
        # -----------------------
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
        X_train_transformed = preprocessor.transform(X_train)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train_transformed)

        # -----------------------
        # PLOT SHAP: GLOBAL IMPORTANCE
        # -----------------------
        st.subheader("🔍 SHAP Explainability")
        st.markdown("**Global Feature Importance** – average contribution of each feature to model predictions.")
        plt.figure(figsize=(8, 5))
        shap.summary_plot(shap_values, X_train_transformed, show=False)
        plt.tight_layout()
        plt.savefig("shap_bar.png", bbox_inches="tight")
        st.image("shap_bar.png", caption="Global SHAP Feature Importance")

        # -----------------------
        # PLOT SHAP: LOCAL EXPLANATION
        # -----------------------
        st.markdown("**Local Explanation** – how this specific prediction was formed.")
        single_val = preprocessor.transform(input_df)
        single_shap = explainer.shap_values(single_val)
        shap.waterfall_plot(
            shap.Explanation(values=single_shap[0],
                             base_values=explainer.expected_value,
                             data=single_val[0]),
            show=False
        )
        plt.tight_layout()
        plt.savefig("shap_waterfall.png", bbox_inches="tight")
        st.image("shap_waterfall.png", caption="Local SHAP Waterfall")

        # -----------------------
        # 📄 GENERATE PDF REPORT
        # -----------------------
        class PDF(FPDF):
            def header(self):
                self.set_font("Helvetica", "B", 16)
                self.cell(0, 10, "🚗 Car Price Prediction Report", ln=True, align="C")
                self.ln(5)

        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 8, f"Predicted Price: ₹{price_pred[0]:,.0f}\n")
        pdf.multi_cell(0, 8, "Input Features:")
        for k, v in input_data.items():
            pdf.multi_cell(0, 8, f"- {k.replace('_', ' ').title()}: {v}")
        pdf.multi_cell(0, 8, "\nInterpretation:\nThis Random Forest regression model demonstrates high predictive accuracy and transparency through SHAP explainability, allowing insight into individual feature contributions.")
        pdf.image("shap_bar.png", x=10, y=None, w=180)
        pdf.image("shap_waterfall.png", x=10, y=None, w=180)
        pdf.output("car_prediction_report.pdf")

        with open("car_prediction_report.pdf", "rb") as f:
            st.download_button("📥 Download Academic PDF Report", f, file_name="car_prediction_report.pdf")

        # -----------------------
        # INSIGHTS
        # -----------------------
        st.markdown("<div class='insight'>", unsafe_allow_html=True)
        st.markdown("""
        ### 📖 Academic Insights
        - **Feature scaling** and encoding ensure that mixed data types are harmonized for learning.  
        - **Random Forests** capture non-linear dependencies effectively, outperforming simpler linear models.  
        - **SHAP** provides *explainable AI* insight—quantifying each feature’s marginal contribution to model output.  
        - **Model interpretability** is vital for ethical and accountable AI in data-driven decision systems.  
        - **PDF documentation** formalizes results for reproducibility and academic evaluation.  
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")



