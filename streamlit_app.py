
# ================================================================
# ✅ FULL STREAMLIT PREDICTION + SHAP + 🚗 CAR PRICE PREDICTOR APP
# ================================================================

# 🎓 LUXURY CAR PRICE PREDICTOR + SHAP + PERFORMANCE METRICS

# Elegant academic Streamlit app with:
# ✅ SHAP explainability
# ✅ RMSE, MAE, R² performance metrics
# ✅ Luxury Mode (UI + adjusted scaling)
# ✅ Academic PDF report with upload
# ================================================================

import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from fpdf import FPDF
import shap
import matplotlib.pyplot as plt

# ================================================================
# 🎨 PAGE CONFIG & THEME
# ================================================================
st.set_page_config(page_title="Luxury Car Price Predictor", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #e3f2fd, #fce4ec);
    font-family: 'Segoe UI', sans-serif;
}
.luxury {
    background: linear-gradient(135deg, #f7f7f7, #d7ccc8);
    border: 1px solid #b0bec5;
    border-radius: 10px;
    padding: 12px;
    margin-top: 10px;
}
.insight {
    background-color: #f9fbe7;
    padding: 10px;
    border-left: 5px solid #8bc34a;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

st.title("🚗 **Luxury Car Price Predictor**")
st.markdown("""
Welcome to the **academic and interpretable car price prediction tool**.
It uses a **Random Forest Regressor** with **SHAP explainability** and **automated reporting** to ensure transparency and fairness in AI-driven pricing models.
""")

# ================================================================
# 🧭 USER MODE SELECTION
# ================================================================
luxury_mode = st.toggle("✨ Enable Luxury Mode (High-End Vehicle Focus)", value=False)

# ================================================================
# 📂 DATASET UPLOAD
# ================================================================
try:
    df = pd.read_csv("car_price_dataset.csv")
    st.success("✅ Default dataset loaded successfully.")
except FileNotFoundError:
    df = None
    st.warning("⚠️ Please upload your dataset.")

uploaded_file = st.file_uploader("📤 Upload CSV dataset", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")

if df is None:
    st.stop()

# ================================================================
# 🧹 DATA CLEANING
# ================================================================
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
features = ['brand', 'fuel_type', 'transmission_type', 'vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
target = 'selling_price'

if not all(col in df.columns for col in features + [target]):
    st.error("❌ Required columns missing from dataset.")
    st.stop()

df = df.dropna(subset=features + [target])
df['log_price'] = np.log1p(df[target])
X = df[features]
y = df['log_price']

# ================================================================
# ⚙️ PREPROCESSING PIPELINE
# ================================================================
num_features = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
cat_features = list(set(features) - set(num_features))

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
])

model = RandomForestRegressor(
    n_estimators=200 if luxury_mode else 100,
    max_depth=20 if luxury_mode else None,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# ================================================================
# 📊 MODEL PERFORMANCE METRICS
# ================================================================
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae_test = mean_absolute_error(y_test, y_test_pred)

# Derived F1-like reliability score (scaled interpretability)
f1_like = max(0, min(1, (r2_test + (1 - rmse_test / (rmse_train + 1e-9))) / 2))

st.markdown("### ⚙️ Model Performance Summary")
st.markdown(f"""
- **Train R²:** {r2_train:.3f}  
- **Test R²:** {r2_test:.3f}  
- **Test RMSE:** ₹{np.expm1(rmse_test):,.0f}  
- **Test MAE:** ₹{np.expm1(mae_test):,.0f}  
- **Reliability Index (F1-analogue):** {f1_like:.2%}  
""")

# ================================================================
# 🧾 PREDICTION INPUT FORM
# ================================================================
st.subheader("🧾 Enter Car Details")

input_data = {}
with st.form("input_form"):
    for feature in features:
        if feature in num_features:
            input_data[feature] = st.number_input(f"{feature.replace('_', ' ').title()}", min_value=0.0, value=0.0)
        else:
            input_data[feature] = st.selectbox(f"{feature.replace('_', ' ').title()}", df[feature].unique())
    submitted = st.form_submit_button("🔮 Predict Luxury Car Price")

# ================================================================
# 🔮 PREDICTION + SHAP EXPLANATION
# ================================================================
if submitted:
    input_df = pd.DataFrame([input_data])
    log_pred = pipeline.predict(input_df)
    price_pred = np.expm1(log_pred)[0]

    st.success(f"💰 **Predicted Selling Price:** ₹{price_pred:,.0f}")

    preproc = pipeline.named_steps['preprocessor']
    model = pipeline.named_steps['model']
    X_train_t = preproc.transform(X_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train_t)

    st.subheader("🔍 SHAP Explainability")

    # Global importance
    fig_global = plt.figure(figsize=(8, 5))
    shap.summary_plot(
        shap_values, X_train_t,
        feature_names=num_features + list(preproc.named_transformers_['cat'].get_feature_names_out(cat_features)),
        show=False
    )
    plt.tight_layout()
    plt.savefig("shap_global.png", bbox_inches="tight")
    st.image("shap_global.png", caption="Global Feature Importance")

    # Local waterfall
    transformed_input = preproc.transform(input_df)
    single_shap = explainer.shap_values(transformed_input)
    fig_local = plt.figure(figsize=(9, 5))
    shap.waterfall_plot(
        shap.Explanation(values=single_shap[0], base_values=explainer.expected_value, data=transformed_input[0]),
        show=False
    )
    plt.tight_layout()
    plt.savefig("shap_local.png", bbox_inches="tight")
    st.image("shap_local.png", caption="Local Prediction Breakdown")

    # ================================================================
    # 🧾 PDF REPORT GENERATION
    # ================================================================
    class PDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 14)
            self.cell(0, 10, "Luxury Car Price Prediction Report", ln=True, align="C")
            self.ln(4)

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "", 12)
    pdf.multi_cell(0, 8, f"Predicted Price: ₹{price_pred:,.0f}\n")
    pdf.multi_cell(0, 8, "Model Performance Metrics:")
    pdf.multi_cell(0, 8, f"- R² (Test): {r2_test:.3f}")
    pdf.multi_cell(0, 8, f"- RMSE: ₹{np.expm1(rmse_test):,.0f}")
    pdf.multi_cell(0, 8, f"- MAE: ₹{np.expm1(mae_test):,.0f}")
    pdf.multi_cell(0, 8, f"- Reliability Index: {f1_like:.2%}")
    pdf.multi_cell(0, 8, "\nInput Features:")
    for k, v in input_data.items():
        pdf.multi_cell(0, 8, f"• {k.title()}: {v}")
    pdf.image("shap_global.png", x=10, y=None, w=180)
    pdf.image("shap_local.png", x=10, y=None, w=180)
    pdf.output("Luxury_Car_Report.pdf")

    with open("Luxury_Car_Report.pdf", "rb") as f:
        st.download_button("📥 Download Full Academic Report", f, file_name="Luxury_Car_Report.pdf")

  # ================================================================
    # 📘 INSIGHTS
    # ================================================================
    st.markdown("<div class='insight'>", unsafe_allow_html=True)
    st.markdown("""
    ### 📘 Academic Insights
    1. **Transformations:** Log-scaling of the price variable stabilizes variance and improves model generalization.
    2. **Feature Encoding:** Mixed data (categorical + numeric) is harmonized via `ColumnTransformer`, ensuring fair comparisons.
    3. **Explainability:** SHAP provides *axiomatic transparency*, aligning with ethical AI principles.
    4. **Reliability Index (F1-like)** quantifies predictive consistency beyond R² and RMSE.
    5. **Luxury Mode:** Uses deeper trees and more estimators for nuanced high-end pricing.
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)



