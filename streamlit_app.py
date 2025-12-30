# ================================================================
# ✅ FULL STREAMLIT PREDICTION + SHAP + 🚗 CAR PRICE PREDICTOR APP
# Academic Edition — Ready for Deployment & GitHub LFS
# ================================================================


# ================================================================
# 🚗 Luxury Car Price Predictor — Academic Edition
# ================================================================

# -----------------------
# IMPORTS
# -----------------------
import os
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import shap
from fpdf import FPDF
import matplotlib.pyplot as plt
import pickle

# ================================================================
# ✅ Page Config & Custom CSS
# ================================================================
st.set_page_config(page_title="Luxury Car Price Predictor", layout="wide")
st.markdown("""
<style>
.stApp {background: linear-gradient(to right, #f3f7fb, #ffffff); font-family: 'Segoe UI', sans-serif;}
.luxury {background: linear-gradient(135deg,#f7f7f7,#d7ccc8); border:1px solid #b0bec5; border-radius:10px; padding:12px; margin-top:10px;}
.insight {background-color:#f9fbe7; padding:10px; border-left:5px solid #8bc34a; margin-top:15px;}
.metric-card {background:#ffffffcc; padding:10px; border-radius:8px; text-align:center;}
</style>
""", unsafe_allow_html=True)

st.title("🚗 Luxury Car Price Predictor — Academic Edition")

# ================================================================
# ✅ Tabs
# ================================================================
tabs = st.tabs([
    "📊 Data Overview",
    "🧾 Single Prediction",
    "🔍 SHAP Explainability",
    "📚 Academic Insights",
    "📄 Export PDF Report"
])

# ================================================================
# ✅ Load Dataset
# ================================================================
dataset_loaded = False
uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    dataset_loaded = True
elif Path("car_price_dataset.csv").exists():
    df = pd.read_csv("car_price_dataset.csv")
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    dataset_loaded = True
else:
    df = None

with tabs[0]:
    st.subheader("📊 Dataset Overview")
    if dataset_loaded:
        st.success("✅ Dataset loaded successfully.")
        st.dataframe(df.head(10))
        st.write("**Summary statistics:**")
        st.write(df.describe())
        st.write("**Missing values per column:**")
        st.write(df.isna().sum())
    else:
        st.warning("⚠️ Please upload a CSV dataset to proceed.")

if not dataset_loaded:
    st.stop()

# ================================================================
# ✅ Data Cleaning & Feature Setup
# ================================================================
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
features = ['brand', 'fuel_type', 'transmission_type', 'vehicle_age',
            'km_driven', 'mileage', 'engine', 'max_power', 'seats']
target = 'selling_price'

missing_cols = [c for c in features + [target] if c not in df.columns]
if missing_cols:
    st.error(f"❌ Missing required columns: {missing_cols}")
    st.stop()

df = df.dropna(subset=features + [target]).copy()
df['price_orig'] = df[target].astype(float)
df['log_price'] = np.log1p(df['price_orig'])

X = df[features]
y = df['log_price']

num_features = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
cat_features = [f for f in features if f not in num_features]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
])

# ================================================================
# ✅ Luxury Mode & Pipeline Training
# ================================================================
luxury_mode = st.checkbox("💎 Enable Luxury Mode (High-End Vehicle Focus)", value=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(
        n_estimators=200 if luxury_mode else 100,
        max_depth=20 if luxury_mode else None,
        random_state=42,
        n_jobs=-1
    ))
])

pipeline.fit(X_train, y_train)

# ================================================================
# ✅ Model Evaluation
# ================================================================
y_test_pred_log = pipeline.predict(X_test)
y_test_true_orig = np.expm1(y_test)
y_test_pred_orig = np.expm1(y_test_pred_log)

r2_real = r2_score(y_test_true_orig, y_test_pred_orig)
rmse_real = np.sqrt(mean_squared_error(y_test_true_orig, y_test_pred_orig))
mae_real = mean_absolute_error(y_test_true_orig, y_test_pred_orig)
practical_accuracy = ((np.abs(y_test_true_orig - y_test_pred_orig) / (y_test_true_orig + 1e-9)) <= 0.10).mean()

# ================================================================
# ✅ Save pipeline
# ================================================================
def save_pipeline(pipeline, filename="models/random_forest_pipeline.pkl"):
    os.makedirs("models", exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(pipeline, f)
    st.success(f"✅ Trained pipeline saved at: {filename}")

save_pipeline(pipeline)

# ================================================================
# ✅ SHAP Precomputation
# ================================================================
explainer = shap.TreeExplainer(pipeline.named_steps['model'])
X_train_tr = pipeline.named_steps['preprocessor'].transform(X_train)
sample_size = min(200, X_train_tr.shape[0])
sample = X_train_tr[:sample_size]
shap_values = explainer.shap_values(sample)
cat_names = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_features)
feature_names = list(num_features) + list(cat_names)

# ================================================================
# ✅ Single Prediction Tab
# ================================================================
input_data = {}
pred_price = None

with tabs[1]:
    st.subheader("🧾 Single Prediction Form")
    with st.form("predict_form"):
        for feat in features:
            if feat in num_features:
                if feat == "seats":
                    input_data[feat] = st.number_input(feat.title(), min_value=1, max_value=20, value=5, step=1)
                else:
                    input_data[feat] = st.number_input(feat.title(), min_value=0.0, value=0.0, step=0.1)
            else:
                input_data[feat] = st.selectbox(feat.title(), options=df[feat].dropna().unique().tolist())
        submitted = st.form_submit_button("🔮 Predict Price")

    if submitted:
        try:
            input_df = pd.DataFrame([input_data])
            log_pred = pipeline.predict(input_df)
            pred_price = float(np.expm1(log_pred)[0])

            # Luxury adjustment
            luxury_brands = {"bmw","audi","mercedes","porsche","jaguar","land rover"}
            brand_val = str(input_data.get("brand", "")).lower()
            if luxury_mode and brand_val in luxury_brands:
                pred_price *= 1.12
                st.success("💎 Luxury brand adjustment applied (+12%)")
            elif luxury_mode:
                pred_price *= 1.03
                st.info("✅ Market premium applied (+3%)")

            st.markdown(f"### 💰 Predicted Price: **₹{pred_price:,.0f}**")

            # Derived metrics
            age = input_data.get("vehicle_age", 0)
            km = input_data.get("km_driven", 0)
            est_depr_pct = age * 8.0
            cost_per_km = pred_price / (km + 1)
            c1, c2, c3 = st.columns(3)
            c1.metric("Estimated Depreciation (ann.)", f"{est_depr_pct:.1f}%")
            c2.metric("Estimated Cost per km", f"₹{cost_per_km:,.2f}")
            c3.metric("Value Category", "Premium" if pred_price > df['price_orig'].median() else "Value")

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ================================================================
# ✅ SHAP Explainability Tab
# ================================================================
with tabs[2]:
    st.subheader("🔍 Global SHAP Explainability")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) quantifies **feature contributions** to predictions.
    """)
    plt.figure(figsize=(8,4))
    shap.summary_plot(shap_values, sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", bbox_inches="tight")
    plt.close()
    st.image("shap_summary.png", caption="Global SHAP Summary", use_column_width=True)

# ================================================================
# ✅ Academic Insights Tab
# ================================================================
with tabs[3]:
    st.subheader("📚 Academic Insights")
    st.markdown("### 📈 Model Evaluation Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{rmse_real:.2f}")
    c2.metric("R² Score", f"{r2_real:.3f}")
    c3.metric("MAE", f"{mae_real:.2f}")

    st.markdown("### 💎 Luxury Market Analysis")
    luxury_brands_list = ["BMW","Mercedes-Benz","Audi","Jaguar","Land Rover"]
    df["Luxury Mode"] = df["brand"].apply(lambda x: "Luxury" if x in luxury_brands_list else "Standard")
    st.bar_chart(df["Luxury Mode"].value_counts())

# ================================================================
# ✅ PDF Export Tab
# ================================================================
with tabs[4]:
    st.subheader("📄 Export Academic Report as PDF")
    if pred_price:
        if st.button("🖨️ Generate PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(True, margin=15)
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Luxury Car Price Prediction Report", ln=True, align="C")
            pdf.ln(5)
            pdf.set_font("Helvetica", "", 11)
            pdf.cell(0, 10, f"Timestamp: {pd.Timestamp.now()}", ln=True)
            pdf.ln(5)
            pdf.cell(0, 10, f"Luxury Mode Enabled: {'Yes' if luxury_mode else 'No'}", ln=True)
            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 10, f"Predicted Price: ₹{pred_price:,.0f}", ln=True)
            pdf.ln(5)
            pdf.set_font("Helvetica", "", 11)
            pdf.cell(0, 8, "Input Features:", ln=True)
            for k, v in input_data.items():
                pdf.cell(0, 6, f"  • {k}: {v}", ln=True)
            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 10, "Model Evaluation Metrics:", ln=True)
            pdf.set_font("Helvetica", "", 11)
            pdf.cell(0, 8, f"  • RMSE: {rmse_real:.2f}", ln=True)
            pdf.cell(0, 8, f"  • R² Score: {r2_real:.3f}", ln=True)
            pdf.cell(0, 8, f"  • MAE: {mae_real:.2f}", ln=True)
            pdf.ln(5)
            if os.path.exists("shap_summary.png"):
                pdf.cell(0, 10, "Global SHAP Summary:", ln=True)
                pdf.image("shap_summary.png", w=170)
            pdf_path = "luxury_car_report.pdf"
            pdf.output(pdf_path)
            with open(pdf_path, "rb") as f:
                st.download_button("📥 Download PDF Report", f, file_name=pdf_path, mime="application/pdf")
    else:
        st.info("Predict a car price first to generate a PDF report.")

# ================================================================
# 🚀 Deployment Notes
# ================================================================
st.markdown("---")
st.subheader("🚀 Deployment Guide")
st.markdown("""
1. Push project to GitHub.
2. Track models via Git LFS: `git lfs track "models/*.pkl"`.
3. Deploy via [Streamlit Cloud](https://share.streamlit.io).
4. Ensures reproducibility, transparency, and academic rigor.
""")
