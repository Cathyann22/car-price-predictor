
# ================================================================
# ✅ FULL STREAMLIT PREDICTION + SHAP + 🚗 CAR PRICE PREDICTOR APP
# ================================================================

# app.py
# ================================================================
# ✅  LUXURY CAR PRICE PREDICTOR — Final, cleaned & academic-ready
# Features:
# - Robust preprocessing
# - Proper SHAP feature-name alignment
# - RMSE/MAE/R2 on original ₹ scale (and on log-scale where stated)
# - Practical accuracy (within ±10%) for end users
# - Luxury Mode, academic PDF report, safe plotting
# ================================================================

import os
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from fpdf import FPDF
import shap
import matplotlib.pyplot as plt

# -----------------------
# ✅ Page config + styles
# -----------------------
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
st.markdown("Interpretable Random Forest regression with SHAP explainability and academic PDF reporting.")

# -----------------------
# Luxury mode toggle (use checkbox for Streamlit compatibility)
# -----------------------
luxury_mode = st.checkbox("💎 Enable Luxury Mode (High-End Vehicle Focus)", value=False)
if luxury_mode:
    st.markdown('<div class="luxury"><strong>Luxury Mode active — deeper trees & premium presentation</strong></div>', unsafe_allow_html=True)

# -----------------------
# ✅ Define features + target
# -----------------------
features = ['brand', 'fuel_type', 'transmission_type', 'vehicle_age',
            'km_driven', 'mileage', 'engine', 'max_power', 'seats']
target = 'selling_price'

# -----------------------
# ✅ Load dataset (local or upload)
# -----------------------
try:
    df = pd.read_csv("car_price_dataset.csv")
    st.success("✅ Dataset loaded from local path.")
except FileNotFoundError:
    df = None
    st.warning("⚠️ Local dataset not found. Please upload a CSV.")

uploaded_file = st.file_uploader("📤 Upload CSV dataset", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Uploaded dataset loaded.")

if df is None:
    st.stop()

# -----------------------
# ✅ Clean & validate
# -----------------------
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
missing = [c for c in features + [target] if c not in df.columns]
if missing:
    st.error(f"❌ Missing required columns: {missing}")
    st.stop()

df = df.dropna(subset=features + [target]).copy()
# Keep original price for final metrics and reporting
df['price_orig'] = df[target].astype(float)
# Log-transform target for modeling (recommended)
df['log_price'] = np.log1p(df['price_orig'])

X = df[features]
y = df['log_price']

# -----------------------
# ✅ Preprocessing pipeline
# -----------------------
num_features = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
cat_features = [f for f in features if f not in num_features]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
])

# model hyperparams tuned slightly for luxury mode
model = RandomForestRegressor(
    n_estimators=200 if luxury_mode else 100,
    max_depth=20 if luxury_mode else None,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
st.success("✅ Model trained successfully.")

# -----------------------
# ✅ Evaluation (log-scale + original scale)
# -----------------------
# Predictions (log-scale)
y_train_pred_log = pipeline.predict(X_train)
y_test_pred_log = pipeline.predict(X_test)

# Inverse transform to original currency scale
y_train_true_orig = np.expm1(y_train)
y_test_true_orig = np.expm1(y_test)
y_train_pred_orig = np.expm1(y_train_pred_log)
y_test_pred_orig = np.expm1(y_test_pred_log)

# Metrics on original scale (more interpretable for stakeholders)
r2_log_train = r2_score(y_train, y_train_pred_log)
r2_log_test = r2_score(y_test, y_test_pred_log)

r2_real = r2_score(y_test_true_orig, y_test_pred_orig)
rmse_real = np.sqrt(mean_squared_error(y_test_true_orig, y_test_pred_orig))
mae_real = mean_absolute_error(y_test_true_orig, y_test_pred_orig)

# Practical accuracy: percent within ±10% of true price
threshold = 0.10
within_pct = (np.abs(y_test_true_orig - y_test_pred_orig) / (y_test_true_orig + 1e-9)) <= threshold
practical_accuracy = within_pct.mean()

# -----------------------
#✅  Show metrics + interpretations
# -----------------------
st.subheader("✅  Model Performance (Test set)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("R² (log-scale)", f"{r2_log_test:.3f}")
col2.metric("R² (original)", f"{r2_real:.3f}")
col3.metric("RMSE (₹)", f"₹{rmse_real:,.0f}")
col4.metric("MAE (₹)", f"₹{mae_real:,.0f}")

st.markdown(f"- **Practical accuracy (within ±{int(threshold*100)}%)**: **{practical_accuracy:.1%}** — proportion of test predictions within ±{int(threshold*100)}% of actual selling price.")
st.markdown("""
**Interpretation (plain + academic):**
- **R² (log-scale)**: how well the model explains variance in log(price); useful for multiplicative processes.  
- **R² (original)**: explanatory power on the real currency scale. Closer to 1 is better.  
- **RMSE / MAE**: typical monetary error sizes (RMSE penalizes large errors more). Lower = better.  
- **Practical accuracy**: user-centric reliability — proportion of predictions that would be 'close enough' for many consumer decisions.
""")

# Guidance based on metrics (simple rules)
notes = []
if r2_real >= 0.75:
    notes.append("Strong predictive performance — suitable for production use with standard caveats.")
elif r2_real >= 0.5:
    notes.append("Moderate performance — useful for guidance; consider more features for high-stakes pricing.")
else:
    notes.append("Low performance — model likely needs better features, more data, or different algorithms.")

if practical_accuracy >= 0.8:
    notes.append("High practical accuracy: most predictions are within ±10% of actual prices.")
elif practical_accuracy >= 0.5:
    notes.append("Moderate practical accuracy: validate predictions for high-value sales.")
else:
    notes.append("Low practical accuracy: use predictions only as rough guidance and inspect SHAP explanations.")

st.info("\n\n".join(notes))

# -----------------------
# ✅ Single prediction form
# -----------------------
st.markdown("---")
st.subheader("🧾 Make a Single Prediction")

input_data = {}
with st.form("predict_form"):
    for feat in features:
        if feat in num_features:
            input_data[feat] = st.number_input(f"{feat.replace('_', ' ').title()}", min_value=0.0, value=0.0)
        else:
            # ensure options drawn from dataset
            options = df[feat].dropna().unique().tolist()
            input_data[feat] = st.selectbox(f"{feat.replace('_', ' ').title()}", options)
    submit = st.form_submit_button("🔮 Predict Price")

if submit:
    try:
        input_df = pd.DataFrame([input_data])
        log_pred = pipeline.predict(input_df)
        pred_price = float(np.expm1(log_pred)[0])

        # Luxury adjustments (conservative)
        luxury_brands = {"bmw", "audi", "mercedes", "porsche", "jaguar", "land rover"}
        brand_val = str(input_data.get("brand", "")).strip().lower()
        if luxury_mode and brand_val in luxury_brands:
            pred_price *= 1.12  # 12% uplift for well-known luxury brands
            st.success("💎 Luxury brand adjustment applied (+12%)")
        elif luxury_mode:
            pred_price *= 1.03  # small market premium
            st.info("✅ Market premium applied (+3%)")

        st.markdown(f"### 💰 Predicted Selling Price: **₹{pred_price:,.0f}**")

        # user metrics for this single car
        age = input_data.get("vehicle_age", 0)
        km = input_data.get("km_driven", 0)
        est_depr_pct = age * 8.0  # 8% per year heuristic
        cost_per_km = pred_price / (km + 1)

        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated Depreciation (ann.)", f"{est_depr_pct:.1f}%")
        c2.metric("Estimated Cost per km", f"₹{cost_per_km:,.2f}")
        c3.metric("Value Category", "Premium" if pred_price > df['price_orig'].median() else "Value")

        # -----------------------
        # ✅ SHAP explainability (global + local)
        # -----------------------
        st.markdown("---")
        st.subheader("🔍 SHAP Explainability")

        preproc = pipeline.named_steps['preprocessor']
        model = pipeline.named_steps['model']

        # transformed training matrix (for SHAP global plots)
        X_train_tr = preproc.transform(X_train)

        # Build feature names aligned with transformed columns
        cat_names = preproc.named_transformers_['cat'].get_feature_names_out(cat_features)
        feature_names = list(num_features) + list(cat_names)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train_tr)

        # Global SHAP summary (save & display)
        plt.figure(figsize=(8, 4))
        shap.summary_plot(shap_values, X_train_tr, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig("shap_summary.png", bbox_inches="tight")
        plt.close()
        st.image("shap_summary.png", caption="Global SHAP Summary", use_column_width=True)

        # Local SHAP waterfall for this prediction
        single_tr = preproc.transform(input_df)
        single_shap = explainer.shap_values(single_tr)
        expl = shap.Explanation(values=single_shap[0], base_values=explainer.expected_value, data=single_tr[0])

        plt.figure(figsize=(8, 4))
        shap.waterfall_plot(expl, show=False)
        plt.tight_layout()
        plt.savefig("shap_local.png", bbox_inches="tight")
        plt.close()
        st.image("shap_local.png", caption="Local SHAP Waterfall (this prediction)", use_column_width=True)

        # -----------------------
        # ✅ PDF report generation (academic header + metrics)
        # -----------------------
        class LuxuryPDF(FPDF):
            def header(self):
                # Academic header with author & course
                if luxury_mode:
                    # gold-on-dark header
                    self.set_fill_color(20, 25, 30)
                    self.rect(0, 0, 210, 16, 'F')
                    self.set_text_color(255, 215, 0)
                    self.set_font("Helvetica", "B", 12)
                    self.cell(0, 10, "  🚗 Luxury Car Price Prediction — Academic Report", ln=True, align="L")
                    self.ln(4)
                    self.set_text_color(0, 0, 0)
                else:
                    self.set_font("Helvetica", "B", 12)
                    self.cell(0, 10, "Car Price Prediction — Academic Report", ln=True, align="C")
                    self.ln(4)

        pdf = LuxuryPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        # Academic metadata
        pdf.multi_cell(0, 7, "Author: Cathy Annabella Masentle Mahumane")
        pdf.multi_cell(0, 7, "Module: PDDS Major Project — Car Price Predictor")
        pdf.multi_cell(0, 7, f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
        pdf.ln(4)

        pdf.multi_cell(0, 7, f"Predicted Price: ₹{pred_price:,.0f}")
        pdf.multi_cell(0, 7, "Input Features:")
        for k, v in input_data.items():
            pdf.multi_cell(0, 7, f"- {k.replace('_', ' ').title()}: {v}")
        pdf.ln(3)
        pdf.multi_cell(0, 7, "Model Performance (test set, original scale):")
        pdf.multi_cell(0, 7, f"- R² (original): {r2_real:.3f}")
        pdf.multi_cell(0, 7, f"- RMSE: ₹{rmse_real:,.0f}")
        pdf.multi_cell(0, 7, f"- MAE: ₹{mae_real:,.0f}")
        pdf.multi_cell(0, 7, f"- Practical accuracy (±{int(threshold*100)}%): {practical_accuracy:.1%}")
        pdf.ln(4)
        pdf.multi_cell(0, 7, "Academic Observations:")
        pdf.multi_cell(0, 7, "- Log-transform stabilizes variance and reduces skewness for price prediction.")
        pdf.multi_cell(0, 7, "- SHAP explainability provides feature-level attribution necessary for transparent AI.")
        pdf.multi_cell(0, 7, "- For luxury brands, consider adding external market indicators and more samples.")
        # add SHAP images if present
        if Path("shap_summary.png").exists():
            pdf.image("shap_summary.png", x=10, w=180)
        pdf.add_page()
        if Path("shap_local.png").exists():
            pdf.image("shap_local.png", x=10, w=180)

        report_path = "luxury_car_report.pdf"
        pdf.output(report_path)

        # download & upload widgets
        with open(report_path, "rb") as f:
            st.download_button("📥 Download Academic PDF Report", f, file_name=report_path)

        st.markdown("**Archive or upload a report (optional)**")
        uploaded_pdf = st.file_uploader("Upload a PDF report to save with this app (optional)", type=["pdf"])
        if uploaded_pdf is not None:
            os.makedirs("uploads", exist_ok=True)
            savepath = Path("uploads") / uploaded_pdf.name
            with open(savepath, "wb") as out:
                out.write(uploaded_pdf.getbuffer())
            st.success(f"Saved uploaded PDF to {savepath}")

    except Exception as exc:
        st.error(f"Prediction error: {exc}")

# -----------------------
# Insights / conclusion
# -----------------------
st.markdown("---")
st.subheader("Conclusion & Next Steps")
st.markdown("""
- Log-transform improves model stability for skewed price distributions.  
- If negative R² or very large RMSE/MAE appear, review preprocessing, feature set (add brand-level features, region, trim), and consider boosting models (XGBoost/CatBoost) or ensembling.  
- For luxury vehicles, collect more labelled examples and include features related to rare options, provenance, and market demand.  
- Always examine local SHAP plots for per-case fairness and to detect biases.
""")



