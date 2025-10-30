
# ================================================================
# ✅ FULL STREAMLIT PREDICTION + SHAP + 🚗 CAR PRICE PREDICTOR APP
# ================================================================


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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from fpdf import FPDF
import shap
import matplotlib.pyplot as plt

# -----------------------
# Page config & styles
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

st.title("🚗 Luxury Car Price Predictor — Interactive Academic Edition")

# -----------------------
# Tabs for interactivity
# -----------------------
tabs = st.tabs(["📊 Data Overview", "🧾 Single Prediction", "🔍 SHAP Explainability", "📚 Academic Insights"])

# -----------------------
# Load dataset
# -----------------------
try:
    df = pd.read_csv("car_price_dataset.csv")
    dataset_loaded = True
except FileNotFoundError:
    df = None
    dataset_loaded = False

with tabs[0]:
    st.subheader("📊 Dataset Overview")
    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        dataset_loaded = True

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

# -----------------------
# Clean & prepare
# -----------------------
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
features = ['brand', 'fuel_type', 'transmission_type', 'vehicle_age',
            'km_driven', 'mileage', 'engine', 'max_power', 'seats']
target = 'selling_price'
missing = [c for c in features + [target] if c not in df.columns]
if missing:
    st.error(f"❌ Missing required columns: {missing}")
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

# -----------------------
# Luxury mode
# -----------------------
luxury_mode = st.checkbox("💎 Enable Luxury Mode (High-End Vehicle Focus)", value=False)
model = RandomForestRegressor(
    n_estimators=200 if luxury_mode else 100,
    max_depth=20 if luxury_mode else None,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# -----------------------
# Evaluate
# -----------------------
y_test_pred_log = pipeline.predict(X_test)
y_test_true_orig = np.expm1(y_test)
y_test_pred_orig = np.expm1(y_test_pred_log)
r2_real = r2_score(y_test_true_orig, y_test_pred_orig)
rmse_real = np.sqrt(mean_squared_error(y_test_true_orig, y_test_pred_orig))
mae_real = mean_absolute_error(y_test_true_orig, y_test_pred_orig)
practical_accuracy = ((np.abs(y_test_true_orig - y_test_pred_orig) / (y_test_true_orig + 1e-9)) <= 0.10).mean()

with tabs[1]:
    st.subheader("🧾 Single Prediction Form")
    input_data = {}
    with st.form("predict_form"):
        for feat in features:
            if feat in num_features:
                if feat == "seats":
                    input_data[feat] = st.number_input(f"{feat.title()}", min_value=1, max_value=20, value=5, step=1, format="%d")
                else:
                    input_data[feat] = st.number_input(f"{feat.title()}", min_value=0.0, value=0.0, step=0.1)
            else:
                input_data[feat] = st.selectbox(f"{feat.title()}", df[feat].dropna().unique().tolist())
        submit = st.form_submit_button("🔮 Predict Price")

    if submit:
        input_df = pd.DataFrame([input_data])
        log_pred = pipeline.predict(input_df)
        pred_price = float(np.expm1(log_pred)[0])
        # Luxury adjustment
        luxury_brands = {"bmw","audi","mercedes","porsche","jaguar","land rover"}
        brand_val = str(input_data.get("brand","")).lower()
        if luxury_mode and brand_val in luxury_brands:
            pred_price *= 1.12
            st.success("💎 Luxury brand adjustment applied (+12%)")
        elif luxury_mode:
            pred_price *= 1.03
            st.info("✅ Market premium applied (+3%)")

        st.markdown(f"### 💰 Predicted Price: **₹{pred_price:,.0f}**")

        # Depreciation & cost per km
        age = input_data.get("vehicle_age",0)
        km = input_data.get("km_driven",0)
        est_depr_pct = age*8.0
        cost_per_km = pred_price/(km+1)
        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated Depreciation (ann.)", f"{est_depr_pct:.1f}%")
        c2.metric("Estimated Cost per km", f"₹{cost_per_km:,.2f}")
        c3.metric("Value Category", "Premium" if pred_price>df['price_orig'].median() else "Value")

with tabs[2]:
    st.subheader("🔍 SHAP Explainability")
    preproc = pipeline.named_steps['preprocessor']
    model = pipeline.named_steps['model']
    X_train_tr = preproc.transform(X_train)
    cat_names = preproc.named_transformers_['cat'].get_feature_names_out(cat_features)
    feature_names = list(num_features)+list(cat_names)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train_tr)

    st.write("### Global Feature Importance")
    plt.figure(figsize=(8,4))
    shap.summary_plot(shap_values, X_train_tr, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", bbox_inches="tight")
    plt.close()
    st.image("shap_summary.png", use_column_width=True)

    st.write("### Local SHAP (Example Prediction)")
    single_tr = preproc.transform(input_df)
    single_shap = explainer.shap_values(single_tr)
    expl = shap.Explanation(values=single_shap[0], base_values=explainer.expected_value, data=single_tr[0])
    plt.figure(figsize=(8,4))
    shap.waterfall_plot(expl, show=False)
    plt.tight_layout()
    plt.savefig("shap_local.png", bbox_inches="tight")
    plt.close()
    st.image("shap_local.png", use_column_width=True)

with tabs[3]:
    st.subheader("📚 Academic Insights")
    with st.expander("Why Log-Transform the Target?"):
        st.markdown("""
        - Price distributions are often skewed; log-transform reduces skewness.
        - Stabilizes variance and improves regression performance.
        - Multiplicative relationships (e.g., age depreciation) become linear.
        """)

    with st.expander("Understanding Metrics"):
        st.markdown(f"""
        - **R² (original scale)**: {r2_real:.3f} — proportion of variance explained.
        - **RMSE**: ₹{rmse_real:,.0f} — penalizes large errors more than MAE.
        - **MAE**: ₹{mae_real:,.0f} — average absolute prediction error.
        - **Practical Accuracy (±10%)**: {practical_accuracy:.1%} — likelihood of predictions being “close enough” for real-world decisions.
        """)

    with st.expander("SHAP Explainability"):
        st.markdown("""
        - Provides per-feature contribution to predictions.
        - **Global summary plots** highlight most important drivers across the dataset.
        - **Local waterfall plots** explain individual predictions.
        - Ensures transparency, fairness, and interpretability in AI-driven price prediction.
        """)

    with st.expander("Luxury Mode Adjustments"):
        st.markdown("""
        - Luxury brands (BMW, Audi, Mercedes, Porsche, Jaguar, Land Rover) receive a market premium.
        - Enables better prediction alignment with real-world high-end car valuations.
        - Encourages feature expansion for rare options, trims, and market demand indicators.
        """)

    with st.expander("Next Steps for Academic Research"):
        st.markdown("""
        - Collect more labeled data for rare luxury models.
        - Consider ensemble models or gradient boosting for improved R².
        - Continuously monitor SHAP explanations to detect biases.
        - Validate models on new market data before production deployment.
        """)



