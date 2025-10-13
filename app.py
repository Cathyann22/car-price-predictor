# ============================================================
# 🚗 Car Price Prediction App — Streamlit + Diagnostics
# ============================================================

# 📦 Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import tempfile
from weasyprint import HTML
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ============================================================
# 🔮 Predict Car Price from Input
# ============================================================

def predict_price(pipeline, input_df):
    """
    Predicts the car price using the trained pipeline.
    
    Parameters:
        pipeline (Pipeline): Trained ML pipeline with preprocessing and model
        input_df (DataFrame): Single-row DataFrame with user input features
    
    Returns:
        float: Predicted price in rupees
    """
    predicted_price = pipeline.predict(input_df)[0]
    return predicted_price

# ============================================================
# 📊 Evaluate Model on Hold-Out Test Set
# ============================================================

def evaluate_model_on_test_set(pipeline, X_test, y_test):
    """
    Evaluates model performance on hold-out test data.
    
    Parameters:
        pipeline (Pipeline): Trained ML pipeline
        X_test (DataFrame): Feature matrix for test set
        y_test (array-like): True target values (log-transformed)
    
    Returns:
        tuple: (y_pred, rmse, mae)
            y_pred (array): Predicted prices (after inverse log transform)
            rmse (float): Root Mean Squared Error
            mae (float): Mean Absolute Error
    """
    expected_columns = [
        'Unnamed: 0', 'car_name', 'brand', 'model', 'vehicle_age', 'km_driven',
        'seller_type', 'fuel_type', 'transmission_type', 'mileage', 'engine',
        'max_power', 'seats'
    ]
    X_test_aligned = X_test.copy()
    for col in expected_columns:
        if col not in X_test_aligned.columns:
            X_test_aligned[col] = np.nan
    X_test_aligned = X_test_aligned[expected_columns]

    categorical_cols = ['brand', 'model', 'seller_type', 'fuel_type', 'transmission_type']
    X_test_aligned[categorical_cols] = X_test_aligned[categorical_cols].astype(str)

    y_pred_log = pipeline.predict(X_test_aligned)
    y_pred = np.expm1(y_pred_log)

    y_true = np.ravel(y_test)
    y_pred = np.ravel(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    return y_pred, rmse, mae

# ============================================================
# ⚙️ Configure Streamlit page
# ============================================================
st.set_page_config(page_title="Car Price Predictor", layout="wide")
st.title("🚗 Car Price Prediction App")
st.markdown("Use the sidebar to enter car details and estimate its market price.")

# ============================================================
# 🔧 Sidebar: Input widgets for user data
# ============================================================
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

# ============================================================
# 🧾 Prepare input DataFrame for prediction
# ============================================================
input_df = pd.DataFrame([{
    "Brand": brand,
    "Year": year,
    "Mileage": mileage,
    "Fuel_Type": fuel_type,
    "Transmission": transmission
}])

# ============================================================
# 🗂️ Create tabs for Prediction, Export, and Diagnostics
# ============================================================
tab1, tab2, tab3 = st.tabs(["📈 Prediction", "📄 Export Report", "🧠 Diagnostics"])

# ============================================================
# 📈 Tab 1: Prediction and SHAP diagnostics
# ============================================================
with tab1:
    st.subheader("📈 Predict Car Price")
    if st.button("Predict Price"):
        predicted_price = predict_price(pipeline, input_df)
        st.success(f"Estimated Price: ₹{int(predicted_price):,}")

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

# ============================================================
# 📄 Tab 2: PDF Export of Prediction
# ============================================================
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

# ============================================================
# 🧠 Tab 3: Model Diagnostics on Hold-Out Test Set
# ============================================================
try:
    X_test = pd.read_csv("X_test.csv")
    y_test = pd.read_csv("y_test.csv").values.ravel()

    with tab3:
        st.header("Model Diagnostics")
        y_pred, rmse, mae = evaluate_model_on_test_set(pipeline, X_test, y_test)

        st.subheader("Sample Predictions")
        for i in range(min(5, len(y_pred))):
            st.write(f"Sample {i+1}: ₹{y_pred[i]:,.2f}")

        st.subheader("Evaluation Metrics")
        st.metric("RMSE", f"₹{rmse:,.0f}")
        st.metric("MAE", f"₹{mae:,.0f}")

        st.markdown("""
        **Interpretation Guide**  
        - ✅ Points near the red line: accurate predictions  
        - ⚠️ Points far from the line: over/underestimation  
        - 📉 RMSE penalizes large errors  
        - 📌 MAE shows average error magnitude  
        """)

        st.image("shap_summary_plot.png", caption="Feature Importance (SHAP)", use_column_width=True)

except Exception as e:
    with tab3:
        st.warning("Diagnostics unavailable — test data not found or improperly formatted.")
        st.text(str(e))

# ============================================================
# 🚀 Deployment Confirmation Block
# ============================================================
st.title("🚀 Deployment Test")
st.write("If you're seeing this, your app deployed successfully!")

st.dataframe(pd.DataFrame({
    "Brand": ["Toyota", "BMW", "Audi"],
    "Price": [500000, 1200000, 1500000]
}))

       

       
