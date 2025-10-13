# 🚗 Car Price Predictor

This Streamlit app predicts the selling price of a car using a trained machine learning pipeline.  
It supports SHAP-based interpretability, luxury mode diagnostics, and PDF export for stakeholder-friendly reporting.

---

## Features

- ✅ Predict car prices from user input  
- 💎 SHAP waterfall plots for luxury vehicles  
- 🖨️ PDF export of predictions and diagnostics  
- 📊 Diagnostics tab with RMSE and MAE metrics  
- 🧠 Modular pipeline with preprocessing and model  

---

## Setup Instructions

To run the app locally:

```bash
pip install -r requirements.txt
streamlit run app.py

Unnamed: 0,car_name,brand,model,vehicle_age,km_driven,seller_type,fuel_type,transmission_type,mileage,engine,max_power,seats
0,Toyota Corolla,Toyota,Corolla,5,60000,Individual,Petrol,Manual,18.0,1200,85,5


