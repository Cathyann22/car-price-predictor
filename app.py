import streamlit as st
import pandas as pd
import numpy as np

st.title("🚗 Minimal Deployment Test")
st.write("If you're seeing this, your app deployed successfully!")

df = pd.DataFrame({
    "Brand": ["Toyota", "BMW", "Audi"],
    "Price": [500000, 1200000, 1500000]
})
st.dataframe(df)




