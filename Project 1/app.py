# Run in Terminal -->  python -m streamlit run app.py "D:\Class Laptop\Studys\app.py"          

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved pipeline
try:
    model_pipeline = joblib.load('property_price_model.joblib')
except FileNotFoundError:
    st.error("Model file not found. Run 'train_model.py' first.")
    st.stop()

st.title("Property Price Predictor")
st.write("Enter the property details below to estimate its price.")

# User Inputs
area_sqft = st.number_input("Area (Square Feet)", min_value=100, max_value=200000, value=1500, step=100)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=20, value=3, step=1)
neighborhood = st.selectbox("Neighborhood Type", options=['Rural', 'Suburb', 'Urban'])
built_date = st.date_input("Built Date")
property_tax = st.number_input("Property Tax ($)", min_value=0.0, value=2500.0, step=100.0)

# Prediction execution
if st.button("Predict Price"):
    input_data = pd.DataFrame({
        'Area_sqft': [area_sqft],
        'Bedrooms': [bedrooms],
        'Neighborhood': [neighborhood],
        'Built_Date': [pd.to_datetime(built_date)],
        'Property_Tax': [property_tax]
    })
    
    # Manual preprocessing required before hitting the pipeline
    input_data['Built_Year'] = input_data['Built_Date'].dt.year
    input_data['Built_Month'] = input_data['Built_Date'].dt.month
    input_data = input_data.drop('Built_Date', axis=1)
    
    if bedrooms > 0:
        input_data['Area_per_Bedroom'] = input_data['Area_sqft'] / input_data['Bedrooms']
    else:
        input_data['Area_per_Bedroom'] = input_data['Area_sqft']
        
    prediction = model_pipeline.predict(input_data)
    st.success(f"Estimated Property Price: ${prediction[0]:,.2f}")