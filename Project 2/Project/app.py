# run --> python -m streamlit run app.py "D:\Practice\Project 2\Project\app.py"  

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the trained Credit Risk Model Pipeline
@st.cache_resource
def load_model():
    return joblib.load('best_model_lightgbm.joblib') # Replace with your saved model filename if needed

model = load_model()

# 2. App Title and Description
st.title("Credit Risk Prediction App")
st.write("Provide applicant details below to evaluate the likelihood of loan default.")

# 3. User Input Fields
# Grouped into logical sections for simplicity
st.header("Applicant Demographics")
age = st.number_input("Age", min_value=18, max_value=100, value=35)
income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
employment_status = st.selectbox("Employment Status", ["Salaried", "Self-Employed", "Unemployed"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
residence_type = st.selectbox("Residence Type", ["Owned", "Rented", "Mortgage"])

st.header("Loan Details")
loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Home", "Auto", "Education"])
loan_tenure_months = st.number_input("Loan Tenure (Months)", min_value=1, max_value=360, value=36)
loan_to_income = st.number_input("Loan-to-Income Ratio", min_value=0.0, max_value=10.0, value=0.3, step=0.05)

st.header("Bureau & Credit History")
credit_utilization_ratio = st.number_input("Credit Utilization Ratio (%)", min_value=0, max_value=100, value=30)
delinquent_ratio = st.number_input("Delinquent Ratio (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
avg_dpd_per_delinquency = st.number_input("Avg DPD per Delinquency", min_value=0.0, value=0.0, step=1.0)
enquiry_count = st.number_input("Enquiry Count", min_value=0, max_value=50, value=2)
number_of_open_accounts = st.number_input("Number of Open Accounts", min_value=0, max_value=50, value=3)
number_of_closed_accounts = st.number_input("Number of Closed Accounts", min_value=0, max_value=50, value=1)

# 4. Prediction Button and Logic
if st.button("Evaluate Credit Risk"):
    # Create input dataframe matching the exact feature structure expected by the model
    input_data = pd.DataFrame({
        'age': [age],
        'income': [income],
        'employment_status': [employment_status],
        'marital_status': [marital_status],
        'residence_type': [residence_type],
        'loan_purpose': [loan_purpose],
        'loan_tenure_months': [loan_tenure_months],
        'loan_to_income': [loan_to_income],
        'credit_utilization_ratio': [credit_utilization_ratio],
        'delinquent_ratio': [delinquent_ratio],
        'avg_dpd_per_delinquency': [avg_dpd_per_delinquency],
        'enquiry_count': [enquiry_count],
        'number_of_open_accounts': [number_of_open_accounts],
        'number_of_closed_accounts': [number_of_closed_accounts]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] # Probability of default (class 1)
    
    # Display Results
    st.subheader("Evaluation Result")
    if prediction == 1:
        st.error(f"⚠️ High Risk: This applicant is **likely to default**.")
    else:
        st.success(f"✅ Low Risk: This applicant is **unlikely to default**.")
        
    st.info(f"Default Probability Score: **{probability * 100:.2f}%**")