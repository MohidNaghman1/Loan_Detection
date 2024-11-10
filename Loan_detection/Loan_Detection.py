import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np

# Load the trained model
model_path = '/mount/src/loan_detection/Loan_detection/model.pkl'
model = None

# Check if the model file exists
if os.path.exists(model_path):
    st.success("Model file found.")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"An error occurred while loading the model: {str(e)}")
else:
    st.error(f"Model file not found at {model_path}. Please check the path.")

if model is not None:
    st.success("Model loaded successfully.")
    st.write(f"Model type: {type(model)}")

def preprocess_data(data):
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    data['Married'] = data['Married'].map({'Yes': 1, 'No': 0})
    data['Dependents'] = data['Dependents'].map({'0': 0, '1': 1, '2': 2, '3+': 3})
    data['Education'] = data['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    data['Self_Employed'] = data['Self_Employed'].map({'Yes': 1, 'No': 0})
    data['Property_Area'] = data['Property_Area'].map({'Urban': 1, 'Semiurban': 2, 'Rural': 3})

    return data

def predict_loan_status(data):
    if model is None:
        st.error("Model is not loaded. Please check the loading process.")
        return None

    # Preprocess the input data
    data = preprocess_data(data)

    # Ensure data is in the correct format
    st.write("Input data for prediction:", data)

    try:
        # Make prediction
        prediction = model.predict(data)
        st.write("Prediction output:", prediction)  # Debugging output

        # Check if prediction is an array-like structure
        if isinstance(prediction, (list, np.ndarray)):
            if len(prediction) > 0:
                result = "Approved" if prediction[0] == 1 else "Rejected"
            else:
                st.error("Prediction output is empty.")
                result = "Unknown"
        else:
            result = "Approved" if prediction == 1 else "Rejected"

        return result

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        return None

# Streamlit UI
st.title("Loan Status Predictor")
st.write("Please enter the required details to predict the loan status")

# Collect user input
with st.form("loan_form"):
    # Input fields for user data
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    
    with col2:
        applicant_income = st.number_input("Applicant Income", min_value=0, step=100)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0, step=100)
        loan_amount = st.number_input("Loan Amount", min_value=10, max_value=1000000, step=10)
        loan_amount_term = st.number_input("Loan Amount Term", min_value=0, step=1)
        credit_history = st.selectbox("Credit History", ["0", "1"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    # Example: gender = st.selectbox("Gender", ["Male", "Female"])
    submitted = st.form_submit_button("Predict")

if submitted:
    # Create a DataFrame from user input
    data = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area]
    })
    
    prediction_result = predict_loan_status(data)
    
    if prediction_result is not None:
        st.write("Loan Status:", prediction_result)
