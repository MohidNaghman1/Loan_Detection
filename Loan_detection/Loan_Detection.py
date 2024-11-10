import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('Loan_detection/model.pkl', 'rb') as f:
    model = pickle.load(f)




# Create a function to preprocess the input data
def preprocess_data(data):
    # Map categorical variables to numerical values
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    data['Married'] = data['Married'].map({'Yes': 1, 'No': 0})
    data['Dependents'] = data['Dependents'].map({'0': 0, '1': 1, '2': 2, '3+': 3})
    data['Education'] = data['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    data['Self_Employed'] = data['Self_Employed'].map({'Yes': 1, 'No': 0})
    data['Property_Area'] = data['Property_Area'].map({'Urban': 1, 'Semiurban': 2, 'Rural': 3})

    return data
    


# Create a function to make predictions
def predict_loan_status(data):
    data = preprocess_data(data)
    prediction = model.predict(data)
    return prediction

# Create a Streamlit app
st.title("Loan Status Predictor")
st.write("Please enter the required details to predict the loan status")

# Create input fields for the user to enter the required details
with st.form("loan_form"):
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
        loan_amount = st.number_input("Loan Amount", min_value=10000, max_value=1000000, step=1000)
        loan_amount_term = st.number_input("Loan Amount Term (months)", min_value=0, step=1)
        credit_history = st.selectbox("Credit History", ["0", "1"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    
    submitted = st.form_submit_button("Predict")

# Make predictions using the input data
if submitted:
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
    
    prediction = predict_loan_status(data)
    
    st.write("Loan Status:", "Approved" if prediction[0] == 1 else "Rejected")
    st.write("Prediction Confidence:", prediction)
