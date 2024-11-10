import streamlit as st
import pandas as pd
import pickle
import os

# Define the model path
model_path = 'Loan_detection/model.pkl'  # Update this path as necessary

# Check if the model file exists
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please check the path and ensure the file exists.")
else:
    try:
        # Load the trained model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        st.error(f"An error occurred while loading the model: {str(e)}")


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
    if 'model' not in globals():
       st.error("Model is not loaded. Please check the loading process.")
    return None
    # Preprocess the input data
    data = preprocess_data(data)

    # Debugging output to check the structure of the input data
    st.write("Input data for prediction:", data)

    # Check shape and columns
    st.write("Data shape:", data.shape)
    st.write("Data columns:", data.columns.tolist())

    expected_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                        'Loan_Amount_Term', 'Credit_History', 'Property_Area']
    
    if list(data.columns) != expected_columns:
        st.error("Input data columns do not match the expected columns.")
        return None

    # Ensure there are no NaN values
    if data.isnull().any().any():
        st.error("Input data contains NaN values.")
        return None
    
    try:
        # Make the prediction
        prediction = model.predict(data)
        return prediction
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        return None

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
    
    if isinstance(prediction, (list, np.ndarray)):
        result = "Approved" if prediction[0] == 1 else "Rejected"
    else:
        result = "Approved" if prediction == 1 else "Rejected"
    st.write("Loan Status:", result)
