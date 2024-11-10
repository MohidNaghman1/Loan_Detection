# Loan Status Predictor

## Overview

The Loan Status Predictor is a web application built using Streamlit that allows users to predict the approval status of a loan application based on various applicant details. This application leverages a pre-trained machine learning model to provide instant feedback on loan eligibility, helping users understand their chances of approval.

## Features

- **User-Friendly Interface**: A simple and intuitive form for inputting applicant details.
- **Data Encoding**: Categorical variables are converted into numerical format using a mapping strategy to ensure compatibility with the machine learning model.
- **Instant Predictions**: Users receive immediate feedback on loan approval status after submitting their information.
- **Responsive Design**: The application is designed to work seamlessly on both desktop and mobile devices.

## Input Fields

Users can provide the following information:

- **Gender**: Male or Female
- **Marital Status**: Yes (Married) or No (Unmarried)
- **Dependents**: Number of dependents (0, 1, 2, or 3+)
- **Education**: Graduate or Not Graduate
- **Self Employed**: Yes or No
- **Applicant Income**: Monthly income of the applicant
- **Coapplicant Income**: Monthly income of the coapplicant (if applicable)
- **Loan Amount**: Desired loan amount
- **Loan Amount Term**: Duration of the loan in months
- **Credit History**: History of credit (0 or 1)
- **Property Area**: Urban, Semiurban, or Rural



## Model Training

The underlying machine learning model can be trained using historical loan data. Ensure that categorical variables are properly encoded and that the model is saved in a compatible format (e.g., using `pickle`).

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

