import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import shap
import requests
import base64
from PIL import Image
from io import BytesIO

api_url = "http://127.0.0.1:8000/loan"

st.title('🤖 Loan Prediction App')


with st.sidebar:
    st.header("Input Features")
    loan_term = st.number_input("Loan term in years", 1, 10)
    income = st.number_input("Annual income", 200000, 10000000, 200000)
    loan_amount = st.number_input("Loan amount", 300000, 20000000, 300000)
    credit_score = st.number_input("Credit score", 300, 900, 400)
    st.caption("The credit score should be between 300 and 900, where higher scores indicate better creditworthiness.")
    currency = st.radio("Currency", ["Kenyan shilling", "Indian Rupee"])
    

currency_map = {
    "Kenyan shilling": "KES",
    "Indian Rupee": "INR"
}
currency_api_value = currency_map[currency]

with st.expander('Original Data Description'):
    data = {
        "Column name": ["loan_id", "no_of_dependents", "education", "self_employed", "income_unnum",
                    "loan_amount", "loan_term", "cibil_score", "residential_assets_value", "commercial_asset_value"],
        "Description": ["Unique customer identifier", "Number of Dependents of the Applicant", "Education of the Applicant", "Employment Status of the Applicant", "Annual Income of the Applicant",
                    "Loan Amount", "Loan Term in Years", "Credit Score", "Row 9 - Col 2", "Row 10 - Col 2"],
    }
    df = pd.DataFrame(data)
    url = "https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset"
    st.link_button("Dataset link", url)
    st.table(data)


if st.button("Predict"):
    payload = {
    "loan_term": loan_term,
    "annual_income": income,
    "loan_amount": loan_amount,
    "credit_score": credit_score,
    "currency": currency_api_value,
    "probability": True,
    "SHAP": True
}
    response = requests.post(api_url, json=payload)

    if response.status_code == 200:
        result = response.json()
        st.write("### Prediction Result")

        if result.get("prediction") == 1:
            st.success("The loan application is likely to be approved.")
        else:   
            st.error("The loan application is likely to be rejected.")
        
        if "SHAP_plot" in result:
            # Decode the base64 image
            shap_img = Image.open(BytesIO(base64.b64decode(result["SHAP_plot"])))
            st.image(shap_img, caption="SHAP Waterfall Plot")

    else:
        st.write("Error:", response.status_code)
        st.write("Message:", response.text)

