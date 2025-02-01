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

api_url = "http://fastapi_server:8000"

st.title('🤖 Loan Predictor')

st.markdown("This is a loan Predictor App that predicts whether a loan application will be approved or not. It does so by calculating how the following features affect the loan application approval:")
st.markdown("1. Loan term in years")
st.markdown("2. Annual income")
st.markdown("3. Loan amount")
st.markdown("4. Credit score")
st.markdown("These are entered on the side panel to the left and predictions are made once the **predict** button is pressed. ")
st.markdown("This project offers the following advantages:")
st.markdown(" -  Clear and Explainable Predictions by use of a SHAP waterfall diagram.")
st.markdown(" -  Custom API Endpoint for Easy Integration.")
st.markdown(" -  Seamless CI/CD Pipeline for Ongoing Improvement")
st.markdown(" > You can find the full project documentation and resources used [here](https://medium.com/@muchaibriank/end-to-end-machine-learning-project-loan-approval-part-2-development-0e437b084ad5)")
st.markdown(" > The API documentation can be found here [here](https://loanpredictor.gitbook.io/api-docs)")
payload = {
"loan_term": 0,
"annual_income": 0,
"loan_amount": 0,
"credit_score": 0,
"currency": "KES",
"probability": False,
"SHAP": False
}

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

    predict_button = st.button("Predict")

    response = requests.post(f"{api_url}/loan", json=payload)

result_placeholder = st.empty()

with result_placeholder.container():
    st.markdown(
        """
        <div style="border: 2px solid #ddd; border-radius: 10px; padding: 20px; text-align: center; background-color: #0E1117;">
            <h4 style="color: #888;">Predictions are displayed here. Enter features on the side panel to the left.</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

if predict_button:
    with result_placeholder.container():
        with st.spinner("Making prediction ..."):
            payload = {
                "loan_term": loan_term,
                "annual_income": income,
                "loan_amount": loan_amount,
                "credit_score": credit_score,
                "currency": currency_api_value,
                "probability": True,
                "SHAP": True,
            }
            response = requests.post(f"{api_url}/loan", json=payload)

    with result_placeholder.container():
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

st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            text-align: center;
            padding: 10px 0;
            background-color: #0E1117;
            font-size: 16px;
            color: #F9F9F9;
        }
        .footer a {
            color: #007BFF;
            text-decoration: none;
        }
        .footer a:hover {
            text-decoration: underline;
        }
    </style>
    <div class="footer">
        made with ❤️ by <a href="https://www.linkedin.com/in/brian-muchai-7380231a7" target="_blank"> Muchai</a>
    </div>
    """,
    unsafe_allow_html=True,
)
