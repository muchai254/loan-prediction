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

st.title('🤖 Machine Learning App')

st.info('This is app builds a machine learning model!')


with st.sidebar:
    st.header("Input Features")
    loan_term = st.number_input("Loan term", 1, 2800055670)
    income = st.number_input("Annual income", 1, 2800055670)
    loan_amount = st.number_input("Loan amount", 1, 2800055670)
    credit_score = st.number_input("Credit score", 1, 2800055670)
    currency = st.radio("Currency", ["Kenyan shilling", "Indian Rupee"])

currency_map = {
    "Kenyan shilling": "KES",
    "Indian Rupee": "INR"
}
currency_api_value = currency_map[currency]

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
        # Parse the response JSON
        result = response.json()
        
        # Display the prediction result
        st.write("### Prediction Result")
        st.write("Predicted value:", result.get("prediction"))
        
        # Display probability if returned
        if "prediction_proba" in result:
            st.write("Prediction Probability:", result["prediction_proba"])
        
        # Display SHAP waterfall plot if returned
        if "SHAP_plot" in result:
            # Decode the base64 image
            shap_img = Image.open(BytesIO(base64.b64decode(result["SHAP_plot"])))
            st.image(shap_img, caption="SHAP Waterfall Plot")

    else:
        st.write("Error:", response.status_code)
        st.write("Message:", response.text)

    # st.write("Prediction:", prediction[0])
    
    # explainer = shap.Explainer(gb_model, X_train) 
    # shap_values = explainer(processed_input)

    # for i, column in enumerate(processed_data.columns):
    #     shap_values[0].data[i] = processed_data.iloc[0, i]
    # shap.plots.waterfall(shap_values[0])
    # st.pyplot(plt.gcf()) 

    # # Display prediction
    # st.write("Prediction:", prediction[0])
    # st.write("Prediction Probability:", prediction_proba[0])


