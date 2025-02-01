from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional, Literal
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import joblib
import os

base_dir = os.path.dirname(os.path.abspath(__file__)) 

model_path = os.path.join(base_dir, "models", "gb_model.joblib")
data_path = os.path.join(base_dir, "data", "X_train.csv")

model = joblib.load(model_path)  
X_train = pd.read_csv(data_path)  

app = FastAPI()

class LoanRequest(BaseModel):
    loan_term: int
    annual_income: int
    loan_amount: int
    credit_score: int
    currency: Literal['KES', 'INR'] 
    probability: Optional[bool] = Field(False, description="Return prediction probability")
    SHAP: Optional[bool] = Field(False, description="Return SHAP waterfall plot")

def log_transform(*dfs):
    transformed_dfs = []
    for df in dfs:
        df_copy = df.copy()
        numeric_columns = df_copy.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            if (df_copy[column] > 0).all():
                df_copy[column] = np.log(df_copy[column])
        transformed_dfs.append(df_copy)  
    return transformed_dfs

@app.post("/loan")
async def loan_prediction(request: LoanRequest):
    custom_input_data = pd.DataFrame([{
        'loan_term': request.loan_term,
        'annual_income': request.annual_income,
        'loan_amount': request.loan_amount,
        'credit_score': request.credit_score
    }])

    if request.currency == "KES":
        custom_input_data[['annual_income', 'loan_amount']] /= 0.65

    # Preprocess the DataFrame
    processed_data = custom_input_data.copy()
    processed_data['income_loan_ratio'] = custom_input_data['annual_income'] / custom_input_data['loan_amount']
    processed_data['credit_score_income_ratio'] = custom_input_data['credit_score'] / custom_input_data['annual_income']
    processed_data.drop(columns=['loan_amount', 'annual_income'], inplace=True)
    processed_input = log_transform(processed_data)[0]

    prediction = model.predict(processed_input)[0]
    
    response = {"prediction": int(prediction)}

    if request.probability:
        prediction_probability = model.predict_proba(processed_input)
        response["prediction_probability"] = prediction_probability[0].tolist()

    if request.SHAP:
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(processed_input)
        
        # Adjust SHAP values for the original data values
        for i, column in enumerate(processed_data.columns):
            shap_values[0].data[i] = processed_data.iloc[0, i]
        
        # Plot SHAP waterfall and convert to an image
        plt.figure()
        shap.plots.waterfall(shap_values[0], show=False)
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        response["SHAP_plot"] = img_base64

        plt.close()
    return response