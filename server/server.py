from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, constr
from typing import Optional, Literal
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.ensemble import GradientBoostingClassifier  # Import your model library
import joblib

# Load the pre-trained gradient boosting model and training data for SHAP explanations
model = joblib.load("./models/gb_model.joblib")  # Path to your model file
X_train = pd.read_csv("./data/X_train.csv")  # Path to your training data

app = FastAPI()

# Define the input schema with Pydantic
class LoanRequest(BaseModel):
    loan_term: int
    annual_income: int
    loan_amount: int
    credit_score: int
    currency: Literal['KES', 'INR']  # Only allows 'KES' or 'INR'
    probability: Optional[bool] = Field(False, description="Return prediction probability")
    SHAP: Optional[bool] = Field(False, description="Return SHAP waterfall plot")

# Define the log transform function
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
    # Create the input DataFrame
    custom_input_data = pd.DataFrame([{
        'loan_term': request.loan_term,
        'annual_income': request.annual_income,
        'loan_amount': request.loan_amount,
        'credit_score': request.credit_score
    }])

    # Adjust the DataFrame values based on the currency
    if request.currency == "KES":
        custom_input_data[['annual_income', 'loan_amount']] /= 0.65

    # Preprocess the DataFrame
    processed_data = custom_input_data.copy()
    processed_data['income_loan_ratio'] = custom_input_data['annual_income'] / custom_input_data['loan_amount']
    processed_data['credit_score_income_ratio'] = custom_input_data['credit_score'] / custom_input_data['annual_income']
    processed_data.drop(columns=['loan_amount', 'annual_income'], inplace=True)
    processed_input = log_transform(processed_data)[0]

    # Make the prediction
    prediction = model.predict(processed_input)[0]
    
    # Prepare the response
    response = {"prediction": int(prediction)}

    # Optionally add prediction probability
    if request.probability:
        prediction_proba = model.predict_proba(processed_input)
        response["prediction_probability"] = prediction_proba[0].tolist()

    # Optionally add SHAP explanation
    if request.SHAP:
        # Initialize SHAP explainer and calculate SHAP values
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
