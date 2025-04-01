# Import of the necessary libraries
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, conint, confloat

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib
import json
import uvicorn

from pathlib import Path

# Here I load the app configuration (like the model name, version, and ports) from a JSON file.
# This keeps sensitive or changeable settings separate from the code itself.
with open('./config/app.json') as f:
    config = json.load(f)

# This class defines the expected structure and validation for the input data sent to the API.
# I’m using Pydantic to ensure values fall within realistic bounds based on the training data.
class RequestModel(BaseModel):
    LIMIT_BAL: confloat(ge=0, le=1000000.0) = 20000.0
    SEX: conint(ge=1, le=2) = 2
    EDUCATION: conint(ge=0, le=6) = 2  
    MARRIAGE: conint(ge=0, le=3) = 2
    AGE: conint(ge=21, le=79) = 24
    PAY_0: conint(ge=-2, le=8) = 0
    PAY_2: conint(ge=-2, le=8) = 0
    PAY_3: conint(ge=-2, le=8) = 0
    PAY_4: conint(ge=-2, le=8) = 0
    PAY_5: conint(ge=-2, le=8) = 0
    PAY_6: conint(ge=-2, le=8) = 0
    BILL_AMT1: confloat(ge=-165580.0, le=964511.0) = 0.0
    BILL_AMT2: confloat(ge=-69777.0, le=983931.0 ) = 0.0
    BILL_AMT3: confloat(ge=-157264.0, le=1664089.0) = 0.0
    BILL_AMT4: confloat(ge=-170000.0, le=891586.0) = 0.0
    BILL_AMT5: confloat(ge=-81334.0, le= 927171.0) = 0.0
    BILL_AMT6: confloat(ge=-209051.0, le= 961664.0) = 0.0
    PAY_AMT1: confloat(ge=0, le= 873552.0) = 0.0
    PAY_AMT2: confloat(ge=0, le= 1684259.0) = 0.0
    PAY_AMT3: confloat(ge=0, le= 896040.0) = 0.0
    PAY_AMT4: confloat(ge=0, le=621000.0) = 0.0
    PAY_AMT5: confloat(ge=0, le=426529.0) = 0.0
    PAY_AMT6: confloat(ge=0, le= 527143.0) = 0.0

# Here I initialize the FastAPI application
app = FastAPI()

# I enable CORS to allow requests from any origin.
# This is especially useful during local development or if a frontend (e.g., React or Streamlit) is calling the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# This function runs automatically when the app starts.
# I use it to load the model and scaler into memory so predictions are ready immediately.
@app.on_event("startup")
async def startup_event():
    """Load ML model and pre-fitted scaler on app startup."""

    mlflow.set_tracking_uri(f"{config['tracking_base_url']}:{config['tracking_port']}")

    # Load model from MLflow Model Registry
    model_uri = f"models:/{config['model_name']}@{config['model_version']}"
    print(model_uri)
    app.model = mlflow.sklearn.load_model(model_uri=model_uri)
    print(f"Model loaded: {model_uri}")

    # # Load the scaler used during model training to ensure consistent preprocessing
    scaler_path = Path("scaler.pkl")
    if scaler_path.exists():
        app.scaler = joblib.load(scaler_path)
        print("Scaler loaded successfully!")
    else:
        print("Scaler file 'scaler.pkl' not found!")

# This is the main prediction endpoint of the API.
# When a POST request is sent to /predict_default with the required input, it returns the prediction.
@app.post("/predict_default", response_description="Credit Default Prediction")
async def predict(input: RequestModel):
    """
    Predict credit default based on user input.
    Uses a decision threshold of 0.3 as defined during training.
    """
    # Define the order of features to ensure consistency with how the model was trained
    feature_order = [
        "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]

    # Create DataFrame with correct feature order and type
    input_df = pd.DataFrame([input.model_dump()])[feature_order].astype('float64')
    print("\nPrepared input data types:\n", input_df.dtypes)

    # Scale input using pre-fitted scaler
    input_scaled = app.scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=feature_order)

    print("\nScaled input data:")
    print(input_scaled_df.to_string())

    # Predict probability of default (class 1)
    proba = app.model.predict_proba(input_scaled_df)[0][1]

    # Classify the customer as default (1) if the probability is greater than or equal to 0.3
    prediction = int(proba >= 0.3)

    return {
        "default_prediction": prediction,
        "probability": round(proba, 4)
    }

# Run the app if this file is executed directly
if __name__ == "__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=config["service_port"])
