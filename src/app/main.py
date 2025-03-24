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

# Load application configuration
with open('./config/app.json') as f:
    config = json.load(f)

# Define input schema
class RequestModel(BaseModel):
    LIMIT_BAL: confloat(ge=0) = 20000.0
    SEX: conint(ge=1, le=2) = 2
    EDUCATION: conint(ge=1, le=4) = 2
    MARRIAGE: conint(ge=0, le=3) = 1
    AGE: conint(ge=18) = 24
    PAY_0: int = 0
    PAY_2: int = 0
    PAY_3: int = 0
    PAY_4: int = 0
    PAY_5: int = 0
    PAY_6: int = 0
    BILL_AMT1: confloat(ge=0) = 0.0
    BILL_AMT2: confloat(ge=0) = 0.0
    BILL_AMT3: confloat(ge=0) = 0.0
    BILL_AMT4: confloat(ge=0) = 0.0
    BILL_AMT5: confloat(ge=0) = 0.0
    BILL_AMT6: confloat(ge=0) = 0.0
    PAY_AMT1: confloat(ge=0) = 0.0
    PAY_AMT2: confloat(ge=0) = 0.0
    PAY_AMT3: confloat(ge=0) = 0.0
    PAY_AMT4: confloat(ge=0) = 0.0
    PAY_AMT5: confloat(ge=0) = 0.0
    PAY_AMT6: confloat(ge=0) = 0.0

# Create FastAPI app
app = FastAPI()

# Enable CORS (for local testing or front-end integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Load ML model and pre-fitted scaler on app startup."""
    mlflow.set_tracking_uri(f"{config['tracking_base_url']}:{config['tracking_port']}")

    # Load model from MLflow Model Registry
    model_uri = f"models:/{config['model_name']}@{config['model_version']}"
    app.model = mlflow.sklearn.load_model(model_uri=model_uri)
    print(f"Model loaded: {model_uri}")

    # Load pre-trained scaler
    scaler_path = Path("scaler.pkl")
    if scaler_path.exists():
        app.scaler = joblib.load(scaler_path)
        print("Scaler loaded successfully!")
    else:
        print("Scaler file 'scaler.pkl' not found!")

@app.post("/predict_default", response_description="Credit Default Prediction")
async def predict(input: RequestModel):
    """
    Predict credit default based on user input.
    Uses a decision threshold of 0.3 as defined during training.
    """
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

    # Apply threshold of 0.3
    prediction = int(proba >= 0.3)

    return {
        "default_prediction": prediction,
        "probability": round(proba, 4)
    }

# Run the app if this file is executed directly
if __name__ == "__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=config["service_port"])
