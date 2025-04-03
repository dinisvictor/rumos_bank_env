import mlflow
import pytest
import mlflow.sklearn
import pandas as pd
import joblib
import json
from pathlib import Path

# Load configuration
with open('./config/app.json') as f:
    config = json.load(f)

@pytest.fixture(scope="module")
def model():
    """
    Load the trained model from MLflow Model Registry.
    Uses the 'champion' version alias (could also use a specific version).
    """
    with open('./config/app.json') as f:
        config = json.load(f)
    # Set MLflow tracking server URI (e.g., localhost:5001)
    mlflow.set_tracking_uri(f"http://mlflow-tracking:{config['tracking_port']}")
    model_name = config["model_name"]
    model_version = config["model_version"]
    
    # Load the model using the alias 'champion'
    return mlflow.sklearn.load_model(
        model_uri=f"models:/{model_name}@{model_version}"
    )


@pytest.fixture(scope="module")
def scaler():
    """
    Load the scaler object used during training for consistent preprocessing.
    """
    scaler_path = Path("rumos_bank/notebooks/mlflows/scaler.pkl")
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        return scaler
        print("Scaler loaded successfully!")
    else:
        print("Scaler file 'scaler.pkl' not found!")

    return scaler


def test_model_prediction_low_risk(model, scaler):
    """
    Functional test:
    Predicts a customer with good credit behavior.
    Expects a prediction of 0 (no default).
    """
    input_data = {
        "LIMIT_BAL": 300000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 45,
        "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 2000, "BILL_AMT2": 1800, "BILL_AMT3": 1600,
        "BILL_AMT4": 1500, "BILL_AMT5": 1200, "BILL_AMT6": 1000,
        "PAY_AMT1": 1000, "PAY_AMT2": 1000, "PAY_AMT3": 1000,
        "PAY_AMT4": 1000, "PAY_AMT5": 1000, "PAY_AMT6": 1000
    }
    df = pd.DataFrame([input_data])
    scaled = scaler.transform(df)
    proba = model.predict_proba(scaled)[0][1] # Probability of class 1 (default)
    prediction = int(proba >= 0.3)

    assert prediction == 0
    assert 0.0 <= proba <= 1.0 # Valid probability range


def test_model_prediction_high_risk(model, scaler):
    """
    Functional test:
    Predicts a high-risk customer with poor payment history.
    Expects a prediction of 1 (default).
    """
    input_data = {
        "LIMIT_BAL": 10000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 21,
        "PAY_0": 8, "PAY_2": 8, "PAY_3": 8, "PAY_4": 8, "PAY_5": 8, "PAY_6": 8,
        "BILL_AMT1": 900000, "BILL_AMT2": 900000, "BILL_AMT3": 900000,
        "BILL_AMT4": 900000, "BILL_AMT5": 900000, "BILL_AMT6": 900000,
        "PAY_AMT1": 0, "PAY_AMT2": 0, "PAY_AMT3": 0,
        "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
    }
    df = pd.DataFrame([input_data])
    scaled = scaler.transform(df)
    proba = model.predict_proba(scaled)[0][1]
    prediction = int(proba >= 0.3)

    assert prediction == 1
    assert 0.0 <= proba <= 1.0


def test_model_output_shape(model, scaler):
    """
    Structural test:
    Ensures the model output shape is correct (predict returns array of shape (1,)).
    """
    input_data = {
        "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
        "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 5000, "BILL_AMT2": 4000, "BILL_AMT3": 3000,
        "BILL_AMT4": 2000, "BILL_AMT5": 1000, "BILL_AMT6": 500,
        "PAY_AMT1": 0, "PAY_AMT2": 0, "PAY_AMT3": 0,
        "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
    }
    df = pd.DataFrame([input_data])
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)

    assert prediction.shape == (1,) # Must return a single prediction

