import json
import pytest
import requests

# Load service configuration
with open('./config/app.json') as f:
    config = json.load(f)

# Define the base URL of the API based on the configured port
BASE_URL = f"http://localhost:{config['service_port']}/predict_default"

def test_default_prediction_expects_0():
    """
    Positive test case:
    Sends a valid request with expected non-default features.
    Should return a prediction of 0 (no default).
    """
    response = requests.post(BASE_URL, json={
        "LIMIT_BAL": 300000, # High credit limit
        "SEX": 2, # Female
        "EDUCATION": 2, # University
        "MARRIAGE": 1, # Married
        "AGE": 45,
        "PAY_0": 0,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0, # No late payments
        "BILL_AMT1": 5000,
        "BILL_AMT2": 4000,
        "BILL_AMT3": 3000,
        "BILL_AMT4": 2000,
        "BILL_AMT5": 1000,
        "BILL_AMT6": 500,
        "PAY_AMT1": 5000,
        "PAY_AMT2": 5000,
        "PAY_AMT3": 5000,
        "PAY_AMT4": 5000,
        "PAY_AMT5": 5000,
        "PAY_AMT6": 5000
    })
    
     # Ensure a successful HTTP response
    assert response.status_code == 200, f"Error: Status code {response.status_code}"
    
    # Verify that prediction result is 0 (no default)
    json_response = response.json()
    assert json_response["default_prediction"]==0, "Error: Invalid prediction (should be 0)"

def test_default_prediction_expects_1():
    """
    Positive test case:
    Sends a valid request with expected default features.
    Should return a prediction of 1 (default).
    """
    response = requests.post(BASE_URL, json={
        "LIMIT_BAL": 10000, # Low credit limit
        "SEX": 1, # Male
        "EDUCATION": 2,
        "MARRIAGE": 2, # Single
        "AGE": 21,
        "PAY_0": 8,
        "PAY_2": 8,
        "PAY_3": 8,
        "PAY_4": 8,
        "PAY_5": 8,
        "PAY_6": 8, # Severe delay
        "BILL_AMT1": 900000,
        "BILL_AMT2": 900000,
        "BILL_AMT3": 900000,
        "BILL_AMT4": 800000,
        "BILL_AMT5": 900000,
        "BILL_AMT6": 900000,
        "PAY_AMT1": 0,
        "PAY_AMT2": 0,
        "PAY_AMT3": 0,
        "PAY_AMT4": 0,
        "PAY_AMT5": 0,
        "PAY_AMT6": 0
    })
    
    #import pdb 
    #pdb.set_trace ()

    assert response.status_code == 200, f"Erro: Status code {response.status_code}"
    
    # Verify that prediction result is 1 (default)
    json_response = response.json()
    assert json_response["default_prediction"]==1, "Error: Invalid prediction (should be 1)"


def test_invalid_request():
    """
    Negative test case:
    Sends a request with invalid types and missing values.
    API should return 422 for Unprocessable Entity due to validation errors.
    """
    response = requests.post(BASE_URL, json={
        "LIMIT_BAL": -500,  # Invalid value
        "SEX": "male",  # Wrong type (string instead of number)
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 20,
        # Missing a required field (e.g. PAY_0)
        "BILL_AMT1": "string instead of number", # Invalid type
        "BILL_AMT2": None, # Null value
        "BILL_AMT3": 3000,
        "BILL_AMT4": 2000,
        "BILL_AMT5": 1000,
        "BILL_AMT6": 500,
        "PAY_AMT1": 0,
        "PAY_AMT2": 0,
        "PAY_AMT3": 0,
        "PAY_AMT4": 0,
        "PAY_AMT5": 0,
        "PAY_AMT6": 0
    })
    
    # Validate error response due to invalid input
    assert response.status_code == 422, f"Error: Status code {response.status_code}" # Incorrect parameter