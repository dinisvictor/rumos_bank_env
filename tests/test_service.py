import json
import pytest
import requests

# Load service configuration
with open('./config/app.json') as f:
    config = json.load(f)

BASE_URL = f"http://localhost:{config['service_port']}/predict_default"

def test_default_prediction_expects_0():
    """
    Test for the /predict_default endpoint with valid data.
    Should return a prediction (0 = no default, 1 = default).
    """
    response = requests.post(BASE_URL, json={
        "LIMIT_BAL": 300000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 45,
        "PAY_0": 0,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
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
    
    assert response.status_code == 200, f"Error: Status code {response.status_code}"
    
    json_response = response.json()
    assert json_response["default_prediction"]==0, "Error: Invalid prediction (should be 0)"

def test_default_prediction_expects_1():
    """
    Test for the /predict_default endpoint with valid data.
    Should return a prediction (0 = no default, 1 = default).
    """
    response = requests.post(BASE_URL, json={
        "LIMIT_BAL": 10000,
        "SEX": 1,
        "EDUCATION": 2,
        "MARRIAGE": 2,
        "AGE": 21,
        "PAY_0": 8,
        "PAY_2": 8,
        "PAY_3": 8,
        "PAY_4": 8,
        "PAY_5": 8,
        "PAY_6": 8,
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
    import pdb 
    pdb.set_trace ()
    assert response.status_code == 200, f"Erro: Status code {response.status_code}"
    
    json_response = response.json()
    assert json_response["default_prediction"]==1, "Error: Invalid prediction (should be 1)"


def test_invalid_request():
    """
    Test for the /predict_default endpoint with invalid data.
    The server should handle bad input correctly and return a 422 error.
    """
    response = requests.post(BASE_URL, json={
        "LIMIT_BAL": -500,  # Invalid value
        "SEX": "male",  # Wrong type (string instead of number)
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 20,
        # Missing a required field (e.g. PAY_0)
        "BILL_AMT1": "string instead of number",
        "BILL_AMT2": None,
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
    
    assert response.status_code == 422, f"Error: Status code {response.status_code}" # Incorrect parameter