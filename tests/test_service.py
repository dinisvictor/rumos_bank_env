import json
import pytest
import requests

# Carregar as configurações do serviço
with open('./config/app.json') as f:
    config = json.load(f)

def test_default_prediction():
    """
    Teste para o endpoint /predict_default com dados válidos.
    Deve retornar uma previsão (0 = paga, 1 = inadimplente).
    """
    response = requests.post(f"http://localhost:{config['service_port']}/predict_default", json={
        "Age": 35,
        "Income": 5000.0,
        "LoanAmount": 20000.0,
        "CreditScore": 750,
        "LoanDuration": 36
    })

    assert response.status_code == 200
    assert "default_prediction" in response.json()
    assert isinstance(response.json()["default_prediction"], (int, float))
    assert response.json()["default_prediction"] in [0, 1]


def test_invalid_request():
    """
    Teste para o endpoint /predict_default com dados inválidos.
    O servidor deve lidar com a entrada errada corretamente.
    """
    response = requests.post(f"http://localhost:{config['service_port']}/predict_default", json={
        "Age": -5,  # Idade inválida
        "Income": "invalid",  # Tipo errado
        "LoanAmount": None,  # Falta um valor
        "CreditScore": 300,  # Valor válido
        "LoanDuration": 36
    })

    assert response.status_code == 422  # O FastAPI deve rejeitar a requisição inválida
