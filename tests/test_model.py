# import mlflow
# import pytest
# import pandas as pd
# import json
# import joblib

# # Carregar configurações da aplicação
# with open('./config/app.json') as f:
#     config = json.load(f)

# @pytest.fixture
# def model():
#     """
#     Fixture para carregar o modelo do MLflow antes dos testes.
#     """
#     mlflow.set_tracking_uri(f"{config['tracking_base_url']}:{config['tracking_port']}")
#     model_uri = f"models:/{config['model_name']}@{config['model_version']}"
#     model = mlflow.sklearn.load_model(model_uri=model_uri)
#     return model

# @pytest.fixture
# def scaler():
#     """
#     Fixture para carregar o MinMaxScaler treinado antes dos testes.
#     """
#     scaler = joblib.load("scaler.pkl")
#     return scaler


# def test_model_prediction(model, scaler):
#     """
#     Testa se o modelo retorna previsões corretamente para um input válido.
#     """
#     # Criar um input de teste
#     input_data = {
#         "LIMIT_BAL": 20000,
#         "SEX": 2,
#         "EDUCATION": 2,
#         "MARRIAGE": 1,
#         "AGE": 35,
#         "PAY_0": 0,
#         "PAY_2": 0,
#         "PAY_3": 0,
#         "PAY_4": 0,
#         "PAY_5": 0,
#         "PAY_6": 0,
#         "BILL_AMT1": 5000,
#         "BILL_AMT2": 4000,
#         "BILL_AMT3": 3000,
#         "BILL_AMT4": 2000,
#         "BILL_AMT5": 1000,
#         "BILL_AMT6": 500,
#         "PAY_AMT1": 0,
#         "PAY_AMT2": 0,
#         "PAY_AMT3": 0,
#         "PAY_AMT4": 0,
#         "PAY_AMT5": 0,
#         "PAY_AMT6": 0
#     }
#     input_df = pd.DataFrame.from_dict(input_data)

#     # Normalizar os dados antes da previsão
#     input_scaled = scaler.transform(input_df)

#     # Fazer a previsão
#     prediction = model.predict(input_scaled)

#     # Verificar se a previsão retorna um único valor (1 ou 0)
#     assert isinstance(prediction[0], (int, float)), "A previsão não é um número válido"
#     assert prediction[0] in [0, 1], "A previsão deve ser 0 (pagamento) ou 1 (inadimplente)"


# def test_invalid_input(load_model, load_scaler):
#     """
#     Testa se o modelo lida corretamente com entradas inválidas.
#     """
#     model = load_model
#     scaler = load_scaler

#     # Criar um input inválido (valores negativos)
#     input_data = {
#         "LIMIT_BAL": -500,
#         "SEX": 2,
#         "EDUCATION": 2,
#         "MARRIAGE": 1,
#         "AGE": -10,  # Idade inválida
#         "PAY_0": 0,
#         "PAY_2": 0,
#         "PAY_3": 0,
#         "PAY_4": 0,
#         "PAY_5": 0,
#         "PAY_6": 0,
#         "BILL_AMT1": -5000,  # Valor negativo
#         "BILL_AMT2": 4000,
#         "BILL_AMT3": 3000,
#         "BILL_AMT4": 2000,
#         "BILL_AMT5": 1000,
#         "BILL_AMT6": 500,
#         "PAY_AMT1": 0,
#         "PAY_AMT2": 0,
#         "PAY_AMT3": 0,
#         "PAY_AMT4": 0,
#         "PAY_AMT5": 0,
#         "PAY_AMT6": 0
#     }
#     input_df = pd.DataFrame.from_dict(input_data)

#     # Normalizar os dados antes da previsão
#     input_scaled = scaler.transform(input_df)

#     # Fazer a previsão e garantir que não falha
#     try:
#         prediction = model.predict(input_scaled)
#         assert len(prediction) == 1, "O modelo não retornou uma previsão válida"
#     except Exception as e:
#         pytest.fail(f"O modelo falhou ao lidar com input inválido: {e}")

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

# @pytest.fixture(scope="module")
# def model():
#     """Load trained model from MLflow Model Registry (sklearn flavor)."""
#     #mlflow.set_tracking_uri(f"{config['tracking_base_url']}:{config['tracking_port']}")
#     model_uri = f"models:/{config['model_name']}@{config['model_version']}"
#     return mlflow.sklearn.load_model(model_uri)

@pytest.fixture(scope="module")
def model():
    with open('./config/app.json') as f:
        config = json.load(f)
    mlflow.set_tracking_uri(f"http://localhost:{config['tracking_port']}")
    model_name = config["model_name"]
    model_version = config["model_version"]
    return mlflow.sklearn.load_model(
        model_uri=f"models:/{model_name}@champion"#model_uri=f"models:/{model_name}@{model_version}"
    )


@pytest.fixture(scope="module")
def scaler():
    # import pdb
    # pdb.set_trace()
     # Load pre-trained scaler
    scaler_path = Path("rumos_bank/notebooks/mlflows/scaler.pkl")
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        return scaler
        print("Scaler loaded successfully!")
    else:
        print("Scaler file 'scaler.pkl' not found!")

    return scaler


def test_model_prediction_low_risk(model, scaler):
    """Expect 0 (no default) for a customer with good payment behavior."""
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
    proba = model.predict_proba(scaled)[0][1]
    prediction = int(proba >= 0.3)

    assert prediction == 0
    assert 0.0 <= proba <= 1.0


def test_model_prediction_high_risk(model, scaler):
    """Expect 1 (default) for a high-risk customer with overdue payments."""
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
    """Model should return a single value (shape = (1,))"""
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

    assert prediction.shape == (1,)


# def test_invalid_input_handling(model, scaler):
#     """Model should raise or handle gracefully invalid input (e.g. negative values)."""
#     input_data = {
#         "LIMIT_BAL": -500, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": ,
#         "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
#         "BILL_AMT1": -5000, "BILL_AMT2": 4000, "BILL_AMT3": 3000,
#         "BILL_AMT4": 2000, "BILL_AMT5": 1000, "BILL_AMT6": 500,
#         "PAY_AMT1": 0, "PAY_AMT2": 0, "PAY_AMT3": 0,
#         "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
#     }
#     df = pd.DataFrame([input_data])
#     with pytest.raises(Exception):
#         scaled = scaler.transform(df)
#         model.predict(scaled)

# def test_model_prediction_threshold_boundary():
#     input_data = {
#         # Dados artificiais que resultam em proba ~0.3
#         ...
#     }
#     ...
#     # Apenas garantir que o modelo retorna um valor válido
#     assert prediction in [0, 1]
