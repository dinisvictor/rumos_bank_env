import mlflow
import pytest
import pandas as pd
import json

# Carregar configurações da aplicação
with open('./config/app.json') as f:
    config = json.load(f)

@pytest.fixture
def load_model():
    """
    Fixture para carregar o modelo do MLflow antes dos testes.
    """
    mlflow.set_tracking_uri(f"{config['tracking_base_url']}:{config['tracking_port']}")
    model_uri = f"models:/{config['model_name']}@{config['model_version']}"
    model = mlflow.pyfunc.load_model(model_uri=model_uri)
    return model


def test_model_prediction(load_model):
    """
    Testa se o modelo retorna previsões corretamente para um input válido.
    """
    model = load_model

    # Criar um input de teste
    input_data = {
        "Age": [35],
        "Income": [5000.0],
        "LoanAmount": [20000.0],
        "CreditScore": [750],
        "LoanDuration": [36]
    }
    input_df = pd.DataFrame.from_dict(input_data)

    # Fazer a previsão
    prediction = model.predict(input_df)

    # Verificar se a previsão retorna um único valor (1 ou 0)
    assert isinstance(prediction[0], (int, float)), "A previsão não é um número válido"
    assert prediction[0] in [0, 1], "A previsão deve ser 0 (pagamento) ou 1 (inadimplente)"


def test_invalid_input(load_model):
    """
    Testa se o modelo lida corretamente com entradas inválidas.
    """
    model = load_model

    # Criar um input inválido (valores negativos)
    input_data = {
        "Age": [-10],
        "Income": [-5000.0],
        "LoanAmount": [-20000.0],
        "CreditScore": [-750],
        "LoanDuration": [-36]
    }
    input_df = pd.DataFrame.from_dict(input_data)

    # Fazer a previsão e garantir que não falha
    try:
        prediction = model.predict(input_df)
        assert len(prediction) == 1, "O modelo não retornou uma previsão válida"
    except Exception as e:
        pytest.fail(f"O modelo falhou ao lidar com input inválido: {e}")
