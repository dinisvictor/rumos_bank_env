import fastapi
from fastapi.middleware.cors import CORSMiddleware

import mlflow
from pydantic import BaseModel
import pandas as pd
import json
import uvicorn
import os

# Encontrar o caminho absoluto correto do ficheiro `config/app.json`
current_dir = os.path.dirname(os.path.abspath(__file__))  # Diretório onde `main.py` está
config_path = os.path.abspath(os.path.join(current_dir, "../../config/app.json"))

# Verificar se o ficheiro existe antes de carregar
if not os.path.exists(config_path):
    raise FileNotFoundError(f"O ficheiro de configuração não foi encontrado: {config_path}")

# Carregar a configuração corretamente
with open(config_path, "r") as f:
    config = json.load(f)

print(f"Configuração carregada de: {config_path}")


# Definir os inputs esperados na requisição
class Request(BaseModel):
    Age: int
    Income: float
    LoanAmount: float
    CreditScore: int
    LoanDuration: int


# Criar a aplicação FastAPI
app = fastapi.FastAPI()

# Permitir CORS para testes locais
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """ Configura o MLflow para carregar o modelo armazenado no Model Registry. """
    mlflow.set_tracking_uri(f"{config['tracking_base_url']}:{config['tracking_port']}")

    # Carregar o modelo registado no MLflow
    model_uri = f"models:/{config['model_name']}@{config['model_version']}"
    app.model = mlflow.pyfunc.load_model(model_uri=model_uri)
    
    print(f"✅ Modelo carregado: {model_uri}")


@app.post("/predict_default")
async def predict(input: Request):  
    """ Recebe dados e retorna a previsão do modelo. """
    input_df = pd.DataFrame.from_dict({k: [v] for k, v in input.model_dump().items()})
    prediction = app.model.predict(input_df)
    return {"default_prediction": prediction.tolist()[0]}


# Iniciar a aplicação na porta definida no config.json
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config["service_port"])

