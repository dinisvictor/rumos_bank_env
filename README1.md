# Descrição
O Rumos Bank tem enfrentado desafios devido à inadimplência dos clientes. Este projeto visa prever quais clientes têm maior probabilidade de não cumprir os prazos de pagamento, permitindo ao banco tomar decisões mais informadas.

# Estrutura do Projecto

OML-trabalho-master/
│── rumos_bank/
│   ├── data/                  # Dados utilizados no projeto
│   │   ├── lending_data.csv    # Dataset principal
│   ├── notebooks/              # Notebooks para análise e modelagem
│   │   ├── rumos_bank_lending_prediction.ipynb
│── .gitignore                  # Arquivos ignorados pelo Git
│── conda.yaml                   # Definição do ambiente Conda
│── docker-compose.yaml          # Configuração de containers (a definir)
│── Dockerfile                   # Dockerfile para o serviço (a definir)
│── README.md                    # Este arquivo

# Configuração do ambiente

1. Criar o ambiente Conda
Execute os seguintes comandos no terminal:
conda env create -f conda.yaml
conda activate rumos_bank_env

2. Instalar dependências adicionais (se necessário)
Caso precise instalar novas bibliotecas depois de ativar o ambiente:
conda install -n rumos_bank_env <package-name>
pip install <package-name>

3. Executar os Notebooks
Com o ambiente ativado, abra o Jupyter Notebook:
jupyter lab


# Treinar e Versionamento do Modelo
Em breve: integração com MLflow para versionamento de modelos.

# Servir o modelo
Em breve: API utilizando FastAPI.

# Containerização e Deploy
Em breve: configuração do Docker e CI/CD.

# Autores
👤 Dinis Guerreiro
📅 Projeto entregue até 01/05/2025