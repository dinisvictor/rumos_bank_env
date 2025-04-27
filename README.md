# Rumos Bank

This project was developed as part of my evaluation for the **Machine Learning Operationalization module**. My goal was to design and implement a full MLOps pipeline to support **Rumos Bank** in predicting the likelihood of customers defaulting on their credit obligations.
I built this solution from end to end — from data analysis and model experimentation to production deployment and testing — using tools and best practices in MLOps.

---

## Project Summary

- Data analysis and model training in **Jupyter Notebook**
- Experiment tracking and model registry with **MLflow**
- The best model is served via an **API using FastAPI**
- The project includes unit and functional tests with **Pytest**
- Fully containerised with **Docker** and orchestrated using **Docker Compose**
- CI/CD pipeline implemented with **GitHub Actions**
- Docker images are published to **GitHub Container Registry (GHCR)**

---

# Data Analysis and Model Training

The notebook `notebooks/rumos_bank_lending_prediction.ipynb` includes:

- Data exploration, visualization, and cleaning
- Feature engineering and scaling  
- Training and comparison of models:
  - Logistic Regression
  - KNN Classifier
  - Decision Tree
  - Random Forest
  - SVC
  - MLP Classifier
- Tracking and comparing experiments using **MLflow**  
- Registering the top-performing model (`random_forest`) in MLflow under the alias:  
  ```bash
  models:/random_forest@champion

  ---

# Testing

I implemented two levels of testing using Pytest:

A. Model Tests (in **tests/test_model.py**)

- Validates predictions for both low-risk and high-risk profiles
- Ensures output shape and valid probability values

B. Service Tests (in **tests/test_service.py**)

- Sends test requests to /predict_default endpoint
- Checks for status codes, valid structure, and input validation

Tests can be run locally using:

```bash
pytest tests/
```

All tests also run automatically in the GitHub Actions pipeline. 

---

# Running the Project Locally

## 1. Create the Conda environment
```bash 
conda env create -f conda.yaml
conda activate rumos_bank_env
```

## 2. Run Docker Compose

```bash
docker compose up --build
```

## 3. Access the services

- **API documentation:** http://127.0.0.1:5002/docs  
- **MLflow Tracking UI:** http://127.0.0.1:5001  
- **HTML form UI:** http://127.0.0.1:5003  

---

# CI/CD Pipeline (GitHub Actions)

## Description

The pipeline is defined in **.github/workflows/main.yml.** It executes the following steps:

1.	Checks out the repository
2.	Builds and runs all Docker services using Docker Compose
3.	Waits for the API to become available
4.	Runs Pytest on both the model and the API
5.	Pushes Docker images to GitHub Container Registry (GHCR)
6.	Shuts down all running containers

## Secrets Setup

To enable image pushes to GHCR:

A. Create a Personal Access Token (PAT)

1. Go to GitHub Settings > Developer Settings > Tokens

2. Create a token with the following permissions:

  - write:packages
  - read:packages
  - delete:packages

3. Add the token as a secret

B. In the repository:

4. Go to Settings > Secrets and variables > Actions

5. Click New repository secret

   - Name: GHCR_PAT
   - Value: paste the PAT created

This token will be used in the pipeline step:

```bash
  name: Log in to GitHub Container Registry
  uses: docker/login-action@v2
  with:
    registry: ghcr.io
    username: ${{ github.actor }}
    password: ${{ secrets.GHCR_PAT }}
```

---

# Docker Services Summary

```bash
services:
  mlflow-tracking:
    image: ghcr.io/mlflow/mlflow
    command: mlflow ui --port 5001 --host 0.0.0.0 --backend-store-uri ./mlruns --artifacts-destination ./mlruns
    ports: ["5001:5001"]

  rumos-bank-lending-service:
    build:
      context: .
      dockerfile: Dockerfile.Service
    image: ghcr.io/dinisvictor/rumos_bank_service
    ports: ["5002:5002"]

  rumos-bank-ui:
    build:
      context: .
      dockerfile: Dockerfile.UI
    image: ghcr.io/dinisvictor/rumos_bank_ui
    ports: ["5003:5003"]

volumes:
  mlruns_data:
```

---
# Packages

Please go to the packages section in GitHub to see all the images.

# Final Considerations

As a data analyst in training, this project was an great opportunity for me to go beyond my knowledge and try to build probably my first pipeline (long journey).
This project delivers a complete MLOps pipeline for credit default prediction, aligned with the requirements:

- Track experiments reproducibly with MLflow

- Deploy models with a real-world API using FastAPI

- Test ML applications for both functionality and reliability

- Use Docker for production-ready packaging.

- Automate everything through GitHub Actions

Docker images are successfully built and pushed to the GitHub Container Registry (GHCR):

```bash
    ghcr.io/dinisvictor/rumos_bank_service
    ghcr.io/dinisvictor/rumos_bank_ui
    ghcr.io/dinisvictor/mlruns 
```

All components are public: both the GitHub repository and the containeres packages.

This project helped me understand the full lifecycle of a machine learning solution, from exploration to deployment — all following industry-standard tools and workflows.
