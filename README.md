# Rumos Bank

This project was developed to support Rumos Bank in predicting the probability of customers defaulting on their credit obligations. The solution involves a full machine learning pipeline – from training and experiment tracking to deployment and testing – using modern MLOps tools and practices.

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

- Data exploration and preprocessing  
- Feature engineering and scaling  
- Training and comparison of models (e.g., Logistic Regression, Random Forest)  
- Logging experiments with MLflow  
- Registering the best model in MLflow under the tag:  
  ```bash
  models:/random_forest@champion

# Running Locally (Optional)

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


# CI/CD Pipeline (GitHub Actions)

## Description

The pipeline is defined in **.github/workflows/main.yml.** It executes the following steps:

1.	Checks out the repository
2.	Builds and runs all Docker services using Docker Compose
3.	Waits for the API to become available
4.	Runs Pytest on both the model and the API
5.	Pushes Docker images to GitHub Container Registry (GHCR)
6.	Shuts down all running containers

## Secrets Configuration

To allow the pipeline to push to GHCR, you need to:

A. Create a Personal Access Token (PAT)

1. Go to GitHub Settings > Developer Settings > Tokens

2. Create a token with the following permissions:

  - write:packages
  - read:packages
  - delete:packages (optional)

3. Add the token as a secret

In the repository:

4. Go to Settings > Secrets and variables > Actions

5. Click New repository secret

   - Name: GHCR_PAT
   - Value: paste the PAT you created

This token will be used in the pipeline step:

```bash
  name: Log in to GitHub Container Registry
  uses: docker/login-action@v2
  with:
    registry: ghcr.io
    username: ${{ github.actor }}
    password: ${{ secrets.GHCR_PAT }}
```
# Testing

Unit and integration tests are written using Pytest.

A. Model Tests (in **tests/test_model.py**)

- Validates predictions for low-risk and high-risk profiles
- Ensures output shape and valid probability values

B. Service Tests (in **tests/test_service.py**)

- Tests POST requests to the /predict_default endpoin
- Checks for correct response structure and status codes

You can run tests locally with:

```bash
pytest tests/
```

Note: All tests are also executed in the CI/CD pipeline.

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

# Final Considerations

This project delivers a complete MLOps pipeline for credit default prediction, fully aligned with the course requirements:

All steps — from data analysis to deployment — are documented, automated, and reproducible using industry-standard tools such as MLflow, FastAPI, and Docker.

The service is thoroughly tested with unit and functional tests, covering both the machine learning model and the inference API, ensuring reliability and correctness.

The best-performing model is tracked and versioned in the MLflow Model Registry and is served using the champion tag for easy access and consistent deployment.

The entire system is containerised using Docker and orchestrated via Docker Compose, allowing it to be launched with a single command in any environment.

A CI/CD pipeline built with GitHub Actions automates the entire lifecycle: building images, running tests, deploying the service, and publishing the artifacts.

Docker images are successfully built and pushed to the GitHub Container Registry (GHCR):

```bash
    ghcr.io/dinisvictor/rumos_bank_service
    ghcr.io/dinisvictor/rumos_bank_ui
    ghcr.io/dinisvictor/mlruns 
```

All components are public: both the GitHub repository and the associated container packages, ensuring that the entire project can be evaluated and reproduced without requiring any authentication or access credentials.

This project demonstrates a robust, end-to-end MLOps solution and adheres to best practices in data science, software development, and cloud-native deployment.
