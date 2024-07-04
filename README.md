# MLflow
This repository demonstrates an end-to-end machine learning workflow using an XGBoost model optimized by Optuna, tracked and managed with MLflow. The setup includes three key components:
1. **MLflow Tracking Server**: Used for logging experiments, parameters, and training results.
2. **PostgreSQL Database**: Serves as the backend for storing MLflow metadata.
3. **MinIO Object Storage**: Acts as a an artifact store.

Additionally, the repository outlines steps to deploy the optimized model on a Kubernetes cluster using KServe, enabling scalable, serverless model serving. 

## Prerequisites
Before you start, ensure you have the following installed:
- Python (tested on > 3.10)
- Docker
- Kubernetes cluster

## Getting Started
Clone the repo and start the services:
```
git clone https://github.com/mmtzlares/MLflow.git
cd Mlflow
pip3 install -r requirements.txt
docker-compose --env-file .env up -d --build
```
Navigate to http://localhost:9001, create a bucket named "mlflow", a ```MINIO_ACCESS_KEY``` ,  ```MINIO_SECRET_ACCESS_KEY```, and update the ```.env``` file with these values. Finally, restart the containers with theses new values. 
