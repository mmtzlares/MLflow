# MLflow
Local dev set up using Postgres and MinIO as backend store solutions. Start the services with: 
```
$ docker-compose --env-file .env up -d --build
```
and create access keys for MinIO. Then, update the ```.env``` file, create a bucket named "mlflow" and restart the containers. To access the tracking server from your scripts, set the URI to: 
```python
import mlflow

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5001")
