FROM python:3.10-slim

RUN apt-get update && apt-get install -y curl
RUN pip install mlflow[extras] psycopg2 boto3

EXPOSE 5000