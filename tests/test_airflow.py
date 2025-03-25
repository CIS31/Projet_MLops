import pytest
import psycopg2
from airflow import DAG
from airflow.models import TaskInstance, DagRun, DagBag
from airflow.utils.state import State
from airflow.utils.types import DagRunType
from datetime import datetime
import pendulum
import os
from minio import Minio
from minio.error import S3Error
import boto3 


MINIO_HOST = "127.0.0.1:9000"  # Nom du service Minio dans Docker et le port
MINIO_ACCESS_KEY = 'minio'
MINIO_SECRET_KEY = 'miniopassword'
MINIO_BUCKET = 'bucket.romain'


# Configure la connexion à PostgreSQL pour les tests
DB_CONN = {
    "dbname": "airflow",
    "user": "airflow",
    "password": "airflow",
    "host": "0.0.0.0",
    "port": 5432,
}


@pytest.fixture
def postgres_connection():
    # Crée une connexion PostgreSQL pour les tests
    try:
        conn = psycopg2.connect(**DB_CONN)
        yield conn
    finally:
        conn.close()

# Configurer la connexion à Minio pour les tests
@pytest.fixture
def boto3_client():
    # Configurer la connexion à Minio avec boto3
    return boto3.client(
        's3',
        endpoint_url=f"http://{MINIO_HOST}",  # Utiliser HTTP car Minio est en local
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
    )


def test_postgres_connection(postgres_connection):
    # Vérifier la connexion à PostgreSQL
    try:
        with postgres_connection.cursor() as cursor:
            cursor.execute("SELECT 1;")  # Exécuter une requête simple
            result = cursor.fetchone()
            assert result is not None, "La requête n'a retourné aucun résultat."
            assert result[0] == 1, "La requête n'a pas retourné le résultat attendu."
            print("\nConnexion à PostgreSQL réussie et requête exécutée avec succès.")
    except Exception as e:
        pytest.fail(f"Échec de la connexion à PostgreSQL : {e}")


def test_boto3_minio_connection(boto3_client):
    # Vérifier la connexion à Minio avec boto3
    try:
        response = boto3_client.list_buckets()
        buckets = response.get('Buckets', [])
        print("Connexion à Minio réussie avec boto3. Buckets disponibles :")
        for bucket in buckets:
            print(f" - {bucket['Name']}")
        assert len(buckets) > 0, "Aucun bucket trouvé dans Minio."
    except Exception as e:
        pytest.fail(f"Échec de la connexion à Minio avec boto3 : {e}")


