import os
import pytest
import psycopg2
import boto3
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv(dotenv_path=".env")

# Récupérer les variables d'environnement
MINIO_HOST = os.getenv("MINIO_HOST")
MINIO_ACCESS_KEY = os.getenv("MINIO_USER")
MINIO_SECRET_KEY = os.getenv("MINIO_PASS")
MINIO_BUCKET = os.getenv("MINIO_BUCKET")

# Configuration PostgreSQL
DB_CONN = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT")),
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
    except Exception as e:
        pytest.fail(f"Échec de la connexion à Minio avec boto3 : {e}")


