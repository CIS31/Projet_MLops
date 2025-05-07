import os
import pytest
import psycopg2
from minio import Minio
from dotenv import load_dotenv
from airflow.dags.fonctions import (
    list_github_files,
    create_bucket_if_not_exists,
    image_exists_in_minio,
    create_table_if_not_exists,
    insert_metadata_into_postgresql,
)

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


# === Fixtures ===

@pytest.fixture
def postgres_connection():
    """Fixture pour créer une connexion PostgreSQL pour les tests."""
    try:
        conn = psycopg2.connect(**DB_CONN)
        yield conn
    finally:
        conn.close()


@pytest.fixture
def minio_client():
    """Fixture pour configurer la connexion à MinIO avec le SDK MinIO."""
    client = Minio(
        endpoint=MINIO_HOST,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False  # Utiliser HTTP pour une connexion locale
    )
    return client


# === Tests ===

def test_postgres_connection(postgres_connection):
    """Test pour vérifier la connexion à PostgreSQL."""
    try:
        with postgres_connection.cursor() as cursor:
            cursor.execute("SELECT 1;")  # Exécuter une requête simple
            result = cursor.fetchone()
            assert result is not None, "La requête n'a retourné aucun résultat."
            assert result[0] == 1, "La requête n'a pas retourné le résultat attendu."
            print("\nConnexion à PostgreSQL réussie et requête exécutée avec succès.")
    except Exception as e:
        pytest.fail(f"Échec de la connexion à PostgreSQL : {e}")

def test_minio_connection(minio_client):
    """Test pour vérifier la connexion à MinIO avec le SDK MinIO."""
    try:
        buckets = minio_client.list_buckets()
        print("Connexion à MinIO réussie. Buckets disponibles :")
        for bucket in buckets:
            print(f" - {bucket.name}")
    except Exception as e:
        pytest.fail(f"Échec de la connexion à MinIO : {e}")

def test_list_github_files(mocker):
    """Test pour vérifier la récupération des fichiers depuis GitHub."""
    # Mocker la réponse de l'API GitHub
    mock_response = mocker.patch("requests.get")
    mock_response.return_value.status_code = 200
    mock_response.return_value.json.return_value = [
        {"name": "image1.jpg"},
        {"name": "image2.png"},
        {"name": "not_an_image.txt"}
    ]

    # Appeler la fonction
    files = list_github_files("dandelion")

    # Vérifier les résultats
    assert files == ["image1.jpg", "image2.png"], "Les fichiers retournés ne sont pas corrects."

def test_create_bucket_if_not_exists(mocker):
    """Test pour vérifier la création d'un bucket dans Minio."""
    # Mocker le client Minio
    mock_client = mocker.patch("airflow.dags.fonctions.client")
    mock_client.bucket_exists.return_value = False

    # Appeler la fonction
    create_bucket_if_not_exists("test-bucket")

    # Vérifier que le bucket a été créé
    mock_client.make_bucket.assert_called_once_with("test-bucket")

def test_image_exists_in_minio(mocker):
    """Test pour vérifier si une image existe dans Minio."""
    # Mocker le client Minio
    mock_client = mocker.patch("airflow.dags.fonctions.client")
    mock_client.stat_object.return_value = True

    # Appeler la fonction
    exists = image_exists_in_minio("folder", "image.jpg")

    # Vérifier le résultat
    assert exists is True, "L'image devrait exister dans Minio."

def test_create_table_if_not_exists(mocker):
    """Test pour vérifier la création d'une table dans PostgreSQL."""
    # Mocker la connexion PostgreSQL
    mock_connect = mocker.patch("psycopg2.connect")
    mock_cursor = mock_connect.return_value.cursor.return_value

    # Appeler la fonction
    create_table_if_not_exists("test_table")

    # Vérifier que la requête SQL a été exécutée
    expected_query = """
        CREATE TABLE IF NOT EXISTS test_table (
            id SERIAL PRIMARY KEY,
            url_source VARCHAR(255) NOT NULL,
            url_s3 VARCHAR(255) NOT NULL,
            label VARCHAR(100) NOT NULL
        );
    """.strip()  # Supprimer les espaces et les sauts de ligne inutiles

    # Vérifier que la requête SQL a été exécutée
    try:
        # Comparer les requêtes en supprimant les espaces et les sauts de ligne
        actual_query = mock_cursor.execute.call_args[0][0].strip()
        assert actual_query == expected_query, f"Requête SQL incorrecte : {actual_query}"
        print("La requête SQL pour créer la table est correcte.")
    except AssertionError as e:
        pytest.fail(f"Erreur dans la requête SQL : {e}")

    # Vérifier que la connexion a été commitée
    mock_connect.return_value.commit.assert_called_once()
    print("La transaction a été commitée avec succès.")

def test_insert_metadata_into_postgresql(mocker):
    """Test pour vérifier l'insertion des métadonnées dans PostgreSQL."""
    # Mocker la connexion PostgreSQL
    mock_connect = mocker.patch("psycopg2.connect")
    mock_cursor = mock_connect.return_value.cursor.return_value

    # Appeler la fonction
    insert_metadata_into_postgresql(
        url_source="http://source.com/image.jpg",
        url_s3="http://s3.com/image.jpg",
        label="dandelion"
    )

    # Vérifier que la requête SQL a été exécutée
    expected_query = """
        INSERT INTO plants_data (url_source, url_s3, label)
        VALUES (%s, %s, %s)
    """.strip()

    try:
        # Comparer les requêtes en supprimant les espaces et les sauts de ligne
        actual_query = mock_cursor.execute.call_args[0][0].strip()
        assert actual_query == expected_query, f"Requête SQL incorrecte : {actual_query}"
        print("La requête SQL pour insérer les métadonnées est correcte.")
    except AssertionError as e:
        pytest.fail(f"Erreur dans la requête SQL : {e}")

    # Vérifier que les paramètres passés sont corrects
    expected_params = ("http://source.com/image.jpg", "http://s3.com/image.jpg", "dandelion")
    actual_params = mock_cursor.execute.call_args[0][1]
    assert actual_params == expected_params, f"Paramètres incorrects : {actual_params}"

    # Vérifier que la transaction a été commitée
    mock_connect.return_value.commit.assert_called_once()
    print("La transaction a été commitée avec succès.")

