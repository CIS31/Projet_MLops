import os
import requests
import psycopg2
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
from io import BytesIO
from minio import Minio
from minio.error import S3Error
from functools import partial

# Configuration
GITHUB_REPO = 'btphan95/greenr-airflow'  # Le dépôt GitHub
GITHUB_BRANCH = 'master'  # La branche où se trouvent les fichiers
GITHUB_PATH_DANDELION = 'data/dandelion'  # Le chemin du répertoire contenant les images dandelion
GITHUB_PATH_GRASS = 'data/grass'  # Le chemin du répertoire contenant les images grass

# Variables Minio (récupérées depuis l'environnement Docker)
MINIO_HOST = "minio:9000"  # Nom du service Minio dans Docker et le port
MINIO_ACCESS_KEY = 'minio'
MINIO_SECRET_KEY = 'miniopassword'
MINIO_BUCKET = 'bucket.romain'  # Le bucket Minio où les images seront stockées (composé uniquement de : lettres minuscules, de chiffres, de tirets et de points)

# Variables PostgreSQL
PG_HOST = 'postgres'  # Nom du service défini dans docker-compose.yml et non pas localhost
PG_PORT = '5432'  # Port PostgreSQL
PG_DATABASE = 'airflow'  # Nom de la base de données
PG_USER = 'airflow'  # Utilisateur PostgreSQL
PG_PASSWORD = 'airflow'  # Mot de passe PostgreSQL

# Initialisation de Minio
client = Minio(
    MINIO_HOST,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False  # False car nous utilisons HTTP dans Docker (si HTTPS, mets à True)
)

# Fonction pour récupérer les fichiers d'un répertoire sur GitHub
def list_github_files(path):
    url = f'https://api.github.com/repos/{GITHUB_REPO}/contents/{path}?ref={GITHUB_BRANCH}'
    response = requests.get(url)
    
    if response.status_code == 200:
        files = response.json()
        image_files = [
            file['name'] for file in files if file['name'].lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        return image_files
    else:
        print(f"Erreur lors de la récupération des fichiers GitHub : {response.status_code}")
        return []

# Fonction pour créer le bucket si nécessaire
def create_bucket_if_not_exists():
    try:
        # Vérifie si le bucket existe déjà
        if not client.bucket_exists(MINIO_BUCKET):
            # Si le bucket n'existe pas, le créer
            client.make_bucket(MINIO_BUCKET)
            print(f"Le bucket {MINIO_BUCKET} a été créé avec succès.")
        else:
            print(f"Le bucket {MINIO_BUCKET} existe déjà.")
    except S3Error as e:
        print(f"Erreur lors de la création du bucket: {e}")

# Fonction pour vérifier si l'image existe déjà sur Minio
def image_exists_in_minio(folder, image_name):
    try:
        # Vérifie si l'objet existe déjà dans le bucket Minio
        client.stat_object(MINIO_BUCKET, f'{folder}/{image_name}')
        return True  # L'image existe déjà
    except S3Error as e:
        if e.code == 'NoSuchKey':
            return False  # L'image n'existe pas
        else:
            print(f"Erreur lors de la vérification de l'existence de l'image {image_name}: {e}")
            return False
        
# Fonction pour crer la table POSTGRES si nécessaire
def create_table_if_not_exists():
    try:
        # Connexion à la base de données PostgreSQL
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DATABASE,
            user=PG_USER,
            password=PG_PASSWORD
        )
        cursor = conn.cursor()

        # Vérifier si la table 'plants_data' existe
        create_table_query = """
        CREATE TABLE IF NOT EXISTS plants_data (
            id SERIAL PRIMARY KEY,
            url_source VARCHAR(255) NOT NULL,
            url_s3 VARCHAR(255) NOT NULL,
            label VARCHAR(100) NOT NULL
        );
        """
        # Exécuter la requête pour créer la table si elle n'existe pas
        cursor.execute(create_table_query)
        conn.commit()

        cursor.close()
        conn.close()
        print("La table 'plants_data' a été vérifiée/créée.")
    except Exception as e:
        print(f"Erreur lors de la vérification de la table dans PostgreSQL : {e}")

# Fonction pour enregistrer les métadonnées dans PostgreSQL
def insert_metadata_into_postgresql(url_source, url_s3, label):
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DATABASE,
            user=PG_USER,
            password=PG_PASSWORD
        )
        cursor = conn.cursor()

        insert_query = """
        INSERT INTO plants_data (url_source, url_s3, label)
        VALUES (%s, %s, %s)
        """
        cursor.execute(insert_query, (url_source, url_s3, label))
        conn.commit()

        cursor.close()
        conn.close()
        print(f"Les métadonnées de {url_source} ont été insérées dans PostgreSQL.")
    except Exception as e:
        print(f"Erreur lors de l'insertion dans PostgreSQL : {e}")

# Fonction pour télécharger et envoyer les images vers Minio, et enregistrer les métadonnées dans PostgreSQL
def download_and_upload_images_from_github(path, label):
    print(f"Début de la fonction de téléchargement des images depuis {path}")

    # Dossier du bucket
    folderMinio = os.path.basename(path)

    # Créer le bucket MINIO s'il n'existe pas
    create_bucket_if_not_exists()

    # Vérifier si la table POSTGRES existe et la créer si nécessaire
    create_table_if_not_exists()

    # Lister les fichiers d'images sur GitHub
    image_files = list_github_files(path)
    
    if not image_files:
        print(f"Aucune image trouvée dans le répertoire {path}.")
        return
    
    # Télécharger chaque fichier, les envoyer vers Minio et insérer les métadonnées dans PostgreSQL
    for image in image_files:
        url = f'https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/{path}/{image}'

        # Vérifier si l'image existe déjà dans Minio
        if image_exists_in_minio(folderMinio, image):
            print(f"L'image {image} existe déjà sur Minio. Elle ne sera pas téléchargée.")
            continue  # Passe à l'image suivante
        
        response = requests.get(url)
        if response.status_code == 200:
            print(f"Image {image} téléchargée avec succès depuis {path}.")
            # Télécharger l'image vers Minio
            client.put_object(
                MINIO_BUCKET, 
                f'{folderMinio}/{image}', 
                BytesIO(response.content), 
                len(response.content)
            )
            print(f"Image {image} uploadée vers Minio dans le dossier {folderMinio}.")
            
            # URL de l'image dans Minio
            url_s3 = f'http://{MINIO_HOST}/{MINIO_BUCKET}/{folderMinio}/{image}'
            
            # Enregistrer les métadonnées dans PostgreSQL
            insert_metadata_into_postgresql(url, url_s3, label)
        else:
            print(f"Erreur de téléchargement pour {image}, code HTTP {response.status_code}")


# Définir le DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 3, 18),
    'retries': 1,
}

dag = DAG(
    'download_and_upload_images_to_minio_and_postgresql',
    default_args=default_args,
    schedule_interval=None,  # Cela peut être planifié si nécessaire
    catchup=False
)

# Tâche pour télécharger et uploader les images dandelion et enregistrer les métadonnées dans PostgreSQL
download_task1 = PythonOperator(
    task_id='imagesDandelion',
    python_callable=partial(download_and_upload_images_from_github, GITHUB_PATH_DANDELION, 'dandelion'),
    dag=dag
)

# Tâche pour télécharger et uploader les images grass et enregistrer les métadonnées dans PostgreSQL
download_task2 = PythonOperator(
    task_id='imagesGrass',
    python_callable=partial(download_and_upload_images_from_github, GITHUB_PATH_GRASS, 'grass'),
    dag=dag
)

download_task1 >> download_task2
