import os
import requests
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
from io import BytesIO
from minio import Minio
from airflow.hooks.base_hook import BaseHook
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

# Initialise Minio Client
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

# Fonction pour télécharger et envoyer les images vers Minio
def download_images_from_github(path):
    print(f"Début de la fonction de téléchargement des images depuis {path}")

    # Dossier du bucket
    folderMinio = os.path.basename(path)

    # Créer le bucket s'il n'existe pas
    create_bucket_if_not_exists()

    # Lister les fichiers d'images sur GitHub
    image_files = list_github_files(path)
    
    if not image_files:
        print(f"Aucune image trouvée dans le répertoire {path}.")
        return
    
    # Télécharger chaque fichier et les envoyer vers Minio
    for image in image_files:
        url = f'https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/{path}/{image}'
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
    'download_and_upload_images_to_minio',
    default_args=default_args,
    schedule_interval=None,  # Cela peut être planifié si nécessaire
    catchup=False
)

# Tâche pour télécharger et uploader les images dandelion
download_task1 = PythonOperator(
    task_id='imagesDandelion',
    python_callable=partial(download_images_from_github, GITHUB_PATH_DANDELION),
    dag=dag
)

# Tâche pour télécharger et uploader les images grass
download_task2 = PythonOperator(
    task_id='imagesGrass',
    python_callable=partial(download_images_from_github, GITHUB_PATH_GRASS),
    dag=dag
)


download_task1 >> download_task2
