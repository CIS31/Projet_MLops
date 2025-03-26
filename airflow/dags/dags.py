import os
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
from fonctions import download_and_upload_images_from_github
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv(dotenv_path="/sources/.env")
GITHUB_PATH_DANDELION = os.getenv("GITHUB_PATH_DANDELION")
GITHUB_PATH_GRASS = os.getenv("GITHUB_PATH_GRASS")

# Définir le DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 3, 18),
    'retries': 1,
}

dag = DAG(
    dag_id='download_and_upload_images_to_minio_and_postgresql',
    default_args=default_args,
    schedule_interval=None,  # Cela peut être planifié si nécessaire
    catchup=False
)

# Tâche pour télécharger et uploader les images dandelion et enregistrer les métadonnées dans PostgreSQL
download_task1 = PythonOperator(
    task_id='imagesDandelion',
    python_callable=download_and_upload_images_from_github,
    op_args=[GITHUB_PATH_DANDELION, 'dandelion'],
    dag=dag
)

# Tâche pour télécharger et uploader les images grass et enregistrer les métadonnées dans PostgreSQL
download_task2 = PythonOperator(
    task_id='imagesGrass',
    python_callable=download_and_upload_images_from_github,
    op_args=[GITHUB_PATH_GRASS, 'grass'],
    dag=dag
)

download_task1 >> download_task2
