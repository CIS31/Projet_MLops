import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from fonctions import download_and_upload_images_from_github
from regressionLogistique import train_logistic_regression_model
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

# ---------------------------
# Charger les variables d'environnement
# ---------------------------
if os.getenv("DOCKER_ENV"):
    env_path = "/sources/.env"
else:
    env_path = ".env"
load_dotenv(dotenv_path=env_path)

# ---------------------------
# Configuration de base
# ---------------------------
GITHUB_PATH_DANDELION  = os.getenv("GITHUB_PATH_DANDELION")
GITHUB_PATH_GRASS      = os.getenv("GITHUB_PATH_GRASS")
MLFLOW_TRACKING_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME             = os.getenv("MODEL_NAME", "Dandelion_vs_Grass")
PROMOTION_DELTA        = float(os.getenv("PROMOTION_DELTA", 0.01))

# ---------------------------
# Arguments par défaut du DAG
# ---------------------------
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 3, 18),
    'retries': 1,
}

# ---------------------------
# Définition du DAG: ingestion, entraînement et promotion
# ---------------------------
with DAG(
    dag_id='download_and_upload_images_to_minio_and_postgresql',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
) as dag:

    # 1️⃣ Télécharger et uploader images Dandelion
    download_dandelion = PythonOperator(
        task_id='download_dandelion',
        python_callable=download_and_upload_images_from_github,
        op_args=[GITHUB_PATH_DANDELION, 'dandelion'],
    )

    # 2️⃣ Télécharger et uploader images Grass
    download_grass = PythonOperator(
        task_id='download_grass',
        python_callable=download_and_upload_images_from_github,
        op_args=[GITHUB_PATH_GRASS, 'grass'],
    )

    # 3️⃣ Entraîner le modèle et logger dans MLflow
    train_model = PythonOperator(
        task_id='createModel',
        python_callable=train_logistic_regression_model,
        op_args=[['dandelion', 'grass']],
    )

    # 4️⃣ Comparer les runs MLflow et auto-stager / promouvoir
    def compare_and_promote():
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

        # Récupérer l'expérience
        exp = client.get_experiment_by_name(MODEL_NAME)

        # Trouver le meilleur run selon validation F1
        best_run = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.validation_f1_score DESC"],
            max_results=1
        )[0]
        new_f1 = best_run.data.metrics.get('validation_f1_score', 0.0)

        # Trouver la version enregistrée correspondant à ce run
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        new_version = None
        for mv in versions:
            if mv.run_id == best_run.info.run_id:
                new_version = mv.version
                break
        if new_version is None:
            raise ValueError(f"No registered model version found for run {best_run.info.run_id}")

        # Récupérer la version actuelle en production
        prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        current_f1 = 0.0
        if prod_versions:
            prod_run_id = prod_versions[0].run_id
            prod_run = client.get_run(prod_run_id)
            current_f1 = prod_run.data.metrics.get('validation_f1_score', 0.0)

        # Auto-staging et promotion
        if new_f1 > current_f1 + PROMOTION_DELTA:
            # Promotion en Production + archivage des anciennes
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=new_version,
                stage="Production",
                archive_existing_versions=True,
            )
        else:
            # Mise en Staging sans archiver la prod
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=new_version,
                stage="Staging",
            )

        # Rolling restart de l'API FastAPI pour recharger le modèle
        os.system("kubectl rollout restart deployment plants-api")

    promote_model = PythonOperator(
        task_id='compare_and_promote',
        python_callable=compare_and_promote,
    )

    # Orchestration des tâches
    download_dandelion >> download_grass >> train_model >> promote_model
