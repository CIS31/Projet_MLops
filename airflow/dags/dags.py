from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# Fonction qui sera exécutée
def print_hello_world():
    print("Hello World")

# Définition du DAG
dag = DAG(
    'hello_world_dag',  # Nom du DAG
    description='Un simple DAG qui imprime Hello World',
    schedule_interval=None,  # Pas de planification automatique
    start_date=datetime(2025, 3, 18),  # Date de début
    catchup=False,  # Pas d'exécution des DAGs passés
)

# Définition de l'opérateur qui appelle la fonction
hello_world_task = PythonOperator(
    task_id='print_hello_world',  # ID de la tâche
    python_callable=print_hello_world,  # Fonction à exécuter
    dag=dag,
)

# Cette tâche est la seule du DAG
hello_world_task
