import pytest
import psycopg2
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.models import TaskInstance, DagRun
from airflow.utils.state import State
from airflow.utils.types import DagRunType
from datetime import datetime
import pendulum
import os

# Configure la connexion à PostgreSQL pour les tests
DB_CONN = {
    "dbname": "airflow",
    "user": "airflow",
    "password": "airflow",
    "host": "0.0.0.0",
    "port": 5432,
}

@pytest.fixture
def dag():
    # Crée un DAG simple pour les tests
    dag = DAG(
        "test_dag",
        schedule="@daily",
        start_date=pendulum.datetime(2025, 3, 22, tz="UTC"),  # Ajout du fuseau horaire
        catchup=False,
    )
    
    # Ajouter des tâches
    task = DummyOperator(task_id="dummy_task", dag=dag)
    return dag

@pytest.fixture
def db_connection():
    # Crée une connexion PostgreSQL pour les tests
    conn = psycopg2.connect(**DB_CONN)
    yield conn
    conn.close()

def test_task_execution(dag, db_connection):
    # Vérifier la connexion à la base de données
    with db_connection.cursor() as cursor:
        cursor.execute("SELECT 1;")
        result = cursor.fetchall()
        assert len(result) > 0, "La connexion à la base de données a échoué."
        print("Connexion à la base de données réussie.")

    # Créer un DagRun pour le DAG
    execution_date = pendulum.datetime(2025, 3, 22, tz="UTC")  # Ajout du fuseau horaire
    with db_connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO dag_run (dag_id, run_id, run_type, execution_date, state, clear_number)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id;
            """,
            ("test_dag", "test_run", DagRunType.MANUAL, execution_date, State.RUNNING, 0),
        )
        dag_run_id = cursor.fetchone()[0]
        db_connection.commit()

    # Créer une instance de la tâche
    task = dag.get_task("dummy_task")
    task_instance = TaskInstance(task=task, run_id=f"manual__{execution_date.isoformat()}")
    task_instance.dag_run_id = dag_run_id

    # Exécuter la tâche et vérifier son état
    task_instance.run()

    # Vérifier que la tâche a bien été exécutée avec succès
    assert task_instance.state == State.SUCCESS