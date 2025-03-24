import pytest
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.models import TaskInstance, DagRun
from airflow.utils.state import State
from airflow.utils.types import DagRunType
from airflow.utils.db import create_session
from datetime import datetime
import pendulum

@pytest.fixture
def dag():
    # Crée un DAG simple pour les tests
    dag = DAG(
        "test_dag",
        schedule_interval="@daily",
        start_date=datetime(2025, 3, 22),
        catchup=False,
    )
    
    # Ajouter des tâches
    task = DummyOperator(task_id="dummy_task", dag=dag)
    return dag

@pytest.fixture
def session():
    # Crée une session SQLAlchemy pour les tests
    with create_session() as session:
        yield session

def test_task_execution(dag, session):
    # Créer un DagRun pour le DAG
    execution_date = pendulum.datetime(2025, 3, 22, tz="UTC")  # Ajout du fuseau horaire
    dag_run = DagRun(
        dag_id=dag.dag_id,
        run_id="test_run",
        run_type=DagRunType.MANUAL,
        execution_date=execution_date,
        state=State.RUNNING,
    )
    session.add(dag_run)
    session.commit()

    # Créer une instance de la tâche
    task = dag.get_task("dummy_task")
    task_instance = TaskInstance(task=task, execution_date=execution_date)
    task_instance.dag_run = dag_run

    # Exécuter la tâche et vérifier son état
    task_instance.run()

    # Vérifier que la tâche a bien été exécutée avec succès
    assert task_instance.state == State.SUCCESS