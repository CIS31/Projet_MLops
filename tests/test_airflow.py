import pytest
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.models import TaskInstance
from datetime import datetime

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

def test_task_execution(dag):
    # Créer une instance de la tâche
    task = dag.get_task("dummy_task")
    task_instance = TaskInstance(task=task, execution_date=datetime(2025, 3, 22))
    
    # Exécuter la tâche et vérifier son état
    task_instance.run()
    
    # Vérifier que la tâche a bien été exécutée avec succès
    assert task_instance.state == "success"
