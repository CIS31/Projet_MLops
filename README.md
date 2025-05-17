# Projet MLOps : Classification d'Images (Dandelion vs Grass)

##  Description du Projet
Ce projet met en place une pipeline MLOps complète pour un modèle de classification d'images (pissenlits vs herbe) en utilisant diverses technologies modernes. Il intègre le prétraitement des données, l'entraînement et la gestion du modèle, le déploiement, la surveillance et l'automatisation de la pipeline.

##  Technologies Utilisées
| **Composant**          | **Technologie**        |
|----------------------|----------------------|
| **Orchestration**     | Apache Airflow       |
| **Deep Learning**     | pyTorch             |
| **Feature Store**     | PostgreSQL          |
| **ML Metadata Store** | MLflow              |
| **Stockage Modèle**   | AWS S3 (MinIO)      |
| **Source Repo**       | GitHub              |
| **CI/CD**            | GitHub Actions      |
| **Conteneurisation**  | Docker / Docker Compose |
| **Registry Docker**   | DockerHub           |
| **API Serving**       | FastAPI             |
| **Déploiement**       | Kubernetes          |

  
##  Pipeline MLOps

1. **Extraction & Prétraitement des Données**
   - Téléchargement des images depuis des URLs
   - Nettoyage et transformation des données
   - Stockage des features dans PostgreSQL

![image](https://github.com/user-attachments/assets/d45f2c7c-de66-45b4-ab70-6de4637f2ef0)

2. **Entraînement du Modèle**
   - Entraînement avec pyTorch
   - Enregistrement des expériences avec MLflow
   - Sauvegarde du modèle sur AWS S3 (MinIO)

![image](https://github.com/user-attachments/assets/7b31f124-64d0-492a-91a1-58e9757b5433)

3. **Déploiement & API**
   - API FastAPI pour servir le modèle
   - WebApp (Streamlit) pour l'interface utilisateur

![image](https://github.com/user-attachments/assets/cb7febd0-f4c9-4507-8b97-f93664e33d86)

4. **Automatisation avec Apache Airflow**
   - DAG pour réentraîner et mettre à jour le modèle

![image](https://github.com/user-attachments/assets/a5e94aa6-e3eb-4dae-958e-f902eb34d4fd)
![image](https://github.com/user-attachments/assets/8daa1eec-8c9c-42b6-a7f3-d7887ad45259)

5. **CI/CD & Déploiement sur Kubernetes**
   - Tests unitaires et d’intégration avec GitHub Actions
   - Conteneurisation avec Docker
   - Déploiement de l'API et de la Webapp sur Kubernetes en local (Docker Desktop) 

![image](https://github.com/user-attachments/assets/7c548a76-08cc-4136-9f98-d1d8a832eaf1)

##  Configuration et Installation
### Prérequis
- Python 3.8+
- Git
- Docker Desktop installé avec Kubernetes activé
- Toutes les images Docker nécessaires doivent être buildées et visibles localement 


## Commandes utiles :

   **Lancer le projet sur docker :**
   - docker-compose -f docker_compose.yaml up -d
     
   **Lancer le deploiement sur kubernetes Local:**
    Assurez-vous que Kubernetes est bien activé dans Docker Desktop  

   - kubectl apply -f deployement.yaml
     
     **Exposer le service minio :**
     
   - kubectl port-forward svc/minio 9101:9101

   **Lancer les tests :**
   - PYTHONPATH=$(pwd) pytest -v tests/

   **Vérifier les bibliothèques dans airflow-webserveur :**
   - docker exec -it projet_mlops-airflow-webserver-1 bash
   - pip list
   - pip install -r /sources/requirements.txt

   **Ajouter une bibliotèque compatible au projet (par example savoir quelle version de mlflow est compatible) :**
   - source venv/bin/activate
   - pip list #mlflow n'est pas présent
   - pip install mlflow
   - pip list #mlflow est présent et on ajoute la version compatible dans le fichier requierements.txt, mlflow==2.22.0

##  Accès aux Services Docker et Kubernetes

   **Docker**
| Service     | URL                             |
|-------------|----------------------------------|
| **API FastAPI** | [http://localhost:8000/docs](http://localhost:8000/docs) |
| **WebApp Streamlit** | [http://localhost:8501](http://localhost:8501) |
| **MinIO UI** | [http://localhost:9000](http://localhost:9000) |
| **Airflow**  | [http://localhost:8080](http://localhost:8080) |
| **MLflow UI**| [http://localhost:5000](http://localhost:5000) |


   **Kubernetes**
| Service     | URL                             |
|-------------|----------------------------------|
| **API FastAPI** | [http://localhost:30080/docs](http://localhost:30080/docs) |
| **WebApp Streamlit** | [http://localhost:30085](http://localhost:30085) |
| **MinIO UI** | [http://localhost:9101](http://localhost:9101) |







