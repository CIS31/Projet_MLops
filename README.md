# Projet MLOps : Classification d'Images (Dandelion vs Grass)

##  Description du Projet
Ce projet met en place une pipeline MLOps complète pour un modèle de classification d'images (pissenlits vs herbe) en utilisant diverses technologies modernes. Il intègre le prétraitement des données, l'entraînement et la gestion du modèle, le déploiement, la surveillance et l'automatisation du pipeline.

##  Structure du Repository

projet-mlops

|--- api  
|   |--src  
|   |--Dockerfile  
|--- webapp  
|   |--src  
|   |--Dockerfile  
|--- airflow  
|   |--dag.py  
|   |--Dockerfile  
|--- docs  
|--- tests  
|--- .gitignore  
|--- docker-compose.yml  
|--- github-action.yaml  
|--- README.md  


##  Technologies Utilisées
| **Composant**          | **Technologie**        |
|----------------------|----------------------|
| **Orchestration**     | Apache Airflow       |
| **Deep Learning**     | FastAI / TensorFlow  |
| **Feature Store**     | PostgreSQL          |
| **ML Metadata Store** | MLflow              |
| **Stockage Modèle**   | AWS S3 (MinIO)      |
| **Source Repo**       | GitHub              |
| **CI/CD**            | GitHub Actions      |
| **Conteneurisation**  | Docker / Docker Compose |
| **Registry Docker**   | DockerHub           |
| **API Serving**       | FastAPI             |
| **Déploiement**       | Kubernetes          |
| **Monitoring**        | Elasticsearch & Kibana |


##  Configuration et Installation
### Prérequis
- Docker & Docker Compose
- Python 3.8+
- Kubernetes (MiniKube ou Docker Desktop)
- Git
- AWS CLI (pour MinIO)
- MLflow
- Apache Airflow

##  Pipeline MLOps

1. **Extraction & Prétraitement des Données**
   - Téléchargement des images depuis des URLs
   - Nettoyage et transformation des données
   - Stockage des features dans PostgreSQL

2. **Entraînement du Modèle**
   - Entraînement avec FastAI ou TensorFlow
   - Enregistrement des expériences avec MLflow
   - Sauvegarde du modèle sur AWS S3 (MinIO)

3. **Déploiement & API**
   - API FastAPI pour servir le modèle
   - WebApp (Gradio/Streamlit) pour l'interface utilisateur

4. **Automatisation avec Apache Airflow**
   - DAG pour réentraîner et mettre à jour le modèle

5. **CI/CD & Déploiement sur Kubernetes**
   - Tests unitaires et d’intégration avec GitHub Actions
   - Conteneurisation avec Docker
   - Déploiement sur Kubernetes

6. **Monitoring**
   - Logs et métriques avec Elasticsearch & Kibana

