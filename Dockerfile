# Utilisation de l'image de base d'Apache Airflow
FROM apache/airflow:2.8.2-python3.9

# Copier le fichier requirements.txt dans l'image
COPY requirements.txt /requirements.txt

# Installer les dépendances via pip
RUN pip install --no-cache-dir -r /requirements.txt