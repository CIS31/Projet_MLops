import os
import requests
import psycopg2
from io import BytesIO
from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv

# Déterminer le chemin du fichier .env
if os.getenv("DOCKER_ENV"):  # Si la variable DOCKER_ENV est définie, on est dans Docker
    env_path = "/sources/.env"
else:  # Sinon, on est en local, notamment pour les tests
    env_path = ".env"

# Charger les variables d'environnement depuis le chemin déterminé
load_dotenv(dotenv_path=env_path)
GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH")
MINIO_HOST = os.getenv("MINIO_HOST_DAG")
MINIO_ACCESS_KEY = os.getenv("MINIO_USER")
MINIO_SECRET_KEY = os.getenv("MINIO_PASS")
MINIO_BUCKET = os.getenv("MINIO_BUCKET")
PG_HOST = os.getenv("DB_HOST_DAG")
PG_PORT = os.getenv("DB_PORT")
PG_DATABASE = os.getenv("DB_NAME")
PG_USER = os.getenv("DB_USER")
PG_PASSWORD = os.getenv("DB_PASS")

# Initialisation de Minio
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
def create_bucket_if_not_exists(bucket_name):
    try:
        # Vérifie si le bucket existe déjà
        if not client.bucket_exists(bucket_name):
            # Si le bucket n'existe pas, le créer
            client.make_bucket(bucket_name)
            print(f"Le bucket {bucket_name} a été créé avec succès.")
        else:
            print(f"Le bucket {bucket_name} existe déjà.")
    except S3Error as e:
        print(f"Erreur lors de la création du bucket: {e}")

# Fonction pour vérifier si l'image existe déjà sur Minio
def image_exists_in_minio(folder, image_name):
    try:
        # Vérifie si l'objet existe déjà dans le bucket Minio
        client.stat_object(MINIO_BUCKET, f'{folder}/{image_name}')
        return True  # L'image existe déjà
    except S3Error as e:
        if e.code == 'NoSuchKey':
            return False  # L'image n'existe pas
        else:
            print(f"Erreur lors de la vérification de l'existence de l'image {image_name}: {e}")
            return False
        
# Fonction pour crer la table POSTGRES si nécessaire
def create_table_if_not_exists(table_name):
    try:
        # Connexion à la base de données PostgreSQL
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DATABASE,
            user=PG_USER,
            password=PG_PASSWORD
        )
        cursor = conn.cursor()

        # Vérifier si la table 'plants_data' existe
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            url_source VARCHAR(255) NOT NULL,
            url_s3 VARCHAR(255) NOT NULL,
            label VARCHAR(100) NOT NULL
        );
        """

        # Exécuter la requête pour créer la table si elle n'existe pas
        cursor.execute(create_table_query)
        conn.commit()

        cursor.close()
        conn.close()
        print("La table 'plants_data' a été vérifiée/créée.")
    except Exception as e:
        print(f"Erreur lors de la vérification de la table dans PostgreSQL : {e}")

# Fonction pour enregistrer les métadonnées dans PostgreSQL
def insert_metadata_into_postgresql(url_source, url_s3, label):
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DATABASE,
            user=PG_USER,
            password=PG_PASSWORD
        )
        cursor = conn.cursor()

        insert_query = """
        INSERT INTO plants_data (url_source, url_s3, label)
        VALUES (%s, %s, %s)
        """
        cursor.execute(insert_query, (url_source, url_s3, label))
        conn.commit()

        cursor.close()
        conn.close()
        print(f"Les métadonnées de {url_source} ont été insérées dans PostgreSQL.")
    except Exception as e:
        print(f"Erreur lors de l'insertion dans PostgreSQL : {e}")

# Fonction pour télécharger et envoyer les images vers Minio, et enregistrer les métadonnées dans PostgreSQL
def download_and_upload_images_from_github(path, label):
    print(f"Début de la fonction de téléchargement des images depuis {path}")

    # Dossier du bucket
    folderMinio = os.path.basename(path)

    # Créer le bucket MINIO s'il n'existe pas
    create_bucket_if_not_exists(MINIO_BUCKET)

    # Vérifier si la table plants_data existe dans POSTGRES et la créer si nécessaire
    create_table_if_not_exists("plants_data")

    # Lister les fichiers d'images sur GitHub
    image_files = list_github_files(path)
    
    if not image_files:
        print(f"Aucune image trouvée dans le répertoire {path}.")
        return
    
    # Télécharger chaque fichier, les envoyer vers Minio et insérer les métadonnées dans PostgreSQL
    for image in image_files:
        url = f'https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/{path}/{image}'

        # Vérifier si l'image existe déjà dans Minio
        if image_exists_in_minio(folderMinio, image):
            print(f"L'image {image} existe déjà sur Minio. Elle ne sera pas téléchargée.")
            continue  # Passe à l'image suivante
        
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
            
            # URL de l'image dans Minio
            url_s3 = f'http://{MINIO_HOST}/{MINIO_BUCKET}/{folderMinio}/{image}'
            
            # Enregistrer les métadonnées dans PostgreSQL
            insert_metadata_into_postgresql(url, url_s3, label)
        else:
            print(f"Erreur de téléchargement pour {image}, code HTTP {response.status_code}")
