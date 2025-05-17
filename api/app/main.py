from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
import boto3
import os
from model import LogisticRegressionModel

# Configuration MinIO, a mettre dans le fichier .env
MINIO_ENDPOINT = os.getenv("MINIO_HOST", "http://minio:9000")  
MINIO_USER = os.getenv("MINIO_USER", "minio")
MINIO_PASS = os.getenv("MINIO_PASS", "miniopassword")
BUCKET_NAME = "bucket.romain"  

app = FastAPI(title="Image Classification API")
app = FastAPI(title="Image Classification API")

# Chargement du modèle une seule fois au démarrage
@app.on_event("startup")
def load_model():
    global model
    print("Connexion à MinIO et chargement du modèle...")
           
    s3 = boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_USER,
        aws_secret_access_key=MINIO_PASS,
        region_name="us-east-1"
    )

    try:
        # Lister les objets du dossier "model/"
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix="model/")
        all_keys = [obj['Key'] for obj in response.get("Contents", []) if obj['Key'].endswith(".pth")]

        if not all_keys:
            raise RuntimeError("Aucun fichier .pth trouvé dans le bucket.")

        # Prendre le premier fichier .pth trouvé (un peu lent, mlflow ne selectionne pas le meilleur modèle)
        selected_key = all_keys[0]
        print(f"Modèle trouvé : {selected_key}")

        # Télécharger ce modèle
        response = s3.get_object(Bucket=BUCKET_NAME, Key=selected_key)
        buffer = BytesIO(response['Body'].read())

        # Charger dans le modèle
        model = LogisticRegressionModel()
        model.load_state_dict(torch.load(buffer, map_location=torch.device("cpu")))
        model.eval()

        print("Modèle chargé avec succès depuis MinIO.")

    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        raise RuntimeError("Échec du téléchargement du modèle depuis MinIO.")


# Prétraitement de l'image
def preprocess_image(image_data: bytes):
    image = Image.open(BytesIO(image_data)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # correspond à ce que le modèle attend
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# Prédiction
def predict(image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        prob = output.item()
        return "dandelion" if prob >= 0.5 else "grass"

# Endpoint POST
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image.")

    image_bytes = await file.read()
    print(f"Fichier reçu : {file.filename}, {len(image_bytes)} octets")

    try:
        image_tensor = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de lecture de l'image : {e}")

    prediction = predict(image_tensor)
    return JSONResponse(content={"prediction": prediction})


