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
MINIO_ENDPOINT = "localhost:9000"  # Sans http:// ici pour boto3
MINIO_USER = os.getenv("MINIO_USER", "minio")
MINIO_PASS = os.getenv("MINIO_PASS", "miniopassword")
BUCKET_NAME = "bucket.romain"  
OBJECT_KEY = "model/model_2025-05-11_17-09-11_acc_75.00.pth"  

app = FastAPI(title="Image Classification API")
app = FastAPI(title="Image Classification API")

# Chargement du modèle une seule fois au démarrage
@app.on_event("startup")
def load_model():
    global model
    print("Connexion à MinIO et chargement du modèle...")

    s3 = boto3.client(
        's3',
        endpoint_url=f"http://{MINIO_ENDPOINT}",
        aws_access_key_id=MINIO_USER,
        aws_secret_access_key=MINIO_PASS,
        region_name="us-east-1"
    )

    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=OBJECT_KEY)
        model_data = response['Body'].read()
        buffer = BytesIO(model_data)

        model = LogisticRegressionModel()
        model.load_state_dict(torch.load(buffer, map_location=torch.device("cpu")))
        model.eval()

        print("Modèle chargé depuis MinIO.")
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
    image_tensor = preprocess_image(image_bytes)
    prediction = predict(image_tensor)
    return JSONResponse(content={"prediction": prediction})


# Test de l'API
# curl -X POST "http://localhost:8000/predict" \
#   -H "accept: application/json" \
#   -H "Content-Type: multipart/form-data" \
#   -F "file=@photo.jpg"

# Requirements pour l'API
# fastapi
# uvicorn
# torch
# torchvision
# boto3
# pillow
# python-multipart


# Pour lancer l'API :
# uvicorn main:app --reload --port 8000