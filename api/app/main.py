import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
import os

# --- Configuration ---
MINIO_ENDPOINT = "http://localhost:9000"
MODEL_PATH = "bucket.romain/model/model_2025-05-10_16-58-15_acc_80.00.pth"
MINIO_USER = os.getenv("MINIO_USER", "minio")
MINIO_PASS = os.getenv("MINIO_PASS", "minio123")
MODEL_URL = f"{MINIO_ENDPOINT}/{MODEL_PATH}"

IMAGE_PATH = "photo.jpg"  # Assurez-vous que l’image est bien présente

# --- Fonction pour charger le modèle depuis MinIO en RAM ---
def load_model_from_minio():
    print("Téléchargement du modèle depuis MinIO...")
    response = requests.get(MODEL_URL, auth=(MINIO_USER, MINIO_PASS))
    if response.status_code != 200:
        raise Exception(f"Échec du téléchargement ({response.status_code})")

    buffer = BytesIO(response.content)
    model = torch.load(buffer, map_location=torch.device("cpu"))
    model.eval()
    print("Modèle chargé en mémoire.")
    return model

# --- Fonction de traitement de l'image ---
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Ajout de la dimension batch

# --- Fonction de prédiction ---
def predict(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        predicted_class = torch.argmax(outputs, dim=1).item()
        return "dandelion" if predicted_class == 0 else "grass"

# --- Main ---
if __name__ == "__main__":
    print("Lancement de l'application de prédiction")

    model = load_model_from_minio()
    image_tensor = preprocess_image(IMAGE_PATH)
    result = predict(model, image_tensor)

    print(f"Résultat de la prédiction : {result}")
