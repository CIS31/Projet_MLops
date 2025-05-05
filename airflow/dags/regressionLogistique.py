import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from minio import Minio
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)
from dotenv import load_dotenv
from datetime import datetime
from io import BytesIO

# ---------------------------
# Charger les variables d'environnement
# ---------------------------
if os.getenv("DOCKER_ENV"):  # Dans Docker
    env_path = "/sources/.env"
else:  # En local
    env_path = ".env"
load_dotenv(dotenv_path=env_path)

MINIO_HOST          = os.getenv("MINIO_HOST_DAG")
MINIO_ACCESS_KEY    = os.getenv("MINIO_USER")
MINIO_SECRET_KEY    = os.getenv("MINIO_PASS")
MINIO_BUCKET        = os.getenv("MINIO_BUCKET")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME          = os.getenv("MODEL_NAME", "Dandelion_vs_Grass")

# ---------------------------
# Exporter les credentials AWS pour MLflow → S3 (MinIO)
# ---------------------------
os.environ["AWS_ACCESS_KEY_ID"]     = MINIO_ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_SECRET_KEY
# Indiquer l'endpoint S3 que MLflow doit utiliser
os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{MINIO_HOST}"

# ---------------------------
# Connexion au client MinIO
# ---------------------------
minio_client = Minio(
    MINIO_HOST,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,  # HTTP en local
)

# ---------------------------
# Configuration MLflow
# ---------------------------
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Dandelion_vs_Grass")

# ---------------------------
# Dataset personnalisé pour charger les images depuis MinIO
# ---------------------------
class MinIODataset(Dataset):
    def __init__(self, client, bucket, folders, transform=None):
        self.client = client
        self.bucket = bucket
        self.folders = folders
        self.transform = transform
        self.image_paths = []
        self.labels = []
        for label, folder in enumerate(folders):
            for obj in client.list_objects(bucket, prefix=folder + '/', recursive=True):
                self.image_paths.append(obj.object_name)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        obj = self.client.get_object(self.bucket, path)
        img = Image.open(obj).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ---------------------------
# Fonction principale d'entraînement et évaluation
# ---------------------------
def train_logistic_regression_model(
    folders=['dandelion', 'grass'],
    num_epochs=50,
    batch_size=32,
    lr=0.01
):
    # 1. Prétraitement des images
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    # 2. Chargement des données depuis MinIO
    dataset = MinIODataset(minio_client, MINIO_BUCKET, folders, transform)
    n = len(dataset)
    train_size = int(0.8 * n)
    val_size   = int(0.1 * n)
    test_size  = n - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    print(f"Total: {n}, Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # 3. Définition du modèle de régression logistique
    input_size = 128 * 128 * 3
    class LogisticRegressionModel(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.fc = nn.Linear(input_size, 1)
        def forward(self, x):
            x = x.view(x.size(0), -1)
            return torch.sigmoid(self.fc(x))

    model = LogisticRegressionModel(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # 4. Lancer un run MLflow pour tracker params, metrics et modèle
    with mlflow.start_run(run_name="LogisticRegression_Airflow"):
        # 4.1 Log des hyperparamètres
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", lr)
        mlflow.pytorch.autolog()

        # 4.2 Entraînement
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for imgs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(imgs).squeeze()
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()
                running_loss = loss.item()
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss:.4f}")
            mlflow.log_metric("train_loss", running_loss, step=epoch)

        # 4.3 Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                preds = (model(imgs).squeeze() > 0.5).float()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1  = f1_score(val_labels, val_preds)
        print(f"Validation Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        print("Val Confusion Matrix:\n", confusion_matrix(val_labels, val_preds))
        print("Val Report:\n", classification_report(val_labels, val_preds))
        mlflow.log_metric("validation_accuracy", val_acc)
        mlflow.log_metric("validation_f1_score", val_f1)

        # 4.4 Test final
        test_preds, test_labels = [], []
        with torch.no_grad():
            for imgs, labels in test_loader:
                preds = (model(imgs).squeeze() > 0.5).float()
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
        test_acc = accuracy_score(test_labels, test_preds)
        test_f1  = f1_score(test_labels, test_preds)
        print(f"Test Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
        print("Test Confusion Matrix:\n", confusion_matrix(test_labels, test_preds))
        print("Test Report:\n", classification_report(test_labels, test_preds))
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1_score", test_f1)

        # 4.5 Log du modèle dans MLflow
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME  # ex: "Dandelion_vs_Grass"
        )

        # 5. Backup optionnel sur MinIO
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fn = f"model_{date_str}_acc_{test_acc*100:.2f}.pth"
        buf = BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)
        prefix = "model/"
        # Créer le dossier virtuel si nécessaire
        if not any(o.object_name.startswith(prefix)
                   for o in minio_client.list_objects(MINIO_BUCKET, prefix=prefix, recursive=True)):
            minio_client.put_object(MINIO_BUCKET, prefix + ".keep", BytesIO(), 0)
        minio_client.put_object(MINIO_BUCKET, prefix + fn, buf, buf.getbuffer().nbytes)
        print(f"Modèle sauvegardé sur MinIO : {prefix + fn}")
