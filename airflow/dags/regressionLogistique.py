import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from minio import Minio
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from dotenv import load_dotenv
from datetime import datetime
from io import BytesIO


# Déterminer le chemin du fichier .env
if os.getenv("DOCKER_ENV"):  # Si la variable DOCKER_ENV est définie, on est dans Docker
    env_path = "/sources/.env"
else:  # Sinon, on est en local, notamment pour les tests
    env_path = ".env"

# Charger les variables d'environnement depuis le chemin déterminé
load_dotenv(dotenv_path=env_path)
MINIO_HOST = os.getenv("MINIO_HOST_DAG")
MINIO_ACCESS_KEY = os.getenv("MINIO_USER")
MINIO_SECRET_KEY = os.getenv("MINIO_PASS")
MINIO_BUCKET = os.getenv("MINIO_BUCKET")

# Connexion au client MinIO
minio_client = Minio(
    MINIO_HOST,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False  # Utiliser HTTP pour une connexion locale
)

# Classe pour charger les images depuis MinIO
class MinIODataset(Dataset):
    def __init__(self, minio_client, bucket_name, folders, transform=None):
        self.minio_client = minio_client
        self.bucket_name = bucket_name
        self.folders = folders  # Liste des dossiers à explorer (par exemple ['dandelion', 'grass'])
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Lister les objets dans MinIO et collecter les chemins d'images et leurs labels
        for label, folder in enumerate(folders):
            objects = minio_client.list_objects(bucket_name, prefix=folder + '/', recursive=True)
            for obj in objects:
                self.image_paths.append(obj.object_name)
                self.labels.append(label)  # Dandelion -> 0, Grass -> 1 (par exemple)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Télécharger l'image depuis MinIO
        obj = self.minio_client.get_object(self.bucket_name, image_path)
        img = Image.open(obj)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

# Fonction pour entraîner et évaluer le modèle
def train_logistic_regression_model(folders=['dandelion', 'grass'], num_epochs=50, batch_size=32, lr=0.01):
    # Prétraitement des images et transformation
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Redimensionner les images
        transforms.ToTensor(),  # Convertir les images en tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalisation
    ])

    # Charger les images depuis MinIO via le Dataset personnalisé
    dataset = MinIODataset(minio_client, MINIO_BUCKET, folders, transform=transform)
    
    # Split des données en Train, Validation, Test (80% train, 10% val, 10% test)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    print(f"Dataset size : {len(dataset)}")
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")

    # Créer des DataLoader pour les différents ensembles
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Définition du modèle de régression logistique
    class LogisticRegressionModel(nn.Module):
        def __init__(self, input_size):
            super(LogisticRegressionModel, self).__init__()
            self.fc = nn.Linear(input_size, 1)  # Une seule sortie pour la régression logistique

        def forward(self, x):
            x = x.view(x.size(0), -1)  # Aplatir l'image en un vecteur
            x = torch.sigmoid(self.fc(x))  # Appliquer la fonction sigmoïde pour obtenir une probabilité
            return x

    # Définir la taille d'entrée après le redimensionnement des images (128x128x3)
    input_size = 128 * 128 * 3

    # Créer une instance du modèle
    model = LogisticRegressionModel(input_size)

    # Définir la fonction de perte et l'optimiseur
    criterion = nn.BCELoss()  # Binary Cross Entropy pour la régression logistique
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Entraînement du modèle
    for epoch in range(num_epochs):
        model.train()  # Mode entraînement
        for images, labels in train_loader:
            optimizer.zero_grad()  # Initialiser les gradients
            outputs = model(images)  # Propagation avant
            loss = criterion(outputs.squeeze(), labels.float())  # Calcul de la perte
            loss.backward()  # Propagation arrière
            optimizer.step()  # Mise à jour des poids

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Évaluation sur l'ensemble de validation
    model.eval()  # Mode évaluation
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            preds = (outputs.squeeze() > 0.5).float()  # Convertir en classes (0 ou 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calcul de la matrice de confusion et rapport de classification
    print("Classification report (Validation set):")
    print(classification_report(all_labels, all_preds))
    
    print("Confusion Matrix (Validation set):")
    print(confusion_matrix(all_labels, all_preds))

    # Évaluation finale sur l'ensemble de test
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            preds = (outputs.squeeze() > 0.5).float()  # Convertir en classes (0 ou 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Rapport de classification et matrice de confusion sur le test set
    print("Classification report (Test set):")
    print(classification_report(all_labels, all_preds))

    print("Confusion Matrix (Test set):")
    print(confusion_matrix(all_labels, all_preds))

    # Calcul de la précision (accuracy) sur l'ensemble de test
    accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"Précision sur l'ensemble de test : {accuracy * 100:.2f}%")

     # Créer le nom du fichier pour sauvegarder le modèle (incluant la date et la précision)
    date_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_filename = f"model_{date_str}_accuracy_{accuracy * 100:.2f}%.pth"

    # Créer le dossier "model" dans MinIO s'il n'existe pas
    model_folder = "model"
    objects = minio_client.list_objects(MINIO_BUCKET, prefix=model_folder + '/', recursive=True)
    empty_file = BytesIO()
    if not any(obj.object_name.startswith(model_folder) for obj in objects):
        # Créer le dossier virtuel "model" (MinIO permet de créer des préfixes)
        minio_client.put_object(MINIO_BUCKET, f"{model_folder}/.keep", empty_file, len(empty_file.getvalue())) # Ajouter un fichier vide pour créer le dossier

    # Sauvegarder le modèle dans MinIO
    with BytesIO() as model_buffer:
        torch.save(model.state_dict(), model_buffer)
        model_buffer.seek(0)
        minio_client.put_object(MINIO_BUCKET, f"{model_folder}/{model_filename}", model_buffer, model_buffer.tell())

    print(f"Modèle sauvegardé sous {model_folder}/{model_filename} dans le bucket {MINIO_BUCKET}")
