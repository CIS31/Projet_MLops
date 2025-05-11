import torch.nn as nn
import torch

# 💡 Remarque :
# Actuellement, seuls les poids du modèle sont enregistrés dans le fichier .pth (via model.state_dict()).
# Cela nécessite de redéfinir manuellement l'architecture dans l'API pour pouvoir recharger les poids.
# Une alternative serait de sauvegarder l'objet modèle complet avec torch.save(model),
# ce qui inclut à la fois la structure et les poids. Cela simplifierait le chargement dans l'API,
# mais pourrait rendre le démarrage légèrement plus lent et moins flexible.
# À tester selon les besoins de maintenance et de performance.


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size=128 * 128 * 3):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        return torch.sigmoid(self.fc(x))
