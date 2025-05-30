import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import shutil
import glob

# --- Konfiguration ---
# Pfad zu deinem trainierten Modell
MODEL_PATH = Path(r"C:\Users\pssol\PycharmProjects\Hackatron\trained_models\best_tuned_cnn_model_long.pth")

# Ordner mit den neuen, ungelabelten Bildern (die von download_ocm_images_wms_sfoe.py erstellt wurden)
UNLABELED_IMAGES_DIR = Path(r"C:\Users\pssol\PycharmProjects\Hackatron\get_data_withmodel\unlabeled_sfoe_images")

# Ausgabeordner für Bilder, die das Modell als "Ja" (Ladestation) klassifiziert
PSEUDO_LABELED_YES_DIR = Path(r"C:\Users\pssol\PycharmProjects\Hackatron\get_data_withmodel\model_labeled_sfoe_images")

# Transformationspipeline (exakt wie im Validierungsset deines Trainingsskripts)
IMG_SIZE = 250
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalize
])

# Threshold für die Entscheidung "Ja" (Wahrscheinlichkeit > threshold)
PREDICTION_THRESHOLD = 0.8 # Beginne mit einem relativ hohen Threshold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# --- Ende Konfiguration ---

# --- Definiere deine Modellarchitektur HIER ---
# WICHTIG: Das muss EXAKT dieselbe Architektur sein wie beim Training!
# Kopiere die Klassendefinition von CustomCNN aus deinem Trainingsskript hierher.
class CustomCNN(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.5,
                 num_conv_blocks=4, first_layer_filters=64, filter_increase_factor=2.0):
        super(CustomCNN, self).__init__()
        self.conv_blocks = nn.ModuleList()
        current_channels = 3
        next_channels = first_layer_filters
        for i in range(num_conv_blocks):
            block = nn.Sequential(
                nn.Conv2d(current_channels, next_channels, kernel_size=3, stride=1, padding='same', bias=False),
                nn.BatchNorm2d(next_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.conv_blocks.append(block)
            current_channels = next_channels
            if i < num_conv_blocks - 1:
                 next_channels = int(current_channels * filter_increase_factor)
            next_channels = max(next_channels, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(current_channels, num_classes)
        # _initialize_weights() wird hier nicht benötigt, da wir Gewichte laden

    def forward(self, x):
        for block in self.conv_blocks: x = block(x)
        x = self.avgpool(x); x = self.flatten(x)
        x = self.dropout(x); x = self.fc(x)
        return x
# --- Ende Modelldefinition ---

def predict_on_images():
    """ Lädt das Modell und klassifiziert Bilder im UNLABELED_IMAGES_DIR. """
    PSEUDO_LABELED_YES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Modell laden ---
    print(f"Lade Modell von: {MODEL_PATH.resolve()}")
    if not MODEL_PATH.exists():
        print(f"FEHLER: Modelldatei nicht gefunden: {MODEL_PATH.resolve()}")
        return


    try:

        model = CustomCNN(
            num_classes=1,  # Das ist korrekt für binäre Klassifikation
            dropout_rate=0.4725254690626744,  # Dein bester Optuna-Wert
            num_conv_blocks=5,  # Dein bester Optuna-Wert
            first_layer_filters=32,  # Dein bester Optuna-Wert
            filter_increase_factor=1.6265635488885444  # Dein bester Optuna-Wert
        ).to(device)

        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()  # Wichtig: In den Evaluationsmodus setzen!
        print("Modell erfolgreich geladen mit den besten Optuna-Architekturparametern.")
    except Exception as e:
        print(f"FEHLER beim Laden des Modells oder der Initialisierung: {e}")
        import traceback
        traceback.print_exc()  # Gibt mehr Details zum Fehler aus
        return

    # --- Bilder verarbeiten ---
    image_files = glob.glob(str(UNLABELED_IMAGES_DIR / "*.png"))
    if not image_files:
        print(f"Keine PNG-Bilder in '{UNLABELED_IMAGES_DIR.resolve()}' gefunden.")
        return

    print(f"{len(image_files)} ungelabelte Bilder gefunden. Starte Vorhersagen...")
    yes_candidates_count = 0
    processed_count = 0

    with torch.no_grad(): # Keine Gradientenberechnung für Inferenz
        for img_path_str in image_files:
            img_path = Path(img_path_str)
            processed_count += 1
            if processed_count % 50 == 0: # Fortschritt alle 50 Bilder
                print(f"  Verarbeitet {processed_count}/{len(image_files)} Bilder...")

            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = val_transforms(image).unsqueeze(0).to(device) # Batch-Dimension hinzufügen

                with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    output = model(image_tensor)

                probability = torch.sigmoid(output).item() # Wahrscheinlichkeit für Klasse "Ja"

                if probability > PREDICTION_THRESHOLD:
                    yes_candidates_count += 1
                    target_path = PSEUDO_LABELED_YES_DIR / img_path.name
                    shutil.copy(str(img_path), str(target_path))
                    # print(f"  -> 'Ja'-Kandidat ({probability:.2f}): {img_path.name} -> kopiert nach {PSEUDO_LABELED_YES_DIR.name}")

            except Exception as e:
                print(f"FEHLER beim Verarbeiten von Bild {img_path.name}: {e}")

    print("\n--- Vorhersagen abgeschlossen ---")
    print(f"Gesamtzahl verarbeiteter Bilder: {processed_count}")
    print(f"Anzahl 'Ja'-Kandidaten (Wahrscheinlichkeit > {PREDICTION_THRESHOLD}): {yes_candidates_count}")
    print(f"'Ja'-Kandidaten wurden in '{PSEUDO_LABELED_YES_DIR.resolve()}' gespeichert.")

if __name__ == "__main__":
    predict_on_images()