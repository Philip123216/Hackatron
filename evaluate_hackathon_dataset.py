import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from pathlib import Path
import numpy as np

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # Fallback, falls __file__ nicht definiert ist (z.B. in manchen interaktiven Umgebungen)
    SCRIPT_DIR = Path(".").resolve() # Nimmt das aktuelle Arbeitsverzeichnis
    print(f"WARNUNG: __file__ nicht definiert, SCRIPT_DIR auf aktuelles Arbeitsverzeichnis gesetzt: {SCRIPT_DIR}")
    print("         Bitte stelle sicher, dass das Skript aus dem Haupt-Projektordner ausgeführt wird.")

# 1. Konfiguration
MODEL_PATH = SCRIPT_DIR / "experiment_runs" / "retrain_v3" / "final_trained_model" / "best_model_retrain_v3.pth"
EVAL_DATA_DIR = SCRIPT_DIR / "eparking"
IMG_SIZE = 250
BATCH_SIZE_EVAL = 32 # Kann je nach Speicher angepasst werden
NORMALIZE_MEAN = [0.485, 0.456, 0.406] # Wie beim Training verwendet
NORMALIZE_STD = [0.229, 0.224, 0.225] # Wie beim Training verwendet

# Architekturparameter des geladenen Modells (basierend auf Optuna-Ergebnissen)
BEST_DROPOUT_RATE = 0.32584188120291535
BEST_NUM_CONV_BLOCKS = 6
BEST_FIRST_LAYER_FILTERS = 16
BEST_FILTER_INCREASE_FACTOR = 1.4239117513348847

# 2. Modellarchitektur-Definition (CustomCNN)
class CustomCNN(nn.Module):
    def __init__(self, num_classes: int = 1, dropout_rate: float = 0.5,
                 num_conv_blocks: int = 4, first_layer_filters: int = 64,
                 filter_increase_factor: float = 2.0):
        super(CustomCNN, self).__init__()
        self.conv_blocks = nn.ModuleList()
        current_channels = 3
        next_channels = int(first_layer_filters)

        for i in range(int(num_conv_blocks)):
            block = nn.Sequential(
                nn.Conv2d(current_channels, next_channels,
                          kernel_size=3, stride=1, padding='same', bias=False),
                nn.BatchNorm2d(next_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.conv_blocks.append(block)
            current_channels = next_channels
            if i < int(num_conv_blocks) - 1:
                next_channels = int(current_channels * filter_increase_factor)
                next_channels = max(next_channels, 1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=float(dropout_rate))
        self.fc = nn.Linear(current_channels, num_classes)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.conv_blocks:
            x = block(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def evaluate_model():
    print(f"--- Starte Evaluation für Modell: {MODEL_PATH.name} ---")
    print(f"Evaluationsdatensatz: {EVAL_DATA_DIR.resolve()}")

    # 3. Laden des trainierten Modells
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Verwendetes Gerät: {device}")

    if not MODEL_PATH.exists():
        print(f"FEHLER: Modelldatei nicht gefunden: {MODEL_PATH.resolve()}")
        return
    if not EVAL_DATA_DIR.exists():
        print(f"FEHLER: Evaluationsdaten-Ordner nicht gefunden: {EVAL_DATA_DIR.resolve()}")
        return
    if not (EVAL_DATA_DIR / 'y').exists() or not (EVAL_DATA_DIR / 'n').exists():
        print(f"FEHLER: Unterordner 'y' und/oder 'n' im Evaluationsdaten-Ordner nicht gefunden.")
        return

    model = CustomCNN(
        num_classes=1, # Für binäre Klassifikation mit BCEWithLogitsLoss
        dropout_rate=BEST_DROPOUT_RATE,
        num_conv_blocks=BEST_NUM_CONV_BLOCKS,
        first_layer_filters=BEST_FIRST_LAYER_FILTERS,
        filter_increase_factor=BEST_FILTER_INCREASE_FACTOR
    ).to(device)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Modellgewichte erfolgreich geladen.")
    except Exception as e:
        print(f"FEHLER beim Laden der Modellgewichte: {e}")
        print("Stelle sicher, dass die Architekturparameter im Skript mit dem trainierten Modell übereinstimmen.")
        return

    model.eval()

    # 4. Datentransformationen und DataLoader
    eval_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])

    try:
        eval_dataset = datasets.ImageFolder(str(EVAL_DATA_DIR), transform=eval_transforms)
        if not eval_dataset.samples:
            print(f"FEHLER: Keine Bilder im Evaluationsdatensatz gefunden: {EVAL_DATA_DIR.resolve()}")
            return
        print(f"Gefundene Klassen im Evaluationsdatensatz: {eval_dataset.classes}")
        print(f"Klassen-Indizes: {eval_dataset.class_to_idx}")
        # Sicherstellen, dass 'y' (positive Klasse) Index 1 hat für die Metrikberechnung
        # Diese Annahme wird bei sklearn.metrics getroffen, wenn pos_label=1 ist
        if eval_dataset.class_to_idx.get('y') != 1 or eval_dataset.class_to_idx.get('n') != 0 :
             print("\nACHTUNG: Unerwartetes Klassen-Mapping im Evaluationsdatensatz!")
             print(f"  Erwartet {{'n': 0, 'y': 1}}, aber erhalten: {eval_dataset.class_to_idx}")
             print("  Dies kann die Interpretation von Precision, Recall und F1-Score beeinflussen.")
             print("  Stelle sicher, dass die Ordnernamen im Eval-Set mit der Erwartung übereinstimmen ('y' für positiv, 'n' für negativ).")


    except Exception as e:
        print(f"FEHLER beim Erstellen des ImageFolder für Evaluationsdaten: {e}")
        return

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE_EVAL,
        shuffle=False, # Wichtig: Keine Mischung für Evaluation
        num_workers=2 # Kann angepasst werden
    )

    # 5. Inferenz und Metrikberechnung
    all_labels_list = []
    all_preds_probs_list = []
    print(f"\nStarte Inferenz auf {len(eval_dataset)} Bildern...")

    with torch.no_grad(): # Keine Gradientenberechnung während der Inferenz
        for inputs, labels in eval_loader:
            inputs = inputs.to(device)
            # labels werden nicht auf device verschoben, da sie direkt für sklearn verwendet werden
            
            outputs = model(inputs)
            
            # Sigmoid anwenden, um Wahrscheinlichkeiten zu erhalten
            probs = torch.sigmoid(outputs).cpu() # Auf CPU für spätere Numpy-Konvertierung
            
            all_labels_list.extend(labels.numpy())
            all_preds_probs_list.extend(probs.numpy())

    all_labels_np = np.array(all_labels_list)
    all_preds_probs_np = np.array(all_preds_probs_list).flatten() # Sicherstellen, dass es 1D ist

    # Umwandlung der Wahrscheinlichkeiten in binäre Vorhersagen
    all_preds_binary_np = (all_preds_probs_np > 0.5).astype(int)

    print("\n--- Evaluationsergebnisse ---")

    # Accuracy
    accuracy = accuracy_score(all_labels_np, all_preds_binary_np)
    print(f"Accuracy: {accuracy:.4f}")

    # Metriken für die positive Klasse 'y' (angenommen, Index 1)
    # Stelle sicher, dass class_to_idx korrekt ist und 'y' wirklich 1 ist.
    # Sklearn verwendet pos_label=1 standardmäßig für binäre Klassifikation,
    # wenn Labels 0 und 1 sind.
    positive_class_label_name = 'y'
    positive_class_idx = eval_dataset.class_to_idx.get(positive_class_label_name, 1)


    precision = precision_score(all_labels_np, all_preds_binary_np, pos_label=positive_class_idx, zero_division=0)
    recall = recall_score(all_labels_np, all_preds_binary_np, pos_label=positive_class_idx, zero_division=0)
    f1 = f1_score(all_labels_np, all_preds_binary_np, pos_label=positive_class_idx, zero_division=0)

    print(f"Precision (für Klasse '{positive_class_label_name}'): {precision:.4f}")
    print(f"Recall (für Klasse '{positive_class_label_name}'): {recall:.4f}")
    print(f"F1-Score (für Klasse '{positive_class_label_name}'): {f1:.4f}")

    # Confusion Matrix
    print("\nConfusion Matrix:")
    # Labels für die Confusion Matrix sollten die Klassennamen sein
    cm_labels = [k for k, v in sorted(eval_dataset.class_to_idx.items(), key=lambda item: item[1])]
    cm = confusion_matrix(all_labels_np, all_preds_binary_np, labels=list(eval_dataset.class_to_idx.values()))
    print(f"Labels für CM: {cm_labels} (entsprechend Indizes {list(eval_dataset.class_to_idx.values())})")
    print(cm)

    # Classification Report
    print("\nClassification Report:")
    # target_names sollten die Klassennamen in der richtigen Reihenfolge der Indizes sein
    report_target_names = [k for k, v in sorted(eval_dataset.class_to_idx.items(), key=lambda item: item[1])]
    print(classification_report(all_labels_np, all_preds_binary_np, target_names=report_target_names, zero_division=0))

    print("\n--- Evaluation abgeschlossen ---")

if __name__ == '__main__':
    evaluate_model() 