# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import glob
import random
import shutil
from pathlib import Path
import time
import copy
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import math
# Import für AMP (wird auch für Inferenz mit autocast benötigt)
from torch.cuda.amp import autocast, GradScaler

# --- 1. Konfiguration ---
# Pfade
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent # Geht 3 Ebenen hoch von src/evaluation/ zum Hackatron-Root
ANNOTATED_DIR = PROJECT_ROOT / "data_annotated"
BASE_DATA_DIR = PROJECT_ROOT / "data_split_for_tuning" # Oder "data", je nachdem
TRAIN_DIR = BASE_DATA_DIR / "train"
VAL_DIR = BASE_DATA_DIR / "validation"
MODEL_TO_EVALUATE_PATH = PROJECT_ROOT / "models" / "best_tuned_cnn_model.pth"

# Trainingsparameter (werden teilweise noch für DataLoader gebraucht)
BATCH_SIZE = 128      # Sollte die gleiche sein wie beim Training für konsistente Val-Loader Erstellung
NUM_WORKERS = 8       # Wie beim Training
IMG_SIZE = 250
RANDOM_SEED = 42      # Konsistent halten

# --- HIER FEHLT NICHTS MEHR ---
TRAIN_VAL_SPLIT = 0.8 # Wird hier nicht direkt für Split, aber ggf. für Logging genutzt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# --- 2. Datenaufteilung (Funktion wird hier nicht mehr gebraucht, aber Definitionen bleiben) ---
def split_data(source_dir, train_dir, val_dir, split_ratio=0.8, seed=42):
   # Diese Funktion wird beim reinen Evaluieren nicht aufgerufen
   pass

# --- 3. Datentransformationen & Laden (Definitionen werden gebraucht) ---
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# Nur Validation Transforms nötig für Evaluation
val_transforms = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), normalize])

def create_dataloaders(train_dir, val_dir, batch_size, num_workers):
    # Nur der Validation Loader wird wirklich gebraucht, aber wir erstellen beide für Konsistenz
    # und um class_to_idx zu bekommen.
    try:
        # Verwende hier die val_transforms für beide, da keine Augmentation nötig
        train_dataset_dummy = datasets.ImageFolder(train_dir, transform=val_transforms)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    except FileNotFoundError:
        print(f"FEHLER: Datenordner nicht gefunden unter {train_dir.parent}. Stelle sicher, dass BASE_DATA_DIR korrekt ist.")
        exit()

    print(f"Using {num_workers} workers for Validation DataLoader.")
    persistent = num_workers > 0
    pin_memory = device.type == 'cuda'
    # Train Loader wird nicht wirklich genutzt, aber erstellt für class_to_idx
    train_loader_dummy = DataLoader(train_dataset_dummy, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent)

    print(f"\nDataLoaders created (for evaluation): Val={len(val_dataset)}/{len(val_loader)} batches")
    if hasattr(train_dataset_dummy, 'class_to_idx'):
         class_to_idx = train_dataset_dummy.class_to_idx
         print(f"  Classes: {list(class_to_idx.keys())} (Mapping: {class_to_idx})")
         if class_to_idx.get('yes', -1) != 1 or class_to_idx.get('no', -1) != 0:
             print(f"\n*** WARNING: Class mapping incorrect! Actual: {class_to_idx}\n")
    else: # Fallback, falls train_dir leer war
         try:
             class_to_idx = val_dataset.class_to_idx
             print(f"  Classes (from val): {list(class_to_idx.keys())} (Mapping: {class_to_idx})")
         except:
             print("ERROR: Could not determine class mapping.")
             exit()


    return train_loader_dummy, val_loader, class_to_idx # Gebe dummy train loader zurück

# --- 4. Modell Definition (Muss exakt die gleiche sein wie beim Training!) ---
class CustomCNN(nn.Module):
    # <<< WICHTIG: Verwende hier die Architektur + Dropout Rate des Modells, das du evaluieren willst! >>>
    def __init__(self, num_classes=1, dropout_rate=0.5): # Annahme: 0.5 war der Standard / beste Wert
        super(CustomCNN, self).__init__()
        def _make_block(in_c, out_c): return nn.Sequential(nn.Conv2d(in_c, out_c, 3, 1, 'same', bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        self.conv1, self.conv2, self.conv3, self.conv4 = _make_block(3, 64), _make_block(64, 128), _make_block(128, 256), _make_block(256, 512)
        self.avgpool, self.flatten, self.dropout, self.fc = nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout_rate), nn.Linear(512, num_classes)
        self._initialize_weights() # Initialisierung ist für Laden nicht relevant, aber schadet nicht
    def forward(self, x): return self.fc(self.dropout(self.flatten(self.avgpool(self.conv4(self.conv3(self.conv2(self.conv1(x))))))))
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0, 0.01); nn.init.constant_(m.bias, 0)

# --- 5. Metrik-Berechnung (wie vorher) ---
def calculate_metrics(outputs, labels):
    probs = torch.sigmoid(outputs).detach().cpu().numpy()
    preds = (probs > 0.5).astype(int).flatten() # Standard Threshold 0.5
    labels = labels.detach().cpu().numpy().flatten()
    try:
        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='binary', pos_label=1, zero_division=0)
        recall = recall_score(labels, preds, average='binary', pos_label=1, zero_division=0)
        f1 = f1_score(labels, preds, average='binary', pos_label=1, zero_division=0)
    except Exception as e: print(f" Error metrics: {e}"); acc, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
    return acc, precision, recall, f1

# --- Funktion zum Anzeigen von False Negatives (wie vorher) ---
def find_and_show_false_negatives(model, val_loader, device, class_to_idx, num_to_show=10):
    print("\n--- Searching for False Negatives ---")
    model.eval()
    false_negative_files = []
    yes_label_index = class_to_idx.get('yes', -1)
    if yes_label_index == -1 or not hasattr(val_loader, 'dataset'): return
    filepaths = [s[0] for s in val_loader.dataset.samples]; all_labels_list = [s[1] for s in val_loader.dataset.samples]
    all_preds_list = []
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            # Wichtig: autocast auch bei Inferenz, wenn Modell mit AMP trainiert wurde
            with autocast(enabled=(device.type == 'cuda')):
                 outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().flatten().tolist() # Standard Threshold 0.5
            all_preds_list.extend(preds)
    for i in range(len(filepaths)):
        if all_labels_list[i] == yes_label_index and all_preds_list[i] == 0: false_negative_files.append(filepaths[i])
    print(f"Found {len(false_negative_files)} FN images.")
    if not false_negative_files: print("No False Negatives found!"); return
    num_to_show = min(num_to_show, len(false_negative_files)); num_cols=5; num_rows=math.ceil(num_to_show/num_cols)
    if num_to_show == 0: return
    print(f"Showing up to {num_to_show} FN images...")
    plt.figure(figsize=(15, 3 * num_rows))
    for i, img_path in enumerate(false_negative_files[:num_to_show]):
        try:
            img = Image.open(img_path); ax = plt.subplot(num_rows, num_cols, i + 1)
            ax.imshow(img); ax.set_title(f"FN: {Path(img_path).name}\nTrue: yes, Pred: no", fontsize=8); ax.axis("off")
        except Exception as e: print(f"Could not load/show {img_path}: {e}")
    plt.tight_layout(); plt.show()

# --- Hauptausführung (nur Evaluation) ---
if __name__ == "__main__":
    # 1. Daten-Split wird NICHT ausgeführt. Prüfe, ob der Ordner existiert.
    if not BASE_DATA_DIR.exists() or not (VAL_DIR / 'yes').exists():
        print(f"FEHLER: Validierungsdaten nicht gefunden unter {VAL_DIR}.")
        print(f"Stelle sicher, dass der Ordner '{BASE_DATA_DIR.name}' existiert und die Aufteilung enthält.")
        exit()
    else:
        print(f"Found existing data split directory: {BASE_DATA_DIR}")

    # 2. DataLoaders erstellen (brauchen nur Validation Loader)
    _, val_loader, class_to_idx = create_dataloaders(TRAIN_DIR, VAL_DIR, BATCH_SIZE, NUM_WORKERS)

    # 3. Modell instantiieren (mit der Architektur des gespeicherten Modells!)
    # <<< Passe dropout_rate an, falls dein bestes Modell einen anderen Wert hatte! >>>
    model = CustomCNN(num_classes=1, dropout_rate=0.5).to(device) # Annahme 0.5

    # 4. Modell laden
    if MODEL_TO_EVALUATE_PATH.exists():
        print(f"\nLoading model weights from {MODEL_TO_EVALUATE_PATH}...")
        try:
            # Lade die Gewichte
            model.load_state_dict(torch.load(MODEL_TO_EVALUATE_PATH, map_location=device))
            print("Model weights loaded successfully.")
        except Exception as e:
            print(f"FEHLER beim Laden der Modelldatei: {e}")
            exit()
    else:
        print(f"FEHLER: Modelldatei nicht gefunden unter {MODEL_TO_EVALUATE_PATH}")
        exit()

    # 5. Finale Evaluation des geladenen Modells
    print("\n--- Final Evaluation on Validation Set (using loaded model) ---")
    model.eval() # Wichtig: In den Eval-Modus schalten!

    all_final_labels = []
    all_final_outputs = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            # Wichtig: AMP auch hier, falls das Modell damit trainiert wurde
            with autocast(enabled=(device.type == 'cuda')):
                 outputs = model(inputs)
            all_final_labels.append(labels)
            all_final_outputs.append(outputs)

    all_final_outputs = torch.cat(all_final_outputs)
    all_final_labels = torch.cat(all_final_labels)
    final_acc, final_prec, final_rec, final_f1 = calculate_metrics(all_final_outputs, all_final_labels)

    print(f"\nFinal Val Acc: {final_acc:.4f}")
    print(f"Final Val Precision: {final_prec:.4f}")
    print(f"Final Val Recall: {final_rec:.4f}")
    print(f"Final Val F1-Score: {final_f1:.4f}")

    # Confusion Matrix
    final_preds = (torch.sigmoid(all_final_outputs).detach().cpu().numpy() > 0.5).astype(int).flatten()
    final_labels_np = all_final_labels.detach().cpu().numpy().flatten()
    cm = confusion_matrix(final_labels_np, final_preds)
    print("\nConfusion Matrix (Validation Set):")
    print(f"Labels: {list(class_to_idx.keys())} (0: no, 1: yes)")
    print(cm)
    # Detaillierter Report
    print("\nClassification Report (Validation Set):")
    print(classification_report(final_labels_np, final_preds, target_names=list(class_to_idx.keys()), zero_division=0))

    # Falsch klassifizierte Bilder anzeigen
    find_and_show_false_negatives(model, val_loader, device, class_to_idx, num_to_show=15)

    print("\n--- Evaluation Script Finished ---")