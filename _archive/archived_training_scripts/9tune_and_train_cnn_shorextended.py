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
import optuna # Für Hyperparameter-Suche
from torch.cuda.amp import autocast, GradScaler # Für Mixed Precision Training

# --- 0. Projekt-Root und Haupt-Ausgabeordner definieren ---
# Annahme: Dieses Skript liegt direkt im Hackatron-Projektordner
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
print(f"Project Root: {PROJECT_ROOT}")

# Hauptordner für alle Ergebnisse dieses Skripts
EXPERIMENT_OUTPUT_DIR_BASE = PROJECT_ROOT / "optuna_experiment_results"
EXPERIMENT_OUTPUT_DIR_BASE.mkdir(parents=True, exist_ok=True)


# --- 1. Globale Konfiguration (Pfade relativ zum PROJECT_ROOT) ---
ANNOTATED_DIR = PROJECT_ROOT / "data_annotated"
# Eigener Datenordner für diesen Lauf (wird ggf. neu erstellt)
BASE_DATA_DIR = EXPERIMENT_OUTPUT_DIR_BASE / "data_split_for_tuning_long"
TRAIN_DIR = BASE_DATA_DIR / "train"
VAL_DIR = BASE_DATA_DIR / "validation"
# Pfad, wo das finale, mit Optuna getunte Modell gespeichert wird
FINAL_MODEL_SAVE_PATH = EXPERIMENT_OUTPUT_DIR_BASE / "best_tuned_cnn_model_long.pth"
# Optuna SQLite Datenbank Pfad
OPTUNA_DB_PATH = EXPERIMENT_OUTPUT_DIR_BASE / "optuna_study_long.db"


# Trainingsparameter für Optuna und finales Training
BATCH_SIZE = 128 # Kann von Optuna ggf. noch überschrieben werden, wenn du es dort definierst
NUM_WORKERS = 8    # Passe dies an deine CPU-Kerne an
N_TRIALS = 100     # Anzahl der Optuna Trials
EPOCHS_PER_TRIAL = 15 # Epochen für jeden einzelnen Optuna Trial
FINAL_TRAINING_EPOCHS = 75 # Erhöht für das finale Training nach Optuna

# Bildparameter und Seeds
IMG_SIZE = 250
RANDOM_SEED = 42
TRAIN_VAL_SPLIT = 0.8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Reproduzierbarkeit
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# --- 2. Datenaufteilung (Bleibt gleich) ---
def split_data(source_dir, train_dir, val_dir, split_ratio=0.8, seed=42):
    print(f"Splitting data from {source_dir} into {train_dir.parent}...")
    random.seed(seed)
    if train_dir.parent.exists():
        print(f"  Removing existing data directory: {train_dir.parent}")
        shutil.rmtree(train_dir.parent)
    for class_name in ['yes', 'no']:
        source_class_dir = source_dir / class_name
        train_class_dir = train_dir / class_name
        val_class_dir = val_dir / class_name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)
        if not source_class_dir.exists(): print(f"  W: Src {source_class_dir} missing."); continue
        images = glob.glob(str(source_class_dir / "*.png"))
        if not images: print(f"  W: No PNGs in {source_class_dir}."); continue
        random.shuffle(images)
        split_idx = int(len(images) * split_ratio)
        train_images, val_images = images[:split_idx], images[split_idx:]
        print(f"  Cls '{class_name}': {len(images)} tot -> {len(train_images)} tr, {len(val_images)} vl")
        for img_path_str in train_images: shutil.copy(Path(img_path_str), train_class_dir / Path(img_path_str).name)
        for img_path_str in val_images: shutil.copy(Path(img_path_str), val_class_dir / Path(img_path_str).name)
    print("Data splitting complete.")


# --- 3. Datentransformationen & Laden (Bleibt gleich) ---
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15), # Standardrotation
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), # Standard Jitter
    transforms.ToTensor(), normalize
])
val_transforms = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), normalize])

def create_dataloaders(train_dir, val_dir, batch_size, num_workers):
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    persistent = num_workers > 0 and device.type == 'cuda' # persistent_workers nur bei GPU und workers > 0
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent)
    print(f"\nDataLoaders created: Train={len(train_dataset)} ({len(train_loader)} batches), Val={len(val_dataset)} ({len(val_loader)} batches)")
    print(f"  Classes: {train_dataset.classes} (Mapping: {train_dataset.class_to_idx})")
    if train_dataset.class_to_idx.get('yes', -1) != 1 or train_dataset.class_to_idx.get('no', -1) != 0:
       print(f"\n*** WARNING: Class mapping incorrect! Expected {{'no': 0, 'yes': 1}} Actual: {train_dataset.class_to_idx}\n")
    return train_loader, val_loader, train_dataset.class_to_idx

# --- 4. Modell Definition (Bleibt gleich) ---
# --- 4. Modell Definition (Angepasst für Optuna) ---
class CustomCNN(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.5,
                 num_conv_blocks=4, first_layer_filters=64, filter_increase_factor=2.0): # NEUE Parameter
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
            # Erhöhe die Kanäle für den nächsten Block, außer für den letzten
            if i < num_conv_blocks - 1:
                 next_channels = int(current_channels * filter_increase_factor)
            # Stelle sicher, dass next_channels mindestens 1 ist, falls filter_increase_factor klein ist
            next_channels = max(next_channels, 1)


        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        # Der Input für den FC-Layer ist die Anzahl der Kanäle des letzten Conv-Blocks
        self.fc = nn.Linear(current_channels, num_classes)
        self._initialize_weights()

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# --- 5. Metrik-Berechnung (Bleibt gleich) ---
def calculate_metrics(outputs, labels):
    probs = torch.sigmoid(outputs).detach().cpu().numpy()
    preds = (probs > 0.5).astype(int).flatten()
    labels_np = labels.detach().cpu().numpy().flatten()
    try:
        acc = accuracy_score(labels_np, preds)
        precision = precision_score(labels_np, preds, average='binary', pos_label=1, zero_division=0)
        recall = recall_score(labels_np, preds, average='binary', pos_label=1, zero_division=0)
        f1 = f1_score(labels_np, preds, average='binary', pos_label=1, zero_division=0)
    except Exception as e: print(f" Error metrics: {e}"); acc, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
    return acc, precision, recall, f1

# --- NEU: Funktion zum Speichern der Fehleranalyse-Bilder ---
def save_error_analysis_images(model, val_loader, device_obj, class_to_idx, output_dir_session, experiment_tag):
    print(f"  Saving error analysis images for {experiment_tag} to {output_dir_session}...")
    fn_dir = output_dir_session / "false_negatives"
    fp_dir = output_dir_session / "false_positives"
    fn_dir.mkdir(parents=True, exist_ok=True)
    fp_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    yes_idx = class_to_idx.get('yes', 1) # Standardmäßig 1
    no_idx = class_to_idx.get('no', 0)   # Standardmäßig 0
    fn_count = 0
    fp_count = 0

    filepaths = [s[0] for s in val_loader.dataset.samples]
    true_labels_indices = [s[1] for s in val_loader.dataset.samples]
    all_predictions_indices = []
    use_amp_eval = (device_obj.type == 'cuda')

    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device_obj, non_blocking=True)
            with torch.amp.autocast(device_type=device_obj.type, enabled=use_amp_eval):
                outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().flatten().tolist()
            all_predictions_indices.extend(preds)

    for i, img_filepath_str in enumerate(filepaths):
        img_filepath = Path(img_filepath_str)
        true_label_idx = true_labels_indices[i]
        pred_label_idx = all_predictions_indices[i]

        if true_label_idx == yes_idx and pred_label_idx == no_idx: # False Negative
            shutil.copy(img_filepath, fn_dir / img_filepath.name)
            fn_count += 1
        elif true_label_idx == no_idx and pred_label_idx == yes_idx: # False Positive
            shutil.copy(img_filepath, fp_dir / img_filepath.name)
            fp_count += 1
    print(f"    False Negatives saved: {fn_count}, False Positives saved: {fp_count}")


# --- NEU: Funktion zum Plotten der Trainingshistorie ---
def plot_training_history(history, output_dir_session, experiment_tag):
    print(f"  Plotting training history for {experiment_tag} to {output_dir_session}...")
    if not history or 'val_f1' not in history or not history['val_f1']:
        print("    Not enough history data to plot (val_f1 missing or empty).")
        return

    epochs_ran = history.get('epoch', range(1, len(history['val_f1']) + 1))
    if not epochs_ran:
        print("    Epoch data missing in history.")
        return

    plt.figure(figsize=(18, 12)) # Etwas größer für mehr Plots

    # Plot 1: F1, Accuracy, Precision, Recall (Validation)
    plt.subplot(2, 2, 1)
    if 'val_f1' in history: plt.plot(epochs_ran, history['val_f1'], 'b-', label='Val F1')
    if 'val_acc' in history: plt.plot(epochs_ran, history['val_acc'], 'c-', label='Val Acc')
    if 'val_precision' in history: plt.plot(epochs_ran, history['val_precision'], 'm-', label='Val Precision')
    if 'val_recall' in history: plt.plot(epochs_ran, history['val_recall'], 'y-', label='Val Recall')
    plt.title(f'{experiment_tag} - Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    # Plot 2: Loss (Train & Validation)
    plt.subplot(2, 2, 2)
    if 'train_loss' in history: plt.plot(epochs_ran, history['train_loss'], 'r-', label='Train Loss')
    if 'val_loss' in history: plt.plot(epochs_ran, history['val_loss'], 'g-', label='Val Loss')
    plt.title(f'{experiment_tag} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot 3: F1 (Train & Validation)
    plt.subplot(2, 2, 3)
    if 'train_f1' in history: plt.plot(epochs_ran, history['train_f1'], 'r-', label='Train F1')
    if 'val_f1' in history: plt.plot(epochs_ran, history['val_f1'], 'b-', label='Val F1')
    plt.title(f'{experiment_tag} - F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.grid(True)

    # Plot 4: Accuracy (Train & Validation)
    plt.subplot(2, 2, 4)
    if 'train_acc' in history: plt.plot(epochs_ran, history['train_acc'], 'r-', label='Train Acc')
    if 'val_acc' in history: plt.plot(epochs_ran, history['val_acc'], 'b-', label='Val Acc')
    plt.title(f'{experiment_tag} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir_session / f"{experiment_tag}_training_history.png")
    plt.close()
    print(f"  Training history plotted and saved to {output_dir_session}") 