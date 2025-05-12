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
BASE_DATA_DIR = EXPERIMENT_OUTPUT_DIR_BASE / "data_split_for_tuning"
TRAIN_DIR = BASE_DATA_DIR / "train"
VAL_DIR = BASE_DATA_DIR / "validation"
# Pfad, wo das finale, mit Optuna getunte Modell gespeichert wird
FINAL_MODEL_SAVE_PATH = EXPERIMENT_OUTPUT_DIR_BASE / "best_tuned_cnn_model.pth"
# Optuna SQLite Datenbank Pfad
OPTUNA_DB_PATH = EXPERIMENT_OUTPUT_DIR_BASE / "optuna_study.db"


# Trainingsparameter für Optuna und finales Training
BATCH_SIZE = 128 # Kann von Optuna ggf. noch überschrieben werden, wenn du es dort definierst
NUM_WORKERS = 8    # Passe dies an deine CPU-Kerne an
N_TRIALS = 100     # Anzahl der Optuna Trials
EPOCHS_PER_TRIAL = 15 # Epochen für jeden einzelnen Optuna Trial
FINAL_TRAINING_EPOCHS = 50 # Erhöht für das finale Training nach Optuna

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

    # Plot 3: F1 (Train & Validation) - Für Overfitting-Analyse
    plt.subplot(2, 2, 3)
    if 'train_f1' in history: plt.plot(epochs_ran, history['train_f1'], 'r-', label='Train F1')
    if 'val_f1' in history: plt.plot(epochs_ran, history['val_f1'], 'b-', label='Val F1')
    plt.title(f'{experiment_tag} - F1 Score (Train vs Val)')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)

    # Plot 4: Zeit pro Epoche
    plt.subplot(2, 2, 4)
    if 'time_per_epoch' in history: plt.plot(epochs_ran, history['time_per_epoch'], 'k-', label='Time/Epoch (s)')
    plt.title(f'{experiment_tag} - Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Seconds')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_save_path = output_dir_session / f"{experiment_tag}_training_history.png"
    plt.savefig(plot_save_path)
    plt.close() # Wichtig, um Speicher freizugeben und Plot-Überlappungen zu vermeiden
    print(f"    Training history plot saved to {plot_save_path}")


# --- 6. Trainings-Loop (leicht angepasst, um mehr History zu speichern) ---
def train_loop_generic(trial, model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device_obj,
                       loop_name_tag, current_session_output_dir, model_save_filename, is_optuna_trial=False):
    print(f"\n--- Starting Training Loop: {loop_name_tag} ({num_epochs} epochs) ---")
    model_full_save_path = current_session_output_dir / model_save_filename
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_f1 = 0.0
    # Erweiterte History
    history = {'epoch':[], 'train_loss':[], 'train_f1':[], 'val_loss':[], 'val_f1':[], 'val_acc':[], 'val_precision':[], 'val_recall':[], 'lr':[], 'time_per_epoch':[]}

    use_amp = (device_obj.type == 'cuda')
    scaler = GradScaler(enabled=use_amp)
    if use_amp: print(f"  AMP (Mixed Precision) enabled for '{loop_name_tag}'.")

    total_training_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\n  Epoch {epoch+1}/{num_epochs} for '{loop_name_tag}'")

        # --- Trainingsphase ---
        model.train()
        running_loss = 0.0
        all_train_labels_epoch, all_train_outputs_epoch = [], []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device_obj, non_blocking=True), labels.float().unsqueeze(1).to(device_obj, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device_obj.type, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)
            all_train_labels_epoch.append(labels.detach()) # Für Epoch-Metriken
            all_train_outputs_epoch.append(outputs.detach())

        epoch_train_loss = running_loss / len(train_loader.dataset)
        _, _, _, epoch_train_f1 = calculate_metrics(torch.cat(all_train_outputs_epoch), torch.cat(all_train_labels_epoch))
        print(f"    Train Loss: {epoch_train_loss:.4f}, Train F1: {epoch_train_f1:.4f}")

        # --- Validierungsphase ---
        model.eval()
        running_val_loss = 0.0
        all_val_labels_epoch, all_val_outputs_epoch = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device_obj, non_blocking=True), labels.float().unsqueeze(1).to(device_obj, non_blocking=True)
                with torch.amp.autocast(device_type=device_obj.type, enabled=use_amp):
                    outputs = model(inputs)
                    val_batch_loss = criterion(outputs, labels)
                running_val_loss += val_batch_loss.item() * inputs.size(0)
                all_val_labels_epoch.append(labels.detach())
                all_val_outputs_epoch.append(outputs.detach())

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_acc, val_prec, val_rec, val_f1 = calculate_metrics(torch.cat(all_val_outputs_epoch), torch.cat(all_val_labels_epoch))
        print(f"    Val Loss: {epoch_val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} P: {val_prec:.4f} R: {val_rec:.4f}")

        # History speichern
        current_lr = optimizer.param_groups[0]['lr']
        epoch_duration = time.time() - epoch_start_time
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(epoch_train_loss)
        history['train_f1'].append(epoch_train_f1)
        history['val_loss'].append(epoch_val_loss)
        history['val_f1'].append(val_f1)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_rec)
        history['lr'].append(current_lr)
        history['time_per_epoch'].append(epoch_duration)
        print(f"    LR: {current_lr:.1e}, Time: {epoch_duration:.2f}s")


        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_f1) # Basiere auf F1 für ReduceLROnPlateau
            else:
                scheduler.step()

        # Modell speichern
        if val_f1 > best_val_f1:
            print(f"    Best Val F1 for '{loop_name_tag}' improved ({best_val_f1:.4f} -> {val_f1:.4f}). Saving model to {model_full_save_path.name}...")
            best_val_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, model_full_save_path)

        # Optuna Pruning und Reporting (nur wenn es ein Optuna Trial ist)
        if is_optuna_trial and trial:
            trial.report(val_f1, epoch)
            if trial.should_prune():
                print(f"  Trial {trial.number} pruned at epoch {epoch+1}.")
                raise optuna.exceptions.TrialPruned()

    total_training_time = time.time() - total_training_start_time
    print(f"\n  Training for '{loop_name_tag}' complete in {total_training_time // 60:.0f}m {total_training_time % 60:.0f}s")
    print(f"  Best Validation F1 achieved: {best_val_f1:.4f}")

    # Lade das beste Modell, wenn eines gespeichert wurde, sonst das letzte
    if best_val_f1 > 0 or model_full_save_path.exists(): # Prüfe auch ob Datei existiert, falls F1 mal 0 war aber gespeichert wurde
        print(f"  Loading best weights from {model_full_save_path.name} for '{loop_name_tag}'.")
        model.load_state_dict(torch.load(model_full_save_path, map_location=device_obj))
    else:
        print(f"  WARNING: No best model saved for '{loop_name_tag}' (F1 was {best_val_f1:.4f}). Using last model state.")
        model.load_state_dict(best_model_wts) # Lade zumindest das letzte Modell

    return model, history, total_training_time, best_val_f1


# --- 7. Optuna Objective Function (Bleibt im Kern gleich, ruft neuen Loop auf) ---
# --- 7. Optuna Objective Function (Erweitert) ---
def objective(trial, train_loader, val_loader, pos_weight_tensor, device_obj):
    # Hyperparameter für das Training
    lr = trial.suggest_float("lr", 1e-6, 5e-2, log=True) # Erweiterter Lernratenbereich
    dropout_rate = trial.suggest_float("dropout", 0.1, 0.7) # Erweiterter Dropout-Bereich
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 5e-2, log=True) # Erweiterter Weight Decay

    # Hyperparameter für die Architektur
    num_conv_blocks = trial.suggest_int("num_conv_blocks", 3, 6) # 3 bis 6 Conv-Blöcke
    first_layer_filters = trial.suggest_categorical("first_layer_filters", [32, 64, 96, 128]) # Mehr Optionen
    filter_increase_factor = trial.suggest_float("filter_increase_factor", 1.5, 2.5) # Faktor für Filteranstieg

    # Optimizer-Typ und seine spezifischen Parameter
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "SGD", "RMSprop"])

    # Modell mit den vorgeschlagenen Architekturparametern instantiieren
    model = CustomCNN(num_classes=1,
                      dropout_rate=dropout_rate,
                      num_conv_blocks=num_conv_blocks,
                      first_layer_filters=first_layer_filters,
                      filter_increase_factor=filter_increase_factor).to(device_obj)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # Optimizer basierend auf Auswahl erstellen
    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        momentum = trial.suggest_float("sgd_momentum", 0.7, 0.99)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == "RMSprop":
        # RMSprop hat keinen 'weight_decay' direkt im Konstruktor, wird oft separat oder gar nicht verwendet
        alpha = trial.suggest_float("rmsprop_alpha", 0.8, 0.999) # Typischer Parameter für RMSprop
        optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=alpha) # Ohne weight_decay hier
        # Wenn du Weight Decay mit RMSprop willst, musst du es manuell in der Optimierungs-Schleife hinzufügen
        # oder eine L2-Regularisierung direkt in der Loss-Funktion implementieren (komplexer).
        # Für den Anfang lassen wir es hier für RMSprop weg oder du könntest einen kleinen festen WD mit AdamW kombinieren.

    print(f"\n--- Starting Optuna Trial {trial.number} ---")
    param_str = (f"  Params: Arch(blks={num_conv_blocks}, flf={first_layer_filters}, fif={filter_increase_factor:.2f}), "
                 f"Train(lr={lr:.1e}, drp={dropout_rate:.2f}, wd={weight_decay:.1e}, opt={optimizer_name}")
    if optimizer_name == "SGD": param_str += f", sgd_mom={momentum:.2f}"
    if optimizer_name == "RMSprop": param_str += f", rmsp_alpha={alpha:.3f}"
    print(param_str + ")")

    try:
        # Verwende den generischen Trainingsloop
        _, _, _, best_trial_f1 = train_loop_generic(
            trial, model, train_loader, val_loader, criterion, optimizer, None, # Kein Scheduler für kurze Trials
            EPOCHS_PER_TRIAL, device_obj,
            f"OptunaTrial_{trial.number}", EXPERIMENT_OUTPUT_DIR_BASE, "dummy_optuna_model.pth",
            is_optuna_trial=True
        )
        return best_trial_f1
    except optuna.exceptions.TrialPruned:
        print(f"  Trial {trial.number} was pruned. Returning 0.0 as F1 score.")
        return 0.0
    except Exception as e:
        print(f"!! Optuna Trial {trial.number} failed with error: {e}")
        import traceback
        traceback.print_exc() # Gib den kompletten Traceback für den Fehler aus
        return -1.0 # Bei anderem Fehler


# --- 8. Hauptausführung mit Optuna ---
if __name__ == "__main__":
    # Sicherstellen, dass der Haupt-Experimentordner existiert
    EXPERIMENT_OUTPUT_DIR_BASE.mkdir(parents=True, exist_ok=True)

    # Daten aufteilen (nur einmal pro Skriptlauf, wenn nötig)
    if not BASE_DATA_DIR.exists() or not (TRAIN_DIR / 'yes').exists():
        print("Data split directory not found or incomplete, splitting data...")
        split_data(ANNOTATED_DIR, TRAIN_DIR, VAL_DIR, TRAIN_VAL_SPLIT, RANDOM_SEED)
    else:
        print(f"Using existing data split at: {BASE_DATA_DIR}")

    # DataLoaders erstellen
    train_loader, val_loader, class_to_idx = create_dataloaders(TRAIN_DIR, VAL_DIR, BATCH_SIZE, NUM_WORKERS)

    # Positives Gewicht für Loss-Funktion berechnen
    print("\nCalculating weight for positive class ('yes')...")
    try:
        num_no = len(glob.glob(str(TRAIN_DIR / 'no' / '*.png')))
        num_yes = len(glob.glob(str(TRAIN_DIR / 'yes' / '*.png')))
        pos_weight_value = 1.0 if num_yes == 0 or num_no == 0 else num_no / num_yes
        print(f"  Found {num_no} 'no', {num_yes} 'yes' samples in training set. Calculated pos_weight: {pos_weight_value:.2f}")
        pos_weight_tensor = torch.tensor([pos_weight_value], device=device)
    except Exception as e:
        print(f"  Error calculating pos_weight: {e}. Defaulting to 1.0.")
        pos_weight_tensor = torch.tensor([1.0], device=device)


    # --- Optuna Hyperparameter Suche ---
    print(f"\n--- Starting Optuna Hyperparameter Search ({N_TRIALS} Trials, {EPOCHS_PER_TRIAL} epochs/trial) ---")
    study_name = "cnn_tuning_study_v2_arch_opt" # Eindeutiger Name für die Studie (v2 mit Arch + Opt)
    storage_name = f"sqlite:///{OPTUNA_DB_PATH}" # Wiederverwendbare DB
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True,
                                direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=2)) # Robusterer Pruner

    # Optimiere
    study.optimize(lambda trial: objective(trial, train_loader, val_loader, pos_weight_tensor, device),
                   n_trials=N_TRIALS, timeout=None) # Timeout optional setzen (in Sekunden)

    print(f"\n--- Optuna Search Finished ---")
    print(f"Number of finished trials: {len(study.trials)}")

    if not study.best_trial or study.best_value <= -0.9: # <= -0.9 da F1 nicht negativ, -1.0 ist Fehler
        print("\nERROR: Optuna search did not find any valid/successful trials. Cannot proceed to final training.")
        print("Please check the Optuna logs and database for errors or pruning issues.")
        exit()

    print("Best trial found by Optuna:")
    best_trial = study.best_trial
    print(f"  Value (Best Validation F1): {best_trial.value:.4f}")
    print("  Best Parameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Optuna Visualisierungen speichern
    optuna_visuals_dir = EXPERIMENT_OUTPUT_DIR_BASE / "optuna_visualizations"
    optuna_visuals_dir.mkdir(parents=True, exist_ok=True)
    try:
        fig_hist = optuna.visualization.plot_optimization_history(study)
        fig_hist.write_image(str(optuna_visuals_dir / f"{study_name}_optuna_optimization_history.png"))
        fig_imp = optuna.visualization.plot_param_importances(study)
        fig_imp.write_image(str(optuna_visuals_dir / f"{study_name}_optuna_param_importances.png"))
        fig_slice = optuna.visualization.plot_slice(study)
        fig_slice.write_image(str(optuna_visuals_dir / f"{study_name}_optuna_slice_plot.png"))
        print(f"  Optuna visualizations saved to: {optuna_visuals_dir}")
    except Exception as e_vis:
        print(f"  ERROR plotting Optuna visualizations: {e_vis}. (Install plotly and kaleido if missing: pip install plotly kaleido)")


    # --- Finales Training mit den besten Hyperparametern ---
    print("\n--- Starting Final Training with Best Hyperparameters ---")
    best_params = best_trial.params

    # Modell mit den besten Architektur- UND Trainingsparametern instantiieren
    final_model = CustomCNN(
        num_classes=1,
        dropout_rate=best_params['dropout'], # Kommt von Optuna
        num_conv_blocks=best_params['num_conv_blocks'], # Kommt von Optuna
        first_layer_filters=best_params['first_layer_filters'], # Kommt von Optuna
        filter_increase_factor=best_params['filter_increase_factor'] # Kommt von Optuna
    ).to(device)

    final_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # Optimizer mit den besten Parametern erstellen
    final_optimizer_name = best_params.get("optimizer", "AdamW") # .get() mit Default, falls Optuna mal abbricht
    if final_optimizer_name == "AdamW":
        final_optimizer = optim.AdamW(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    elif final_optimizer_name == "SGD":
        final_optimizer = optim.SGD(final_model.parameters(), lr=best_params['lr'],
                                    momentum=best_params.get('sgd_momentum', 0.9), # .get() mit Default
                                    weight_decay=best_params['weight_decay'])
    elif final_optimizer_name == "RMSprop":
        final_optimizer = optim.RMSprop(final_model.parameters(), lr=best_params['lr'],
                                        alpha=best_params.get('rmsprop_alpha', 0.99))
    else: # Fallback, falls Optimizer-Name unerwartet ist
        print(f"Warning: Unknown optimizer '{final_optimizer_name}' from Optuna. Defaulting to AdamW.")
        final_optimizer = optim.AdamW(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])


    # Scheduler für finales Training (optional, aber oft gut)
    final_scheduler = optim.lr_scheduler.ReduceLROnPlateau(final_optimizer, mode='max', factor=0.2, patience=10, min_lr=1e-7) # Erhöhte Patience

    print(f"Training final model for {FINAL_TRAINING_EPOCHS} epochs with best params: {best_params}")
    trained_final_model, final_history, final_train_time, best_final_val_f1 = train_loop_generic(
        None, final_model, train_loader, val_loader, final_criterion, final_optimizer, final_scheduler,
        FINAL_TRAINING_EPOCHS, device,
        "FinalTunedModel", EXPERIMENT_OUTPUT_DIR_BASE, FINAL_MODEL_SAVE_PATH.name, # Nur Dateiname
        is_optuna_trial=False
    )

    # Trainingshistorie des finalen Modells plotten
    if final_history:
        plot_training_history(final_history, EXPERIMENT_OUTPUT_DIR_BASE, "FinalTunedModel_with_best_optuna_params")

    # --- Finale Evaluation des besten, getunten Modells ---
    print("\n--- Final Evaluation on Validation Set (using BEST TUNED and TRAINED model) ---")
    trained_final_model.eval()
    all_final_labels, all_final_outputs = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                outputs = trained_final_model(inputs)
            all_final_labels.append(labels)
            all_final_outputs.append(outputs)

    all_final_outputs_cat = torch.cat(all_final_outputs)
    all_final_labels_cat = torch.cat(all_final_labels)
    final_acc, final_prec, final_rec, final_f1 = calculate_metrics(all_final_outputs_cat, all_final_labels_cat)

    # Zusammenfassende Textausgabe
    summary_text = f"--- Experiment Summary ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---\n"
    summary_text += f"Device: {device}\n"
    summary_text += f"Data Split: {TRAIN_VAL_SPLIT*100:.0f}% Train / {(1-TRAIN_VAL_SPLIT)*100:.0f}% Val (Seed: {RANDOM_SEED})\n"
    summary_text += f"  Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}\n"
    summary_text += f"  Pos_weight for 'yes' class: {pos_weight_tensor.item():.2f}\n"
    summary_text += f"\nOptuna Search ({N_TRIALS} trials, {EPOCHS_PER_TRIAL} epochs/trial):\n"
    summary_text += f"  Study Name: {study_name}\n"
    summary_text += f"  Best Optuna Trial Value (Validation F1): {best_trial.value:.4f}\n"
    summary_text += f"  Best Optuna Parameters:\n"
    for key, value in best_trial.params.items():
        summary_text += f"    {key}: {value}\n"
    summary_text += f"\nFinal Model Training ({FINAL_TRAINING_EPOCHS} epochs with best params):\n"
    summary_text += f"  Model saved to: {FINAL_MODEL_SAVE_PATH}\n"
    summary_text += f"  Best Validation F1 during final training: {best_final_val_f1:.4f}\n"
    summary_text += f"  Training time for final model: {final_train_time // 60:.0f}m {final_train_time % 60:.0f}s\n"
    summary_text += f"\nFinal Evaluation on Validation Set (Best Tuned Model):\n"
    summary_text += f"  Accuracy: {final_acc:.4f}\n"
    summary_text += f"  Precision (yes): {final_prec:.4f}\n"
    summary_text += f"  Recall (yes): {final_rec:.4f}\n"
    summary_text += f"  F1-Score (yes): {final_f1:.4f}\n"

    final_preds_np = (torch.sigmoid(all_final_outputs_cat).detach().cpu().numpy() > 0.5).astype(int).flatten()
    final_labels_np = all_final_labels_cat.detach().cpu().numpy().flatten()
    cm = confusion_matrix(final_labels_np, final_preds_np)
    summary_text += "\nConfusion Matrix (Validation Set):\n"
    summary_text += f"Labels: {list(class_to_idx.keys())}\n"
    summary_text += str(cm) + "\n"
    summary_text += "\nClassification Report (Validation Set):\n"
    summary_text += classification_report(final_labels_np, final_preds_np, target_names=list(class_to_idx.keys()), zero_division=0) + "\n"

    print("\n" + "="*30 + " SCRIPT SUMMARY " + "="*30)
    print(summary_text)
    summary_file_path = EXPERIMENT_OUTPUT_DIR_BASE / "experiment_summary.txt"
    with open(summary_file_path, "w", encoding='utf-8') as f:
        f.write(summary_text)
    print(f"\nSummary saved to: {summary_file_path}")

    # Speichern der Fehleranalyse-Bilder (FN/FP)
    save_error_analysis_images(trained_final_model, val_loader, device, class_to_idx, EXPERIMENT_OUTPUT_DIR_BASE, "FinalTunedModel_with_best_optuna_params")

    print("\n--- Script Finished ---")