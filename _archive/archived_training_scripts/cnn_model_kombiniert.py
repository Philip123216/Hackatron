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
import optuna # Nur zum Laden der Studie, nicht zum Optimieren
from torch.cuda.amp import autocast, GradScaler

# --- 0. Projekt-Root und Haupt-Ausgabeordner definieren ---
# Annahme: Dieses Skript liegt in einem Unterordner wie 'model_training'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
print(f"Project Root: {PROJECT_ROOT}")

# Eindeutiger Name für dieses Retraining-Experiment
EXPERIMENT_NAME = "retrain_kombiniert_v1" # z.B. v1, v2 etc.

# Hauptordner für alle Ergebnisse dieses spezifischen Retraining-Laufs
# Wird unter einem neuen _archive Unterordner gespeichert, um alte Ergebnisse nicht zu überschreiben
CURRENT_EXPERIMENT_DIR = PROJECT_ROOT / "_archive" / f"experiment_{EXPERIMENT_NAME}_{time.strftime('%Y%m%d_%H%M%S')}"
CURRENT_EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Alle Ergebnisse für diesen Lauf werden in '{CURRENT_EXPERIMENT_DIR}' gespeichert.")


# --- 1. Globale Konfiguration ---
# WICHTIG: Pfad zum kombinierten Datensatz, der ALLE Bilder (alt + neu verifiziert) enthält
ANNOTATED_DIR = PROJECT_ROOT / "data_annotated_kombiniert" # Passe diesen Namen ggf. an

# Datenordner für diesen spezifischen Retraining-Lauf (wird neu erstellt)
BASE_DATA_DIR = CURRENT_EXPERIMENT_DIR / "data_split"
TRAIN_DIR = BASE_DATA_DIR / "train"
VAL_DIR = BASE_DATA_DIR / "validation"

# Pfad, wo das neu trainierte Modell gespeichert wird
FINAL_MODEL_SAVE_PATH = CURRENT_EXPERIMENT_DIR / f"best_model_{EXPERIMENT_NAME}.pth"

# WICHTIG: Pfad zu deiner existierenden Optuna SQLite Datenbank, um die besten Parameter zu laden
OPTUNA_DB_PATH_FOR_BEST_PARAMS = PROJECT_ROOT / "_archive" / "optuna_experiment_results" / "optuna_study_long.db"
# Überprüfe, ob dies der korrekte Name deiner vorherigen Studie ist
# In deinem Trainingsskript war es: "cnn_tuning_study_v2_arch_opt"
OPTUNA_STUDY_NAME_TO_LOAD = "cnn_tuning_study_v2_arch_opt"


# Trainingsparameter
BATCH_SIZE = 128 # Dieser Wert kann beibehalten oder basierend auf GPU-Speicher angepasst werden
NUM_WORKERS = 8
# Erhöhe die Epochenanzahl, da wir Early Stopping verwenden
FINAL_TRAINING_EPOCHS = 150 # z.B. 100-200, Early Stopping wird es ggf. früher beenden
EARLY_STOPPING_PATIENCE = 15 # Anzahl der Epochen ohne Verbesserung, bevor gestoppt wird

# Bildparameter und Seeds
IMG_SIZE = 250
RANDOM_SEED = 42
TRAIN_VAL_SPLIT = 0.8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Reproduzierbarkeit (wie gehabt)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# --- 2. Datenaufteilung (Bleibt gleich) ---
def split_data(source_dir, train_dir, val_dir, split_ratio=0.8, seed=42):
    print(f"Splitting data from {source_dir} into {train_dir.parent}...")
    random.seed(seed)
    # Wichtig: Vor dem Splitten sicherstellen, dass der Zielordner (BASE_DATA_DIR) leer ist oder nicht existiert
    if train_dir.parent.exists():
        print(f"  Removing existing data directory: {train_dir.parent}")
        shutil.rmtree(train_dir.parent)
    # Die Ordner werden innerhalb von BASE_DATA_DIR neu erstellt
    train_dir.parent.mkdir(parents=True, exist_ok=True)

    for class_name in ['yes', 'no']:
        source_class_dir = source_dir / class_name
        # Die Unterordner train/yes, train/no etc. werden hier erstellt
        current_train_class_dir = train_dir / class_name
        current_val_class_dir = val_dir / class_name
        current_train_class_dir.mkdir(parents=True, exist_ok=True)
        current_val_class_dir.mkdir(parents=True, exist_ok=True)

        if not source_class_dir.exists():
            print(f"  WARNUNG: Quellordner {source_class_dir} für Klasse '{class_name}' nicht gefunden. Überspringe."); continue
        images = glob.glob(str(source_class_dir / "*.png"))
        if not images:
            print(f"  WARNUNG: Keine PNGs in {source_class_dir} für Klasse '{class_name}' gefunden. Überspringe."); continue

        random.shuffle(images)
        split_idx = int(len(images) * split_ratio)
        train_images, val_images = images[:split_idx], images[split_idx:]
        print(f"  Cls '{class_name}': {len(images)} tot -> {len(train_images)} tr, {len(val_images)} vl")
        for img_path_str in train_images: shutil.copy(Path(img_path_str), current_train_class_dir / Path(img_path_str).name)
        for img_path_str in val_images: shutil.copy(Path(img_path_str), current_val_class_dir / Path(img_path_str).name)
    print("Data splitting complete.")


# --- 3. Datentransformationen & Laden (Bleibt gleich) ---
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.ToTensor(), normalize
])
val_transforms = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), normalize])

def create_dataloaders(train_dir, val_dir, batch_size, num_workers):
    # (Code wie gehabt)
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    persistent = num_workers > 0 and device.type == 'cuda'
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent)
    print(f"\nDataLoaders created: Train={len(train_dataset)} ({len(train_loader)} batches), Val={len(val_dataset)} ({len(val_loader)} batches)")
    print(f"  Classes: {train_dataset.classes} (Mapping: {train_dataset.class_to_idx})")
    if train_dataset.class_to_idx.get('yes', -1) != 1 or train_dataset.class_to_idx.get('no', -1) != 0:
       print(f"\n*** WARNING: Class mapping incorrect! Expected {{'no': 0, 'yes': 1}} Actual: {train_dataset.class_to_idx}\n")
    return train_loader, val_loader, train_dataset.class_to_idx

# --- 4. Modell Definition (EXAKT wie im vorherigen Training) ---
class CustomCNN(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.5,
                 num_conv_blocks=4, first_layer_filters=64, filter_increase_factor=2.0):
        super(CustomCNN, self).__init__()
        self.conv_blocks = nn.ModuleList()
        current_channels = 3; next_channels = first_layer_filters
        for i in range(num_conv_blocks):
            block = nn.Sequential(
                nn.Conv2d(current_channels, next_channels, kernel_size=3, stride=1, padding='same', bias=False),
                nn.BatchNorm2d(next_channels), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.conv_blocks.append(block)
            current_channels = next_channels
            if i < num_conv_blocks - 1: next_channels = int(current_channels * filter_increase_factor)
            next_channels = max(next_channels, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1); self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(current_channels, num_classes)
        self._initialize_weights() # Wichtig für Training von Grund auf
    def forward(self, x):
        for block in self.conv_blocks: x = block(x)
        x = self.avgpool(x); x = self.flatten(x); x = self.dropout(x); x = self.fc(x)
        return x
    def _initialize_weights(self): # (Code wie gehabt)
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0, 0.01); nn.init.constant_(m.bias, 0)

# --- 5. Metrik-Berechnung (Bleibt gleich) ---
def calculate_metrics(outputs, labels): # (Code wie gehabt)
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

# --- Funktion zum Speichern der Fehleranalyse-Bilder (Bleibt gleich) ---
def save_error_analysis_images(model, val_loader, device_obj, class_to_idx, output_dir_session, experiment_tag): # (Code wie gehabt)
    print(f"  Saving error analysis images for {experiment_tag} to {output_dir_session}...")
    fn_dir = output_dir_session / "false_negatives"
    fp_dir = output_dir_session / "false_positives"
    fn_dir.mkdir(parents=True, exist_ok=True)
    fp_dir.mkdir(parents=True, exist_ok=True)
    model.eval(); yes_idx = class_to_idx.get('yes', 1); no_idx = class_to_idx.get('no', 0)
    fn_count = 0; fp_count = 0; filepaths = [s[0] for s in val_loader.dataset.samples]
    true_labels_indices = [s[1] for s in val_loader.dataset.samples]; all_predictions_indices = []
    use_amp_eval = (device_obj.type == 'cuda')
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device_obj, non_blocking=True)
            with torch.amp.autocast(device_type=device_obj.type, enabled=use_amp_eval): outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().flatten().tolist(); all_predictions_indices.extend(preds)
    for i, img_filepath_str in enumerate(filepaths):
        img_filepath = Path(img_filepath_str); true_label_idx = true_labels_indices[i]; pred_label_idx = all_predictions_indices[i]
        if true_label_idx == yes_idx and pred_label_idx == no_idx: shutil.copy(img_filepath, fn_dir / img_filepath.name); fn_count += 1
        elif true_label_idx == no_idx and pred_label_idx == yes_idx: shutil.copy(img_filepath, fp_dir / img_filepath.name); fp_count += 1
    print(f"    False Negatives saved: {fn_count}, False Positives saved: {fp_count}")

# --- Funktion zum Plotten der Trainingshistorie (Bleibt gleich) ---
def plot_training_history(history, output_dir_session, experiment_tag): # (Code wie gehabt)
    print(f"  Plotting training history for {experiment_tag} to {output_dir_session}...")
    if not history or 'val_f1' not in history or not history['val_f1']: print("    Not enough history data to plot."); return
    epochs_ran = history.get('epoch', range(1, len(history['val_f1']) + 1));
    if not epochs_ran: print("    Epoch data missing."); return
    plt.figure(figsize=(18, 12)); plt.subplot(2, 2, 1)
    if 'val_f1' in history: plt.plot(epochs_ran, history['val_f1'], 'b-', label='Val F1')
    if 'val_acc' in history: plt.plot(epochs_ran, history['val_acc'], 'c-', label='Val Acc')
    if 'val_precision' in history: plt.plot(epochs_ran, history['val_precision'], 'm-', label='Val Precision')
    if 'val_recall' in history: plt.plot(epochs_ran, history['val_recall'], 'y-', label='Val Recall')
    plt.title(f'{experiment_tag} - Validation Metrics'); plt.xlabel('Epoch'); plt.ylabel('Score'); plt.legend(); plt.grid(True)
    plt.subplot(2, 2, 2)
    if 'train_loss' in history: plt.plot(epochs_ran, history['train_loss'], 'r-', label='Train Loss')
    if 'val_loss' in history: plt.plot(epochs_ran, history['val_loss'], 'g-', label='Val Loss')
    plt.title(f'{experiment_tag} - Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.subplot(2, 2, 3)
    if 'train_f1' in history: plt.plot(epochs_ran, history['train_f1'], 'r-', label='Train F1')
    if 'val_f1' in history: plt.plot(epochs_ran, history['val_f1'], 'b-', label='Val F1')
    plt.title(f'{experiment_tag} - F1 Score (Train vs Val)'); plt.xlabel('Epoch'); plt.ylabel('F1 Score'); plt.legend(); plt.grid(True)
    plt.subplot(2, 2, 4)
    if 'time_per_epoch' in history: plt.plot(epochs_ran, history['time_per_epoch'], 'k-', label='Time/Epoch (s)')
    plt.title(f'{experiment_tag} - Time per Epoch'); plt.xlabel('Epoch'); plt.ylabel('Seconds'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plot_save_path = output_dir_session / f"{experiment_tag}_training_history.png"
    plt.savefig(plot_save_path); plt.close(); print(f"    Training history plot saved to {plot_save_path}")


# --- 6. Trainings-Loop (mit Early Stopping) ---
def train_loop_retrain(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device_obj,
                       loop_name_tag, current_experiment_dir, model_save_filename, patience): # Hinzugefügt: patience
    print(f"\n--- Starting Retraining Loop: {loop_name_tag} (max {num_epochs} epochs, EarlyStopping Patience: {patience}) ---")
    model_full_save_path = current_experiment_dir / model_save_filename # Speichern direkt im aktuellen Experiment-Ordner
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_f1 = 0.0
    history = {'epoch':[], 'train_loss':[], 'train_f1':[], 'val_loss':[], 'val_f1':[], 'val_acc':[], 'val_precision':[], 'val_recall':[], 'lr':[], 'time_per_epoch':[]}

    # Für Early Stopping
    early_stopping_counter = 0
    best_val_metric_for_early_stopping = 0.0 # Wir optimieren auf F1

    use_amp = (device_obj.type == 'cuda')
    scaler = GradScaler(enabled=use_amp)
    if use_amp: print(f"  AMP (Mixed Precision) enabled for '{loop_name_tag}'.")
    total_training_start_time = time.time()

    for epoch in range(num_epochs):
        # (Trainings- und Validierungsphasen Code wie gehabt)
        epoch_start_time = time.time(); print(f"\n  Epoch {epoch+1}/{num_epochs} for '{loop_name_tag}'")
        model.train(); running_loss = 0.0; all_train_labels_epoch, all_train_outputs_epoch = [], []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device_obj, non_blocking=True), labels.float().unsqueeze(1).to(device_obj, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device_obj.type, enabled=use_amp): outputs = model(inputs); loss = criterion(outputs, labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running_loss += loss.item() * inputs.size(0); all_train_labels_epoch.append(labels.detach()); all_train_outputs_epoch.append(outputs.detach())
        epoch_train_loss = running_loss / len(train_loader.dataset); _, _, _, epoch_train_f1 = calculate_metrics(torch.cat(all_train_outputs_epoch), torch.cat(all_train_labels_epoch))
        print(f"    Train Loss: {epoch_train_loss:.4f}, Train F1: {epoch_train_f1:.4f}")
        model.eval(); running_val_loss = 0.0; all_val_labels_epoch, all_val_outputs_epoch = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device_obj, non_blocking=True), labels.float().unsqueeze(1).to(device_obj, non_blocking=True)
                with torch.amp.autocast(device_type=device_obj.type, enabled=use_amp): outputs = model(inputs); val_batch_loss = criterion(outputs, labels)
                running_val_loss += val_batch_loss.item() * inputs.size(0); all_val_labels_epoch.append(labels.detach()); all_val_outputs_epoch.append(outputs.detach())
        epoch_val_loss = running_val_loss / len(val_loader.dataset); val_acc, val_prec, val_rec, val_f1 = calculate_metrics(torch.cat(all_val_outputs_epoch), torch.cat(all_val_labels_epoch))
        print(f"    Val Loss: {epoch_val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} P: {val_prec:.4f} R: {val_rec:.4f}")
        current_lr = optimizer.param_groups[0]['lr']; epoch_duration = time.time() - epoch_start_time
        history['epoch'].append(epoch + 1); history['train_loss'].append(epoch_train_loss); history['train_f1'].append(epoch_train_f1)
        history['val_loss'].append(epoch_val_loss); history['val_f1'].append(val_f1); history['val_acc'].append(val_acc)
        history['val_precision'].append(val_prec); history['val_recall'].append(val_rec); history['lr'].append(current_lr); history['time_per_epoch'].append(epoch_duration)
        print(f"    LR: {current_lr:.1e}, Time: {epoch_duration:.2f}s")

        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau): scheduler.step(val_f1)
            else: scheduler.step()

        # Early Stopping Logik
        if val_f1 > best_val_metric_for_early_stopping:
            best_val_metric_for_early_stopping = val_f1
            early_stopping_counter = 0
            if val_f1 > best_val_f1: # Nur speichern, wenn es insgesamt das Beste ist
                print(f"    Best Val F1 for '{loop_name_tag}' improved ({best_val_f1:.4f} -> {val_f1:.4f}). Saving model to {model_full_save_path.name}...")
                best_val_f1 = val_f1
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, model_full_save_path)
        else:
            early_stopping_counter += 1
            print(f"    EarlyStopping: No Val F1 improvement for {early_stopping_counter}/{patience} epochs.")

        if early_stopping_counter >= patience:
            print(f"  EARLY STOPPING triggered at epoch {epoch+1} for '{loop_name_tag}'!")
            break
    # (Rest des train_loop_generic wie gehabt, aber ohne Optuna-spezifische Teile)
    total_training_time = time.time() - total_training_start_time
    print(f"\n  Training for '{loop_name_tag}' complete in {total_training_time // 60:.0f}m {total_training_time % 60:.0f}s")
    print(f"  Best Validation F1 achieved: {best_val_f1:.4f}")
    if best_val_f1 > 0 or model_full_save_path.exists():
        print(f"  Loading best weights from {model_full_save_path.name} for '{loop_name_tag}'.")
        model.load_state_dict(torch.load(model_full_save_path, map_location=device_obj))
    else:
        print(f"  WARNING: No best model saved for '{loop_name_tag}'. Using last model state."); model.load_state_dict(best_model_wts)
    return model, history, total_training_time, best_val_f1

# --- 7. Funktion zum Laden der besten Parameter aus Optuna-Studie ---
def load_best_optuna_params(study_db_path, study_name_to_load):
    print(f"\nLade beste Parameter aus Optuna-Studie: {study_name_to_load} von DB: {study_db_path}")
    if not study_db_path.exists():
        print(f"FEHLER: Optuna DB-Datei nicht gefunden: {study_db_path}")
        return None
    try:
        storage_name = f"sqlite:///{study_db_path}"
        study = optuna.load_study(study_name=study_name_to_load, storage=storage_name)
        best_params = study.best_trial.params
        print("Beste Parameter erfolgreich aus Optuna-Studie geladen:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        return best_params
    except Exception as e:
        print(f"FEHLER beim Laden der besten Parameter aus der Optuna-Studie: {e}")
        print("Stelle sicher, dass der study_name und der Pfad zur DB korrekt sind.")
        return None

# --- 8. Hauptausführung (JETZT FÜR RETRAINING) ---
if __name__ == "__main__":
    # Sicherstellen, dass der Haupt-Experimentordner existiert
    CURRENT_EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    # Daten aufteilen (mit dem neuen ANNOTATED_DIR)
    if not BASE_DATA_DIR.exists() or not (TRAIN_DIR / 'yes').exists() or not (VAL_DIR / 'yes').exists():
        print(f"Daten-Split für '{EXPERIMENT_NAME}' nicht gefunden oder unvollständig, führe Split durch...")
        split_data(ANNOTATED_DIR, TRAIN_DIR, VAL_DIR, TRAIN_VAL_SPLIT, RANDOM_SEED)
    else:
        print(f"Verwende existierenden Daten-Split für '{EXPERIMENT_NAME}' unter: {BASE_DATA_DIR}")

    # DataLoaders erstellen
    train_loader, val_loader, class_to_idx = create_dataloaders(TRAIN_DIR, VAL_DIR, BATCH_SIZE, NUM_WORKERS)

    # Positives Gewicht für Loss-Funktion berechnen (wie gehabt)
    print("\nBerechne Gewicht für positive Klasse ('yes')...")
    try:
        num_no = len(glob.glob(str(TRAIN_DIR / 'no' / '*.png'))); num_yes = len(glob.glob(str(TRAIN_DIR / 'yes' / '*.png')))
        pos_weight_value = 1.0 if num_yes == 0 or num_no == 0 else num_no / num_yes
        print(f"  {num_no} 'no', {num_yes} 'yes' im Training. Pos_weight: {pos_weight_value:.2f}")
        pos_weight_tensor = torch.tensor([pos_weight_value], device=device)
    except Exception as e: print(f"  Fehler pos_weight: {e}. Default 1.0."); pos_weight_tensor = torch.tensor([1.0], device=device)

    # --- Lade beste Hyperparameter aus vorheriger Optuna-Studie ---
    best_params = load_best_optuna_params(OPTUNA_DB_PATH_FOR_BEST_PARAMS, OPTUNA_STUDY_NAME_TO_LOAD)
    if best_params is None:
        print("Konnte beste Parameter nicht laden. Breche Retraining ab.")
        exit()

    # --- Retraining mit den besten Hyperparametern ---
    print(f"\n--- Starte Retraining ({EXPERIMENT_NAME}) mit geladenen besten Hyperparametern ---")

    # Modell mit den besten Architektur-Parametern instantiieren
    retrain_model = CustomCNN(
        num_classes=1,
        dropout_rate=best_params['dropout'],
        num_conv_blocks=best_params['num_conv_blocks'],
        first_layer_filters=best_params['first_layer_filters'],
        filter_increase_factor=best_params['filter_increase_factor']
    ).to(device)

    retrain_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # Optimizer mit den besten Trainings-Parametern erstellen
    final_optimizer_name = best_params.get("optimizer", "AdamW")
    if final_optimizer_name == "AdamW":
        retrain_optimizer = optim.AdamW(retrain_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    elif final_optimizer_name == "SGD":
        retrain_optimizer = optim.SGD(retrain_model.parameters(), lr=best_params['lr'], momentum=best_params.get('sgd_momentum', 0.9), weight_decay=best_params['weight_decay'])
    elif final_optimizer_name == "RMSprop":
        retrain_optimizer = optim.RMSprop(retrain_model.parameters(), lr=best_params['lr'], alpha=best_params.get('rmsprop_alpha', 0.99))
    else:
        print(f"Warnung: Unbekannter Optimizer '{final_optimizer_name}'. Verwende AdamW."); retrain_optimizer = optim.AdamW(retrain_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])

    retrain_scheduler = optim.lr_scheduler.ReduceLROnPlateau(retrain_optimizer, mode='max', factor=0.2, patience=7, min_lr=1e-7) # Patience für Scheduler ggf. anpassen

    print(f"Trainiere Modell '{EXPERIMENT_NAME}' für max. {FINAL_TRAINING_EPOCHS} Epochen mit Parametern: {best_params}")
    trained_retrain_model, retrain_history, retrain_time, best_retrain_val_f1 = train_loop_retrain(
        retrain_model, train_loader, val_loader, retrain_criterion, retrain_optimizer, retrain_scheduler,
        FINAL_TRAINING_EPOCHS, device,
        f"RetrainedModel_{EXPERIMENT_NAME}", CURRENT_EXPERIMENT_DIR, FINAL_MODEL_SAVE_PATH.name, # Nur Dateiname
        patience=EARLY_STOPPING_PATIENCE
    )

    if retrain_history:
        plot_training_history(retrain_history, CURRENT_EXPERIMENT_DIR, f"RetrainedModel_{EXPERIMENT_NAME}")

    # --- Finale Evaluation des neu trainierten Modells ---
    print(f"\n--- Finale Evaluation ({EXPERIMENT_NAME}) auf Validierungsset ---")
    trained_retrain_model.eval()
    all_final_labels, all_final_outputs = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')): outputs = trained_retrain_model(inputs)
            all_final_labels.append(labels); all_final_outputs.append(outputs)
    all_final_outputs_cat = torch.cat(all_final_outputs); all_final_labels_cat = torch.cat(all_final_labels)
    final_acc, final_prec, final_rec, final_f1 = calculate_metrics(all_final_outputs_cat, all_final_labels_cat)

    # (Rest der Summary-Ausgabe und Fehleranalyse wie gehabt, angepasst an CURRENT_EXPERIMENT_DIR und EXPERIMENT_NAME)
    summary_text = f"--- Retraining Experiment Summary ({EXPERIMENT_NAME} - {time.strftime('%Y-%m-%d %H:%M:%S')}) ---\n"
    summary_text += f"Device: {device}\nData Source: {ANNOTATED_DIR}\n"
    summary_text += f"Data Split: {TRAIN_VAL_SPLIT*100:.0f}% Train / {(1-TRAIN_VAL_SPLIT)*100:.0f}% Val (Seed: {RANDOM_SEED})\n"
    summary_text += f"  Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}\n"
    summary_text += f"  Pos_weight for 'yes' class: {pos_weight_tensor.item():.2f}\n"
    summary_text += f"\nGeladene beste Optuna Parameter (von Studie '{OPTUNA_STUDY_NAME_TO_LOAD}'):\n"
    for key, value in best_params.items(): summary_text += f"    {key}: {value}\n"
    summary_text += f"\nRetraining ({FINAL_TRAINING_EPOCHS} max. Epochen, EarlyStopping Patience: {EARLY_STOPPING_PATIENCE}):\n"
    summary_text += f"  Modell gespeichert nach: {FINAL_MODEL_SAVE_PATH}\n"
    summary_text += f"  Bester Validierungs-F1 während Retraining: {best_retrain_val_f1:.4f}\n"
    summary_text += f"  Trainingszeit: {retrain_time // 60:.0f}m {retrain_time % 60:.0f}s\n"
    summary_text += f"\nFinale Evaluation auf Validierungsset:\n"
    summary_text += f"  Accuracy: {final_acc:.4f}\nPrecision (yes): {final_prec:.4f}\nRecall (yes): {final_rec:.4f}\nF1-Score (yes): {final_f1:.4f}\n"
    final_preds_np = (torch.sigmoid(all_final_outputs_cat).detach().cpu().numpy() > 0.5).astype(int).flatten()
    final_labels_np = all_final_labels_cat.detach().cpu().numpy().flatten()
    cm = confusion_matrix(final_labels_np, final_preds_np); summary_text += "\nConfusion Matrix:\n"; summary_text += f"Labels: {list(class_to_idx.keys())}\n"; summary_text += str(cm) + "\n"
    summary_text += "\nClassification Report:\n"; summary_text += classification_report(final_labels_np, final_preds_np, target_names=list(class_to_idx.keys()), zero_division=0) + "\n"
    print("\n" + "="*30 + f" SCRIPT SUMMARY ({EXPERIMENT_NAME}) " + "="*30); print(summary_text)
    summary_file_path = CURRENT_EXPERIMENT_DIR / f"experiment_summary_{EXPERIMENT_NAME}.txt"
    with open(summary_file_path, "w", encoding='utf-8') as f: f.write(summary_text)
    print(f"\nSummary gespeichert nach: {summary_file_path}")
    save_error_analysis_images(trained_retrain_model, val_loader, device, class_to_idx, CURRENT_EXPERIMENT_DIR, f"RetrainedModel_{EXPERIMENT_NAME}")
    print("\n--- Script Finished ---") 