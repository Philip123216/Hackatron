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

# --- 0. Projekt-Root definieren ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent # Annahme: Skript liegt im Projekt-Root
print(f"Project Root (angenommen): {PROJECT_ROOT}")

# --- 1. Globale Konfiguration ---
ANNOTATED_DIR = PROJECT_ROOT / "data_annotated"
BASE_DATA_DIR_FOR_RUN = PROJECT_ROOT / "data_split_for_run_v3" # Eigener Datenordner für diesen Lauf
TRAIN_DIR_RUN = BASE_DATA_DIR_FOR_RUN / "train"
VAL_DIR_RUN = BASE_DATA_DIR_FOR_RUN / "validation"
EXPERIMENTS_V3_OUTPUT_DIR = PROJECT_ROOT / "experiment_v3_results" # Hauptordner für Ergebnisse

TRAIN_VAL_SPLIT = 0.8
RANDOM_SEED = 42
IMG_SIZE = 250
DEFAULT_BATCH_SIZE = 128 # Kann pro Experiment/Optuna überschrieben werden
DEFAULT_NUM_WORKERS = 8  # Kann pro Experiment/Optuna überschrieben werden

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Reproduzierbarkeit
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    # torch.backends.cudnn.deterministic = True # Kann langsamer machen
    # torch.backends.cudnn.benchmark = False   # Kann langsamer machen

# --- Hilfsfunktionen (Daten, Modell, Metriken, Plotting, Fehleranalyse) ---

def split_data(source_dir: Path, train_dir: Path, val_dir: Path, split_ratio=0.8, seed=42):
    # (Code aus deinem run_experiments.py, leicht angepasst)
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

        if not source_class_dir.exists():
            print(f"  Warning: Source directory {source_class_dir} not found for class {class_name}. Skipping.")
            continue
        images = glob.glob(str(source_class_dir / "*.png"))
        if not images:
            print(f"  Warning: No PNG images found in {source_class_dir} for class {class_name}. Skipping.")
            continue
        random.shuffle(images)
        split_idx = int(len(images) * split_ratio)
        train_images, val_images = images[:split_idx], images[split_idx:]
        print(f"  Class '{class_name}': {len(images)} total -> {len(train_images)} train, {len(val_images)} val")
        for img_path_str in train_images:
            img_path_obj = Path(img_path_str)
            shutil.copy(img_path_obj, train_class_dir / img_path_obj.name)
        for img_path_str in val_images:
            img_path_obj = Path(img_path_str)
            shutil.copy(img_path_obj, val_class_dir / img_path_obj.name)
    print("Data splitting complete.")


def create_dataloaders(train_dir, val_dir, batch_size, num_workers, train_transforms, val_transforms):
    # (Code aus deinem run_experiments.py, leicht angepasst)
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    print(f"Using {num_workers} workers for DataLoaders.")
    persistent = num_workers > 0 and device.type == 'cuda' # persistent_workers nur mit GPU und workers > 0 sinnvoll
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent, drop_last=True if len(train_dataset) % batch_size == 1 else False) # Drop last wenn nur 1 sample übrig bleibt
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent)
    print(f"  DataLoaders: Train={len(train_dataset)} ({len(train_loader)} batches), Val={len(val_dataset)} ({len(val_loader)} batches)")
    print(f"  Classes: {train_dataset.classes} (Mapping: {train_dataset.class_to_idx})")
    if train_dataset.class_to_idx.get('yes', -1) != 1 or train_dataset.class_to_idx.get('no', -1) != 0:
       print(f"\n*** WARNING: Class mapping might be incorrect! Expected {{'no':0, 'yes':1}}. Actual: {train_dataset.class_to_idx}\n")
    return train_loader, val_loader, train_dataset.class_to_idx


def calculate_pos_weight(train_dir, device_obj):
    # (Code aus deinem run_experiments.py)
    try:
        n_no=len(glob.glob(str(train_dir/'no'/'*.png'))); n_yes=len(glob.glob(str(train_dir/'yes'/'*.png')))
        if n_yes == 0: # Division durch Null vermeiden
            pw = 1.0
            print(f"  Warning: No 'yes' samples found for pos_weight calculation. Defaulting to 1.0.")
        else:
            pw = n_no / n_yes
        print(f"  Samples for pos_weight: {n_no} 'no', {n_yes} 'yes'. Calculated pos_weight: {pw:.2f}")
        return torch.tensor([pw], device=device_obj)
    except Exception as e:
        print(f"  Error calculating pos_weight: {e}. Defaulting to 1.0.")
        return torch.tensor([1.0], device=device_obj)


class CustomCNN(nn.Module):
    # (Code aus deinem run_experiments.py, leicht angepasst für Initialisierung)
    def __init__(self, num_classes=1, dropout_rate=0.5, num_conv_blocks=4):
        super(CustomCNN, self).__init__()
        def _make_block(in_c, out_c): return nn.Sequential(nn.Conv2d(in_c,out_c,3,1,'same',bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True), nn.MaxPool2d(2,2))
        self.conv_blocks = nn.ModuleList()
        current_channels = 3
        block_channels = [64, 128, 256, 512, 512, 512] # Max 6 Blöcke vordefiniert
        actual_num_blocks = min(num_conv_blocks, len(block_channels))
        for i in range(actual_num_blocks):
            self.conv_blocks.append(_make_block(current_channels, block_channels[i]))
            current_channels = block_channels[i]
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(current_channels, num_classes)
        self._initialize_weights()
    def forward(self, x):
        for block in self.conv_blocks: x = block(x)
        return self.fc(self.dropout(self.flatten(self.avgpool(x))))
    def _initialize_weights(self): # Aus deinem tune_and_train_cnn.py
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0, 0.01); nn.init.constant_(m.bias, 0)


def calculate_metrics(outputs, labels):
    # (Code aus deinem tune_and_train_cnn.py, leicht robuster gemacht)
    try:
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        preds = (probs > 0.5).astype(int).flatten()
        labels_np = labels.detach().cpu().numpy().flatten()
        acc = accuracy_score(labels_np, preds)
        precision = precision_score(labels_np, preds, average='binary', pos_label=1, zero_division=0)
        recall = recall_score(labels_np, preds, average='binary', pos_label=1, zero_division=0)
        f1 = f1_score(labels_np, preds, average='binary', pos_label=1, zero_division=0)
    except Exception as e:
        print(f"  Error calculating metrics: {e}")
        acc, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
    return acc, precision, recall, f1


def save_error_analysis(model, val_loader, device_obj, class_to_idx, output_dir, experiment_name):
    # (Code aus deinem run_experiments.py, leicht angepasst)
    print(f"  Saving error analysis for {experiment_name} to {output_dir}...")
    fn_dir=output_dir/"false_negatives"; fp_dir=output_dir/"false_positives"
    fn_dir.mkdir(parents=True, exist_ok=True); fp_dir.mkdir(parents=True, exist_ok=True)
    model.eval(); yes_idx=class_to_idx.get('yes',1); no_idx=class_to_idx.get('no',0); fn_c=0; fp_c=0
    filepaths=[s[0] for s in val_loader.dataset.samples]
    true_labels=[s[1] for s in val_loader.dataset.samples]
    preds_list=[]
    use_amp_eval = (device_obj.type == 'cuda')
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs=inputs.to(device_obj, non_blocking=True)
            with autocast(enabled=use_amp_eval): outputs=model(inputs)
            preds_list.extend((torch.sigmoid(outputs)>0.5).int().cpu().flatten().tolist())
    for i, fp_str in enumerate(filepaths):
        fp=Path(fp_str)
        if true_labels[i]==yes_idx and preds_list[i]==no_idx: shutil.copy(fp, fn_dir/fp.name); fn_c+=1
        elif true_labels[i]==no_idx and preds_list[i]==yes_idx: shutil.copy(fp, fp_dir/fp.name); fp_c+=1
    print(f"    FNs saved: {fn_c}, FPs saved: {fp_c}")


def plot_experiment_history(history, output_dir, experiment_name):
    # (Code aus deinem run_experiments.py)
    print(f"  Plotting training history for {experiment_name} to {output_dir}...")
    if not history or 'val_f1' not in history or not history['val_f1']: print("    No history to plot or missing 'val_f1'."); return
    epochs = history.get('epoch', range(1, len(history['val_f1']) + 1))
    plt.figure(figsize=(12,8)); plt.subplot(2,1,1)
    if 'val_f1' in history: plt.plot(epochs, history['val_f1'], 'b-', label='Val F1');
    if 'val_acc' in history: plt.plot(epochs, history['val_acc'], 'c-', label='Val Acc')
    if 'val_precision' in history: plt.plot(epochs, history['val_precision'], 'm-', label='Val Precision')
    if 'val_recall' in history: plt.plot(epochs, history['val_recall'], 'y-', label='Val Recall')
    plt.legend(); plt.title(f'{experiment_name} - Validation Metrics'); plt.grid(True); plt.subplot(2,1,2)
    if 'train_loss' in history: plt.plot(epochs, history['train_loss'], 'r-', label='Train Loss');
    if 'val_loss' in history: plt.plot(epochs, history['val_loss'], 'g-', label='Val Loss')
    plt.legend(); plt.title(f'{experiment_name} - Loss'); plt.xlabel('Epoch'); plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"{experiment_name}_training_history.png"); plt.close()


# --- Trainings-Loop für ein MANUELLES Experiment (aus run_experiments.py) ---
def train_manual_experiment_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device_obj, experiment_name, experiment_dir_path):
    # (Code aus deinem train_experiment_loop in run_experiments.py, leicht angepasst)
    print(f"\n--- Training Manual Experiment: {experiment_name} ({num_epochs} epochs) ---")
    model_save_path = experiment_dir_path / f"{experiment_name}_best_model.pth"
    best_model_wts = copy.deepcopy(model.state_dict()); best_val_f1 = 0.0
    history = {'epoch':[],'train_loss':[],'val_loss':[],'val_f1':[],'val_acc':[],'val_precision':[],'val_recall':[],'time_per_epoch':[]}
    use_amp = (device_obj.type == 'cuda'); scaler = GradScaler(enabled=use_amp)
    if use_amp: print("  AMP (Mixed Precision) enabled for manual experiment.")
    total_train_time_start = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time(); print(f"\n  Manual Exp. Epoch {epoch+1}/{num_epochs}")
        model.train(); running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device_obj,non_blocking=True), labels.float().unsqueeze(1).to(device_obj,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp): outputs = model(inputs); loss = criterion(outputs, labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running_loss += loss.item() * inputs.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset); print(f"    Train Loss: {epoch_train_loss:.4f}")

        model.eval(); running_val_loss = 0.0; all_val_labels, all_val_outputs = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device_obj,non_blocking=True), labels.float().unsqueeze(1).to(device_obj,non_blocking=True)
                with autocast(enabled=use_amp): outputs = model(inputs); loss = criterion(outputs, labels) # Loss auch hier berechnen
                running_val_loss += loss.item()*inputs.size(0); all_val_labels.append(labels); all_val_outputs.append(outputs)
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_acc,val_prec,val_rec,val_f1 = calculate_metrics(torch.cat(all_val_outputs), torch.cat(all_val_labels))
        print(f"    Val Loss: {epoch_val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} P: {val_prec:.4f} R: {val_rec:.4f}")

        epoch_time = time.time() - epoch_start_time
        history['epoch'].append(epoch+1)
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_f1'].append(val_f1)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_rec)
        history['time_per_epoch'].append(epoch_time)

        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau): scheduler.step(val_f1) # Monitor F1
            else: scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"    Current LR: {current_lr:.1e}")


        if val_f1 > best_val_f1:
            print(f"    Best Val F1 improved ({best_val_f1:.4f} -> {val_f1:.4f}). Saving model...");
            best_val_f1=val_f1; best_model_wts=copy.deepcopy(model.state_dict()); torch.save(best_model_wts,model_save_path)

    total_time = time.time() - total_train_time_start
    print(f"\n  Training for {experiment_name} done in {total_time//60:.0f}m {total_time%60:.0f}s. Best Val F1: {best_val_f1:.4f}.")
    print(f"  Best model for this experiment saved to: {model_save_path.relative_to(PROJECT_ROOT)}")
    if best_val_f1 > 0: # Nur laden, wenn ein Modell gespeichert wurde
        model.load_state_dict(torch.load(model_save_path))
    else:
        print("  Warning: No best model saved (best_val_f1 was 0.0). Using last model state.")
        model.load_state_dict(best_model_wts) # Lade den letzten Zustand, falls kein "bester" gespeichert wurde
    return model, history, total_time


# --- Trainings-Loop für einen Optuna Trial (aus tune_and_train_cnn.py) ---
def train_optuna_trial(trial, model, train_loader, val_loader, criterion, optimizer, num_epochs, device_obj, experiment_name_base):
    # (Code aus deinem train_trial in tune_and_train_cnn.py, leicht angepasst für Logging)
    best_trial_val_f1 = 0.0
    use_amp = (device_obj.type == 'cuda'); scaler = GradScaler(enabled=use_amp)
    print(f"    Starting Optuna Trial {trial.number} for '{experiment_name_base}' ({num_epochs} epochs per trial)...")
    if use_amp: print("    AMP (Mixed Precision) enabled for Optuna trial.")

    for epoch in range(num_epochs):
        model.train(); running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device_obj, non_blocking=True), labels.float().unsqueeze(1).to(device_obj, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp): outputs = model(inputs); loss = criterion(outputs, labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running_loss += loss.item() * inputs.size(0)
        # Kein Train Loss pro Epoche im Trial loggen, um Ausgabe kurz zu halten
        model.eval(); running_val_loss = 0.0; all_val_labels, all_val_outputs = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device_obj, non_blocking=True), labels.float().unsqueeze(1).to(device_obj, non_blocking=True)
                with autocast(enabled=use_amp): outputs = model(inputs); loss = criterion(outputs, labels) # Loss auch hier berechnen
                running_val_loss += loss.item() * inputs.size(0); all_val_labels.append(labels); all_val_outputs.append(outputs)
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        _, _, _, val_f1 = calculate_metrics(torch.cat(all_val_outputs), torch.cat(all_val_labels)) # Nur F1 für Trial
        print(f"      Trial {trial.number}, Epoch {epoch+1}/{num_epochs} - Val Loss: {epoch_val_loss:.4f}, Val F1: {val_f1:.4f}")
        best_trial_val_f1 = max(best_trial_val_f1, val_f1)
        trial.report(val_f1, epoch) # Optuna mitteilen, wie gut es läuft
        if trial.should_prune():
            print(f"      Trial {trial.number} pruned at epoch {epoch+1}."); raise optuna.exceptions.TrialPruned()
    return best_trial_val_f1


# --- Optuna Objective Function (aus tune_and_train_cnn.py) ---
def objective(trial, train_loader_obj, val_loader_obj, pos_weight_tensor_obj, device_obj, optuna_config_obj, experiment_name_base):
    # (Code aus deinem objective in tune_and_train_cnn.py, leicht angepasst)
    # Hyperparameter, die Optuna variieren soll
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout", 0.2, 0.6) # Leicht angepasster Bereich
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    num_conv_blocks = trial.suggest_int("num_conv_blocks", optuna_config_obj.get("min_conv_blocks", 3), optuna_config_obj.get("max_conv_blocks", 5))

    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "SGD", "RMSprop"])

    # Modell mit den Trial-Parametern bauen
    model = CustomCNN(num_classes=1, dropout_rate=dropout_rate, num_conv_blocks=num_conv_blocks).to(device_obj)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor_obj)

    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        momentum = trial.suggest_float("sgd_momentum", 0.8, 0.99)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == "RMSprop":
        alpha = trial.suggest_float("rmsprop_alpha", 0.9, 0.99)
        optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=alpha, weight_decay=weight_decay)

    print(f"    Trial {trial.number}: lr={lr:.1e}, dropout={dropout_rate:.2f}, wd={weight_decay:.1e}, blocks={num_conv_blocks}, opt={optimizer_name}")

    try:
        best_trial_f1 = train_optuna_trial(trial, model, train_loader_obj, val_loader_obj,
                                           criterion, optimizer, optuna_config_obj["epochs_per_trial"], device_obj, experiment_name_base)
        return best_trial_f1
    except optuna.exceptions.TrialPruned:
        return 0.0 # Pruned trials bekommen einen schlechten Score
    except Exception as e:
        print(f"    !! Trial {trial.number} failed with error: {e}")
        return -1.0 # Fehlerhafte Trials bekommen einen noch schlechteren Score

# --- Finale Trainingsfunktion nach Optuna (aus tune_and_train_cnn.py) ---
def train_final_optuna_model(best_params, train_loader, val_loader, pos_weight_tensor, device_obj, final_train_config, save_path, experiment_name_base):
    # (Code aus deinem train_final_model in tune_and_train_cnn.py, umbenannt und leicht angepasst)
    print(f"\n--- Starting Final Training for '{experiment_name_base}' with Optuna's Best Params ---")
    print(f"  Best params: {best_params}")

    num_blocks = best_params.get("num_conv_blocks", final_train_config.get("base_num_conv_blocks", 4))
    model = CustomCNN(num_classes=1, dropout_rate=best_params['dropout'], num_conv_blocks=num_blocks).to(device_obj)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    optimizer_name = best_params.get("optimizer", "AdamW")
    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=best_params['lr'], momentum=best_params.get('sgd_momentum',0.9), weight_decay=best_params['weight_decay'])
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=best_params['lr'], alpha=best_params.get('rmsprop_alpha',0.99), weight_decay=best_params['weight_decay'])

    # Für das finale Training verwenden wir keinen Scheduler, um es einfach zu halten, oder einen simplen
    scheduler = None # optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    num_epochs_final = final_train_config["epochs"]
    print(f"  Training final model for {num_epochs_final} epochs...")

    # Wir verwenden hier den generischen Trainingsloop für manuelle Experimente
    final_trained_model, history, train_time = train_manual_experiment_loop(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs_final, device_obj, f"{experiment_name_base}_final_optuna", Path(save_path).parent
    )
    # Stelle sicher, dass das Modell unter dem richtigen finalen Namen gespeichert wird,
    # train_manual_experiment_loop speichert unter experiment_name_best_model.pth
    final_model_path = Path(save_path)
    if (Path(save_path).parent / f"{experiment_name_base}_final_optuna_best_model.pth").exists():
        shutil.move(Path(save_path).parent / f"{experiment_name_base}_final_optuna_best_model.pth", final_model_path)
        print(f"  Final Optuna model explicitly saved to: {final_model_path.relative_to(PROJECT_ROOT)}")
    else:
        print(f"  Warning: Expected best model file for final optuna training not found. Saving current state to {final_model_path.relative_to(PROJECT_ROOT)}")
        torch.save(final_trained_model.state_dict(), final_model_path)


    return final_trained_model, history, train_time


# --- Haupt-Experimentier-Skript ---
if __name__ == "__main__":
    EXPERIMENTS_V3_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Daten einmalig aufteilen ---
    if not BASE_DATA_DIR_FOR_RUN.exists() or \
       not (TRAIN_DIR_RUN / 'yes').exists() or \
       not (VAL_DIR_RUN / 'no').exists() : # Sicherer Check
        print("Data split directory for this run not found or incomplete, splitting data...")
        split_data(ANNOTATED_DIR, TRAIN_DIR_RUN, VAL_DIR_RUN, TRAIN_VAL_SPLIT, RANDOM_SEED)
    else:
        print(f"Using existing data split from: {BASE_DATA_DIR_FOR_RUN}")

    # --- Basis-Transformationen definieren ---
    normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    base_val_transforms = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), normalize_transform])
    base_train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(), normalize_transform])

    # --- Definition der Experiment-Sitzungen ---
    all_experiment_sessions = [
        {
            "session_name": "manual_baseline_4blocks_adamw",
            "session_type": "manual_experiment",
            "config": {
                "train_transforms": base_train_transforms, "val_transforms": base_val_transforms,
                "optimizer_type": "AdamW", "lr": 0.0001, "weight_decay": 3e-5,
                "scheduler_type": "ReduceLROnPlateau", "scheduler_params": {"mode": 'max', "factor": 0.1, "patience": 5, "min_lr": 1e-6},
                "dropout_rate": 0.4, "epochs": 30, "num_conv_blocks": 4, "batch_size": 128
            }
        },
        {
            "session_name": "optuna_search_lr_dropout_wd_blocks_optimizer",
            "session_type": "optuna_search",
            "optuna_config": {
                "n_trials": 500, # Für einen echten Lauf erhöhen (z.B. 100-200)
                "epochs_per_trial": 25, # Kurz für schnelle Trials
                "min_conv_blocks": 3, # Optuna variiert Anzahl der Blöcke
                "max_conv_blocks": 5,
                "batch_size": 128, # Für Optuna konstant halten oder auch tunen?
                "num_workers": DEFAULT_NUM_WORKERS,
                "train_transforms": base_train_transforms, # Konstante Augmentation für diese Suche
                "val_transforms": base_val_transforms
            },
            "final_train_config": {
                "epochs": 100, # Längeres Training für das finale Modell
                "batch_size": 128,
                "num_workers": DEFAULT_NUM_WORKERS,
                "train_transforms": base_train_transforms,
                "val_transforms": base_val_transforms
            }
        },
        {
            "session_name": "manual_stronger_aug_5blocks",
            "session_type": "manual_experiment",
            "config": {
                "train_transforms": transforms.Compose([ # Stärkere Augmentation
                    transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.RandomHorizontalFlip(p=0.5),transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomAffine(degrees=25,translate=(0.15,0.15),scale=(0.85,1.15),shear=15),
                    transforms.ColorJitter(brightness=0.25,contrast=0.25,saturation=0.25,hue=0.15),
                    transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
                    transforms.ToTensor(),normalize_transform]),
                "val_transforms": base_val_transforms,
                "optimizer_type": "AdamW", "lr": 0.00005, "weight_decay": 5e-5,
                "scheduler_type": "ReduceLROnPlateau", "scheduler_params": {"mode": 'max', "factor": 0.2, "patience": 7, "min_lr": 1e-7},
                "dropout_rate": 0.55, "epochs": 40, "num_conv_blocks": 5, "batch_size": 64 # Kleinere Batch Size für größeres Modell
            }
        },
    ]

    # --- Durchführung aller Experiment-Sitzungen ---
    overall_summary = []
    for session_idx, session_details in enumerate(all_experiment_sessions):
        session_name = session_details["session_name"]
        session_type = session_details["session_type"]
        current_session_dir = EXPERIMENTS_V3_OUTPUT_DIR / session_name
        current_session_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n\n{'='*20} RUNNING SESSION {session_idx+1}/{len(all_experiment_sessions)}: {session_name} (Type: {session_type}) {'='*20}")

        # DataLoaders und pos_weight für diese Sitzung (könnte variieren, wenn BatchSize etc. pro Sitzung anders)
        current_bs = DEFAULT_BATCH_SIZE
        current_nw = DEFAULT_NUM_WORKERS
        current_train_transforms = base_train_transforms
        current_val_transforms = base_val_transforms

        if session_type == "manual_experiment":
            current_bs = session_details["config"].get("batch_size", DEFAULT_BATCH_SIZE)
            current_nw = session_details["config"].get("num_workers", DEFAULT_NUM_WORKERS)
            current_train_transforms = session_details["config"]["train_transforms"]
            current_val_transforms = session_details["config"]["val_transforms"]
        elif session_type == "optuna_search":
            current_bs = session_details["optuna_config"].get("batch_size", DEFAULT_BATCH_SIZE)
            current_nw = session_details["optuna_config"].get("num_workers", DEFAULT_NUM_WORKERS)
            current_train_transforms = session_details["optuna_config"]["train_transforms"]
            current_val_transforms = session_details["optuna_config"]["val_transforms"]


        train_loader_sess, val_loader_sess, class_to_idx_sess = create_dataloaders(
            TRAIN_DIR_RUN, VAL_DIR_RUN, current_bs, current_nw,
            current_train_transforms, current_val_transforms
        )
        pos_weight_tensor_sess = calculate_pos_weight(TRAIN_DIR_RUN, device)

        trained_model_sess = None
        history_sess = None
        train_time_sess = 0.0
        best_params_from_optuna = {}


        if session_type == "manual_experiment":
            cfg = session_details["config"]
            model_manual = CustomCNN(num_classes=1, dropout_rate=cfg["dropout_rate"], num_conv_blocks=cfg["num_conv_blocks"]).to(device)
            criterion_manual = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor_sess)

            if cfg["optimizer_type"] == "AdamW": optimizer_manual = optim.AdamW(model_manual.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay",0.01))
            elif cfg["optimizer_type"] == "SGD": optimizer_manual = optim.SGD(model_manual.parameters(), lr=cfg["lr"], momentum=cfg.get("momentum",0.9), weight_decay=cfg.get("weight_decay",0))
            elif cfg["optimizer_type"] == "RMSprop": optimizer_manual = optim.RMSprop(model_manual.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay",0), alpha=cfg.get("alpha",0.99), eps=cfg.get("eps",1e-08))
            else: optimizer_manual = optim.AdamW(model_manual.parameters(), lr=cfg["lr"])

            scheduler_manual = None
            if cfg.get("scheduler_type") == "StepLR": scheduler_manual = optim.lr_scheduler.StepLR(optimizer_manual, **cfg["scheduler_params"])
            elif cfg.get("scheduler_type") == "ReduceLROnPlateau": scheduler_manual = optim.lr_scheduler.ReduceLROnPlateau(optimizer_manual, **cfg["scheduler_params"])

            trained_model_sess, history_sess, train_time_sess = train_manual_experiment_loop(
                model_manual, train_loader_sess, val_loader_sess, criterion_manual, optimizer_manual, scheduler_manual,
                cfg["epochs"], device, session_name, current_session_dir
            )

        elif session_type == "optuna_search":
            optuna_cfg = session_details["optuna_config"]
            final_train_cfg = session_details["final_train_config"]
            optuna_study_name = f"optuna_study_{session_name}"
            print(f"\n  --- Starting Optuna Hyperparameter Search for '{session_name}' ({optuna_cfg['n_trials']} Trials) ---")

            # Verwende einen SQLite-Speicher für die Studie, um sie fortsetzen zu können
            # storage_name = f"sqlite:///{current_session_dir / optuna_study_name}.db"
            # study = optuna.create_study(study_name=optuna_study_name, storage=storage_name, load_if_exists=True, direction="maximize", pruner=optuna.pruners.MedianPruner())
            # Für Einfachheit erstmal ohne persistenten Speicher:
            study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())

            study.optimize(lambda trial: objective(trial, train_loader_sess, val_loader_sess, pos_weight_tensor_sess, device, optuna_cfg, session_name),
                           n_trials=optuna_cfg["n_trials"])

            print(f"\n  --- Optuna Search Finished for '{session_name}' ---")
            print(f"  Finished trials: {len(study.trials)}")

            if not study.best_trial or study.best_value <= -1.0: # -1.0 ist der Fehler-Score
                print(f"  ERROR: Optuna search for '{session_name}' did not find any valid/successful trials. Skipping final training.")
            else:
                print(f"  Best Optuna trial for '{session_name}':")
                best_trial = study.best_trial
                print(f"    Value (Best Val F1 during search): {best_trial.value:.4f}")
                print("    Params: ")
                for key, value in best_trial.params.items(): print(f"      {key}: {value}")
                best_params_from_optuna = best_trial.params

                # Finales Training mit den besten Optuna-Parametern
                final_model_save_path = current_session_dir / f"{session_name}_optuna_final_best_model.pth"
                trained_model_sess, history_sess, train_time_sess = train_final_optuna_model(
                    best_params_from_optuna, train_loader_sess, val_loader_sess, pos_weight_tensor_sess, device,
                    final_train_cfg, final_model_save_path, session_name
                )
        else:
            print(f"  Unknown session_type: {session_type}. Skipping.")
            continue

        # --- Finale Evaluation für diese Sitzung ---
        if trained_model_sess is not None:
            print(f"\n--- Final Evaluation for Session: {session_name} ---")
            trained_model_sess.eval()
            all_final_labels, all_final_outputs = [], []
            with torch.no_grad():
                for inputs, labels in val_loader_sess:
                    inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
                    with autocast(enabled=(device.type=='cuda')): outputs = trained_model_sess(inputs)
                    all_final_labels.append(labels); all_final_outputs.append(outputs)
            all_final_outputs_cat=torch.cat(all_final_outputs); all_final_labels_cat=torch.cat(all_final_labels)
            final_acc, final_prec, final_rec, final_f1 = calculate_metrics(all_final_outputs_cat, all_final_labels_cat)

            results_text = (f"Session: {session_name} (Type: {session_type})\n")
            if session_type == "optuna_search" and best_params_from_optuna:
                results_text += f"  Best Optuna Params: {best_params_from_optuna}\n"
            results_text += (f"  Validation Accuracy: {final_acc:.4f}\n"+
                             f"  Validation Precision (yes): {final_prec:.4f}\n"+
                             f"  Validation Recall (yes): {final_rec:.4f}\n"+
                             f"  Validation F1-Score (yes): {final_f1:.4f}\n"+
                             f"  Total Training Time: {train_time_sess//60:.0f}m {train_time_sess%60:.0f}s\n")
            final_preds=(torch.sigmoid(all_final_outputs_cat).detach().cpu().numpy()>0.5).astype(int).flatten()
            final_labels_np=all_final_labels_cat.detach().cpu().numpy().flatten()
            cm = confusion_matrix(final_labels_np, final_preds)
            results_text += "\nCM:\n"+f"Labels: {list(class_to_idx_sess.keys())}\n"+str(cm)+"\n\nReport:\n"
            results_text += classification_report(final_labels_np,final_preds,target_names=list(class_to_idx_sess.keys()),zero_division=0)+"\n"
            print(results_text)
            with open(current_session_dir / f"{session_name}_summary.txt", "w") as f: f.write(results_text)

            overall_summary.append({"session_name":session_name, "type": session_type, "accuracy":final_acc,
                                    "precision":final_prec,"recall":final_rec,"f1_score":final_f1,
                                    "training_time_seconds":train_time_sess,
                                    "best_optuna_params": best_params_from_optuna if best_params_from_optuna else "N/A"})
            save_error_analysis(trained_model_sess,val_loader_sess,device,class_to_idx_sess,current_session_dir,session_name)
            if history_sess: plot_experiment_history(history_sess,current_session_dir,session_name)
        else:
            print(f"  No model trained for session {session_name}. Skipping final evaluation.")
            overall_summary.append({"session_name":session_name, "type": session_type, "f1_score":0.0, "training_time_seconds":0, "error": "Training skipped or failed"})


    # --- Gesamtzusammenfassung aller Experiment-Sitzungen ---
    print("\n\n{'='*20} OVERALL EXPERIMENT SUMMARY {'='*20}")
    for metrics in sorted(overall_summary, key=lambda x: x['f1_score'], reverse=True):
        print(f"\nSession: {metrics['session_name']} (Type: {metrics['type']})")
        if "error" in metrics:
            print(f"  Error: {metrics['error']}")
        else:
            print(f"  F1: {metrics['f1_score']:.4f} Acc: {metrics['accuracy']:.4f} P: {metrics['precision']:.4f} R: {metrics['recall']:.4f}")
            print(f"  Time: {metrics['training_time_seconds']//60:.0f}m {metrics['training_time_seconds']%60:.0f}s")
            if metrics['type'] == "optuna_search": print(f"  Best Optuna Params: {metrics['best_optuna_params']}")
    print("\n--- Experiment Script V3 Finished ---")