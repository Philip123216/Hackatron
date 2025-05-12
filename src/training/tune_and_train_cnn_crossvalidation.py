# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset # Subset wird für CV benötigt
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
from sklearn.model_selection import StratifiedKFold # <<< NEUER IMPORT für CV
import matplotlib.pyplot as plt
import math
import optuna
from torch.cuda.amp import autocast, GradScaler

# --- 1. Konfiguration ---
# === Path ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent # Passe an, falls dein Skript woanders liegt
ANNOTATED_DIR = PROJECT_ROOT / "data_annotated"
BASE_DATA_DIR = PROJECT_ROOT / "data_split_for_optuna_cv" # Eigener Ordner für Optuna/CV Split
TRAIN_DIR = BASE_DATA_DIR / "train" # Für den initialen Optuna-Split
VAL_DIR = BASE_DATA_DIR / "validation"  # Für den initialen Optuna-Split
OPTUNA_FINAL_MODEL_SAVE_PATH = PROJECT_ROOT / "best_optuna_tuned_cnn_model_with_cv.pth"
OPTUNA_STUDY_DB_PATH = PROJECT_ROOT / "optuna_study_with_cv.db"

# === Cross-Validation Konfiguration ===
DO_CROSS_VALIDATION = True
N_SPLITS_CV = 3  # Reduziert für schnellere Tests, erhöhe auf 5 für robustere Ergebnisse
EPOCHS_PER_CV_FOLD = 30 # Sollte ähnlich zu FINAL_TRAINING_EPOCHS sein
CV_RESULTS_DIR = PROJECT_ROOT / "cross_validation_results_final_config"
CV_RESULTS_DIR.mkdir(parents=True, exist_ok=True) # Stelle sicher, dass der Ordner existiert

# === Optuna & Training ===
BATCH_SIZE = 128
NUM_WORKERS = 8 # Passe an deine CPU an
N_TRIALS_OPTUNA = 50 # Reduziere für Tests (z.B. 10), erhöhe für intensive Suche (z.B. 100+)
EPOCHS_PER_OPTUNA_TRIAL = 15
FINAL_TRAINING_EPOCHS = 40
IMG_SIZE = 250
RANDOM_SEED = 42
TRAIN_VAL_SPLIT = 0.8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Project Root: {PROJECT_ROOT}")
print(f"Using device: {device}")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# --- 2. Datenaufteilung (für den initialen Optuna Train/Val Split) ---
def split_data(source_dir, train_dir, val_dir, split_ratio=0.8, seed=42):
    # (Deine bestehende split_data Funktion - unverändert)
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


# --- 3. Datentransformationen & Laden ---
# Globale Definition der Transformationen, damit sie überall verfügbar sind
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.ToTensor(), normalize
])
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(), normalize
])

def create_dataloaders(train_source, val_source, batch_size, num_workers, class_to_idx_map=None, is_cv_fold=False):
    # Akzeptiert jetzt Pfade oder Dataset-Objekte (für CV)
    if isinstance(train_source, Path): # Annahme: Pfade für ImageFolder
        current_train_transforms = train_transforms # Globale verwenden
        current_val_transforms = val_transforms   # Globale verwenden
        train_dataset = datasets.ImageFolder(train_source, transform=current_train_transforms)
        val_dataset = datasets.ImageFolder(val_source, transform=current_val_transforms)
        class_to_idx = train_dataset.class_to_idx
        classes = train_dataset.classes
    elif isinstance(train_source, torch.utils.data.Dataset): # Annahme: Bereits erstellte Dataset-Objekte (Subsets für CV)
        train_dataset = train_source
        val_dataset = val_source
        # Für CV wird class_to_idx von außen übergeben oder vom original_dataset abgeleitet
        if class_to_idx_map:
            class_to_idx = class_to_idx_map
            classes = [k for k, v in sorted(class_to_idx.items(), key=lambda item: item[1])] # Sortiere nach Index für Konsistenz
        else: # Fallback, sollte nicht passieren, wenn class_to_idx_map übergeben wird
            class_to_idx = {'no': 0, 'yes': 1}; classes = ['no', 'yes']
            print("Warning: class_to_idx_map not provided to create_dataloaders for CV fold.")
    else:
        raise ValueError("train_source/val_source must be Path or Dataset object")

    persistent = num_workers > 0 and device.type == 'cuda'
    pin_memory = device.type == 'cuda'
    train_drop_last = len(train_dataset) % batch_size <= 1 and len(train_dataset) > batch_size
    val_drop_last = len(val_dataset) % batch_size <= 1 and len(val_dataset) > batch_size

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=not is_cv_fold, # Kein Shuffle bei CV, da KFold shufflet
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=persistent, drop_last=train_drop_last)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory, persistent_workers=persistent, drop_last=val_drop_last)

    print(f"  DataLoaders {'(CV Fold)' if is_cv_fold else ''}: Tr={len(train_dataset)} ({len(train_loader)} b), Vl={len(val_dataset)} ({len(val_loader)} b)")
    print(f"  Classes: {classes} (Map: {class_to_idx}) W: {num_workers} BS: {batch_size}")
    if class_to_idx.get('yes',-1)!=1 or class_to_idx.get('no',-1)!=0: print(f"*** W: Class map! {class_to_idx}")
    return train_loader, val_loader, class_to_idx

def calculate_pos_weight_from_dataset(dataset_obj_or_path, class_to_idx_map, device_obj):
    # (Deine angepasste Funktion aus der vorherigen Antwort)
    print(f"  Calculating pos_weight...")
    try:
        n_yes, n_no = 0, 0
        yes_idx = class_to_idx_map['yes']
        no_idx = class_to_idx_map['no']

        if isinstance(dataset_obj_or_path, Path): # Wenn Pfad, zähle Dateien
            n_yes = len(glob.glob(str(dataset_obj_or_path / 'yes' / '*.png')))
            n_no = len(glob.glob(str(dataset_obj_or_path / 'no' / '*.png')))
        elif isinstance(dataset_obj_or_path, torch.utils.data.Dataset): # Wenn Dataset-Objekt
            if isinstance(dataset_obj_or_path, Subset):
                labels_in_subset = [dataset_obj_or_path.dataset.targets[i] for i in dataset_obj_or_path.indices]
            elif hasattr(dataset_obj_or_path, 'targets'):
                labels_in_subset = dataset_obj_or_path.targets
            elif hasattr(dataset_obj_or_path, 'labels'):
                labels_in_subset = dataset_obj_or_path.labels
            else:
                labels_in_subset = [s[1] for s in dataset_obj_or_path]
            n_yes = sum(1 for label_idx in labels_in_subset if label_idx == yes_idx)
            n_no = sum(1 for label_idx in labels_in_subset if label_idx == no_idx)
        else:
            raise ValueError("Input to calculate_pos_weight must be Path or Dataset")

        if n_yes == 0 or n_no == 0:
            pw = 1.0; print(f"  W: One class 0 samples ({n_no} no, {n_yes} yes). Default pw=1.0.")
        else:
            pw = n_no / n_yes
        print(f"  Pos_weight samples: {n_no} 'no', {n_yes} 'yes'. Calculated pw: {pw:.2f}")
        return torch.tensor([pw], device=device_obj)
    except Exception as e: print(f"  Err calc pos_weight: {e}. Defaulting."); return torch.tensor([1.0], device=device_obj)


# --- Plotting Funktion (NEU EINGEFÜGT) ---
def plot_training_history(history, output_dir, plot_name_prefix):
    # (Code aus meiner vorherigen Antwort für plot_training_history)
    print(f"  Plotting training history for {plot_name_prefix} to {output_dir}...")
    if not history or 'val_f1' not in history or not history['val_f1']:
        print("    No history to plot or 'val_f1' is missing.")
        return
    epochs = history.get('epoch', range(1, len(history['val_f1']) + 1))
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(15, 10)); plt.subplot(2, 1, 1)
    if 'val_f1' in history and history['val_f1']: plt.plot(epochs, history['val_f1'], 'b-', label='Val F1')
    if 'val_acc' in history and history['val_acc']: plt.plot(epochs, history['val_acc'], 'c-', label='Val Acc')
    if 'val_precision' in history and history['val_precision']: plt.plot(epochs, history['val_precision'], 'm-', label='Val Precision')
    if 'val_recall' in history and history['val_recall']: plt.plot(epochs, history['val_recall'], 'y-', label='Val Recall')
    plt.title(f'{plot_name_prefix} - Validation Metrics'); plt.xlabel('Epoch'); plt.ylabel('Metric Value'); plt.legend(); plt.grid(True)
    plt.subplot(2, 1, 2)
    if 'train_loss' in history and history['train_loss']: plt.plot(epochs, history['train_loss'], 'r-', label='Train Loss')
    if 'val_loss' in history and history['val_loss']: plt.plot(epochs, history['val_loss'], 'g-', label='Val Loss')
    plt.title(f'{plot_name_prefix} - Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(output_dir / f"{plot_name_prefix}_training_history.png"); plt.close()
    print(f"    History plot saved to {output_dir / f'{plot_name_prefix}_training_history.png'}")


# --- Deine bestehenden Funktionen ---
# CustomCNN, calculate_metrics, find_and_show_false_negatives,
# train_trial, objective, train_final_model
# Stelle sicher, dass CustomCNN hier die ist, die du CVen willst
# (z.B. die mit den von Optuna getuneten num_conv_blocks etc., oder die fixe 4-Block)
# In diesem Beispiel verwende ich die Original-CustomCNN aus deinem Skript.
# Wenn Optuna `num_conv_blocks` tuned, musst du das in `model_params_from_optuna` übergeben.

# (HIER DEINE FUNKTIONEN CustomCNN, calculate_metrics, find_and_show_false_negatives,
#  train_trial, objective, train_final_model EINFÜGEN - sie sind identisch zu deinem Skript)
# --- Für Kürze hier weggelassen, aber sie MÜSSEN im finalen Skript sein ---
# WICHTIG: train_final_model (und der neue train_cv_fold) sollte `history` zurückgeben.

# (DEINE FUNKTIONEN VON OBEN HIER EINFÜGEN)
# CustomCNN, calculate_metrics, find_and_show_false_negatives, train_trial, objective, train_final_model
# ... (angenommen, sie sind hier definiert und train_final_model gibt history zurück) ...
# --- 6. Trainings-Loop für einen Optuna Trial ---
def train_trial(trial, model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_f1 = 0.0
    use_amp = (device.type == 'cuda')
    scaler = GradScaler(enabled=use_amp)
    for epoch in range(num_epochs):
        model.train(); running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.float().unsqueeze(1).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp): outputs = model(inputs); loss = criterion(outputs, labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running_loss += loss.item() * inputs.size(0)
        model.eval(); running_val_loss = 0.0; all_val_labels, all_val_outputs = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.float().unsqueeze(1).to(device, non_blocking=True)
                with autocast(enabled=use_amp): outputs = model(inputs); loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0); all_val_labels.append(labels); all_val_outputs.append(outputs)
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_acc, val_prec, val_rec, val_f1 = calculate_metrics(torch.cat(all_val_outputs), torch.cat(all_val_labels))
        print(f"  Trial {trial.number} Epoch {epoch+1}/{num_epochs} - Val Loss: {epoch_val_loss:.4f} Val F1: {val_f1:.4f}")
        best_val_f1 = max(best_val_f1, val_f1)
        trial.report(val_f1, epoch)
        if trial.should_prune(): print(f"  Trial {trial.number} pruned."); raise optuna.exceptions.TrialPruned()
    return best_val_f1

# --- 7. Optuna Objective Function ---
def objective(trial, train_loader, val_loader, pos_weight_tensor, device):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout", 0.1, 0.6)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    optimizer_name = "AdamW"
    model = CustomCNN(num_classes=1, dropout_rate=dropout_rate).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"  Params: lr={lr:.6f}, dropout={dropout_rate:.3f}, weight_decay={weight_decay:.6f}")
    try:
        # Training für diesen Trial starten
        best_trial_f1 = train_trial(trial, model, train_loader, val_loader,
                                    criterion, optimizer, EPOCHS_PER_TRIAL, device)
        return best_trial_f1
    except optuna.exceptions.TrialPruned:
        print(f"  Trial {trial.number} was pruned. Returning 0.0 as score.")
        # Gib einen schlechten Wert zurück, da der Trial nicht zu Ende geführt wurde
        return 0.0  # Oder einen anderen sehr schlechten Wert, den Optuna nicht als "gut" interpretiert
    except Exception as e:
        print(f"!! Trial {trial.number} failed with error: {e}")
        # Bei anderem Fehler ebenfalls schlechten Wert zurückgeben
        return 0.0


# --- NEUE Funktion für einen CV-Fold Training ---
def train_cv_fold(model_config_params, fold_train_loader, fold_val_loader, fold_pos_weight,
                  num_epochs, device, fold_idx, current_cv_fold_dir, class_to_idx_cv):
    # (Code aus meiner vorherigen Antwort für train_cv_fold, leicht angepasst)
    print(f"\n--- Training CV Fold {fold_idx + 1}/{N_SPLITS_CV} ({num_epochs} epochs) ---")
    # Modell für jeden Fold neu instantiieren mit den besten Optuna-Parametern
    # Passe `CustomCNN` an, um alle getunten Architekturparameter zu akzeptieren
    model = CustomCNN(num_classes=1,
                      dropout_rate=model_config_params['dropout']
                      # num_conv_blocks=model_config_params.get('num_conv_blocks', 4) # Beispiel
                      ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=fold_pos_weight)
    optimizer_name = model_config_params.get('optimizer', 'AdamW') # Falls Optuna den Optimizer tuned
    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=model_config_params['lr'], weight_decay=model_config_params['weight_decay'])
    # Hier ggf. andere Optimizer aus best_params initialisieren (z.B. SGD, RMSprop)
    else: # Fallback
        optimizer = optim.AdamW(model.parameters(), lr=model_config_params['lr'], weight_decay=model_config_params['weight_decay'])

    best_val_f1_fold = 0.0
    history_fold = {'epoch':[],'train_loss':[],'val_loss':[],'val_f1':[],'val_acc':[],'val_precision':[],'val_recall':[],'time_per_epoch':[]}
    use_amp = (device.type == 'cuda'); scaler = GradScaler(enabled=use_amp)
    start_time_fold = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time(); model.train(); running_loss = 0.0
        for inputs, labels in fold_train_loader:
            inputs,labels=inputs.to(device,non_blocking=True),labels.float().unsqueeze(1).to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type,enabled=use_amp):outputs=model(inputs);loss=criterion(outputs,labels)
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
            running_loss += loss.item()*inputs.size(0)
        epoch_train_loss = running_loss/len(fold_train_loader.dataset)
        model.eval();running_val_loss=0.0;all_val_labels,all_val_outputs=[],[]
        with torch.no_grad():
            for inputs,labels in fold_val_loader:
                inputs,labels=inputs.to(device,non_blocking=True),labels.float().unsqueeze(1).to(device,non_blocking=True)
                with torch.amp.autocast(device_type=device.type,enabled=use_amp):outputs=model(inputs);val_batch_loss=criterion(outputs,labels)
                running_val_loss+=val_batch_loss.item()*inputs.size(0);all_val_labels.append(labels);all_val_outputs.append(outputs)
        epoch_val_loss=running_val_loss/len(fold_val_loader.dataset)
        val_acc,val_prec,val_rec,val_f1=calculate_metrics(torch.cat(all_val_outputs),torch.cat(all_val_labels))
        print(f"    CVF{fold_idx+1} Ep{epoch+1}/{num_epochs} TrL:{epoch_train_loss:.4f} VaL:{epoch_val_loss:.4f} VaF1:{val_f1:.4f} VaP:{val_prec:.4f} VaR:{val_rec:.4f}")
        epoch_time=time.time()-epoch_start_time
        for k,v in zip(history_fold.keys(),[epoch+1,epoch_train_loss,epoch_val_loss,val_f1,val_acc,val_prec,val_rec,epoch_time]): history_fold[k].append(v)
        if val_f1 > best_val_f1_fold: best_val_f1_fold = val_f1
    time_elapsed_fold = time.time()-start_time_fold
    print(f"  CVF{fold_idx+1} Train done {time_elapsed_fold//60:.0f}m {time_elapsed_fold%60:.0f}s. BestF1:{best_val_f1_fold:.4f}")
    model.eval();all_f_val_lab,all_f_val_out=[],[] # Finale Eval mit letztem Zustand
    with torch.no_grad():
        for inputs,labels in fold_val_loader:
            inputs,labels=inputs.to(device),labels.float().unsqueeze(1).to(device)
            with torch.amp.autocast(device_type=device.type,enabled=use_amp): outputs=model(inputs)
            all_f_val_lab.append(labels);all_f_val_out.append(outputs)
    f_acc,f_prec,f_rec,f_f1=calculate_metrics(torch.cat(all_f_val_out),torch.cat(all_f_val_lab))
    plot_training_history(history_fold,current_cv_fold_dir,f"fold_{fold_idx+1}")
    f_preds=(torch.sigmoid(torch.cat(all_f_val_out)).detach().cpu().numpy()>0.5).astype(int).flatten()
    f_labs_np=torch.cat(all_f_val_lab).detach().cpu().numpy().flatten()
    cm_f=confusion_matrix(f_labs_np,f_preds);report_f=classification_report(f_labs_np,f_preds,target_names=list(class_to_idx_cv.keys()),zero_division=0)
    sum_txt_f=f"Fold {fold_idx+1} Final Metrics:\nF1={f_f1:.4f} P={f_prec:.4f} R={f_rec:.4f} Acc={f_acc:.4f}\n\nCM:\n{cm_f}\n\nReport:\n{report_f}"
    with open(current_cv_fold_dir/f"fold_{fold_idx+1}_summary.txt","w") as f:f.write(sum_txt_f)
    print(sum_txt_f)
    return {'accuracy':f_acc,'precision':f_prec,'recall':f_rec,'f1':f_f1,'history':history_fold,'cm':cm_f}


# --- Hauptausführung ---
if __name__ == "__main__":
    print("DEBUG: Entered main execution block.")
    # --- Initialer Daten-Split für Optuna ---
    if not BASE_DATA_DIR.exists() or not (TRAIN_DIR / 'yes').exists() or not (VAL_DIR / 'no').exists():
        print("Data split directory for Optuna not found or incomplete, splitting data...")
        split_data(ANNOTATED_DIR, TRAIN_DIR, VAL_DIR, TRAIN_VAL_SPLIT, RANDOM_SEED)
    else: print(f"Using existing data split for Optuna from: {BASE_DATA_DIR}")

    # Erstelle DataLoaders für den initialen Optuna-Lauf
    optuna_train_loader, optuna_val_loader, optuna_class_to_idx = create_dataloaders(
        TRAIN_DIR, VAL_DIR, BATCH_SIZE, NUM_WORKERS
    )
    optuna_pos_weight = calculate_pos_weight_from_dataset(TRAIN_DIR, optuna_class_to_idx, device)

    # --- Optuna Hyperparameter Suche ---
    print(f"\n--- Starting Optuna Hyperparameter Search ({N_TRIALS_OPTUNA} Trials, {EPOCHS_PER_OPTUNA_TRIAL} epochs/trial) ---")
    optuna_storage_name = f"sqlite:///{OPTUNA_STUDY_DB_PATH}"
    study = optuna.create_study(
        study_name="cnn_hyperparam_tuning_v_cv", storage=optuna_storage_name,
        load_if_exists=True, direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1)
    )
    study.optimize(lambda trial: objective(trial, optuna_train_loader, optuna_val_loader, optuna_pos_weight, device),
                   n_trials=N_TRIALS_OPTUNA, timeout=None) # Optional: timeout=3600*X für X Stunden

    print(f"\n--- Optuna Search Finished ---"); print(f"Finished trials: {len(study.trials)}")
    if not study.best_trial or study.best_value <= -1.0: # -1.0 ist der Fehler-Score in objective
         print("\nERROR: Optuna search did not find any valid/successful trials. Check logs. Cannot proceed with CV or final training.")
         best_params = None # Wichtig, damit CV nicht startet, wenn Optuna fehlschlägt
    else:
        print("Best Optuna trial found:"); best_trial = study.best_trial
        print(f"  Value (Best Val F1 during search): {best_trial.value:.4f}"); print("  Params: ")
        for key, value in best_trial.params.items(): print(f"    {key}: {value}")
        best_params = best_trial.params # Speichere die besten Parameter

        # --- Finales Training mit den besten Optuna Parametern (Optional, aber gut für ein finales Modell) ---
        print("\n--- Starting Final Training with Best Optuna Hyperparameters ---")
        # Modell mit den besten Parametern erstellen
        final_model = CustomCNN(num_classes=1,
                                dropout_rate=best_params['dropout']
                                # Hier weitere Architekturparameter von Optuna einfügen
                                # z.B. num_conv_blocks=best_params.get('num_conv_blocks', 4)
                                ).to(device)
        final_criterion = nn.BCEWithLogitsLoss(pos_weight=optuna_pos_weight) # Nutze pos_weight vom initialen Split
        optimizer_name_final = best_params.get('optimizer', 'AdamW')
        if optimizer_name_final == "AdamW":
            final_optimizer = optim.AdamW(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
        # ... (ggf. andere Optimizer aus best_params initialisieren) ...
        else: # Fallback
            final_optimizer = optim.AdamW(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])

        # Einfacher Scheduler für finales Training
        final_scheduler = optim.lr_scheduler.ReduceLROnPlateau(final_optimizer, mode='max', factor=0.2, patience=7, min_lr=1e-7)

        # Wir verwenden hier den generischen train_cv_fold, um Konsistenz zu wahren, nennen den Output aber anders
        trained_final_model, history_final_train, time_final_train, best_f1_final_train = train_cv_fold(
            best_params, optuna_train_loader, optuna_val_loader, optuna_pos_weight, # Nutze die Optuna Split Loader
            FINAL_TRAINING_EPOCHS, device, -1, # fold_idx=-1 für "finales Training"
            PROJECT_ROOT, # Basis-Output-Dir für Plots etc.
            optuna_class_to_idx
        )
        torch.save(trained_final_model.state_dict(), OPTUNA_FINAL_MODEL_SAVE_PATH) # Speichere das finale Modell explizit
        print(f"Final Optuna-tuned model saved to: {OPTUNA_FINAL_MODEL_SAVE_PATH}")
        plot_training_history(history_final_train, PROJECT_ROOT, "optuna_final_model") # Plot für finales Training


        print("\n--- Final Evaluation on Validation Set (using BEST OPTUNA TUNED model) ---")
        # (Dein Code für die finale Evaluation des trained_final_model bleibt hier)
        # ...
        find_and_show_false_negatives(trained_final_model, optuna_val_loader, device, optuna_class_to_idx, num_to_show=15)


    # =========== K-FOLD CROSS-VALIDATION (nur wenn Optuna erfolgreich war und best_params existieren) ===========
    if DO_CROSS_VALIDATION and best_params: # Prüfe, ob best_params existiert
        print(f"\n\n{'='*20} STARTING K-FOLD CROSS-VALIDATION ({N_SPLITS_CV} Folds) for Optuna's Best Config {'='*20}")
        print(f"Using best hyperparameters from Optuna search for CV: {best_params}")
        CV_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # Lade alle Datenpunkte und Labels aus dem ANNOTATED_DIR
        all_image_filepaths = []
        all_image_labels_numeric = []
        # Erstelle ein temporäres Dataset nur zum Holen der Pfade und initialen class_to_idx
        # Es ist wichtig, dass class_to_idx_cv konsistent über alle Folds ist
        # und dem entspricht, was deine Modelle erwarten (z.B. 'no':0, 'yes':1)
        # Verwende ANNOTATED_DIR, um alle Daten zu erfassen
        master_dataset_for_cv_paths = datasets.ImageFolder(ANNOTATED_DIR, transform=val_transforms) # Dummy transform
        class_to_idx_cv = master_dataset_for_cv_paths.class_to_idx

        for img_path, label_idx in master_dataset_for_cv_paths.samples:
            all_image_filepaths.append(img_path)
            all_image_labels_numeric.append(label_idx)

        if not all_image_filepaths: print("E: No images for CV. Check ANNOTATED_DIR."); exit()

        skf = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_SEED)
        fold_metrics_summary = []

        for fold_idx, (train_indices, val_indices) in enumerate(skf.split(all_image_filepaths, all_image_labels_numeric)):
            current_cv_fold_output_dir = CV_RESULTS_DIR / f"fold_{fold_idx+1}"
            current_cv_fold_output_dir.mkdir(parents=True, exist_ok=True)

            # Erstelle Subsets für diesen Fold direkt aus dem master_dataset_for_cv_paths
            # Wichtig: Die Transformationen müssen hier korrekt angewendet werden!
            # Wir erstellen neue ImageFolder-Instanzen pro Fold, um sicherzustellen, dass die
            # Transformationen angewendet werden, wenn die DataLoaders die Bilder laden.
            # Alternativ: Eigene Dataset-Klasse, die Indizes und Transformationen verwaltet.

            # Pragmatische Lösung: Temporäre Ordner pro Fold erstellen
            temp_fold_train_dir = current_cv_fold_output_dir / "train_temp_fold_data"
            temp_fold_val_dir = current_cv_fold_output_dir / "val_temp_fold_data"
            if temp_fold_train_dir.exists(): shutil.rmtree(temp_fold_train_dir)
            if temp_fold_val_dir.exists(): shutil.rmtree(temp_fold_val_dir)
            for class_name_cv in class_to_idx_cv.keys():
                (temp_fold_train_dir / class_name_cv).mkdir(parents=True, exist_ok=True)
                (temp_fold_val_dir / class_name_cv).mkdir(parents=True, exist_ok=True)

            print(f"\n--- Preparing Data for CV Fold {fold_idx+1}/{N_SPLITS_CV} ---")
            fold_train_s_count={'yes':0,'no':0}; fold_val_s_count={'yes':0,'no':0}
            inv_class_to_idx_cv = {v: k for k, v in class_to_idx_cv.items()} # Zum Holen des Klassennamens

            for i in train_indices:
                src_path=Path(all_image_filepaths[i]); lbl_idx=all_image_labels_numeric[i]; lbl_name=inv_class_to_idx_cv[lbl_idx]
                shutil.copy(src_path,temp_fold_train_dir/lbl_name/src_path.name); fold_train_s_count[lbl_name]+=1
            for i in val_indices:
                src_path=Path(all_image_filepaths[i]); lbl_idx=all_image_labels_numeric[i]; lbl_name=inv_class_to_idx_cv[lbl_idx]
                shutil.copy(src_path,temp_fold_val_dir/lbl_name/src_path.name); fold_val_s_count[lbl_name]+=1
            print(f"  Fold Train Samples: {fold_train_s_count}");print(f"  Fold Val Samples: {fold_val_s_count}")

            # DataLoaders für diesen Fold erstellen (nutzen die globalen train_transforms/val_transforms)
            fold_train_loader, fold_val_loader, _ = create_dataloaders(
                temp_fold_train_dir, temp_fold_val_dir, BATCH_SIZE, NUM_WORKERS, class_to_idx_cv, is_cv_fold=True
            )
            fold_pos_weight = calculate_pos_weight_from_dataset(temp_fold_train_dir, class_to_idx_cv, device)

            # Modell- und Optimizer-Parameter sind die `best_params` von Optuna
            fold_results = train_cv_fold(
                best_params, fold_train_loader, fold_val_loader, fold_pos_weight,
                EPOCHS_PER_CV_FOLD, device, fold_idx, current_cv_fold_output_dir, class_to_idx_cv
            )
            fold_metrics_summary.append(fold_results)
            # Temporäre Daten dieses Folds löschen
            try: shutil.rmtree(temp_fold_train_dir); shutil.rmtree(temp_fold_val_dir)
            except Exception as e_rm: print(f"W: Could not remove temp fold data: {e_rm}")


        # --- Zusammenfassung der Cross-Validation ---
        if fold_metrics_summary: # Nur wenn CV gelaufen ist
            print(f"\n\n{'='*20} CROSS-VALIDATION OVERALL SUMMARY ({N_SPLITS_CV} Folds) {'='*20}")
            avg_f1=np.mean([r['f1'] for r in fold_metrics_summary if r]);std_f1=np.std([r['f1'] for r in fold_metrics_summary if r])
            avg_prec=np.mean([r['precision'] for r in fold_metrics_summary if r]);std_prec=np.std([r['precision'] for r in fold_metrics_summary if r])
            avg_rec=np.mean([r['recall'] for r in fold_metrics_summary if r]);std_rec=np.std([r['recall'] for r in fold_metrics_summary if r])
            avg_acc=np.mean([r['accuracy'] for r in fold_metrics_summary if r]);std_acc=np.std([r['accuracy'] for r in fold_metrics_summary if r])
            print(f"Avg Val F1: {avg_f1:.4f} +/- {std_f1:.4f}");print(f"Avg Val P: {avg_prec:.4f} +/- {std_prec:.4f}")
            print(f"Avg Val R: {avg_rec:.4f} +/- {std_rec:.4f}");print(f"Avg Val Acc: {avg_acc:.4f} +/- {std_acc:.4f}")
            cv_sum_txt=f"CV Summary ({N_SPLITS_CV} Folds)\nOptuna Best Params:\n{best_params}\n\n"
            for i,r in enumerate(fold_metrics_summary): cv_sum_txt+=f"F{i+1}: F1={r['f1']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} Acc={r['accuracy']:.4f}\n  CM:\n{r['cm']}\n"
            cv_sum_txt+=f"\nAvg F1:{avg_f1:.4f}+/-{std_f1:.4f}\nAvg P:{avg_prec:.4f}+/-{std_prec:.4f}\nAvg R:{avg_rec:.4f}+/-{std_rec:.4f}\nAvg Acc:{avg_acc:.4f}+/-{std_acc:.4f}\n"
            with open(CV_RESULTS_DIR/"cv_overall_summary.txt","w") as f:f.write(cv_sum_txt)
            print(f"Overall CV summary saved to {CV_RESULTS_DIR/'cv_overall_summary.txt'}")
        else:
            print("Cross-Validation did not produce any results to summarize.")

    # ================= ENDE CV ABSCHNITT =======================
    print("\n--- Script Finished ---")