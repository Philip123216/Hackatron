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
PROJECT_ROOT = Path(__file__).resolve().parent.parent
print(f"Project Root (angenommen): {PROJECT_ROOT}")

# --- 1. Globale Konfiguration ---
ANNOTATED_DIR = PROJECT_ROOT / "data_annotated"
BASE_DATA_DIR_FOR_RUN = PROJECT_ROOT / "data_split_for_run_v4" # Eigener Datenordner für diesen Lauf
TRAIN_DIR_RUN = BASE_DATA_DIR_FOR_RUN / "train"
VAL_DIR_RUN = BASE_DATA_DIR_FOR_RUN / "validation"
EXPERIMENTS_V4_OUTPUT_DIR = PROJECT_ROOT / "experiment_v4_results" # Hauptordner für Ergebnisse

TRAIN_VAL_SPLIT = 0.8
RANDOM_SEED = 42
IMG_SIZE = 250
# Standardwerte - können pro Experiment überschrieben werden
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_WORKERS = 8 # Passe dies an deine CPU-Kerne an (oft Kerne/2 oder Kerne)
DEFAULT_EPOCHS = 40     # Erhöht für manuelle Experimente

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Reproduzierbarkeit
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    # torch.backends.cudnn.deterministic = True # Kann für maximale Reproduzierbarkeit gesetzt werden, aber Leistung kosten
    # torch.backends.cudnn.benchmark = False   # True kann schneller sein, wenn Input-Größen konstant sind

# --- Hilfsfunktionen (Daten, Modell, Metriken, Plotting, Fehleranalyse) ---
# (Die meisten dieser Funktionen sind identisch oder sehr ähnlich zu v3)

def split_data(source_dir: Path, train_dir: Path, val_dir: Path, split_ratio=0.8, seed=42):
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

def create_dataloaders(train_dir, val_dir, batch_size, num_workers, train_transforms, val_transforms):
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    persistent = num_workers > 0 and device.type == 'cuda'
    pin_memory = device.type == 'cuda'
    # Verhindere drop_last, wenn das letzte Batch <= 1 wäre, was BatchNorm Probleme bereiten kann
    train_drop_last = len(train_dataset) % batch_size <= 1 and len(train_dataset) > batch_size
    val_drop_last = len(val_dataset) % batch_size <= 1 and len(val_dataset) > batch_size

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=pin_memory, persistent_workers=persistent, drop_last=train_drop_last)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory, persistent_workers=persistent, drop_last=val_drop_last)
    print(f"  DataLoaders: Tr={len(train_dataset)} ({len(train_loader)} b), Vl={len(val_dataset)} ({len(val_loader)} b)")
    print(f"  Classes: {train_dataset.classes} (Map: {train_dataset.class_to_idx}) Workers: {num_workers} BS: {batch_size}")
    if train_dataset.class_to_idx.get('yes',-1)!=1 or train_dataset.class_to_idx.get('no',-1)!=0: print(f"*** W: Class map! {train_dataset.class_to_idx}")
    return train_loader, val_loader, train_dataset.class_to_idx

def calculate_pos_weight(train_dir, device_obj):
    try:
        n_no=len(glob.glob(str(train_dir/'no'/'*.png'))); n_yes=len(glob.glob(str(train_dir/'yes'/'*.png')))
        pw = 1.0 if n_yes==0 or n_no==0 else n_no/n_yes # Vermeide Division durch Null, falls eine Klasse fehlt
        print(f"  Pos_weight samples: {n_no} 'no', {n_yes} 'yes'. Calculated pos_weight: {pw:.2f}")
        return torch.tensor([pw], device=device_obj)
    except Exception as e: print(f"  Err calc pos_weight: {e}. Defaulting."); return torch.tensor([1.0], device=device_obj)

class CustomCNN(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.5, num_conv_blocks=4, first_layer_filters=64, filter_increase_factor=2.0):
        super(CustomCNN, self).__init__()
        self.conv_blocks = nn.ModuleList()
        current_channels = 3
        next_channels = first_layer_filters

        for i in range(num_conv_blocks):
            # Standard Block: Conv -> BN -> ReLU -> MaxPool
            block = nn.Sequential(
                nn.Conv2d(current_channels, next_channels, kernel_size=3, stride=1, padding='same', bias=False),
                nn.BatchNorm2d(next_channels),
                nn.ReLU(inplace=True),
                # Optional: Zweiten Conv-Layer vor dem Pool hinzufügen für mehr Komplexität
                # nn.Conv2d(next_channels, next_channels, kernel_size=3, stride=1, padding='same', bias=False),
                # nn.BatchNorm2d(next_channels),
                # nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.conv_blocks.append(block)
            current_channels = next_channels
            if i < num_conv_blocks -1 : # Nicht für den letzten Block die Kanäle erhöhen
                 next_channels = int(current_channels * filter_increase_factor)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(current_channels, num_classes) # current_channels ist Output des letzten Conv-Blocks
        self._initialize_weights()
    def forward(self, x):
        for block in self.conv_blocks: x = block(x)
        return self.fc(self.dropout(self.flatten(self.avgpool(x))))
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0, 0.01); nn.init.constant_(m.bias, 0)

def calculate_metrics(outputs, labels):
    try:
        probs=torch.sigmoid(outputs).detach().cpu().numpy();preds=(probs>0.5).astype(int).flatten();labels_np=labels.detach().cpu().numpy().flatten()
        acc=accuracy_score(labels_np,preds);prec=precision_score(labels_np,preds,average='binary',pos_label=1,zero_division=0)
        rec=recall_score(labels_np,preds,average='binary',pos_label=1,zero_division=0);f1=f1_score(labels_np,preds,average='binary',pos_label=1,zero_division=0)
    except Exception as e: print(f"  Err metrics: {e}"); acc,prec,rec,f1=0.,0.,0.,0.
    return acc,prec,rec,f1

def save_error_analysis(model, val_loader, device_obj, class_to_idx, output_dir, experiment_name):
    # (Wie in v3)
    print(f"  Saving error analysis for {experiment_name} to {output_dir}...")
    fn_dir=output_dir/"false_negatives"; fp_dir=output_dir/"false_positives"; fn_dir.mkdir(parents=True,exist_ok=True); fp_dir.mkdir(parents=True,exist_ok=True)
    model.eval(); yes_idx=class_to_idx.get('yes',1); no_idx=class_to_idx.get('no',0); fn_c=0; fp_c=0
    filepaths=[s[0] for s in val_loader.dataset.samples]; true_labels=[s[1] for s in val_loader.dataset.samples]; preds_list=[]
    use_amp_eval=(device_obj.type=='cuda')
    with torch.no_grad():
        for inputs,_ in val_loader:
            inputs=inputs.to(device_obj,non_blocking=True)
            with torch.amp.autocast(device_type=device_obj.type, enabled=use_amp_eval): outputs=model(inputs) # Aktualisiert für neuere PyTorch-Versionen
            preds_list.extend((torch.sigmoid(outputs)>0.5).int().cpu().flatten().tolist())
    for i,fp_str in enumerate(filepaths):
        fp=Path(fp_str)
        if true_labels[i]==yes_idx and preds_list[i]==no_idx: shutil.copy(fp, fn_dir/fp.name); fn_c+=1
        elif true_labels[i]==no_idx and preds_list[i]==yes_idx: shutil.copy(fp, fp_dir/fp.name); fp_c+=1
    print(f"    FNs saved: {fn_c}, FPs saved: {fp_c}")

def plot_training_history(history, output_dir, experiment_name):
    # (Wie in v3, umbenannt zu plot_training_history)
    print(f"  Plotting training history for {experiment_name} to {output_dir}...")
    if not history or 'val_f1' not in history or not history['val_f1']: print("    No history to plot."); return
    epochs=history.get('epoch',range(1,len(history['val_f1'])+1))
    plt.figure(figsize=(15,10));
    plt.subplot(2,1,1);
    if 'val_f1' in history: plt.plot(epochs,history['val_f1'],'b-',label='Val F1');
    if 'val_acc' in history: plt.plot(epochs,history['val_acc'],'c-',label='Val Acc')
    if 'val_precision' in history: plt.plot(epochs,history['val_precision'],'m-',label='Val Precision')
    if 'val_recall' in history: plt.plot(epochs,history['val_recall'],'y-',label='Val Recall')
    plt.legend(); plt.title(f'{experiment_name} - Validation Metrics'); plt.grid(True)
    plt.subplot(2,1,2);
    if 'train_loss' in history: plt.plot(epochs,history['train_loss'],'r-',label='Train Loss');
    if 'val_loss' in history: plt.plot(epochs,history['val_loss'],'g-',label='Val Loss')
    plt.legend(); plt.title(f'{experiment_name} - Loss'); plt.xlabel('Epoch'); plt.grid(True)
    plt.tight_layout();plt.savefig(output_dir/f"{experiment_name}_training_history.png");plt.close()

# --- Trainings-Loop (generisch für manuelle Experimente und finales Optuna-Training) ---
def train_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device_obj, loop_name, output_dir_path, save_model_as):
    # (Kombiniert und angepasst aus train_manual_experiment_loop und train_final_optuna_model)
    print(f"\n--- Training Loop: {loop_name} ({num_epochs} epochs) ---")
    model_save_path = output_dir_path / save_model_as
    best_model_wts = copy.deepcopy(model.state_dict()); best_val_f1 = 0.0
    history = {'epoch':[],'train_loss':[],'val_loss':[],'val_f1':[],'val_acc':[],'val_precision':[],'val_recall':[],'time_per_epoch':[]}
    use_amp = (device_obj.type == 'cuda'); scaler = GradScaler(enabled=use_amp)
    if use_amp: print(f"  AMP (Mixed Precision) enabled for '{loop_name}'.")
    total_train_time_start = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time(); print(f"\n  Epoch {epoch+1}/{num_epochs} for '{loop_name}'")
        model.train(); running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device_obj,non_blocking=True), labels.float().unsqueeze(1).to(device_obj,non_blocking=True)
            optimizer.zero_grad(set_to_none=True) # Effizienter für neuere PyTorch Versionen
            with torch.amp.autocast(device_type=device_obj.type, enabled=use_amp): # Aktualisiert
                outputs = model(inputs); loss = criterion(outputs, labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running_loss += loss.item() * inputs.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset); print(f"    Train Loss: {epoch_train_loss:.4f}")

        model.eval(); running_val_loss = 0.0; all_val_labels, all_val_outputs = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device_obj,non_blocking=True), labels.float().unsqueeze(1).to(device_obj,non_blocking=True)
                with torch.amp.autocast(device_type=device_obj.type, enabled=use_amp): outputs = model(inputs); val_batch_loss = criterion(outputs, labels)
                running_val_loss += val_batch_loss.item()*inputs.size(0); all_val_labels.append(labels); all_val_outputs.append(outputs)
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_acc,val_prec,val_rec,val_f1 = calculate_metrics(torch.cat(all_val_outputs), torch.cat(all_val_labels))
        print(f"    Val Loss: {epoch_val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} P: {val_prec:.4f} R: {val_rec:.4f}")

        epoch_time = time.time() - epoch_start_time
        history['epoch'].append(epoch+1); history['train_loss'].append(epoch_train_loss); history['val_loss'].append(epoch_val_loss)
        history['val_f1'].append(val_f1); history['val_acc'].append(val_acc); history['val_precision'].append(val_prec)
        history['val_recall'].append(val_rec); history['time_per_epoch'].append(epoch_time)

        current_lr = optimizer.param_groups[0]['lr'] # Get current learning rate
        print(f"    Current LR: {current_lr:.1e}")

        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau): scheduler.step(val_f1)
            else: scheduler.step()

        if val_f1 > best_val_f1:
            print(f"    Best Val F1 for '{loop_name}' improved ({best_val_f1:.4f} -> {val_f1:.4f}). Saving model to {model_save_path.name}...");
            best_val_f1=val_f1; best_model_wts=copy.deepcopy(model.state_dict()); torch.save(best_model_wts,model_save_path)

    total_time = time.time() - total_train_time_start
    print(f"\n  Training for '{loop_name}' done in {total_time//60:.0f}m {total_time%60:.0f}s. Best Val F1: {best_val_f1:.4f}.")
    if best_val_f1 > 0: model.load_state_dict(torch.load(model_save_path))
    else: print(f"  W: No best model saved for '{loop_name}' (F1 was 0). Using last state."); model.load_state_dict(best_model_wts)
    return model, history, total_time, best_val_f1


# --- Optuna Objective Function ---
def objective(trial, train_loader_obj, val_loader_obj, pos_weight_tensor_obj, device_obj, optuna_config_obj, experiment_name_base):
    lr = trial.suggest_float("lr", 1e-6, 5e-2, log=True) # Erweiterter LR-Bereich
    dropout_rate = trial.suggest_float("dropout", 0.1, 0.7)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 5e-2, log=True) # Erweiterter WD-Bereich
    num_conv_blocks = trial.suggest_int("num_conv_blocks", optuna_config_obj.get("min_conv_blocks", 3), optuna_config_obj.get("max_conv_blocks", 6)) # Bis zu 6 Blöcke
    first_layer_filters = trial.suggest_categorical("first_layer_filters", [32, 64, 96]) # Mehr Optionen für erste Filter
    filter_increase_factor = trial.suggest_float("filter_increase_factor", 1.5, 2.5) # Wie schnell Filter ansteigen

    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "SGD", "RMSprop"])

    model = CustomCNN(num_classes=1, dropout_rate=dropout_rate, num_conv_blocks=num_conv_blocks,
                      first_layer_filters=first_layer_filters, filter_increase_factor=filter_increase_factor).to(device_obj)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor_obj)

    if optimizer_name == "AdamW": optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        momentum = trial.suggest_float("sgd_momentum", 0.7, 0.99)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == "RMSprop":
        alpha = trial.suggest_float("rmsprop_alpha", 0.8, 0.999)
        optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=alpha, weight_decay=weight_decay)

    print(f"    Optuna Trial {trial.number}: lr={lr:.1e}, drp={dropout_rate:.2f}, wd={weight_decay:.1e}, blks={num_conv_blocks}, opt={optimizer_name}, flf={first_layer_filters}, fif={filter_increase_factor:.1f}")

    try:
        # Verwende hier den generischen train_loop für einen Optuna-Trial (aber ohne Modell speichern)
        # Wir brauchen keinen Scheduler für kurze Optuna-Trials, es sei denn, wir wollen ihn auch tunen
        _, history_trial, _, best_trial_val_f1 = train_loop(
            model, train_loader_obj, val_loader_obj, criterion, optimizer, None, # Kein Scheduler im Trial
            optuna_config_obj["epochs_per_trial"], device_obj,
            f"{experiment_name_base}_trial_{trial.number}", Path("."), "dummy_trial_model.pth" # Dummy Pfad, wird nicht wirklich genutzt
        )
        # Das Modell wird hier nicht gespeichert, nur der F1 Score ist wichtig
        trial.report(best_trial_val_f1, optuna_config_obj["epochs_per_trial"] -1) # Report am Ende des Trials
        if trial.should_prune(): print(f"      Trial {trial.number} pruned."); raise optuna.exceptions.TrialPruned()
        return best_trial_val_f1 # Optuna will den Wert, den es maximieren soll
    except optuna.exceptions.TrialPruned: return 0.0
    except Exception as e: print(f"    !! Trial {trial.number} failed: {e}"); return -1.0


# --- Haupt-Experimentier-Skript ---
if __name__ == "__main__":
    EXPERIMENTS_V4_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not BASE_DATA_DIR_FOR_RUN.exists() or not (TRAIN_DIR_RUN / 'yes').exists():
        print("Data split for this run not found, splitting data...")
        split_data(ANNOTATED_DIR, TRAIN_DIR_RUN, VAL_DIR_RUN, TRAIN_VAL_SPLIT, RANDOM_SEED)
    else: print(f"Using existing data split: {BASE_DATA_DIR_FOR_RUN}")

    normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    base_val_transforms = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), normalize_transform])
    base_train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(20), # Leicht mehr Rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Leicht mehr Jitter
        transforms.ToTensor(), normalize_transform])

    # --- Definition der Experiment-Sitzungen ---
    all_experiment_sessions = [
        {
            "session_name": "manual_v4_baseline_4blocks_adamw",
            "session_type": "manual_experiment",
            "config": {
                "train_transforms": base_train_transforms, "val_transforms": base_val_transforms,
                "optimizer_type": "AdamW", "lr": 0.0001, "weight_decay": 5e-5, # Angepasst
                "scheduler_type": "ReduceLROnPlateau", "scheduler_params": {"mode": 'max', "factor": 0.2, "patience": 7, "min_lr": 1e-7},
                "dropout_rate": 0.5, "epochs": 60, "num_conv_blocks": 4, "batch_size": 128, "first_layer_filters":64, "filter_increase_factor":2.0
            }
        },
        {
            "session_name": "optuna_v4_deep_search",
            "session_type": "optuna_search",
            "optuna_config": {
                "n_trials": 200, # DEUTLICH ERHÖHT
                "epochs_per_trial": 20, # Erhöht
                "min_conv_blocks": 3, "max_conv_blocks": 6, # Erweiterter Bereich
                "batch_size": 128, "num_workers": DEFAULT_NUM_WORKERS,
                "train_transforms": base_train_transforms, "val_transforms": base_val_transforms
            },
            "final_train_config": {
                "epochs": 100, # DEUTLICH ERHÖHT
                "batch_size": 128, "num_workers": DEFAULT_NUM_WORKERS,
                "train_transforms": base_train_transforms, "val_transforms": base_val_transforms
                # num_conv_blocks, first_layer_filters, filter_increase_factor kommen von Optuna
            }
        },
        {
            "session_name": "manual_v4_very_deep_6blocks_strong_aug",
            "session_type": "manual_experiment",
            "config": {
                "train_transforms": transforms.Compose([
                    transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.RandomHorizontalFlip(p=0.5),transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomAffine(degrees=30,translate=(0.2,0.2),scale=(0.8,1.2),shear=20),
                    transforms.ColorJitter(brightness=0.3,contrast=0.3,saturation=0.3,hue=0.15),
                    transforms.RandomPerspective(distortion_scale=0.4, p=0.4),
                    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                    transforms.ToTensor(),normalize_transform]),
                "val_transforms": base_val_transforms,
                "optimizer_type": "AdamW", "lr": 0.00003, "weight_decay": 1e-4,
                "scheduler_type": "ReduceLROnPlateau", "scheduler_params": {"mode":'max', "factor":0.2, "patience":10, "min_lr":1e-8},
                "dropout_rate": 0.6, "epochs": 80, "num_conv_blocks": 6, "batch_size": 64, # Kleinere Batch Size
                "first_layer_filters":64, "filter_increase_factor":1.8
            }
        },
         {
            "session_name": "manual_v4_onecyclelr_5blocks",
            "session_type": "manual_experiment",
            "config": {
                "train_transforms": base_train_transforms, "val_transforms": base_val_transforms,
                "optimizer_type": "AdamW", "lr": 0.001, "weight_decay": 1e-4, # LR ist max_lr für OneCycle
                "scheduler_type": "OneCycleLR",
                "scheduler_params": {"max_lr": 0.001, "total_steps": None}, # total_steps wird später gesetzt
                "dropout_rate": 0.5, "epochs": 50, "num_conv_blocks": 5, "batch_size": 128,
                "first_layer_filters":64, "filter_increase_factor":2.0
            }
        },
    ]

    # --- Durchführung aller Experiment-Sitzungen ---
    # (Die Hauptschleife bleibt sehr ähnlich zu v3, ruft aber die generalisierte train_loop auf)
    overall_summary = []
    for session_idx, session_details in enumerate(all_experiment_sessions):
        session_name = session_details["session_name"]
        session_type = session_details["session_type"]
        current_session_dir = EXPERIMENTS_V4_OUTPUT_DIR / session_name
        current_session_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n\n{'='*25} RUNNING SESSION {session_idx+1}/{len(all_experiment_sessions)}: {session_name} (Type: {session_type}) {'='*25}")

        current_bs = DEFAULT_BATCH_SIZE; current_nw = DEFAULT_NUM_WORKERS
        current_train_transforms = base_train_transforms; current_val_transforms = base_val_transforms

        if session_type == "manual_experiment":
            cfg = session_details["config"]
            current_bs = cfg.get("batch_size", DEFAULT_BATCH_SIZE)
            current_nw = cfg.get("num_workers", DEFAULT_NUM_WORKERS)
            current_train_transforms = cfg["train_transforms"]
            current_val_transforms = cfg["val_transforms"]
        elif session_type == "optuna_search":
            cfg = session_details["optuna_config"]
            current_bs = cfg.get("batch_size", DEFAULT_BATCH_SIZE)
            current_nw = cfg.get("num_workers", DEFAULT_NUM_WORKERS)
            current_train_transforms = cfg["train_transforms"]
            current_val_transforms = cfg["val_transforms"]

        train_loader_sess, val_loader_sess, class_to_idx_sess = create_dataloaders(
            TRAIN_DIR_RUN, VAL_DIR_RUN, current_bs, current_nw, current_train_transforms, current_val_transforms
        )
        pos_weight_tensor_sess = calculate_pos_weight(TRAIN_DIR_RUN, device)

        trained_model_sess = None; history_sess = None; train_time_sess = 0.0; best_params_from_optuna = {}; best_val_f1_sess = 0.0

        if session_type == "manual_experiment":
            cfg = session_details["config"]
            model_manual = CustomCNN(num_classes=1, dropout_rate=cfg["dropout_rate"], num_conv_blocks=cfg["num_conv_blocks"],
                                     first_layer_filters=cfg.get("first_layer_filters",64),
                                     filter_increase_factor=cfg.get("filter_increase_factor",2.0)).to(device)
            criterion_manual = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor_sess)
            optimizer_manual = getattr(optim, cfg["optimizer_type"])(model_manual.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay",0)) # Vereinfacht

            scheduler_manual = None
            if cfg.get("scheduler_type") == "StepLR": scheduler_manual = optim.lr_scheduler.StepLR(optimizer_manual, **cfg["scheduler_params"])
            elif cfg.get("scheduler_type") == "ReduceLROnPlateau": scheduler_manual = optim.lr_scheduler.ReduceLROnPlateau(optimizer_manual, **cfg["scheduler_params"])
            elif cfg.get("scheduler_type") == "OneCycleLR":
                 cfg["scheduler_params"]["total_steps"] = cfg["epochs"] * len(train_loader_sess) # Wichtig für OneCycleLR
                 scheduler_manual = optim.lr_scheduler.OneCycleLR(optimizer_manual, **cfg["scheduler_params"])


            trained_model_sess, history_sess, train_time_sess, best_val_f1_sess = train_loop(
                model_manual, train_loader_sess, val_loader_sess, criterion_manual, optimizer_manual, scheduler_manual,
                cfg["epochs"], device, session_name, current_session_dir, f"{session_name}_best_model.pth"
            )

        elif session_type == "optuna_search":
            optuna_cfg = session_details["optuna_config"]
            final_train_cfg = session_details["final_train_config"]
            optuna_study_name = f"optuna_study_{session_name}"
            print(f"\n  --- Starting Optuna Search for '{session_name}' ({optuna_cfg['n_trials']} Trials, {optuna_cfg['epochs_per_trial']} epochs/trial) ---")
            storage_name = f"sqlite:///{current_session_dir / optuna_study_name}.db"
            study = optuna.create_study(study_name=optuna_study_name, storage=storage_name, load_if_exists=True,
                                        direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=2)) # Angepasster Pruner
            study.optimize(lambda trial: objective(trial, train_loader_sess, val_loader_sess, pos_weight_tensor_sess, device, optuna_cfg, session_name),
                           n_trials=optuna_cfg["n_trials"], timeout=optuna_cfg.get("timeout_hours", 24)*3600) # Timeout in Stunden

            print(f"\n  --- Optuna Search Finished for '{session_name}' ---")
            if not study.best_trial or study.best_value <= -1.0:
                print(f"  ERROR: Optuna for '{session_name}' found no valid trials. Skipping final training.")
            else:
                best_trial = study.best_trial; best_params_from_optuna = best_trial.params
                print(f"  Best Optuna trial for '{session_name}': Value (F1): {best_trial.value:.4f}, Params: {best_params_from_optuna}")

                # Finales Training mit den besten Optuna-Parametern
                num_blocks_final = best_params_from_optuna.get("num_conv_blocks", final_train_cfg.get("base_num_conv_blocks", 4))
                flf_final = best_params_from_optuna.get("first_layer_filters", final_train_cfg.get("first_layer_filters",64))
                fif_final = best_params_from_optuna.get("filter_increase_factor", final_train_cfg.get("filter_increase_factor",2.0))

                model_final = CustomCNN(num_classes=1, dropout_rate=best_params_from_optuna['dropout'], num_conv_blocks=num_blocks_final,
                                        first_layer_filters=flf_final, filter_increase_factor=fif_final).to(device)
                criterion_final = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor_sess)
                optimizer_name_final = best_params_from_optuna.get("optimizer", "AdamW")
                if optimizer_name_final == "AdamW": optimizer_final = optim.AdamW(model_final.parameters(), lr=best_params_from_optuna['lr'], weight_decay=best_params_from_optuna['weight_decay'])
                elif optimizer_name_final == "SGD": optimizer_final = optim.SGD(model_final.parameters(), lr=best_params_from_optuna['lr'], momentum=best_params_from_optuna.get('sgd_momentum',0.9), weight_decay=best_params_from_optuna['weight_decay'])
                elif optimizer_name_final == "RMSprop": optimizer_final = optim.RMSprop(model_final.parameters(), lr=best_params_from_optuna['lr'], alpha=best_params_from_optuna.get('rmsprop_alpha',0.99), weight_decay=best_params_from_optuna['weight_decay'])

                # Einfacher Scheduler für finales Training oder keinen
                scheduler_final = optim.lr_scheduler.ReduceLROnPlateau(optimizer_final, mode='max', factor=0.2, patience=10, min_lr=1e-7)

                trained_model_sess, history_sess, train_time_sess, best_val_f1_sess = train_loop(
                    model_final, train_loader_sess, val_loader_sess, criterion_final, optimizer_final, scheduler_final,
                    final_train_cfg["epochs"], device, f"{session_name}_final_optuna_train", current_session_dir, f"{session_name}_optuna_final_best_model.pth"
                )
                # Optuna Visualisierungen speichern
                try:
                    fig_hist=optuna.visualization.plot_optimization_history(study); fig_hist.write_image(str(current_session_dir/f"{session_name}_optuna_history.png"))
                    fig_imp=optuna.visualization.plot_param_importances(study); fig_imp.write_image(str(current_session_dir/f"{session_name}_optuna_importances.png"))
                    fig_slice=optuna.visualization.plot_slice(study); fig_slice.write_image(str(current_session_dir/f"{session_name}_optuna_slice.png"))
                except Exception as e_vis: print(f"  Err plotting Optuna visuals: {e_vis}. Install plotly if missing.")
        else: print(f"  Unknown type: {session_type}. Skip."); continue

        if trained_model_sess:
            print(f"\n--- Final Eval for Session: {session_name} ---"); trained_model_sess.eval()
            all_final_labels, all_final_outputs = [], []
            with torch.no_grad():
                for inputs, labels in val_loader_sess:
                    inputs,labels = inputs.to(device),labels.float().unsqueeze(1).to(device)
                    with torch.amp.autocast(device_type=device.type,enabled=(device.type=='cuda')): outputs = trained_model_sess(inputs)
                    all_final_labels.append(labels); all_final_outputs.append(outputs)
            all_f_out_cat=torch.cat(all_final_outputs); all_f_lab_cat=torch.cat(all_final_labels)
            final_acc,final_prec,final_rec,final_f1 = calculate_metrics(all_f_out_cat, all_f_lab_cat)
            res_txt=(f"Sess: {session_name} (Type: {session_type})\n")
            if session_type=="optuna_search" and best_params_from_optuna: res_txt+=f"  Optuna Best: {best_params_from_optuna}\n  Optuna Best F1 (search): {study.best_value:.4f}\n"
            res_txt+=(f"  ValAcc: {final_acc:.4f}\n  ValP(yes): {final_prec:.4f}\n  ValR(yes): {final_rec:.4f}\n  ValF1(yes): {final_f1:.4f}\n  TrainTime: {train_time_sess//60:.0f}m {train_time_sess%60:.0f}s\n")
            final_preds=(torch.sigmoid(all_f_out_cat).detach().cpu().numpy()>0.5).astype(int).flatten(); final_labels_np=all_f_lab_cat.detach().cpu().numpy().flatten()
            cm=confusion_matrix(final_labels_np,final_preds); res_txt+="\nCM:\n"+f"Lbls: {list(class_to_idx_sess.keys())}\n"+str(cm)+"\n\nReport:\n"
            res_txt+=classification_report(final_labels_np,final_preds,target_names=list(class_to_idx_sess.keys()),zero_division=0)+"\n"
            print(res_txt)
            with open(current_session_dir / f"{session_name}_summary.txt", "w",
                      encoding='utf-8') as f:
                f.write(res_txt)
            overall_summary.append({"session_name":session_name,"type":session_type,"accuracy":final_acc,"precision":final_prec,"recall":final_rec,"f1_score":final_f1,"train_time_sec":train_time_sess,"best_optuna_params":best_params_from_optuna if best_params_from_optuna else "N/A"})
            save_error_analysis(trained_model_sess,val_loader_sess,device,class_to_idx_sess,current_session_dir,session_name)
            if history_sess: plot_training_history(history_sess,current_session_dir,session_name)
        else: print(f"  No model for {session_name}. Skip eval."); overall_summary.append({"session_name":session_name,"type":session_type,"f1_score":0.0,"train_time_sec":0,"error":"Train fail/skip"})

    print("\n\n{'='*25} OVERALL EXPERIMENT SUMMARY (V4) {'='*25}")
    for metrics in sorted(overall_summary,key=lambda x: x['f1_score'],reverse=True):
        print(f"\nSess: {metrics['session_name']} (Type: {metrics['type']})")
        if "error" in metrics: print(f"  Err: {metrics['error']}")
        else:
            print(f"  F1: {metrics['f1_score']:.4f} Acc: {metrics['accuracy']:.4f} P: {metrics['precision']:.4f} R: {metrics['recall']:.4f}")
            print(f"  Time: {metrics['train_time_sec']//60:.0f}m {metrics['train_time_sec']%60:.0f}s")
            if metrics['type']=="optuna_search" and metrics['best_optuna_params']!="N/A": print(f"  OptunaBest: {metrics['best_optuna_params']}")
    print("\n--- Experiment Script V4 Finished ---")