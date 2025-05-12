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
from torch.cuda.amp import autocast, GradScaler

# --- 0. Projekt-Root definieren (Annahme: Skript liegt im Projekt-Root) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- 1. Globale Konfiguration (Pfade, Basisparameter) ---
ANNOTATED_DIR = PROJECT_ROOT / "data_annotated"
BASE_DATA_DIR_FOR_EXPERIMENTS = PROJECT_ROOT / "data_split_for_experiments" # Daten für diesen Lauf
TRAIN_DIR_EXP = BASE_DATA_DIR_FOR_EXPERIMENTS / "train"
VAL_DIR_EXP = BASE_DATA_DIR_FOR_EXPERIMENTS / "validation"
EXPERIMENTS_OUTPUT_DIR = PROJECT_ROOT / "experiment_results" # Hauptordner für Ergebnisse

TRAIN_VAL_SPLIT = 0.8
RANDOM_SEED = 42
IMG_SIZE = 250
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_WORKERS = 8
DEFAULT_EPOCHS = 30 # Für einzelne Experimente (kann pro Experiment überschrieben werden)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Project Root: {PROJECT_ROOT}")

# Reproduzierbarkeit
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# --- DUMMY-IMPLEMENTIERUNGEN (ERSETZE DIESE MIT DEINEN ECHTEN IMPORTS/FUNKTIONEN AUS src!) ---
# In src/data/preparation.py
# --- 2. Datenaufteilung ---
def split_data(source_dir: Path, train_dir: Path, val_dir: Path, split_ratio=0.8, seed=42):
    print(f"Splitting data from {source_dir} into {train_dir.parent}...")
    random.seed(seed)
    if train_dir.parent.exists():
        print(f"  Removing existing data directory: {train_dir.parent}")
        shutil.rmtree(train_dir.parent) # Löscht den gesamten data_split_for_experiments Ordner

    for class_name in ['yes', 'no']:
        # source_class_dir ist jetzt relativ zum source_dir (z.B. PROJECT_ROOT / "data_annotated" / "yes")
        source_class_dir = source_dir / class_name
        # Zielordner bleiben relativ zum train_dir und val_dir
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
            img_path_obj = Path(img_path_str) # Stelle sicher, dass es ein Path-Objekt ist
            shutil.copy(img_path_obj, train_class_dir / img_path_obj.name)
        for img_path_str in val_images:
            img_path_obj = Path(img_path_str)
            shutil.copy(img_path_obj, val_class_dir / img_path_obj.name)

    print("Data splitting complete.")

def create_dataloaders_experiment(train_dir, val_dir, batch_size, num_workers, train_transforms_exp, val_transforms_exp): # img_size, device_type entfernt
    print("DUMMY: Creating DataLoaders...")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms_exp)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms_exp)
    persistent, pin_memory = num_workers > 0, device.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent)
    print(f"  Loaders: Tr={len(train_dataset)}/{len(train_loader)}b, Vl={len(val_dataset)}/{len(val_loader)}b. Cls: {train_dataset.classes}")
    return train_loader, val_loader, train_dataset.class_to_idx

def calculate_pos_weight(train_dir, device_obj):
    print("DUMMY: Calc pos_weight...")
    try:
        n_no=len(glob.glob(str(train_dir/'no'/'*.png'))); n_yes=len(glob.glob(str(train_dir/'yes'/'*.png')))
        pw = 1.0 if n_yes == 0 else n_no / n_yes
        print(f"  {n_no} no, {n_yes} yes. PW: {pw:.2f}"); return torch.tensor([pw], device=device_obj)
    except: return torch.tensor([1.0], device=device_obj)

# In src/training/model_arch.py
class CustomCNN(nn.Module): # Basisarchitektur
    def __init__(self, num_classes=1, dropout_rate=0.5, num_conv_blocks=4): # num_conv_blocks hinzugefügt
        super(CustomCNN, self).__init__()
        def _make_block(in_c, out_c): return nn.Sequential(nn.Conv2d(in_c,out_c,3,1,'same',bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True), nn.MaxPool2d(2,2))
        self.conv_blocks = nn.ModuleList()
        current_channels = 3
        block_channels = [64, 128, 256, 512, 512, 512] # Max 6 Blöcke vordefiniert
        for i in range(min(num_conv_blocks, len(block_channels))): # Nutze num_conv_blocks
            self.conv_blocks.append(_make_block(current_channels, block_channels[i]))
            current_channels = block_channels[i]
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(current_channels, num_classes) # current_channels ist Output des letzten Conv-Blocks
    def forward(self, x):
        for block in self.conv_blocks: x = block(x)
        return self.fc(self.dropout(self.flatten(self.avgpool(x))))

# In src/evaluation/metrics_and_errors.py
def calculate_metrics(outputs, labels):
    probs = torch.sigmoid(outputs).detach().cpu().numpy(); preds = (probs > 0.5).astype(int).flatten(); labels = labels.detach().cpu().numpy().flatten()
    acc=accuracy_score(labels,preds); prec=precision_score(labels,preds,average='binary',pos_label=1,zero_division=0)
    rec=recall_score(labels,preds,average='binary',pos_label=1,zero_division=0); f1=f1_score(labels,preds,average='binary',pos_label=1,zero_division=0)
    return acc, prec, rec, f1

def save_error_analysis(model, val_loader, device_obj, class_to_idx, output_dir, experiment_name):
    print(f"DUMMY: Saving error analysis for {experiment_name} to {output_dir}...")
    fn_dir=output_dir/"false_negatives"; fp_dir=output_dir/"false_positives"; fn_dir.mkdir(exist_ok=True); fp_dir.mkdir(exist_ok=True)
    model.eval(); yes_idx=class_to_idx.get('yes',1); no_idx=class_to_idx.get('no',0); fn_c=0; fp_c=0
    filepaths=[s[0] for s in val_loader.dataset.samples]; true_labels=[s[1] for s in val_loader.dataset.samples]; preds_list=[]
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs=inputs.to(device_obj)
            with autocast(enabled=(device_obj.type=='cuda')): outputs=model(inputs)
            preds_list.extend((torch.sigmoid(outputs)>0.5).int().cpu().flatten().tolist())
    for i, fp_str in enumerate(filepaths):
        fp=Path(fp_str)
        if true_labels[i]==yes_idx and preds_list[i]==no_idx: shutil.copy(fp, fn_dir/fp.name); fn_c+=1
        elif true_labels[i]==no_idx and preds_list[i]==yes_idx: shutil.copy(fp, fp_dir/fp.name); fp_c+=1
    print(f"    FNs: {fn_c}, FPs: {fp_c}")

def plot_experiment_history(history, output_dir, experiment_name):
    print(f"DUMMY: Plotting history for {experiment_name} to {output_dir}...")
    if not history or 'val_f1' not in history or not history['val_f1']: print(" No history to plot."); return
    epochs = history.get('epoch', range(1, len(history['val_f1']) + 1))
    plt.figure(figsize=(12,8)); plt.subplot(2,1,1)
    plt.plot(epochs, history['val_f1'], 'b-', label='Val F1');
    if 'val_acc' in history: plt.plot(epochs, history['val_acc'], 'c-', label='Val Acc')
    plt.legend(); plt.title(f'{experiment_name} - Val Metrics'); plt.grid(True); plt.subplot(2,1,2)
    plt.plot(epochs, history['train_loss'], 'r-', label='Train Loss'); plt.plot(epochs, history['val_loss'], 'g-', label='Val Loss')
    plt.legend(); plt.title(f'{experiment_name} - Loss'); plt.xlabel('Epoch'); plt.grid(True)
    plt.savefig(output_dir / f"{experiment_name}_training_history.png"); plt.close()

# In src/training/engine.py
def train_experiment_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device_obj, experiment_name, experiment_dir_path):
    print(f"\n--- Training Experiment: {experiment_name} ({num_epochs} epochs) ---")
    model_save_path = experiment_dir_path / f"{experiment_name}_best_model.pth"
    best_model_wts = copy.deepcopy(model.state_dict()); best_val_f1 = 0.0
    history = {'epoch':[],'train_loss':[],'val_loss':[],'val_f1':[],'val_acc':[],'val_precision':[],'val_recall':[],'time_per_epoch':[]}
    use_amp = (device_obj.type == 'cuda'); scaler = GradScaler(enabled=use_amp)
    if use_amp: print("  AMP enabled.")
    total_train_time_start = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time(); print(f"\n  Epoch {epoch+1}/{num_epochs}")
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
                with autocast(enabled=use_amp): outputs = model(inputs); loss = criterion(outputs, labels)
                running_val_loss += loss.item()*inputs.size(0); all_val_labels.append(labels); all_val_outputs.append(outputs)
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_acc,val_prec,val_rec,val_f1 = calculate_metrics(torch.cat(all_val_outputs), torch.cat(all_val_labels))
        print(f"    Val Loss: {epoch_val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} P: {val_prec:.4f} R: {val_rec:.4f}")
        epoch_time = time.time() - epoch_start_time
        for k,v in zip(history.keys(),[epoch+1,epoch_train_loss,epoch_val_loss,val_f1,val_acc,val_prec,val_rec,epoch_time]): history[k].append(v)
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau): scheduler.step(val_f1)
            else: scheduler.step()
        if val_f1 > best_val_f1: print(f"    Best Val F1 imp. ({best_val_f1:.4f} -> {val_f1:.4f}). Save model..."); best_val_f1=val_f1; best_model_wts=copy.deepcopy(model.state_dict()); torch.save(best_model_wts,model_save_path)
    total_time = time.time() - total_train_time_start
    print(f"\n  Training for {experiment_name} done in {total_time//60:.0f}m {total_time%60:.0f}s. Best F1: {best_val_f1:.4f}. Saved: {model_save_path.relative_to(PROJECT_ROOT)}")
    model.load_state_dict(torch.load(model_save_path)); return model, history, total_time
# --- HIER ENDEN DIE DUMMY-FUNKTIONEN ---


# --- Haupt-Experimentier-Skript ---
if __name__ == "__main__":
    EXPERIMENTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not BASE_DATA_DIR_FOR_EXPERIMENTS.exists() or not (TRAIN_DIR_EXP / 'yes').exists():
        print("Data split directory for experiments not found, splitting data...")
        split_data(ANNOTATED_DIR, TRAIN_DIR_EXP, VAL_DIR_EXP, TRAIN_VAL_SPLIT, RANDOM_SEED)
    else:
        print(f"Using existing data split from: {BASE_DATA_DIR_FOR_EXPERIMENTS}")

    normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    base_val_transforms = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), normalize_transform])

    # --- Experiment-Definitionen ---
    experiments = []

    # Exp 1: Baseline
    exp1_train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(), normalize_transform])
    experiments.append(
        {"name": "exp01_baseline", "train_transforms": exp1_train_transforms, "val_transforms": base_val_transforms,
         "optimizer_type": "AdamW", "lr": 0.0001, "weight_decay": 3e-5, "scheduler_type": "ReduceLROnPlateau",
         "scheduler_params": {"mode": 'max', "factor": 0.1, "patience": 5, "min_lr": 1e-6},  # <<< verbose entfernt
         "dropout_rate": 0.4, "epochs": DEFAULT_EPOCHS, "num_conv_blocks": 4})

    # Exp 2: SGD
    experiments.append({"name":"exp02_sgd_momentum_steplr","train_transforms":copy.deepcopy(exp1_train_transforms),"val_transforms":base_val_transforms,
                        "optimizer_type":"SGD","lr":0.01,"momentum":0.9,"weight_decay":5e-4,"scheduler_type":"StepLR",
                        "scheduler_params":{"step_size":10,"gamma":0.1},"dropout_rate":0.4,"epochs":DEFAULT_EPOCHS, "num_conv_blocks": 4})

    # Exp 3: Stärkere Augmentation
    exp3_train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.RandomHorizontalFlip(p=0.5),transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(degrees=25,translate=(0.15,0.15),scale=(0.85,1.15),shear=15),
        transforms.ColorJitter(brightness=0.25,contrast=0.25,saturation=0.25,hue=0.15),
        transforms.ToTensor(),normalize_transform])
    experiments.append(
        {"name": "exp03_stronger_aug", "train_transforms": exp3_train_transforms, "val_transforms": base_val_transforms,
         "optimizer_type": "AdamW", "lr": 0.0001, "weight_decay": 3e-5, "scheduler_type": "ReduceLROnPlateau",
         "scheduler_params": {"mode": 'max', "factor": 0.1, "patience": 5, "min_lr": 1e-6},  # <<< verbose entfernt
         "dropout_rate": 0.5, "epochs": DEFAULT_EPOCHS, "num_conv_blocks": 4})

    # Exp 4: Tiefere Architektur (5 Conv-Blöcke)
    experiments.append(
        {"name": "exp03_stronger_aug", "train_transforms": exp3_train_transforms, "val_transforms": base_val_transforms,
         "optimizer_type": "AdamW", "lr": 0.0001, "weight_decay": 3e-5, "scheduler_type": "ReduceLROnPlateau",
         "scheduler_params": {"mode": 'max', "factor": 0.1, "patience": 5, "min_lr": 1e-6},  # <<< verbose entfernt
         "dropout_rate": 0.5, "epochs": DEFAULT_EPOCHS, "num_conv_blocks": 4})

    # Exp 5: RMSprop Optimizer
    experiments.append({"name": "exp05_rmsprop", "train_transforms": copy.deepcopy(exp1_train_transforms),
                        "val_transforms": base_val_transforms,
                        "optimizer_type": "RMSprop", "lr": 0.0001, "weight_decay": 1e-5,
                        "scheduler_type": "ReduceLROnPlateau",
                        "scheduler_params": {"mode": 'max', "factor": 0.1, "patience": 5, "min_lr": 1e-6},
                        # <<< verbose entfernt
                        "dropout_rate": 0.4, "epochs": DEFAULT_EPOCHS, "num_conv_blocks": 4})

    # Exp 6: Sehr kleine Lernrate mit Plateau Scheduler
    experiments.append({"name": "exp06_vlow_lr_plateau", "train_transforms": copy.deepcopy(exp1_train_transforms),
                        "val_transforms": base_val_transforms,
                        "optimizer_type": "AdamW", "lr": 1e-5, "weight_decay": 1e-5,
                        "scheduler_type": "ReduceLROnPlateau",
                        "scheduler_params": {"mode": 'max', "factor": 0.2, "patience": 3, "min_lr": 1e-7},
                        # <<< verbose entfernt
                        "dropout_rate": 0.3, "epochs": DEFAULT_EPOCHS + 10, "num_conv_blocks": 4})

    # Exp 7: Weniger Augmentation
    exp7_train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),normalize_transform])
    experiments.append(
        {"name": "exp07_less_aug", "train_transforms": exp7_train_transforms, "val_transforms": base_val_transforms,
         "optimizer_type": "AdamW", "lr": 0.0001, "weight_decay": 3e-5, "scheduler_type": "ReduceLROnPlateau",
         "scheduler_params": {"mode": 'max', "factor": 0.1, "patience": 5, "min_lr": 1e-6},  # <<< verbose entfernt
         "dropout_rate": 0.3, "epochs": DEFAULT_EPOCHS, "num_conv_blocks": 4})

    # Exp 8: Höhere Dropout Rate
    experiments.append({"name": "exp08_high_dropout", "train_transforms": copy.deepcopy(exp1_train_transforms),
                        "val_transforms": base_val_transforms,
                        "optimizer_type": "AdamW", "lr": 0.0001, "weight_decay": 3e-5,
                        "scheduler_type": "ReduceLROnPlateau",
                        "scheduler_params": {"mode": 'max', "factor": 0.1, "patience": 5, "min_lr": 1e-6},
                        # <<< verbose entfernt
                        "dropout_rate": 0.65, "epochs": DEFAULT_EPOCHS, "num_conv_blocks": 4})


    all_experiment_summary = []
    for exp_config in experiments:
        exp_name = exp_config["name"]
        current_experiment_dir = EXPERIMENTS_OUTPUT_DIR / exp_name
        current_experiment_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n\n===== RUNNING EXPERIMENT: {exp_name} =====")

        current_batch_size=exp_config.get("batch_size",DEFAULT_BATCH_SIZE); current_num_workers=exp_config.get("num_workers",DEFAULT_NUM_WORKERS)
        train_loader_exp, val_loader_exp, class_to_idx_exp = create_dataloaders_experiment(
            TRAIN_DIR_EXP, VAL_DIR_EXP, current_batch_size, current_num_workers,
            exp_config["train_transforms"], exp_config["val_transforms"]
        )
        # Modell mit num_conv_blocks erstellen
        num_blocks = exp_config.get("num_conv_blocks", 4)
        model_exp = CustomCNN(num_classes=1, dropout_rate=exp_config.get("dropout_rate",0.5), num_conv_blocks=num_blocks).to(device)

        pos_weight_tensor_exp = calculate_pos_weight(TRAIN_DIR_EXP, device)
        criterion_exp = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor_exp)

        if exp_config["optimizer_type"] == "AdamW":
            optimizer_exp = optim.AdamW(model_exp.parameters(), lr=exp_config["lr"], weight_decay=exp_config.get("weight_decay",0.01))
        elif exp_config["optimizer_type"] == "SGD":
            optimizer_exp = optim.SGD(model_exp.parameters(), lr=exp_config["lr"], momentum=exp_config.get("momentum",0.9), weight_decay=exp_config.get("weight_decay",0))
        elif exp_config["optimizer_type"] == "RMSprop": # <<< NEU
             optimizer_exp = optim.RMSprop(model_exp.parameters(), lr=exp_config["lr"], weight_decay=exp_config.get("weight_decay",0), alpha=0.99, eps=1e-08)
        else: optimizer_exp = optim.AdamW(model_exp.parameters(), lr=exp_config["lr"])

        scheduler_exp = None
        if exp_config.get("scheduler_type") == "StepLR": scheduler_exp = optim.lr_scheduler.StepLR(optimizer_exp, **exp_config["scheduler_params"])
        elif exp_config.get("scheduler_type") == "ReduceLROnPlateau":
            scheduler_exp = optim.lr_scheduler.ReduceLROnPlateau(optimizer_exp, **exp_config["scheduler_params"])

        trained_model_exp, history_exp, train_time_exp = train_experiment_loop(
            model_exp, train_loader_exp, val_loader_exp, criterion_exp, optimizer_exp, scheduler_exp,
            exp_config["epochs"], device, exp_name, current_experiment_dir)

        print(f"\n--- Final Evaluation for Experiment: {exp_name} ---")
        trained_model_exp.eval()
        all_final_labels_exp, all_final_outputs_exp = [], []
        with torch.no_grad():
            for inputs, labels in val_loader_exp:
                inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
                with autocast(enabled=(device.type=='cuda')): outputs = trained_model_exp(inputs)
                all_final_labels_exp.append(labels); all_final_outputs_exp.append(outputs)
        all_final_outputs_exp=torch.cat(all_final_outputs_exp); all_final_labels_exp=torch.cat(all_final_labels_exp)
        final_acc, final_prec, final_rec, final_f1 = calculate_metrics(all_final_outputs_exp, all_final_labels_exp)

        results_text = (f"Experiment: {exp_name}\n"+f"  Validation Accuracy: {final_acc:.4f}\n"+
                        f"  Validation Precision (yes): {final_prec:.4f}\n"+f"  Validation Recall (yes): {final_rec:.4f}\n"+
                        f"  Validation F1-Score (yes): {final_f1:.4f}\n"+f"  Training Time: {train_time_exp//60:.0f}m {train_time_exp%60:.0f}s\n")
        final_preds_exp=(torch.sigmoid(all_final_outputs_exp).detach().cpu().numpy()>0.5).astype(int).flatten()
        final_labels_np_exp=all_final_labels_exp.detach().cpu().numpy().flatten()
        cm_exp = confusion_matrix(final_labels_np_exp, final_preds_exp)
        results_text += "\nCM:\n"+f"Labels: {list(class_to_idx_exp.keys())}\n"+str(cm_exp)+"\n\nReport:\n"
        results_text += classification_report(final_labels_np_exp,final_preds_exp,target_names=list(class_to_idx_exp.keys()),zero_division=0)+"\n"
        print(results_text)
        with open(current_experiment_dir / f"{exp_name}_summary.txt", "w") as f:
            f.write(results_text)
        all_experiment_summary.append({"name":exp_name,"accuracy":final_acc,"precision":final_prec,"recall":final_rec,"f1_score":final_f1,"training_time_seconds":train_time_exp})
        save_error_analysis(trained_model_exp,val_loader_exp,device,class_to_idx_exp,current_experiment_dir,exp_name)
        plot_experiment_history(history_exp,current_experiment_dir,exp_name)

    print("\n\n===== ALL EXPERIMENT RESULTS SUMMARY =====")
    for metrics in sorted(all_experiment_summary, key=lambda x: x['f1_score'], reverse=True):
        print(f"\nExperiment: {metrics['name']}\n  F1: {metrics['f1_score']:.4f} Acc: {metrics['accuracy']:.4f} P: {metrics['precision']:.4f} R: {metrics['recall']:.4f} Time: {metrics['training_time_seconds']//60:.0f}m {metrics['training_time_seconds']%60:.0f}s")
    print("\n--- Experiment Script Finished ---")