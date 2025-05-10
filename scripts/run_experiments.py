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

# --- 0. Projekt-Root definieren ---
# Annahme: Dieses Skript liegt in einem Unterordner (z.B. 'scripts') des Projekt-Roots
PROJECT_ROOT = Path(__file__).resolve().parent.parent # Geht eine Ebene hoch (von scripts/ zu Hackatron/)

# --- 1. Globale Konfiguration (Pfade, Basisparameter) ---
ANNOTATED_DIR = PROJECT_ROOT / "data_annotated"
BASE_DATA_DIR_FOR_EXPERIMENTS = PROJECT_ROOT / "data_split_for_experiments"
TRAIN_DIR_EXP = BASE_DATA_DIR_FOR_EXPERIMENTS / "train"
VAL_DIR_EXP = BASE_DATA_DIR_FOR_EXPERIMENTS / "validation"
EXPERIMENTS_OUTPUT_DIR = PROJECT_ROOT / "experiment_results"

TRAIN_VAL_SPLIT = 0.8
RANDOM_SEED = 42
IMG_SIZE = 250
# Standard-Trainingsparameter (können pro Experiment überschrieben werden)
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_WORKERS = 8
DEFAULT_EPOCHS = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Project Root: {PROJECT_ROOT}")

# Reproduzierbarkeit
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# --- Importiere Module aus src ---
# Stelle sicher, dass der src-Ordner im Python-Pfad ist oder verwende relative Imports,
# wenn dieses Skript selbst Teil eines größeren Pakets ist.
# Für einfache Ausführung aus scripts/:
import sys
sys.path.append(str(PROJECT_ROOT)) # Füge Projekt-Root zum sys.path hinzu

from src.data.preparation import split_data, create_dataloaders_experiment
from src.training.model_arch import CustomCNN
from src.evaluation.metrics_and_errors import calculate_metrics, save_error_analysis, plot_experiment_history
from src.training.engine import train_experiment_loop # Neuer Name für den Trainingsloop

# --- (Die Funktionen split_data, CustomCNN, calculate_metrics, save_error_analysis, plot_experiment_history, train_experiment_loop
#      müssen jetzt in den entsprechenden src-Modulen definiert sein und hier importiert werden.
#      Ich füge hier stark gekürzte Dummy-Versionen ein, damit das Skript lauffähig ist.
#      DU MUSST DEN CODE DIESER FUNKTIONEN AUS DEINEN MODULEN IN src/ HIERHER KOPIEREN ODER
#      SICHERSTELLEN, DASS DIE IMPORTS KORREKT FUNKTIONIEREN!)

# --- Beispielhafte Dummy-Funktionen (ERSETZE DIESE MIT DEINEN ECHTEN IMPORTS/FUNKTIONEN AUS src/) ---
# DIES IST NUR, DAMIT DAS SKRIPT HIER ALS GANZES STEHT.
# IN DEINEM PROJEKT SOLLTEN DIESE AUS src IMPORTIERT WERDEN!

# In src/data/preparation.py
# def split_data(source_dir, train_dir, val_dir, split_ratio=0.8, seed=42): ...
# def create_dataloaders_experiment(train_dir, val_dir, batch_size, num_workers, train_transforms_exp, val_transforms_exp, img_size, device_type): ...

# In src/training/model_arch.py
# class CustomCNN(nn.Module): ...

# In src/evaluation/metrics_and_errors.py
# def calculate_metrics(outputs, labels): ...
# def save_error_analysis(model, val_loader, device, class_to_idx, output_dir, experiment_name): ...
# def plot_experiment_history(history, output_dir, experiment_name): ...

# In src/training/engine.py
# def train_experiment_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, experiment_name, experiment_dir_path): ...
# (Diese Funktion train_experiment_loop würde den Inhalt der alten train_experiment Funktion enthalten)

# --- HIER ENDEN DIE DUMMY-FUNKTIONEN ---


# --- Haupt-Experimentier-Skript ---
if __name__ == "__main__":
    # --- Globales Setup ---
    EXPERIMENTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not BASE_DATA_DIR_FOR_EXPERIMENTS.exists() or not (TRAIN_DIR_EXP / 'yes').exists():
        print("Data split directory for experiments not found, splitting data...")
        split_data(ANNOTATED_DIR, TRAIN_DIR_EXP, VAL_DIR_EXP, TRAIN_VAL_SPLIT, RANDOM_SEED)
    else:
        print(f"Using existing data split from: {BASE_DATA_DIR_FOR_EXPERIMENTS}")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    base_val_transforms = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), normalize])

    # --- Experiment-Definitionen ---
    experiments = []

    # Experiment 1: Baseline (ähnlich deinem aktuellen guten Modell)
    exp1_train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(), normalize
    ])
    experiments.append({
        "name": "exp01_baseline_adamw_weighted",
        "train_transforms": exp1_train_transforms, "val_transforms": base_val_transforms,
        "optimizer_type": "AdamW", "lr": 0.0001, "weight_decay": 3e-5,
        "scheduler_type": None, "dropout_rate": 0.4, "epochs": DEFAULT_EPOCHS
    })

    # Experiment 2: SGD mit Momentum und StepLR Scheduler
    experiments.append({
        "name": "exp02_sgd_momentum_steplr",
        "train_transforms": copy.deepcopy(exp1_train_transforms), "val_transforms": base_val_transforms,
        "optimizer_type": "SGD", "lr": 0.01, "momentum": 0.9, "weight_decay": 5e-4,
        "scheduler_type": "StepLR", "scheduler_params": {"step_size": 10, "gamma": 0.1},
        "dropout_rate": 0.4, "epochs": DEFAULT_EPOCHS
    })

    # Experiment 3: Stärkere Augmentation mit RandomAffine
    exp3_train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(), normalize
    ])
    experiments.append({
        "name": "exp03_stronger_aug_adamw_plateau",
        "train_transforms": exp3_train_transforms, "val_transforms": base_val_transforms,
        "optimizer_type": "AdamW", "lr": 0.0001, "weight_decay": 3e-5,
        "scheduler_type": "ReduceLROnPlateau", "scheduler_params": {"mode": 'max', "factor": 0.1, "patience": 5, "verbose":True},
        "dropout_rate": 0.5, "epochs": DEFAULT_EPOCHS
    })

    # --- Experimente durchführen ---
    all_experiment_summary = []

    for exp_config in experiments:
        exp_name = exp_config["name"]
        current_experiment_dir = EXPERIMENTS_OUTPUT_DIR / exp_name
        current_experiment_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n\n===== RUNNING EXPERIMENT: {exp_name} =====")

        train_loader_exp, val_loader_exp, class_to_idx_exp = create_dataloaders_experiment(
            TRAIN_DIR_EXP, VAL_DIR_EXP, DEFAULT_BATCH_SIZE, DEFAULT_NUM_WORKERS,
            exp_config["train_transforms"], exp_config["val_transforms"], IMG_SIZE, device.type
        )
        model_exp = CustomCNN(num_classes=1, dropout_rate=exp_config.get("dropout_rate", 0.5)).to(device)
        pos_weight_tensor_exp = calculate_pos_weight(TRAIN_DIR_EXP, device) # Aus src.data.preparation
        criterion_exp = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor_exp)

        if exp_config["optimizer_type"] == "AdamW":
            optimizer_exp = optim.AdamW(model_exp.parameters(), lr=exp_config["lr"], weight_decay=exp_config.get("weight_decay", 0.01))
        elif exp_config["optimizer_type"] == "SGD":
            optimizer_exp = optim.SGD(model_exp.parameters(), lr=exp_config["lr"], momentum=exp_config.get("momentum", 0.9), weight_decay=exp_config.get("weight_decay", 0))
        else: optimizer_exp = optim.AdamW(model_exp.parameters(), lr=exp_config["lr"])

        scheduler_exp = None
        if exp_config.get("scheduler_type") == "StepLR": scheduler_exp = optim.lr_scheduler.StepLR(optimizer_exp, **exp_config["scheduler_params"])
        elif exp_config.get("scheduler_type") == "ReduceLROnPlateau": scheduler_exp = optim.lr_scheduler.ReduceLROnPlateau(optimizer_exp, **exp_config["scheduler_params"])

        trained_model_exp, history_exp, train_time_exp = train_experiment_loop(
            model_exp, train_loader_exp, val_loader_exp, criterion_exp, optimizer_exp, scheduler_exp,
            exp_config["epochs"], device, exp_name, current_experiment_dir # Übergibt den spezifischen Ordner
        )

        print(f"\n--- Final Evaluation for Experiment: {exp_name} ---")
        trained_model_exp.eval()
        all_final_labels_exp, all_final_outputs_exp = [], []
        with torch.no_grad():
            for inputs, labels in val_loader_exp:
                inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
                with autocast(enabled=(device.type == 'cuda')): outputs = trained_model_exp(inputs)
                all_final_labels_exp.append(labels); all_final_outputs_exp.append(outputs)
        all_final_outputs_exp = torch.cat(all_final_outputs_exp); all_final_labels_exp = torch.cat(all_final_labels_exp)
        final_acc, final_prec, final_rec, final_f1 = calculate_metrics(all_final_outputs_exp, all_final_labels_exp)

        results_text = (
            f"Experiment: {exp_name}\n"
            f"  Validation Accuracy: {final_acc:.4f}\n"
            f"  Validation Precision (yes): {final_prec:.4f}\n"
            f"  Validation Recall (yes): {final_rec:.4f}\n"
            f"  Validation F1-Score (yes): {final_f1:.4f}\n"
            f"  Training Time: {train_time_exp // 60:.0f}m {train_time_exp % 60:.0f}s\n"
        )
        final_preds_exp = (torch.sigmoid(all_final_outputs_exp).detach().cpu().numpy() > 0.5).astype(int).flatten()
        final_labels_np_exp = all_final_labels_exp.detach().cpu().numpy().flatten()
        cm_exp = confusion_matrix(final_labels_np_exp, final_preds_exp)
        results_text += "\nConfusion Matrix (Validation Set):\n"
        results_text += f"Labels: {list(class_to_idx_exp.keys())} (0: no, 1: yes)\n"
        results_text += str(cm_exp) + "\n"
        results_text += "\nClassification Report (Validation Set):\n"
        results_text += classification_report(final_labels_np_exp, final_preds_exp, target_names=list(class_to_idx_exp.keys()), zero_division=0) + "\n"

        print(results_text)
        with open(current_experiment_dir / f"{exp_name}_summary.txt", "w") as f:
            f.write(results_text)

        all_experiment_summary.append({
            "name": exp_name, "accuracy": final_acc, "precision": final_prec, "recall": final_rec, "f1_score": final_f1,
            "training_time_seconds": train_time_exp
        })
        save_error_analysis(trained_model_exp, val_loader_exp, device, class_to_idx_exp, current_experiment_dir, exp_name)
        plot_experiment_history(history_exp, current_experiment_dir, exp_name)


    # --- Ergebnisse zusammenfassen ---
    print("\n\n===== ALL EXPERIMENT RESULTS SUMMARY =====")
    for metrics in sorted(all_experiment_summary, key=lambda x: x['f1_score'], reverse=True): # Sortiere nach F1
        print(f"\nExperiment: {metrics['name']}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  Training Time: {metrics['training_time_seconds'] // 60:.0f}m {metrics['training_time_seconds'] % 60:.0f}s")

    print("\n--- Experiment Script Finished ---")