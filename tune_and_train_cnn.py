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
import optuna
from torch.cuda.amp import autocast, GradScaler

# --- 1. Konfiguration ---
ANNOTATED_DIR = Path("./data_annotated")
BASE_DATA_DIR = Path("./data_split_for_tuning")
TRAIN_DIR = BASE_DATA_DIR / "train"
VAL_DIR = BASE_DATA_DIR / "validation"
FINAL_MODEL_SAVE_PATH = Path("./best_tuned_cnn_model.pth")
BATCH_SIZE = 128
NUM_WORKERS = 8
N_TRIALS = 100
EPOCHS_PER_TRIAL = 15
FINAL_TRAINING_EPOCHS = 30 # Epochen für das finale Training nach Optuna
IMG_SIZE = 250
RANDOM_SEED = 42
TRAIN_VAL_SPLIT = 0.8 # 80% Training, 20% Validation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# --- 2. Datenaufteilung ---
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
        if not source_class_dir.exists(): continue
        images = glob.glob(str(source_class_dir / "*.png"))
        if not images: continue
        random.shuffle(images)
        split_idx = int(len(images) * split_ratio)
        train_images, val_images = images[:split_idx], images[split_idx:]
        print(f"  Class '{class_name}': {len(images)} total -> {len(train_images)} train, {len(val_images)} val")
        for img_path in train_images: shutil.copy(img_path, train_class_dir / Path(img_path).name)
        for img_path in val_images: shutil.copy(img_path, val_class_dir / Path(img_path).name)
    print("Data splitting complete.")

# --- 3. Datentransformationen & Laden ---
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
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    print(f"Using {num_workers} workers for DataLoaders.")
    persistent = num_workers > 0
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent)
    print(f"\nDataLoaders created: Train={len(train_dataset)}/{len(train_loader)} batches, Val={len(val_dataset)}/{len(val_loader)} batches")
    print(f"  Classes: {train_dataset.classes} (Mapping: {train_dataset.class_to_idx})")
    if train_dataset.class_to_idx.get('yes', -1) != 1 or train_dataset.class_to_idx.get('no', -1) != 0:
       print(f"\n*** WARNING: Class mapping incorrect! Actual: {train_dataset.class_to_idx}\n")
    return train_loader, val_loader, train_dataset.class_to_idx

# --- 4. Modell Definition ---
class CustomCNN(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.5):
        super(CustomCNN, self).__init__()
        def _make_block(in_c, out_c): return nn.Sequential(nn.Conv2d(in_c, out_c, 3, 1, 'same', bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        self.conv1, self.conv2, self.conv3, self.conv4 = _make_block(3, 64), _make_block(64, 128), _make_block(128, 256), _make_block(256, 512)
        self.avgpool, self.flatten, self.dropout, self.fc = nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout_rate), nn.Linear(512, num_classes)
        self._initialize_weights()
    def forward(self, x): return self.fc(self.dropout(self.flatten(self.avgpool(self.conv4(self.conv3(self.conv2(self.conv1(x))))))))
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0, 0.01); nn.init.constant_(m.bias, 0)

# --- 5. Metrik-Berechnung ---
def calculate_metrics(outputs, labels):
    probs = torch.sigmoid(outputs).detach().cpu().numpy()
    preds = (probs > 0.5).astype(int).flatten()
    labels = labels.detach().cpu().numpy().flatten()
    try:
        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='binary', pos_label=1, zero_division=0)
        recall = recall_score(labels, preds, average='binary', pos_label=1, zero_division=0)
        f1 = f1_score(labels, preds, average='binary', pos_label=1, zero_division=0)
    except Exception as e: print(f" Error metrics: {e}"); acc, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
    return acc, precision, recall, f1

# --- Funktion zum Anzeigen von False Negatives ---
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
            with autocast(enabled=(device.type == 'cuda')): outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().flatten().tolist()
            all_preds_list.extend(preds)
    for i in range(len(filepaths)):
        if all_labels_list[i] == yes_label_index and all_preds_list[i] == 0: false_negative_files.append(filepaths[i])
    print(f"Found {len(false_negative_files)} FN images.")
    if not false_negative_files: return
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

# --- Funktion für finales Training ---
def train_final_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path):
    best_model_wts = copy.deepcopy(model.state_dict()); best_val_f1_final = 0.0
    use_amp = (device.type == 'cuda'); scaler = GradScaler(enabled=use_amp)
    start_time = time.time()
    for epoch in range(num_epochs):
        print(f"\nFinal Epoch {epoch+1}/{num_epochs}")
        model.train(); running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp): outputs = model(inputs); loss = criterion(outputs, labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running_loss += loss.item() * inputs.size(0)
        print(f"  Train Loss: {running_loss / len(train_loader.dataset):.4f}")
        model.eval(); all_val_labels, all_val_outputs = [], []; running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
                with autocast(enabled=use_amp): outputs = model(inputs); loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0); all_val_labels.append(labels); all_val_outputs.append(outputs)
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_acc, val_prec, val_rec, val_f1 = calculate_metrics(torch.cat(all_val_outputs), torch.cat(all_val_labels))
        print(f"  Val Loss: {epoch_val_loss:.4f} Acc: {val_acc:.4f} Precision: {val_prec:.4f} Recall: {val_rec:.4f} F1: {val_f1:.4f}")
        if val_f1 > best_val_f1_final:
             print(f"  Best Val F1 improved ({best_val_f1_final:.4f} -> {val_f1:.4f}). Saving final model..."); best_val_f1_final = val_f1
             best_model_wts = copy.deepcopy(model.state_dict()); torch.save(best_model_wts, save_path)
    time_elapsed = time.time() - start_time
    print(f"\nFinal Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Val F1: {best_val_f1_final:4f}"); print(f"Final model saved to: {save_path}")
    model.load_state_dict(best_model_wts); return model

# --- 8. Hauptausführung mit Optuna ---
if __name__ == "__main__":
    if not BASE_DATA_DIR.exists() or not (TRAIN_DIR / 'yes').exists():
        print("Data split directory not found or incomplete, splitting data...")
        split_data(ANNOTATED_DIR, TRAIN_DIR, VAL_DIR, TRAIN_VAL_SPLIT, RANDOM_SEED)
    else: print("Data split directory already exists, using existing split.")
    train_loader, val_loader, class_to_idx = create_dataloaders(TRAIN_DIR, VAL_DIR, BATCH_SIZE, NUM_WORKERS)
    print("\nCalculating weight for positive class ('yes')...")
    try:
        num_no = len(glob.glob(str(TRAIN_DIR / 'no' / '*.png'))); num_yes = len(glob.glob(str(TRAIN_DIR / 'yes' / '*.png')))
        pos_weight_value = 1.0 if num_yes == 0 else num_no / num_yes
        print(f"  Found {num_no} 'no', {num_yes} 'yes' samples. Pos weight: {pos_weight_value:.2f}")
        pos_weight_tensor = torch.tensor([pos_weight_value], device=device)
    except Exception as e: print(f" Error calc weights: {e}. Defaulting."); pos_weight_tensor = torch.tensor([1.0], device=device)

    print(f"\n--- Starting Optuna Hyperparameter Search ({N_TRIALS} Trials) ---")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, train_loader, val_loader, pos_weight_tensor, device), n_trials=N_TRIALS)

    print(f"\n--- Optuna Search Finished ---"); print(f"Finished trials: {len(study.trials)}")
    if study.best_trial is None or study.best_value <= 0.0: # <= 0.0 weil F1 score nicht negativ sein kann
         print("\nERROR: Optuna search did not find any valid trials. Check logs. Cannot proceed.")
         exit()
    print("Best trial:"); best_trial = study.best_trial; print(f"  Value (Best Val F1): {best_trial.value:.4f}"); print("  Params: ")
    for key, value in best_trial.params.items(): print(f"    {key}: {value}")

    print("\n--- Starting Final Training with Best Hyperparameters ---")
    best_params = best_trial.params
    final_model = CustomCNN(num_classes=1, dropout_rate=best_params['dropout']).to(device)
    final_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    final_optimizer = optim.AdamW(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    # <<< torch.compile entfernt >>>
    print(f"Training final model for {FINAL_TRAINING_EPOCHS} epochs...")
    trained_final_model = train_final_model(final_model, train_loader, val_loader, final_criterion, final_optimizer, FINAL_TRAINING_EPOCHS, device, FINAL_MODEL_SAVE_PATH)

    print("\n--- Final Evaluation on Validation Set (using BEST TUNED model) ---")
    trained_final_model.eval()
    all_final_labels, all_final_outputs = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
            with autocast(enabled=(device.type == 'cuda')): outputs = trained_final_model(inputs)
            all_final_labels.append(labels); all_final_outputs.append(outputs)
    all_final_outputs = torch.cat(all_final_outputs); all_final_labels = torch.cat(all_final_labels)
    final_acc, final_prec, final_rec, final_f1 = calculate_metrics(all_final_outputs, all_final_labels)
    print(f"Final Tuned Val Acc: {final_acc:.4f}"); print(f"Final Tuned Val Precision: {final_prec:.4f}")
    print(f"Final Tuned Val Recall: {final_rec:.4f}"); print(f"Final Tuned Val F1-Score: {final_f1:.4f}")
    final_preds = (torch.sigmoid(all_final_outputs).detach().cpu().numpy() > 0.5).astype(int).flatten()
    final_labels_np = all_final_labels.detach().cpu().numpy().flatten()
    cm = confusion_matrix(final_labels_np, final_preds)
    print("\nConfusion Matrix (Validation Set - Tuned Model):"); print(f"Labels: {list(class_to_idx.keys())} (0: no, 1: yes)"); print(cm)
    print("\nClassification Report (Validation Set - Tuned Model):"); print(classification_report(final_labels_np, final_preds, target_names=list(class_to_idx.keys()), zero_division=0))
    find_and_show_false_negatives(trained_final_model, val_loader, device, class_to_idx, num_to_show=15)
    print("\n--- Script Finished ---")