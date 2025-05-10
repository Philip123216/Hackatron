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
import copy # Zum Speichern des besten Modells
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import math

# <<< NEU: Imports für AMP >>>
from torch.cuda.amp import autocast, GradScaler

# --- 1. Konfiguration ---
# Pfade
ANNOTATED_DIR = Path("./data_annotated") # Ordner mit deinen 'yes'/'no' Unterordnern
BASE_DATA_DIR = Path("./data")          # Hauptordner für Train/Val Split
TRAIN_DIR = BASE_DATA_DIR / "train"
VAL_DIR = BASE_DATA_DIR / "validation"
MODEL_SAVE_PATH = Path("./best_custom_cnn_model_optimized.pth") # Neuer Name für optimiertes Modell

# Trainingsparameter
# <<< OPTIMIERUNG: Batch-Größe erhöht >>>
BATCH_SIZE = 128      # Erhöht für starke GPU (z.B. 64, 128, 256). Anpassen falls Out-of-Memory.
LEARNING_RATE = 0.001
NUM_EPOCHS = 30       # Kann ggf. angepasst werden
TRAIN_VAL_SPLIT = 0.8
RANDOM_SEED = 42

# Bildparameter
IMG_SIZE = 250

# Gerät wählen (GPU, falls verfügbar)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Reproduzierbarkeit sicherstellen
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# --- 2. Datenaufteilung: Train/Validation Split ---
def split_data(source_dir, train_dir, val_dir, split_ratio=0.8, seed=42):
    """ Teilt Bilder aus source_dir/yes und source_dir/no in train und val Ordner auf. """
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
            print(f"  Warning: Source directory {source_class_dir} not found.")
            continue

        images = glob.glob(str(source_class_dir / "*.png"))
        if not images:
            print(f"  Warning: No PNG images found in {source_class_dir}.")
            continue

        random.shuffle(images)
        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        print(f"  Class '{class_name}': {len(images)} total -> {len(train_images)} train, {len(val_images)} val")

        for img_path in train_images:
            shutil.copy(img_path, train_class_dir / Path(img_path).name)
        for img_path in val_images:
            shutil.copy(img_path, val_class_dir / Path(img_path).name)

    print("Data splitting complete.")

# --- 3. Datentransformationen & Laden ---
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    normalize
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalize
])

def create_dataloaders(train_dir, val_dir, batch_size):
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    # <<< OPTIMIERUNG: num_workers erhöht >>>
    # Erhöhe num_workers (z.B. auf 8, 12, 16 - abhängig von CPU-Kernen)
    # Verwende persistent_workers=True (falls verfügbar) für schnellere Starts nach erster Epoche
    num_dataloader_workers = 8 # Anpassen nach Bedarf und CPU-Kernen
    print(f"Using {num_dataloader_workers} workers for DataLoaders.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_dataloader_workers, pin_memory=True if device == 'cuda' else False,
                              persistent_workers=True if num_dataloader_workers > 0 else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_dataloader_workers, pin_memory=True if device == 'cuda' else False,
                            persistent_workers=True if num_dataloader_workers > 0 else False)

    print(f"\nDataLoaders created:")
    print(f"  Train dataset: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Validation dataset: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Classes: {train_dataset.classes} (Mapping: {train_dataset.class_to_idx})")

    if train_dataset.class_to_idx.get('yes', -1) != 1 or train_dataset.class_to_idx.get('no', -1) != 0:
       print("\n*** WARNING: Class mapping might be incorrect! Expected {'no': 0, 'yes': 1} ***")
       print(f"Actual mapping: {train_dataset.class_to_idx}\n")

    return train_loader, val_loader, train_dataset.class_to_idx

# --- 4. Modell Definition (Custom CNN) ---
class CustomCNN(nn.Module):
    def __init__(self, num_classes=1):
        super(CustomCNN, self).__init__()

        def _make_block(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self.conv1 = _make_block(3, 64)
        self.conv2 = _make_block(64, 128)
        self.conv3 = _make_block(128, 256)
        self.conv4 = _make_block(256, 512)
        # self.conv5 = _make_block(512, 512) # Optional 5. Block

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # if hasattr(self, 'conv5'): x = self.conv5(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# --- 5. Metrik-Berechnung ---
def calculate_metrics(outputs, labels):
    """ Berechnet Metriken für binäre Klassifikation. """
    probs = torch.sigmoid(outputs).detach().cpu().numpy()
    preds = (probs > 0.5).astype(int).flatten()
    labels = labels.detach().cpu().numpy().flatten()
    try:
        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='binary', pos_label=1, zero_division=0)
        recall = recall_score(labels, preds, average='binary', pos_label=1, zero_division=0)
        f1 = f1_score(labels, preds, average='binary', pos_label=1, zero_division=0)
    except Exception as e:
        print(f"  Error calculating metrics: {e}")
        acc, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
    return acc, precision, recall, f1

# --- Funktion zum Anzeigen von False Negatives (wie vorher) ---
def find_and_show_false_negatives(model, val_loader, device, class_to_idx, num_to_show=10):
    """
    Findet False Negatives (echte 'yes', vorhergesagt 'no') im Validation Set
    und zeigt einige davon an.
    """
    print("\n--- Searching for False Negatives (Actual: yes, Predicted: no) ---")
    model.eval()
    false_negative_files = []
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    yes_label_index = class_to_idx.get('yes', -1)
    no_pred_index = 0
    if yes_label_index == -1: return
    if not hasattr(val_loader, 'dataset'): return

    filepaths = [s[0] for s in val_loader.dataset.samples]
    all_labels_list = [s[1] for s in val_loader.dataset.samples]
    all_preds_list = []
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            # <<< OPTIMIERUNG: autocast auch hier für Konsistenz >>>
            with autocast(enabled=(device.type == 'cuda')):
                outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int().cpu().flatten().tolist()
            all_preds_list.extend(preds)

    for i in range(len(filepaths)):
        if all_labels_list[i] == yes_label_index and all_preds_list[i] == no_pred_index:
            false_negative_files.append(filepaths[i])

    print(f"Found {len(false_negative_files)} False Negative images.")
    if not false_negative_files: return

    print(f"Showing up to {num_to_show} False Negative images...")
    num_files_to_show = min(num_to_show, len(false_negative_files))
    if num_files_to_show == 0: return
    num_cols = 5
    num_rows = math.ceil(num_files_to_show / num_cols)
    plt.figure(figsize=(15, 3 * num_rows))
    for i, img_path in enumerate(false_negative_files[:num_files_to_show]):
        try:
            img = Image.open(img_path)
            ax = plt.subplot(num_rows, num_cols, i + 1)
            ax.imshow(img)
            ax.set_title(f"FN: {Path(img_path).name}\nTrue: yes, Pred: no", fontsize=8)
            ax.axis("off")
        except Exception as e: print(f"Could not load/show image {img_path}: {e}")
    plt.tight_layout()
    plt.show()

# --- 6. Trainings- und Validierungs-Loop mit Optimierungen ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_save_path):
    """ Trainiert und validiert das Modell mit AMP. """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_metric = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
               'train_f1': [], 'val_f1': [], 'val_precision': [], 'val_recall': []}

    # <<< OPTIMIERUNG: GradScaler für AMP initialisieren (nur wenn CUDA verwendet wird) >>>
    use_amp = (device.type == 'cuda')
    scaler = GradScaler(enabled=use_amp)
    if use_amp: print("Automatic Mixed Precision (AMP) enabled.")

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 10)

        # --- Trainingsphase ---
        model.train()
        running_loss = 0.0
        all_train_labels = []
        all_train_outputs = []

        batch_start_time = time.time() # Zeitmessung pro Batch
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True) # non_blocking kann bei pin_memory helfen
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True) # set_to_none=True kann etwas schneller sein

            # <<< OPTIMIERUNG: autocast für Forward Pass >>>
            with autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # <<< OPTIMIERUNG: scaler für Backward Pass und Optimizer Step >>>
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            all_train_labels.append(labels.detach())
            all_train_outputs.append(outputs.detach()) # Wichtig: Logits für Metrikberechnung

            # Fortschrittsanzeige
            if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                 batch_time = time.time() - batch_start_time
                 batches_left = len(train_loader) - (i + 1)
                 est_time_left_epoch = batches_left * (batch_time / (i+1)) if (i+1) > 0 else 0
                 print(f"  Batch {i+1}/{len(train_loader)}, Avg Loss: {running_loss / ((i+1)*train_loader.batch_size):.4f}, "
                       f"Time/Batch: {batch_time/(i+1):.2f}s, Est. Epoch Time Left: {est_time_left_epoch:.0f}s")


        epoch_train_loss = running_loss / len(train_loader.dataset)
        all_train_outputs_cpu = torch.cat(all_train_outputs) # Auf CPU für Metriken
        all_train_labels_cpu = torch.cat(all_train_labels)
        train_acc, train_prec, train_rec, train_f1 = calculate_metrics(all_train_outputs_cpu, all_train_labels_cpu)

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        print(f"  Train Loss: {epoch_train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f}")

        # --- Validierungsphase ---
        model.eval()
        running_val_loss = 0.0
        all_val_labels = []
        all_val_outputs = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

                # <<< OPTIMIERUNG: autocast auch hier >>>
                with autocast(enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                running_val_loss += loss.item() * inputs.size(0)
                all_val_labels.append(labels)
                all_val_outputs.append(outputs)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        all_val_outputs_cpu = torch.cat(all_val_outputs)
        all_val_labels_cpu = torch.cat(all_val_labels)
        val_acc, val_prec, val_rec, val_f1 = calculate_metrics(all_val_outputs_cpu, all_val_labels_cpu)

        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_rec)
        history['val_f1'].append(val_f1)
        print(f"  Val Loss: {epoch_val_loss:.4f} Acc: {val_acc:.4f} Precision: {val_prec:.4f} Recall: {val_rec:.4f} F1: {val_f1:.4f}")

        current_val_metric = val_f1
        if current_val_metric > best_val_metric:
            print(f"  Validation F1 improved ({best_val_metric:.4f} --> {current_val_metric:.4f}). Saving model...")
            best_val_metric = current_val_metric
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, model_save_path)

    time_elapsed = time.time() - start_time
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Validation F1-Score: {best_val_metric:4f}")
    print(f"Best model saved to: {model_save_path}")

    model.load_state_dict(best_model_wts)
    return model, history

# --- 7. Hauptausführung ---
if __name__ == "__main__":
    # 1. Daten aufteilen
    if not BASE_DATA_DIR.exists() or not (TRAIN_DIR / 'yes').exists():
        print("Data split directory not found or incomplete, splitting data...")
        split_data(ANNOTATED_DIR, TRAIN_DIR, VAL_DIR, TRAIN_VAL_SPLIT, RANDOM_SEED)
    else:
        print("Data split directory already exists, skipping split.")

    # 2. DataLoaders erstellen
    train_loader, val_loader, class_to_idx = create_dataloaders(TRAIN_DIR, VAL_DIR, BATCH_SIZE)

    # 3. Modell instantiieren und auf Gerät verschieben
    model = CustomCNN(num_classes=1).to(device)

    # 4. Gewicht für Weighted Loss berechnen
    print("\nCalculating weight for positive class ('yes')...")
    try:
        num_no_train = len(glob.glob(str(TRAIN_DIR / 'no' / '*.png')))
        num_yes_train = len(glob.glob(str(TRAIN_DIR / 'yes' / '*.png')))
        if num_yes_train == 0:
            print("  Warning: No 'yes' samples found. Using default weight (1).")
            pos_weight_value = 1.0
        else:
            pos_weight_value = num_no_train / num_yes_train
            print(f"  Found {num_no_train} 'no' samples and {num_yes_train} 'yes' samples in training set.")
            print(f"  Calculated positive weight: {pos_weight_value:.2f}")
        pos_weight_tensor = torch.tensor([pos_weight_value], device=device)
    except Exception as e:
        print(f"  Error calculating weights: {e}. Using default weight (1).")
        pos_weight_tensor = torch.tensor([1.0], device=device)

    # 5. Prüfen ob Training nötig oder übersprungen werden kann
    if MODEL_SAVE_PATH.exists():
        print(f"\nFound existing model at {MODEL_SAVE_PATH}, loading weights and skipping training.")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        trained_model = model
        history = None
    else:
        print(f"\nNo existing model found at {MODEL_SAVE_PATH}, starting training...")
        # Loss und Optimizer definieren
        print("  Setting up criterion with positive weight.")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

        # Training starten
        trained_model, history = train_model(model, train_loader, val_loader,
                                             criterion, optimizer, NUM_EPOCHS,
                                             device, MODEL_SAVE_PATH)

    # 6. Finale Evaluation
    print("\n--- Final Evaluation on Validation Set (using loaded/trained model) ---")
    trained_model.eval()
    all_final_labels = []
    all_final_outputs = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            # <<< OPTIMIERUNG: autocast auch hier >>>
            with autocast(enabled=(device.type == 'cuda')):
                outputs = trained_model(inputs)
            all_final_labels.append(labels)
            all_final_outputs.append(outputs)

    all_final_outputs = torch.cat(all_final_outputs)
    all_final_labels = torch.cat(all_final_labels)
    final_acc, final_prec, final_rec, final_f1 = calculate_metrics(all_final_outputs, all_final_labels)

    print(f"Final Val Acc: {final_acc:.4f}")
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
    print("\nClassification Report (Validation Set):")
    print(classification_report(final_labels_np, final_preds, target_names=list(class_to_idx.keys()), zero_division=0))

    # Falsch klassifizierte Bilder anzeigen
    find_and_show_false_negatives(trained_model, val_loader, device, class_to_idx, num_to_show=15)

    print("\n--- Script Finished ---")