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
import optuna # Import für Optuna
from torch.cuda.amp import autocast, GradScaler

# --- 0. Projekt-Root definieren (Annahme: Skript liegt im Projekt-Root) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- 1. Globale Konfiguration ---
ANNOTATED_DIR = PROJECT_ROOT / "data_annotated"
BASE_DATA_DIR = PROJECT_ROOT / "data_split_for_optuna_tuning" # Eigener Ordner für diesen Optuna-Lauf
TRAIN_DIR = BASE_DATA_DIR / "train"
VAL_DIR = BASE_DATA_DIR / "validation"
# Pfad für das *final* trainierte Modell nach dem Optuna-Tuning
FINAL_MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "best_optuna_tuned_cnn_model.pth"
# Ordner für Optuna-spezifische Ausgaben (beste Parameter, finale Zusammenfassung)
OPTUNA_OUTPUT_DIR = PROJECT_ROOT / "optuna_results"


# Optuna und Trainingsparameter
# Ca. 6 Stunden Laufzeit:
# Wenn ein Trial mit 20 Epochen ~3-4 Minuten dauert:
# 360 Minuten / 3.5 Min_pro_Trial approx = ~100 Trials.
N_TRIALS = 100        # Anzahl der Hyperparameter-Kombinationen
EPOCHS_PER_TRIAL = 20 # Epochen *pro* Optuna Trial (mit Pruning)
# Längeres Training für das finale Modell mit besten Parametern
FINAL_TRAINING_EPOCHS = 40 # Erhöht für potenziell besseres finales Training

# Standard-Parameter
BATCH_SIZE = 128 # Kannst du anpassen
NUM_WORKERS = 8  # Kannst du anpassen
IMG_SIZE = 250
RANDOM_SEED = 42
TRAIN_VAL_SPLIT = 0.8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Project Root: {PROJECT_ROOT}")

# Reproduzierbarkeit
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# --- 2. Datenaufteilung ---
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
        if not source_class_dir.exists(): print(f"  Warn: Src for {class_name} not found."); continue
        images = glob.glob(str(source_class_dir / "*.png"))
        if not images: print(f"  Warn: No PNGs for {class_name} found."); continue
        random.shuffle(images)
        split_idx = int(len(images) * split_ratio)
        train_images, val_images = images[:split_idx], images[split_idx:]
        print(f"  Class '{class_name}': {len(images)} total -> {len(train_images)} train, {len(val_images)} val")
        for img_path_str in train_images: shutil.copy(Path(img_path_str), train_class_dir / Path(img_path_str).name)
        for img_path_str in val_images: shutil.copy(Path(img_path_str), val_class_dir / Path(img_path_str).name)
    print("Data splitting complete.")

# --- 3. Datentransformationen & Laden ---
normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# Verwende hier die Transformationen, die bei deinem besten Modell gut funktioniert haben
# z.B. die "less_aug" oder die Standard-Augmentationen
train_transforms_for_optuna = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5), # Je nachdem, ob es geholfen hat
    # transforms.RandomRotation(10),
    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    normalize_transform
])
val_transforms = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), normalize_transform])

def create_dataloaders(train_dir, val_dir, batch_size, num_workers, train_transforms_to_use):
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms_to_use)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    print(f"Using {num_workers} workers for DataLoaders.")
    persistent = num_workers > 0; pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent)
    print(f"\nDataLoaders created: Train={len(train_dataset)}/{len(train_loader)}b, Val={len(val_dataset)}/{len(val_loader)}b")
    print(f"  Classes: {train_dataset.classes} (Mapping: {train_dataset.class_to_idx})")
    if train_dataset.class_to_idx.get('yes', -1) != 1 or train_dataset.class_to_idx.get('no', -1) != 0: print(f"\n*** WARNING: Class mapping incorrect! Actual: {train_dataset.class_to_idx}\n")
    return train_loader, val_loader, train_dataset.class_to_idx

def calculate_pos_weight(train_dir, device_obj):
    print("Calculating positive weight...")
    try:
        n_no=len(glob.glob(str(train_dir/'no'/'*.png'))); n_yes=len(glob.glob(str(train_dir/'yes'/'*.png')))
        pw = 1.0 if n_yes == 0 else n_no / n_yes
        print(f"  {n_no} no, {n_yes} yes. PW: {pw:.2f}"); return torch.tensor([pw], device=device_obj)
    except: print("  Error calc_pos_weight, defaulting."); return torch.tensor([1.0], device=device_obj)

# --- 4. Modell Definition ---
class CustomCNN(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.5, num_conv_blocks=4):
        super(CustomCNN, self).__init__()
        def _make_block(in_c, out_c): return nn.Sequential(nn.Conv2d(in_c,out_c,3,1,'same',bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True), nn.MaxPool2d(2,2))
        self.conv_blocks = nn.ModuleList(); current_channels = 3
        block_channels = [64,128,256,512,512,512]
        for i in range(min(num_conv_blocks, len(block_channels))):
            self.conv_blocks.append(_make_block(current_channels, block_channels[i])); current_channels = block_channels[i]
        self.avgpool=nn.AdaptiveAvgPool2d(1); self.flatten=nn.Flatten(); self.dropout=nn.Dropout(dropout_rate); self.fc=nn.Linear(current_channels,num_classes)
    def forward(self, x):
        for block in self.conv_blocks: x = block(x)
        return self.fc(self.dropout(self.flatten(self.avgpool(x))))

# --- 5. Metrik-Berechnung & Fehleranalyse ---
def calculate_metrics(outputs, labels):
    probs = torch.sigmoid(outputs).detach().cpu().numpy(); preds = (probs > 0.5).astype(int).flatten(); labels = labels.detach().cpu().numpy().flatten()
    acc=accuracy_score(labels,preds); prec=precision_score(labels,preds,average='binary',pos_label=1,zero_division=0)
    rec=recall_score(labels,preds,average='binary',pos_label=1,zero_division=0); f1=f1_score(labels,preds,average='binary',pos_label=1,zero_division=0)
    return acc, prec, rec, f1

def find_and_show_false_negatives(model, val_loader, device_obj, class_to_idx, num_to_show=10, output_dir=None, exp_name="final_eval"):
    print(f"\n--- Searching for False Negatives ({exp_name}) ---")
    model.eval(); false_negative_files = []
    yes_idx = class_to_idx.get('yes',1); no_pred_idx = 0
    if yes_idx == -1 or not hasattr(val_loader, 'dataset'): return
    filepaths = [s[0] for s in val_loader.dataset.samples]; true_labels = [s[1] for s in val_loader.dataset.samples]; preds_list = []
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device_obj)
            with autocast(enabled=(device_obj.type=='cuda')): outputs = model(inputs)
            preds_list.extend((torch.sigmoid(outputs) > 0.5).int().cpu().flatten().tolist())
    for i, fp_str in enumerate(filepaths):
        if true_labels[i] == yes_idx and preds_list[i] == no_pred_idx: false_negative_files.append(Path(fp_str))
    print(f"Found {len(false_negative_files)} FN images.")
    if not false_negative_files: print("No False Negatives found!"); return
    num_to_show_actual = min(num_to_show, len(false_negative_files))
    if num_to_show_actual == 0: return
    print(f"Saving/Showing up to {num_to_show_actual} FN images...")
    if output_dir:
        fn_save_dir = output_dir / f"{exp_name}_false_negatives"
        fn_save_dir.mkdir(parents=True, exist_ok=True)
        for img_p in false_negative_files[:num_to_show_actual]: shutil.copy(img_p, fn_save_dir / img_p.name)
        print(f"  FN images saved to {fn_save_dir.relative_to(PROJECT_ROOT)}")
    num_cols=5; num_rows=math.ceil(num_to_show_actual/num_cols)
    plt.figure(figsize=(15, 3 * num_rows));
    for i, img_p in enumerate(false_negative_files[:num_to_show_actual]):
        try:
            img = Image.open(img_p); ax = plt.subplot(num_rows,num_cols,i+1)
            ax.imshow(img); ax.set_title(f"FN: {img_p.name}\nTrue:yes,Pred:no",fontsize=8); ax.axis("off")
        except Exception as e: print(f"Could not load/show {img_p}: {e}")
    plt.tight_layout(); plt.show()

def plot_training_history(history, output_dir, plot_name="training_history"):
    if not history or 'val_f1' not in history or not history['val_f1']: print(" No history to plot."); return
    epochs = history.get('epoch', range(1, len(history['val_f1']) + 1))
    plt.figure(figsize=(12,8)); plt.subplot(2,1,1)
    plt.plot(epochs, history['val_f1'], 'b-', label='Val F1');
    if 'val_acc' in history: plt.plot(epochs, history['val_acc'], 'c-', label='Val Acc')
    plt.legend(); plt.title(f'{plot_name} - Validation Metrics'); plt.grid(True); plt.subplot(2,1,2)
    plt.plot(epochs, history['train_loss'], 'r-', label='Train Loss'); plt.plot(epochs, history['val_loss'], 'g-', label='Val Loss')
    plt.legend(); plt.title(f'{plot_name} - Loss'); plt.xlabel('Epoch'); plt.grid(True)
    save_file = output_dir / f"{plot_name}.png"
    plt.savefig(save_file); print(f"  Saved training history plot to {save_file.relative_to(PROJECT_ROOT)}"); plt.close()


# --- 6. Trainings-Loop für einen Optuna Trial ---
def train_trial(trial, model, train_loader, val_loader, criterion, optimizer, num_epochs, device_obj):
    best_val_f1_trial = 0.0
    use_amp = (device_obj.type == 'cuda'); scaler = GradScaler(enabled=use_amp)
    for epoch in range(num_epochs):
        model.train(); running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device_obj,non_blocking=True), labels.float().unsqueeze(1).to(device_obj,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp): outputs = model(inputs); loss = criterion(outputs, labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running_loss += loss.item() * inputs.size(0)
        model.eval(); running_val_loss = 0.0; all_vl, all_vo = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device_obj,non_blocking=True), labels.float().unsqueeze(1).to(device_obj,non_blocking=True)
                with autocast(enabled=use_amp): outputs = model(inputs); loss = criterion(outputs, labels)
                running_val_loss += loss.item()*inputs.size(0); all_vl.append(labels); all_vo.append(outputs)
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_acc, val_prec, val_rec, val_f1 = calculate_metrics(torch.cat(all_vo), torch.cat(all_vl))
        print(f"  Trial {trial.number} Epoch {epoch+1}/{num_epochs} - Val Loss: {epoch_val_loss:.4f} Val F1: {val_f1:.4f}")
        best_val_f1_trial = max(best_val_f1_trial, val_f1)
        trial.report(val_f1, epoch) # Melde den aktuellen F1-Score für Pruning
        if trial.should_prune(): print(f"  Trial {trial.number} pruned at epoch {epoch+1}."); raise optuna.exceptions.TrialPruned()
    return best_val_f1_trial

# --- 7. Optuna Objective Function ---
def objective(trial, train_loader, val_loader, pos_weight_tensor, device_obj, class_to_idx_map):
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True) # Breiterer Suchraum für Lernrate
    dropout_rate = trial.suggest_float("dropout", 0.2, 0.65) # Leicht angepasster Dropout-Bereich
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    num_conv_blocks_trial = trial.suggest_int("num_conv_blocks", 3, 5) # Teste auch Architektur leicht

    model_trial = CustomCNN(num_classes=1, dropout_rate=dropout_rate, num_conv_blocks=num_conv_blocks_trial).to(device_obj)
    criterion_trial = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer_trial = optim.AdamW(model_trial.parameters(), lr=lr, weight_decay=weight_decay)
    print(f"\n--- Starting Optuna Trial {trial.number} ---")
    print(f"  Params: lr={lr:.6f}, dropout={dropout_rate:.3f}, weight_decay={weight_decay:.6f}, conv_blocks={num_conv_blocks_trial}")
    try:
        best_trial_f1 = train_trial(trial, model_trial, train_loader, val_loader, criterion_trial, optimizer_trial, EPOCHS_PER_TRIAL, device_obj)
        return best_trial_f1
    except optuna.exceptions.TrialPruned: print(f"  Trial {trial.number} was pruned. Ret 0.0"); return 0.0
    except Exception as e: print(f"!! Trial {trial.number} failed: {e}"); return 0.0

# --- Funktion für finales Training ---
def train_final_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device_obj, save_path, history_plot_dir):
    best_model_wts = copy.deepcopy(model.state_dict()); best_val_f1_final = 0.0
    use_amp = (device_obj.type == 'cuda'); scaler = GradScaler(enabled=use_amp)
    history = {'epoch':[],'train_loss':[],'val_loss':[],'val_f1':[],'val_acc':[],'val_precision':[],'val_recall':[],'time_per_epoch':[]}
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time(); print(f"\nFinal Training Epoch {epoch+1}/{num_epochs}")
        model.train(); running_loss = 0.0
        for inputs,labels in train_loader:
            inputs,labels = inputs.to(device_obj),labels.float().unsqueeze(1).to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp): outputs=model(inputs); loss=criterion(outputs,labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running_loss += loss.item()*inputs.size(0)
        epoch_train_loss = running_loss/len(train_loader.dataset); print(f"  Train Loss: {epoch_train_loss:.4f}")
        model.eval(); all_vl,all_vo = [],[]; running_val_loss = 0.0
        with torch.no_grad():
            for inputs,labels in val_loader:
                inputs,labels = inputs.to(device_obj),labels.float().unsqueeze(1).to(device_obj)
                with autocast(enabled=use_amp): outputs=model(inputs); loss=criterion(outputs,labels)
                running_val_loss += loss.item()*inputs.size(0); all_vl.append(labels); all_vo.append(outputs)
        epoch_val_loss = running_val_loss/len(val_loader.dataset)
        val_acc,val_prec,val_rec,val_f1 = calculate_metrics(torch.cat(all_vo),torch.cat(all_vl))
        print(f"  Val Loss: {epoch_val_loss:.4f} Acc: {val_acc:.4f} P: {val_prec:.4f} R: {val_rec:.4f} F1: {val_f1:.4f}")
        epoch_time = time.time() - epoch_start_time
        for k,v in zip(history.keys(),[epoch+1,epoch_train_loss,epoch_val_loss,val_f1,val_acc,val_prec,val_rec,epoch_time]): history[k].append(v)
        if val_f1 > best_val_f1_final:
            print(f"  Best Val F1 improved ({best_val_f1_final:.4f} -> {val_f1:.4f}). Saving final model..."); best_val_f1_final=val_f1
            best_model_wts = copy.deepcopy(model.state_dict()); torch.save(best_model_wts,save_path)
    time_elapsed = time.time()-start_time
    print(f"\nFinal Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best Val F1 during final training: {best_val_f1_final:4f}"); print(f"Final model saved to: {save_path.relative_to(PROJECT_ROOT)}")
    model.load_state_dict(best_model_wts); return model, history

# --- 8. Hauptausführung mit Optuna ---
if __name__ == "__main__":
    OPTUNA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True) # Sicherstellen, dass models/ existiert

    if not BASE_DATA_DIR.exists() or not (TRAIN_DIR / 'yes').exists():
        print("Data split directory not found or incomplete, splitting data...")
        split_data(ANNOTATED_DIR, TRAIN_DIR, VAL_DIR, TRAIN_VAL_SPLIT, RANDOM_SEED)
    else: print(f"Using existing data split from: {BASE_DATA_DIR}")

    train_loader, val_loader, class_to_idx = create_dataloaders(TRAIN_DIR, VAL_DIR, BATCH_SIZE, NUM_WORKERS, train_transforms_for_optuna)
    pos_weight_tensor = calculate_pos_weight(TRAIN_DIR, device)

    print(f"\n--- Starting Optuna Hyperparameter Search ({N_TRIALS} Trials for approx. 6 hours) ---")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=1))
    study.optimize(lambda trial: objective(trial, train_loader, val_loader, pos_weight_tensor, device, class_to_idx),
                   n_trials=N_TRIALS, timeout=6*60*60) # Timeout von 6 Stunden

    print(f"\n--- Optuna Search Finished ---"); print(f"Finished trials: {len(study.trials)}")
    if study.best_trial is None or study.best_value <= 0.0:
         print("\nERROR: Optuna search did not find any valid trials. Check logs. Cannot proceed."); exit()
    print("Best trial found by Optuna:"); best_trial = study.best_trial; print(f"  Value (Best Val F1): {best_trial.value:.4f}"); print("  Params: ")
    for key, value in best_trial.params.items(): print(f"    {key}: {value}")

    best_params = best_trial.params
    best_params_save_path = OPTUNA_OUTPUT_DIR / "optuna_best_params.txt"
    with open(best_params_save_path, "w") as f:
        f.write("Best Hyperparameters found by Optuna:\n"); [f.write(f"  {k}: {v}\n") for k,v in best_params.items()]
        f.write(f"Best Validation F1 during search: {best_trial.value:.4f}\n")
    print(f"Best Optuna parameters saved to {best_params_save_path.relative_to(PROJECT_ROOT)}")

    print("\n--- Starting Final Training with Best Hyperparameters from Optuna ---")
    final_model = CustomCNN(num_classes=1, dropout_rate=best_params['dropout'], num_conv_blocks=best_params.get('num_conv_blocks',4)).to(device) # Nutze get für num_conv_blocks
    final_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    final_optimizer = optim.AdamW(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    print(f"Training final model for {FINAL_TRAINING_EPOCHS} epochs...")
    trained_final_model, final_history = train_final_model(final_model, train_loader, val_loader, final_criterion, final_optimizer,
                                           FINAL_TRAINING_EPOCHS, device, FINAL_MODEL_SAVE_PATH, OPTUNA_OUTPUT_DIR)

    print("\n--- Final Evaluation on Validation Set (using BEST OPTUNA TUNED model) ---")
    trained_final_model.eval()
    all_fl, all_fo = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
            with autocast(enabled=(device.type == 'cuda')): outputs = trained_final_model(inputs)
            all_fl.append(labels); all_fo.append(outputs)
    final_o = torch.cat(all_fo); final_l = torch.cat(all_fl)
    final_acc, final_prec, final_rec, final_f1 = calculate_metrics(final_o, final_l)
    print(f"Final Tuned Val Acc: {final_acc:.4f} P: {final_prec:.4f} R: {final_rec:.4f} F1: {final_f1:.4f}")
    final_preds = (torch.sigmoid(final_o).detach().cpu().numpy()>0.5).astype(int).flatten(); final_lnp = final_l.detach().cpu().numpy().flatten()
    cm = confusion_matrix(final_lnp, final_preds); print("\nCM:"); print(f"Lbls: {list(class_to_idx.keys())}"); print(cm)
    report = classification_report(final_lnp, final_preds, target_names=list(class_to_idx.keys()), zero_division=0)
    print("\nReport:"); print(report)
    final_summary_text = (f"Final Model after Optuna ({FINAL_TRAINING_EPOCHS} epochs):\n" +
                          f"Best Optuna F1 (search): {best_trial.value:.4f}\nParams: {best_params}\n" +
                          f"Final Val Acc: {final_acc:.4f}, P: {final_prec:.4f}, R: {final_rec:.4f}, F1: {final_f1:.4f}\n\nCM:\n{cm}\n\nReport:\n{report}")
    with open(OPTUNA_OUTPUT_DIR / "optuna_final_model_summary.txt", "w") as f: f.write(final_summary_text)
    print(f"\nFinal summary saved to {OPTUNA_OUTPUT_DIR / 'optuna_final_model_summary.txt'}")
    find_and_show_false_negatives(trained_final_model, val_loader, device, class_to_idx, num_to_show=15, output_dir=OPTUNA_OUTPUT_DIR, exp_name="final_optuna_tuned_FNs")
    plot_training_history(final_history, OPTUNA_OUTPUT_DIR, "final_optuna_tuned_model_history")
    print("\n--- Optuna Tuning and Final Training Script Finished ---")