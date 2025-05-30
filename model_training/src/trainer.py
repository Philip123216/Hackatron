# src/trainer.py
# Dieses Modul enthält die Kernlogik für den Trainingsprozess des CNN-Modells.
# Es beinhaltet die generische Trainingsschleife, die sowohl für Optuna-Trials
# als auch für das finale Training verwendet wird, sowie die Objective-Funktion für Optuna.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler  # Für Mixed Precision Training, falls GPU verwendet wird
import time
import copy
import optuna  # Für Hyperparameter-Optimierung und TrialPruned Exception
from pathlib import Path

# Importiere Hilfsfunktionen und Klassen aus den anderen Modulen des src-Pakets
from .utils import calculate_metrics
from .model import CustomCNN
from .data_utils import get_transforms_for_trial, create_dataloaders # KORREKTER IMPORT


def run_training_loop(
        trial_or_none: optuna.trial.Trial,  # Optuna Trial Objekt (oder None für finales Training)
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,  # Loss-Funktion
        optimizer: optim.Optimizer,
        scheduler_or_none: torch.optim.lr_scheduler._LRScheduler,  # Lernraten-Scheduler (oder None)
        num_epochs_max: int,  # Maximale Anzahl an Epochen für diesen Loop
        device_obj: torch.device,
        loop_name_tag: str,  # Ein Name für diesen Trainingslauf (z.B. "OptunaTrial_X", "FinalModel")
        current_session_output_dir: Path,  # Basis-Ausgabeordner für diesen Lauf
        model_save_filename: str,  # Dateiname für das zu speichernde Modell
        is_optuna_trial: bool,  # Flag, ob dies ein Optuna-Trial ist (für Pruning/Reporting)
        early_stopping_patience: int,  # Anzahl Epochen ohne Verbesserung, bevor gestoppt wird
        metric_to_optimize: str,  # Name der Validierungsmetrik, die optimiert wird (z.B. 'val_acc')
        higher_is_better: bool  # True, wenn ein höherer Wert der Metrik besser ist
) -> tuple:
    """
    Führt einen kompletten Trainings- und Validierungsloop für eine gegebene Anzahl
    von Epochen durch. Speichert das beste Modell basierend auf der `metric_to_optimize`.
    Implementiert Early Stopping und Optuna Pruning/Reporting.

    Returns:
        tuple: (trainiertes_modell, history_dict, gesamt_trainingszeit, bester_metrik_wert)
    """
    print(f"\n--- Starte Trainings-Loop: '{loop_name_tag}' (max. {num_epochs_max} Epochen) ---")
    print(
        f"  Optimiere für Metrik: '{metric_to_optimize}' (Ziel: {'Maximieren' if higher_is_better else 'Minimieren'})")
    if early_stopping_patience > 0:
        print(f"  Early Stopping aktiviert mit Geduld von {early_stopping_patience} Epochen.")

    model_full_save_path = current_session_output_dir / model_save_filename
    # Speichere initialen Zustand, falls keine Verbesserung erzielt wird
    best_model_wts = copy.deepcopy(model.state_dict())

    # Initialisiere besten Metrikwert (schlechtester möglicher Wert)
    best_metric_value = -float('inf') if higher_is_better else float('inf')

    # Dictionary zum Speichern der Trainingshistorie (Loss, Metriken pro Epoche)
    history = {key: [] for key in ['epoch', 'train_loss', 'train_acc', 'train_f1',
                                   'val_loss', 'val_acc', 'val_f1', 'val_precision', 'val_recall',
                                   'lr', 'time_per_epoch']}
    current_early_stopping_counter = 0
    best_epoch = -1  # Epoche, in der die beste Metrik erreicht wurde

    use_amp = (device_obj.type == 'cuda')  # Automatic Mixed Precision nur auf GPU
    scaler = GradScaler(enabled=use_amp)
    if use_amp: print(f"  AMP (Mixed Precision) für '{loop_name_tag}' aktiviert.")

    total_training_start_time = time.time()

    for epoch in range(num_epochs_max):
        epoch_start_time = time.time()
        print(f"\n  Epoche {epoch + 1}/{num_epochs_max} für '{loop_name_tag}'")

        # --- Trainingsphase einer Epoche ---
        model.train()  # Modell in den Trainingsmodus setzen
        running_train_loss = 0.0
        all_train_labels_epoch, all_train_outputs_epoch = [], []
        for inputs, labels in train_loader:
            inputs = inputs.to(device_obj, non_blocking=True)
            # Labels für BCEWithLogitsLoss: float, unsqueezed
            labels_for_loss = labels.float().unsqueeze(1).to(device_obj, non_blocking=True)
            # Labels für Metrikberechnung: Original-Integer-Labels
            labels_for_metrics = labels.to(device_obj, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # Gradienten zurücksetzen
            with torch.amp.autocast(device_type=device_obj.type, enabled=use_amp):
                outputs = model(inputs)  # Forward Pass
                loss = criterion(outputs, labels_for_loss)  # Loss berechnen

            scaler.scale(loss).backward()  # Gradienten berechnen (skaliert für AMP)
            scaler.step(optimizer)  # Optimizer-Schritt (skaliert)
            scaler.update()  # Scaler aktualisieren

            running_train_loss += loss.item() * inputs.size(0)
            all_train_labels_epoch.append(labels_for_metrics.detach())
            all_train_outputs_epoch.append(outputs.detach())

        if len(train_loader.dataset) > 0:
            epoch_train_loss = running_train_loss / len(train_loader.dataset)
            # Precision und Recall für Trainingsset werden hier nicht explizit geloggt, aber Acc und F1
            train_acc, _, _, train_f1 = calculate_metrics(torch.cat(all_train_outputs_epoch),
                                                          torch.cat(all_train_labels_epoch))
        else:  # Fallback, falls Trainingsset leer ist
            epoch_train_loss, train_acc, train_f1 = 0.0, 0.0, 0.0
            print("  WARNUNG: Trainings-Datensatz ist leer in dieser Epoche.")

        print(f"    Train -> Loss: {epoch_train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        history['train_loss'].append(epoch_train_loss);
        history['train_acc'].append(train_acc);
        history['train_f1'].append(train_f1)

        # --- Validierungsphase einer Epoche ---
        model.eval()  # Modell in den Evaluationsmodus setzen
        running_val_loss = 0.0
        all_val_labels_epoch, all_val_outputs_epoch = [], []

        if len(val_loader.dataset) == 0:  # Fallback, falls Validierungsset leer ist
            epoch_val_loss, val_acc, val_prec, val_rec, val_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
            print("  WARNUNG: Validierungs-Datensatz ist leer in dieser Epoche.")
        else:
            with torch.no_grad():  # Keine Gradientenberechnung während der Validierung
                for inputs, labels in val_loader:
                    inputs = inputs.to(device_obj, non_blocking=True)
                    labels_for_loss = labels.float().unsqueeze(1).to(device_obj, non_blocking=True)
                    labels_for_metrics = labels.to(device_obj, non_blocking=True)

                    with torch.amp.autocast(device_type=device_obj.type, enabled=use_amp):
                        outputs = model(inputs)
                        val_batch_loss = criterion(outputs, labels_for_loss)
                    running_val_loss += val_batch_loss.item() * inputs.size(0)
                    all_val_labels_epoch.append(labels_for_metrics.detach())
                    all_val_outputs_epoch.append(outputs.detach())
            epoch_val_loss = running_val_loss / len(val_loader.dataset)
            val_acc, val_prec, val_rec, val_f1 = calculate_metrics(torch.cat(all_val_outputs_epoch),
                                                                   torch.cat(all_val_labels_epoch))

        print(
            f"    Val   -> Loss: {epoch_val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, P: {val_prec:.4f}, R: {val_rec:.4f}")

        # History und Zeitmessung
        history['epoch'].append(epoch + 1);
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(val_acc);
        history['val_f1'].append(val_f1)
        history['val_precision'].append(val_prec);
        history['val_recall'].append(val_rec)
        history['lr'].append(optimizer.param_groups[0]['lr']);
        history['time_per_epoch'].append(time.time() - epoch_start_time)
        print(f"    LR: {optimizer.param_groups[0]['lr']:.1e}, Epochenzeit: {(time.time() - epoch_start_time):.2f}s")

        # Hole den aktuellen Wert der Metrik, die für Speichern/Early Stopping relevant ist
        current_metric_value_for_optimization = locals().get(metric_to_optimize,
                                                             -1.0 if higher_is_better else float('inf'))

        # Lernraten-Scheduler Schritt (falls vorhanden)
        if scheduler_or_none:
            if isinstance(scheduler_or_none, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler_or_none.step(current_metric_value_for_optimization)
            else:
                scheduler_or_none.step()

        # Überprüfe, ob sich die Zielmetrik verbessert hat
        improved = False
        if higher_is_better:
            if current_metric_value_for_optimization > best_metric_value: improved = True
        else:
            if current_metric_value_for_optimization < best_metric_value: improved = True

        if improved:
            print(f"    Beste '{metric_to_optimize}' für '{loop_name_tag}' verbessert "
                  f"({best_metric_value:.4f} -> {current_metric_value_for_optimization:.4f}). Speichere Modell...")
            best_metric_value = current_metric_value_for_optimization
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, model_full_save_path)
            best_epoch = epoch + 1
            current_early_stopping_counter = 0
        elif early_stopping_patience > 0:
            current_early_stopping_counter += 1
            print(f"    EarlyStopping: Keine Verbesserung in '{metric_to_optimize}' "
                  f"seit {current_early_stopping_counter}/{early_stopping_patience} Epochen.")

        # Optuna Reporting und Pruning (nur wenn dies ein Optuna Trial ist)
        if is_optuna_trial and trial_or_none:
            trial_or_none.report(current_metric_value_for_optimization, epoch)
            if trial_or_none.should_prune():
                print(f"  Trial {trial_or_none.number} (Optuna Pruning) bei Epoche {epoch + 1} "
                      f"(Metrik: {current_metric_value_for_optimization:.4f}).")
                raise optuna.exceptions.TrialPruned()

        # Early Stopping für den aktuellen Loop (Optuna Trial oder finales Training)
        if early_stopping_patience > 0 and current_early_stopping_counter >= early_stopping_patience:
            print(f"  EARLY STOPPING für '{loop_name_tag}' ausgelöst bei Epoche {epoch + 1}!")
            break

    total_training_time = time.time() - total_training_start_time
    print(
        f"\n  Training für '{loop_name_tag}' abgeschlossen in {total_training_time // 60:.0f}m {total_training_time % 60:.0f}s.")
    if best_epoch != -1:
        print(f"  Beste '{metric_to_optimize}' ({best_metric_value:.4f}) wurde in Epoche {best_epoch} erreicht.")
    else:
        print(f"  Keine signifikante Verbesserung der '{metric_to_optimize}' über den initialen Wert hinaus erzielt.")

    # Lade die Gewichte des besten Modells, wenn eine Verbesserung erzielt wurde
    if model_full_save_path.exists() and best_epoch != -1:
        print(f"  Lade beste Gewichte von '{model_full_save_path.name}' für '{loop_name_tag}'.")
        model.load_state_dict(torch.load(model_full_save_path, map_location=device_obj))
    else:
        print(
            f"  WARNUNG: Kein besseres Modell während '{loop_name_tag}' gespeichert oder initialer Wert nicht übertroffen. "
            f"Verwende Gewichte vom Ende des Trainings (oder initial, falls keine Verbesserung).")
        if best_model_wts: model.load_state_dict(best_model_wts)

    return model, history, total_training_time, best_metric_value


def objective(trial: optuna.trial.Trial, config,
              pos_weight_tensor: torch.Tensor, device_obj: torch.device) -> float:
    """
    Optuna Objective-Funktion. Definiert den Suchraum für Hyperparameter,
    instanziiert Modell, Optimizer und DataLoaders (mit getunten Augmentationen)
    und führt einen Trainingslauf durch.
    """
    # --- Hyperparameter-Vorschläge für Architektur und Training ---
    # Diese Bereiche können basierend auf Vorerfahrungen oder spezifischen Anforderungen angepasst werden.
    lr = trial.suggest_float("lr", 5e-4, 8e-3, log=True)  # Lernrate
    dropout_rate = trial.suggest_float("dropout_rate", 0.30, 0.65)  # Dropout-Rate für FC-Layer
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 5e-5, log=True) # L2 Regularisierung
    num_conv_blocks = trial.suggest_int("num_conv_blocks", 4, 6)    # Anzahl Convolutional Blocks
    first_layer_filters = trial.suggest_categorical("first_layer_filters", [16, 32, 48, 64]) # Filter im 1. Block
    filter_increase_factor = trial.suggest_float("filter_increase_factor", 1.4, 2.1) # Faktor für Filteranstieg
    optimizer_name = trial.suggest_categorical("optimizer_name", ["AdamW", "RMSprop"]) # Auswahl des Optimizers

    # --- Hyperparameter für Daten-Augmentation ---
    rotation_degrees = trial.suggest_int("rotation_degrees", 0, 35)        # Max. Rotationswinkel
    cj_brightness = trial.suggest_float("cj_brightness", 0.0, 0.35)     # Helligkeits-Jitter
    cj_contrast = trial.suggest_float("cj_contrast", 0.0, 0.35)       # Kontrast-Jitter
    cj_saturation = trial.suggest_float("cj_saturation", 0.0, 0.35)   # Sättigungs-Jitter
    cj_hue = trial.suggest_float("cj_hue", 0.0, 0.18)                 # Farbton-Jitter
    hflip_p = trial.suggest_float("hflip_p", 0.0, 0.5)                # Wahrscheinlichkeit für Horiz. Flip
    vflip_p = trial.suggest_float("vflip_p", 0.0, 0.5)                # Wahrscheinlichkeit für Vert. Flip

    # --- DataLoaders MIT den Trial-spezifischen Augmentationen erstellen ---
    # Die Transformationen werden für jeden Trial neu generiert.
    current_trial_transforms = get_transforms_for_trial(
        img_size=config.IMG_SIZE,
        normalize_mean=config.NORMALIZE_MEAN, normalize_std=config.NORMALIZE_STD,
        rotation_degrees=rotation_degrees,
        cj_brightness=cj_brightness, cj_contrast=cj_contrast,
        cj_saturation=cj_saturation, cj_hue=cj_hue,
        hflip_p=hflip_p, vflip_p=vflip_p
    )
    # Die DataLoaders verwenden die globalen Pfade aus 'config' für die Datenquelle.
    train_loader_trial, val_loader_trial, _ = create_dataloaders( # KORREKTER FUNKTIONSAUFRUF
        config.TRAIN_DIR, config.VAL_DIR, config.BATCH_SIZE, config.NUM_WORKERS,
        current_trial_transforms['train'], current_trial_transforms['val'], device_obj
    )
    if train_loader_trial is None or val_loader_trial is None:
        print(f"FEHLER: Konnte DataLoaders für Trial {trial.number} nicht erstellen. Trial wird als fehlgeschlagen markiert.")
        return -float('inf') if config.HIGHER_IS_BETTER_FOR_METRIC else float('inf')

    # --- Modell, Criterion, Optimizer erstellen ---
    model = CustomCNN(num_classes=1, dropout_rate=dropout_rate, num_conv_blocks=num_conv_blocks,
                      first_layer_filters=first_layer_filters, filter_increase_factor=filter_increase_factor).to(device_obj)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor) # Berücksichtigt Klassenimbalance

    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "RMSprop":
        alpha = trial.suggest_float("rmsprop_alpha", 0.9, 0.999) # Spezifischer Parameter für RMSprop
        optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=alpha, weight_decay=weight_decay)
    else: # Fallback, falls Konfiguration erweitert wird und hier nicht abgefangen
        print(f"WARNUNG: Unbekannter Optimizer '{optimizer_name}' in Trial {trial.number}. Verwende AdamW als Standard.")
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Logging der für diesen Trial gewählten Parameter
    params_to_log = trial.params.copy()
    params_to_log.update({ # Füge Augmentationsparameter explizit zum Log hinzu
        "aug_rotation": rotation_degrees, "aug_cj_b": cj_brightness, "aug_cj_c": cj_contrast,
        "aug_cj_s": cj_saturation, "aug_cj_h": cj_hue, "aug_hflip_p": hflip_p, "aug_vflip_p": vflip_p
    })
    print(f"\n--- Starte Optuna Trial {trial.number} ---")
    print(f"  Parameter für diesen Trial: {params_to_log}")

    # Temporärer Speicherort für das potenziell beste Modell dieses spezifischen Trials
    temp_model_filename = f"trial_{trial.number}_best_model.pth"

    try:
        # Führe den Trainingsloop für diesen Trial aus
        _, _, _, best_trial_metric = run_training_loop(
            trial_or_none=trial, model=model, train_loader=train_loader_trial, val_loader=val_loader_trial,
            criterion=criterion, optimizer=optimizer, scheduler_or_none=None, # Kein Lernraten-Scheduler für kurze Optuna-Trials
            num_epochs_max=config.EPOCHS_PER_TRIAL_MAX, device_obj=device_obj,
            loop_name_tag=f"OptunaTrial_{trial.number}",
            current_session_output_dir=config.OPTUNA_TEMP_MODELS_DIR, # Speichere temporäre Trial-Modelle
            model_save_filename=temp_model_filename,
            is_optuna_trial=True, # Wichtig für Optuna Pruning/Reporting
            early_stopping_patience=config.OPTUNA_EARLY_STOPPING_PATIENCE_TRIAL, # Early Stopping für diesen Trial
            metric_to_optimize=config.METRIC_TO_OPTIMIZE, # Die zu optimierende Metrik (z.B. 'val_acc')
            higher_is_better=config.HIGHER_IS_BETTER_FOR_METRIC # Optimierungsrichtung
        )
        return best_trial_metric # Optuna wird diesen Wert verwenden, um den Trial zu bewerten
    except optuna.exceptions.TrialPruned:
        print(f"  Trial {trial.number} wurde durch Optuna oder internes Early Stopping gepruned.")
        return -float('inf') if config.HIGHER_IS_BETTER_FOR_METRIC else float('inf')
    except Exception as e: # Fange andere unerwartete Fehler während des Trials ab
        print(f"!! Optuna Trial {trial.number} ist mit einem Fehler fehlgeschlagen: {e}")
        import traceback; traceback.print_exc() # Gib den vollständigen Fehler-Stacktrace aus
        return -float('inf') if config.HIGHER_IS_BETTER_FOR_METRIC else float('inf')