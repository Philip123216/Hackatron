# main_training.py
# Hauptskript zur Orchestrierung des gesamten Trainingsprozesses für das CNN-Modell.
# Dieser Prozess umfasst:
# 1. Laden der globalen Konfiguration aus src.config.
# 2. Aufteilen der annotierten Daten in Trainings- und Validierungssets, falls nicht vorhanden.
# 3. Berechnung des Gewichtungsfaktors für die positive Klasse (zur Behandlung von Klassenimbalance).
# 4. Durchführung einer Hyperparameter-Suche mit Optuna, um die besten Modellarchitektur-,
#    Trainings- und Augmentationsparameter zu finden. Die Optimierung zielt auf die
#    in config.METRIC_TO_OPTIMIZE festgelegte Validierungsmetrik ab.
# 5. Training eines finalen Modells unter Verwendung der besten von Optuna gefundenen Hyperparameter,
#    inklusive der optimierten Daten-Augmentationsstrategie.
# 6. Evaluation des finalen Modells und Speicherung detaillierter Ergebnisse, Plots und einer Fehleranalyse.

import torch
import torch.nn as nn  # Für Type Hinting und Kontext
import torch.optim as optim
import optuna
import time
import glob  # Für Dateisuche bei der pos_weight Berechnung
from pathlib import Path
import random  # Für das Setzen des globalen Seeds
import shutil  # Für das optionale Löschen alter Daten-Splits

# Importiere modulare Komponenten aus dem 'src' Verzeichnis
# Annahme: Dieses Skript (main_training.py) liegt im Ordner 'model_training',
# und 'src' ist ein direkter Unterordner von 'model_training'.
from src import config  # Lädt alle Konfigurationen aus src/config.py
from src.data_utils import split_data, get_fixed_transforms, create_dataloaders, get_transforms_for_trial
from src.model import CustomCNN  # Die Definition der CNN-Architektur
from src.trainer import run_training_loop, objective  # Kern-Trainingslogik und Optuna Objective-Funktion
from src.utils import plot_training_history, save_error_analysis_images, calculate_metrics  # Hilfsfunktionen
from sklearn.metrics import confusion_matrix, classification_report  # Für die finale Auswertung


def main():
    """
    Hauptfunktion, die den gesamten Trainings-Workflow von Datenaufbereitung
    über Hyperparameter-Optimierung bis zum finalen Training und Evaluation steuert.
    """
    print(f"--- Starte Experiment: {config.EXPERIMENT_NAME} ---")
    print(f"Verwendetes PyTorch-Gerät: {config.DEVICE}")
    print(f"Alle Ausgaben für dieses Experiment werden gespeichert in: {config.EXPERIMENT_OUTPUT_DIR.resolve()}")

    # Setze Seeds für Zufallsgeneratoren zur Gewährleistung der Reproduzierbarkeit des Laufs
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)  # Setzt Seed für alle GPUs
    random.seed(config.RANDOM_SEED)
    import numpy as np  # Importiere NumPy erst nach dem Setzen der Seeds von Python/Torch
    np.random.seed(config.RANDOM_SEED)

    # --- 1. Datenaufteilung in Trainings- und Validierungssets ---
    # Überprüft, ob ein valider Datensplit für dieses Experiment bereits existiert.
    # Ein neuer Split wird durchgeführt, wenn der Zielordner nicht existiert oder leer ist.
    print("\n--- Schritt 1: Überprüfe/Erstelle Datenaufteilung ---")
    train_yes_dir = config.TRAIN_DIR / 'yes'
    train_no_dir = config.TRAIN_DIR / 'no'
    perform_split = True  # Flag, ob ein neuer Split nötig ist
    if config.BASE_DATA_SPLIT_DIR.exists() and train_yes_dir.exists() and train_no_dir.exists():
        if len(list(train_yes_dir.iterdir())) > 0 and len(list(train_no_dir.iterdir())) > 0:
            print(f"Verwende existierenden Datensplit in: {config.BASE_DATA_SPLIT_DIR.resolve()}")
            perform_split = False
        else:
            print(
                f"Datensplit-Verzeichnis '{config.BASE_DATA_SPLIT_DIR.resolve()}' existiert, aber Klassenordner ('yes'/'no') sind leer. Neuer Split wird durchgeführt.")
    else:
        print(
            f"Datensplit-Verzeichnis '{config.BASE_DATA_SPLIT_DIR.resolve()}' nicht gefunden. Neuer Split wird durchgeführt.")

    if perform_split:
        # Überprüfe, ob der Quellordner für die Annotationen (Input für den Split) korrekt ist
        if not config.ANNOTATED_DATA_SOURCE_DIR.exists() or \
                not (config.ANNOTATED_DATA_SOURCE_DIR / 'yes').exists() or \
                not (config.ANNOTATED_DATA_SOURCE_DIR / 'no').exists():
            print(f"FEHLER: Quell-Datenordner für Split '{config.ANNOTATED_DATA_SOURCE_DIR.resolve()}' "
                  f"ist nicht vorhanden oder enthält keine 'yes'/'no' Unterordner. \n"
                  f"Bitte Daten vorbereiten und Pfad 'ANNOTATED_DATA_SOURCE_DIR' in 'src/config.py' prüfen.")
            return  # Beende das Skript, da keine Daten zum Aufteilen vorhanden sind

        split_data(config.ANNOTATED_DATA_SOURCE_DIR, config.TRAIN_DIR, config.VAL_DIR,
                   config.TRAIN_VAL_SPLIT, config.RANDOM_SEED)

    # --- 2. Berechnung des Gewichtungsfaktors für die positive Klasse ---
    # Dies hilft, mit unausgewogenen Datensätzen (Klassenimbalance) umzugehen,
    # indem die seltenere Klasse in der Loss-Funktion stärker gewichtet wird.
    print("\n--- Schritt 2: Berechne Gewicht für positive Klasse ('yes') in der Loss-Funktion ---")
    try:
        num_no_samples = len(glob.glob(str(config.TRAIN_DIR / 'no' / '*.png')))
        num_yes_samples = len(glob.glob(str(config.TRAIN_DIR / 'yes' / '*.png')))

        if num_yes_samples == 0:
            print(
                "  WARNUNG: Keine 'yes'-Samples im Trainingsset gefunden! 'pos_weight' wird auf 1.0 gesetzt (keine Gewichtung).")
            pos_weight_value = 1.0
        elif num_no_samples == 0:
            print(
                "  WARNUNG: Keine 'no'-Samples im Trainingsset gefunden! 'pos_weight' wird auf 1.0 gesetzt (keine Gewichtung).")
            pos_weight_value = 1.0
        else:
            # Formel: Gewicht = (Anzahl der negativen Samples) / (Anzahl der positiven Samples)
            pos_weight_value = num_no_samples / num_yes_samples

        print(f"  {num_no_samples} 'no' Samples, {num_yes_samples} 'yes' Samples im Trainingsdatensatz.")
        print(f"  Berechnetes 'pos_weight' für die 'yes'-Klasse: {pos_weight_value:.2f}")
        pos_weight_tensor = torch.tensor([pos_weight_value], device=config.DEVICE)
    except Exception as e:
        print(f"  Fehler bei der Berechnung von pos_weight: {e}. Setze Standardwert 1.0.")
        pos_weight_tensor = torch.tensor([1.0], device=config.DEVICE)

    # --- 3. Optuna Hyperparameter-Suche ---
    # Die DataLoaders für die Optuna-Trials werden dynamisch innerhalb der 'objective'-Funktion
    # (in src/trainer.py) erstellt, da Augmentationsparameter ebenfalls optimiert werden.
    print(f"\n--- Schritt 3: Starte Optuna Hyperparameter-Suche ---")
    print(f"  Studienname: {config.OPTUNA_STUDY_NAME}")
    print(f"  Datenbank-Speicherort: sqlite:///{config.OPTUNA_DB_PATH.resolve()}")
    print(f"  Anzahl der durchzuführenden Trials: {config.N_TRIALS}")
    print(
        f"  Maximale Epochen pro Trial: {config.EPOCHS_PER_TRIAL_MAX} (mit Early Stopping Geduld von {config.OPTUNA_EARLY_STOPPING_PATIENCE_TRIAL} Epochen)")
    print(
        f"  Zielmetrik für Optimierung: Validierungs-{config.METRIC_TO_OPTIMIZE} (Ziel: {'Maximieren' if config.HIGHER_IS_BETTER_FOR_METRIC else 'Minimieren'})")

    study = optuna.create_study(
        study_name=config.OPTUNA_STUDY_NAME,
        storage=f"sqlite:///{config.OPTUNA_DB_PATH.resolve()}",
        load_if_exists=True,  # Ermöglicht das Fortsetzen einer unterbrochenen Studie
        direction="maximize" if config.HIGHER_IS_BETTER_FOR_METRIC else "minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,  # Anzahl Trials, bevor der Pruner aktiv wird
            n_warmup_steps=7,  # Anzahl Epochen innerhalb eines Trials, bevor Pruning für diesen Trial einsetzen kann
            interval_steps=3  # Pruning-Prüfung erfolgt alle 'interval_steps' Epochen
        )
    )

    # Starte die Optimierung. 'config' wird an die Objective-Funktion übergeben.
    study.optimize(lambda trial_obj: objective(trial_obj, config, pos_weight_tensor, config.DEVICE),
                   n_trials=config.N_TRIALS, timeout=None)  # Optional: timeout in Sekunden für die gesamte Suche

    print(f"\n--- Optuna Hyperparameter-Suche abgeschlossen ---")
    if not study.best_trial:
        print(
            "FEHLER: Optuna konnte keinen besten Trial finden. Überprüfe die Optuna-Logs und die Datenbank. Breche ab.")
        return

    # Überprüfung, ob der beste Wert sinnvoll ist
    best_value_is_valid = True
    if config.HIGHER_IS_BETTER_FOR_METRIC and study.best_value <= -float('inf') + 1e-9: best_value_is_valid = False
    if not config.HIGHER_IS_BETTER_FOR_METRIC and study.best_value >= float('inf') - 1e-9: best_value_is_valid = False
    if not best_value_is_valid:
        print(f"FEHLER: Optuna hat keinen gültigen besten Wert gefunden (Wert: {study.best_value}). "
              "Möglicherweise sind alle Trials fehlgeschlagen oder wurden gepruned. Bitte Logs prüfen. Breche ab.")
        return

    print(f"Bester Trial von Optuna (optimiert für Validierungs-{config.METRIC_TO_OPTIMIZE}):")
    best_trial = study.best_trial
    print(f"  Wert der Zielmetrik im besten Trial: {best_trial.value:.4f}")
    print("  Beste gefundene Hyperparameter:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Speichere Optuna Visualisierungen
    try:
        if optuna.visualization.is_available():
            # Stelle sicher, dass der Zielordner für Visualisierungen existiert
            config.OPTUNA_VISUALS_DIR.mkdir(parents=True, exist_ok=True)

            fig_history = optuna.visualization.plot_optimization_history(study)
            fig_history.write_image(str(config.OPTUNA_VISUALS_DIR / "optuna_optimization_history.png"))

            fig_importance = optuna.visualization.plot_param_importances(study)
            fig_importance.write_image(str(config.OPTUNA_VISUALS_DIR / "optuna_param_importances.png"))

            # Slice Plot kann sehr viele Subplots erzeugen und ist optional
            # fig_slice = optuna.visualization.plot_slice(study)
            # fig_slice.write_image(str(config.OPTUNA_VISUALS_DIR / "optuna_slice_plot.png"))
            print(f"  Optuna Visualisierungen gespeichert in: {config.OPTUNA_VISUALS_DIR.resolve()}")
        else:
            print("  WARNUNG: Optuna Visualisierungen können nicht erstellt werden. "
                  "Möglicherweise fehlen Abhängigkeiten wie 'plotly' und 'kaleido'.")
    except Exception as e_vis:  # Fange spezifische Fehler beim Plotten ab
        print(f"  FEHLER beim Erstellen oder Speichern der Optuna Visualisierungen: {e_vis}. "
              "Überprüfe Installation von 'plotly' und 'kaleido'.")

    # --- 4. Finales Training mit den besten Hyperparametern ---
    print("\n--- Schritt 4: Starte finales Training mit den besten Hyperparametern von Optuna ---")
    best_params = best_trial.params  # Die als optimal ermittelten Parameter

    # Instanziiere das Modell mit den besten Architekturparametern
    final_model = CustomCNN(
        num_classes=1,  # Binäre Klassifikation
        dropout_rate=best_params['dropout_rate'],
        num_conv_blocks=best_params['num_conv_blocks'],
        first_layer_filters=best_params['first_layer_filters'],
        filter_increase_factor=best_params['filter_increase_factor']
    ).to(config.DEVICE)

    final_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # Erstelle den Optimizer mit den besten Trainingsparametern
    final_optimizer_name = best_params.get("optimizer_name", "AdamW")  # .get für sicheren Zugriff mit Fallback
    final_lr = best_params['lr']
    final_weight_decay = best_params['weight_decay']

    if final_optimizer_name == "AdamW":
        final_optimizer = optim.AdamW(final_model.parameters(), lr=final_lr, weight_decay=final_weight_decay)
    elif final_optimizer_name == "RMSprop":
        final_rmsprop_alpha = best_params.get('rmsprop_alpha', 0.99)  # Fallback für Alpha
        final_optimizer = optim.RMSprop(final_model.parameters(), lr=final_lr, alpha=final_rmsprop_alpha,
                                        weight_decay=final_weight_decay)
    elif final_optimizer_name == "SGD":
        final_sgd_momentum = best_params.get('sgd_momentum', 0.9)  # Fallback für Momentum
        final_optimizer = optim.SGD(final_model.parameters(), lr=final_lr, momentum=final_sgd_momentum,
                                    weight_decay=final_weight_decay)
    else:
        print(f"WARNUNG: Unbekannter Optimizer '{final_optimizer_name}' aus Optuna. Verwende AdamW als Standard.")
        final_optimizer = optim.AdamW(final_model.parameters(), lr=final_lr, weight_decay=final_weight_decay)

    # Lernraten-Scheduler für das finale Training
    final_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        final_optimizer,
        mode='max' if config.HIGHER_IS_BETTER_FOR_METRIC else 'min',
        factor=0.2, patience=10, min_lr=1e-7  # Erhöhte Geduld für längeres finales Training
    )

    # Erstelle die Trainings-Transformationen mit den besten von Optuna gefundenen Augmentationsparametern
    final_train_transforms_obj = get_transforms_for_trial(
        img_size=config.IMG_SIZE,
        normalize_mean=config.NORMALIZE_MEAN, normalize_std=config.NORMALIZE_STD,
        rotation_degrees=best_params.get('rotation_degrees', 20),
        # Fallback-Werte, falls Augmentation nicht getuned wurde
        cj_brightness=best_params.get('cj_brightness', 0.15),
        cj_contrast=best_params.get('cj_contrast', 0.15),
        cj_saturation=best_params.get('cj_saturation', 0.15),
        cj_hue=best_params.get('cj_hue', 0.08),
        hflip_p=best_params.get('hflip_p', 0.5),
        vflip_p=best_params.get('vflip_p', 0.5)
    )['train']  # Hole nur die 'train'-Komponente

    # Standard-Validierungstransformationen (ohne Augmentation)
    fixed_val_transforms = get_fixed_transforms(config.IMG_SIZE, config.NORMALIZE_MEAN, config.NORMALIZE_STD)['val']

    # Batch-Größe für finales Training (kann von Optuna getuned werden oder Standardwert nehmen)
    final_batch_size = best_params.get('batch_size_trial', config.BATCH_SIZE)  # Falls 'batch_size_trial' getuned wurde

    # DataLoaders für das finale Training erstellen
    final_train_loader, final_val_loader, final_class_to_idx = create_dataloaders(  # Verwende create_dataloaders
        config.TRAIN_DIR, config.VAL_DIR, final_batch_size, config.NUM_WORKERS,
        final_train_transforms_obj,  # Beste Augmentationen für Training
        fixed_val_transforms,  # Standard-Transformationen für Validierung
        config.DEVICE
    )
    if final_train_loader is None or final_val_loader is None:
        print("FEHLER: Konnte DataLoaders für finales Training nicht erstellen. Breche ab.")
        return

    print(f"Trainiere finales Modell für maximal {config.FINAL_TRAINING_EPOCHS_MAX} Epochen mit den besten Parametern.")
    print(f"  Verwendete Parameter für finales Training: {best_params}")

    trained_final_model, final_history, final_train_time, best_final_metric_val = run_training_loop(
        trial_or_none=None,  # Kein Optuna-Trial
        model=final_model,
        train_loader=final_train_loader, val_loader=final_val_loader,
        criterion=final_criterion, optimizer=final_optimizer, scheduler_or_none=final_scheduler,
        num_epochs_max=config.FINAL_TRAINING_EPOCHS_MAX,
        device_obj=config.DEVICE,
        loop_name_tag=f"FinalModel_{config.EXPERIMENT_NAME}",
        current_session_output_dir=config.FINAL_MODEL_OUTPUT_SUBDIR,  # Speichere im spezifischen Unterordner
        model_save_filename=config.FINAL_MODEL_SAVE_PATH.name,  # Nur der Dateiname
        is_optuna_trial=False,
        early_stopping_patience=config.FINAL_EARLY_STOPPING_PATIENCE,
        metric_to_optimize=config.METRIC_TO_OPTIMIZE,
        higher_is_better=config.HIGHER_IS_BETTER_FOR_METRIC
    )

    # Plotte die Trainingshistorie des finalen Modells
    if final_history:  # Nur plotten, wenn History-Daten vorhanden sind
        plot_training_history(final_history, config.EXPERIMENT_OUTPUT_DIR,
                              f"FinalModel_{config.EXPERIMENT_NAME}_history")

    # --- 5. Finale Evaluation des trainierten Modells ---
    print(f"\n--- Schritt 5: Finale Evaluation des besten Modells auf dem Validierungsset ---")
    trained_final_model.eval()  # Sicherstellen, dass Modell im Eval-Modus ist
    all_final_labels_list, all_final_outputs_list = [], []

    if final_val_loader and len(final_val_loader.dataset) > 0:  # Nur wenn ValLoader existiert und nicht leer ist
        with torch.no_grad():
            for inputs, labels in final_val_loader:
                inputs = inputs.to(config.DEVICE)
                labels_for_metrics = labels.to(config.DEVICE)
                with torch.amp.autocast(device_type=config.DEVICE.type, enabled=(config.DEVICE.type == 'cuda')):
                    outputs = trained_final_model(inputs)
                all_final_labels_list.append(labels_for_metrics)
                all_final_outputs_list.append(outputs)
    else:
        print("WARNUNG: Validierungs-DataLoader ist leer oder nicht vorhanden. Finale Evaluation übersprungen.")

    # Metriken nur berechnen, wenn Daten vorhanden sind
    if all_final_outputs_list and all_final_labels_list:
        all_final_outputs_cat = torch.cat(all_final_outputs_list)
        all_final_labels_cat = torch.cat(all_final_labels_list)
        final_acc, final_prec, final_rec, final_f1 = calculate_metrics(all_final_outputs_cat, all_final_labels_cat)

        final_preds_np = (torch.sigmoid(all_final_outputs_cat).cpu().numpy() > 0.5).astype(int).flatten()
        final_labels_np = all_final_labels_cat.cpu().numpy().flatten()
        cm = confusion_matrix(final_labels_np, final_preds_np)
        # Stelle sicher, dass final_class_to_idx existiert und gültig ist
        target_names_report = list(final_class_to_idx.keys()) if final_class_to_idx else ['Klasse 0', 'Klasse 1']
        class_report_str = classification_report(final_labels_np, final_preds_np,
                                                 target_names=target_names_report, zero_division=0)
    else:  # Fallback-Werte, falls keine Evaluation möglich war
        final_acc, final_prec, final_rec, final_f1 = 0.0, 0.0, 0.0, 0.0
        cm = np.array([[0, 0], [0, 0]])
        class_report_str = "Keine Daten für Classification Report verfügbar (Validierungsset leer oder fehlerhaft)."
        final_class_to_idx = {'no': 0, 'yes': 1}  # Dummy für Summary

    # Erstelle eine detaillierte Zusammenfassung des gesamten Experiments
    summary_text = f"--- Experiment Zusammenfassung ({config.EXPERIMENT_NAME} @ {time.strftime('%Y-%m-%d %H:%M:%S')}) ---\n\n"
    # ... (Restliche Summary-Infos wie zuvor, stelle sicher, dass alle Variablen existieren oder Fallbacks haben)

    summary_text += f"Hardware-Gerät: {config.DEVICE}\n"
    summary_text += f"Zufalls-Seed: {config.RANDOM_SEED}\n"
    summary_text += f"Datenquelle für Aufteilung: '{config.ANNOTATED_DATA_SOURCE_DIR.name}'\n"
    if final_train_loader and hasattr(final_train_loader, 'dataset'):
        summary_text += f"  Anzahl Trainingsbilder: {len(final_train_loader.dataset)}, Anzahl Validierungsbilder: {len(final_val_loader.dataset)}\n"
        summary_text += f"  Aufteilungsverhältnis (Training): {config.TRAIN_VAL_SPLIT * 100:.0f}%\n"
    summary_text += f"  Gewicht für positive Klasse ('yes') in Loss-Funktion: {pos_weight_tensor.item():.2f}\n"

    summary_text += f"\nOptuna Hyperparameter-Suche:\n"
    summary_text += f"  Anzahl durchgeführter Trials: {len(study.trials)}\n"
    summary_text += f"  Maximale Epochen pro Trial: {config.EPOCHS_PER_TRIAL_MAX} (Early Stopping Patience: {config.OPTUNA_EARLY_STOPPING_PATIENCE_TRIAL})\n"
    summary_text += f"  Optimierte Metrik: Validierungs-{config.METRIC_TO_OPTIMIZE} (Ziel: {'Maximieren' if config.HIGHER_IS_BETTER_FOR_METRIC else 'Minimieren'})\n"
    summary_text += f"  Bester Optuna Trial Wert ({config.METRIC_TO_OPTIMIZE}): {best_trial.value:.4f}\n"
    summary_text += f"  Beste gefundene Hyperparameter:\n"
    for key, value in best_trial.params.items(): summary_text += f"    {key}: {value}\n"

    summary_text += f"\nFinales Modell Training:\n"
    summary_text += f"  Maximale Epochen: {config.FINAL_TRAINING_EPOCHS_MAX} (Early Stopping Patience: {config.FINAL_EARLY_STOPPING_PATIENCE})\n"
    summary_text += f"  Modell gespeichert unter: {config.FINAL_MODEL_SAVE_PATH.resolve()}\n"
    summary_text += f"  Beste Validierungs-{config.METRIC_TO_OPTIMIZE} während finalem Training: {best_final_metric_val:.4f}\n"
    summary_text += f"  Trainingszeit für finales Modell: {final_train_time // 60:.0f} Min {final_train_time % 60:.0f} Sek\n"

    summary_text += f"\nFinale Evaluation des besten Modells auf dem Validierungsset:\n"
    summary_text += f"  Accuracy: {final_acc:.4f}\n"
    summary_text += f"  Precision (für Klasse 'yes'): {final_prec:.4f}\n"
    summary_text += f"  Recall (für Klasse 'yes'): {final_rec:.4f}\n"
    summary_text += f"  F1-Score (für Klasse 'yes'): {final_f1:.4f}\n"
    if final_class_to_idx:
        summary_text += f"\nConfusion Matrix (Validierungsset):\n  Labels: {list(final_class_to_idx.keys())}\n{str(cm)}\n"
    summary_text += f"\nClassification Report (Validierungsset):\n{class_report_str}\n"

    print("\n" + "=" * 40 + " SCRIPT ZUSAMMENFASSUNG " + "=" * 40)
    print(summary_text)

    # Speichere die Zusammenfassung in einer Datei
    summary_file_path = config.EXPERIMENT_OUTPUT_DIR / f"experiment_summary_{config.EXPERIMENT_NAME}.txt"
    with open(summary_file_path, "w", encoding='utf-8') as f:
        f.write(summary_text)
    print(f"\nZusammenfassung des Experiments gespeichert: {summary_file_path.resolve()}")

    # Speichere Bilder für die Fehleranalyse (False Positives/Negatives)
    if final_val_loader and hasattr(final_val_loader, 'dataset') and len(
            final_val_loader.dataset) > 0 and final_class_to_idx:
        save_error_analysis_images(
            trained_final_model, final_val_loader, config.DEVICE, final_class_to_idx,
            config.ERROR_ANALYSIS_DIR, f"FinalModel_{config.EXPERIMENT_NAME}"
        )
    else:
        print("Fehleranalyse-Bilder nicht gespeichert, da Validierungs-DataLoader oder Klassen-Mapping ungültig ist.")

    print("\n--- Skript vollständig beendet ---")


if __name__ == "__main__":
    main()