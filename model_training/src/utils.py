# src/utils.py
# Dieses Modul enthält allgemeine Hilfsfunktionen, die im Trainingsprozess verwendet werden:
# - Berechnung von Klassifikationsmetriken.
# - Speichern von Bildern für die Fehleranalyse (False Positives/Negatives).
# - Plotten der Trainings- und Validierungshistorie.

import torch
import torch.nn # Nur für Type Hinting von model in save_error_analysis_images
from torch.utils.data import DataLoader # Nur für Type Hinting
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg') # Wichtig: Nicht-interaktives Backend für matplotlib, um Fehler auf Servern/ohne GUI zu vermeiden
import matplotlib.pyplot as plt
from pathlib import Path
import shutil # Für das Kopieren von Dateien in save_error_analysis_images
import numpy as np # Für np.unique in calculate_metrics Fehlerbehandlung (optional)

def calculate_metrics(outputs: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> tuple:
    """
    Berechnet gängige Klassifikationsmetriken (Accuracy, Precision, Recall, F1-Score)
    für eine binäre Klassifikationsaufgabe.

    Args:
        outputs (torch.Tensor): Die rohen Modell-Ausgaben (Logits).
        labels (torch.Tensor): Die wahren Labels (0 oder 1).
        threshold (float): Schwellenwert, um Wahrscheinlichkeiten in binäre Vorhersagen umzuwandeln.

    Returns:
        tuple: (accuracy, precision, recall, f1_score)
    """
    # Stelle sicher, dass Tensoren auf der CPU sind und NumPy-Arrays werden können
    if outputs.is_cuda: outputs = outputs.cpu()
    if labels.is_cuda: labels = labels.cpu()

    # Umwandlung der Logits in Wahrscheinlichkeiten und dann in binäre Vorhersagen
    try:
        probs = torch.sigmoid(outputs).detach().numpy() # .detach() um Gradientenverfolgung zu stoppen
        preds = (probs > threshold).astype(int).flatten()
        labels_np = labels.detach().numpy().flatten()
    except Exception as e_convert:
        print(f"  FEHLER bei der Konvertierung von Tensoren für Metrikberechnung: {e_convert}. Setze Metriken auf 0.")
        return 0.0, 0.0, 0.0, 0.0

    # Berechne Metriken
    try:
        acc = accuracy_score(labels_np, preds)
        # Für binäre Klassifikation: pos_label=1, wenn 'yes' als Klasse 1 codiert ist.
        # zero_division=0 verhindert Fehler, falls eine Klasse keine Vorhersagen/Labels hat (z.B. in sehr kleinen Batches).
        precision = precision_score(labels_np, preds, average='binary', pos_label=1, zero_division=0)
        recall = recall_score(labels_np, preds, average='binary', pos_label=1, zero_division=0)
        f1 = f1_score(labels_np, preds, average='binary', pos_label=1, zero_division=0)
    except ValueError as e_val:
        # Dieser Fehler tritt oft auf, wenn in einem Batch nur eine Klasse vorkommt (z.B. alle Labels sind 0).
        # print(f"  WARNUNG bei Metrikberechnung (ValueError): {e_val}. Setze Metriken auf 0.")
        # print(f"    Labels (unique): {np.unique(labels_np)}, Vorhersagen (unique): {np.unique(preds)}")
        acc, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
    except Exception as e_general:
        print(f"  FEHLER bei Metrikberechnung (Allgemein): {e_general}. Setze Metriken auf 0.")
        acc, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
    return acc, precision, recall, f1

def save_error_analysis_images(
    model: torch.nn.Module,
    val_loader: DataLoader, # Expliziter Typ für Klarheit
    device_obj: torch.device,
    class_to_idx: dict,
    output_dir_session: Path, # Sollte der spezifische Experiment-Ordner sein
    experiment_tag: str
) -> None:
    """
    Führt Inferenzen auf dem Validierungsset durch und speichert Bilder,
    die vom Modell falsch klassifiziert wurden (False Negatives und False Positives).

    Args:
        model (torch.nn.Module): Das trainierte Modell.
        val_loader (DataLoader): DataLoader für das Validierungsset.
        device_obj (torch.device): Das Gerät (CPU/GPU), auf dem die Inferenz läuft.
        class_to_idx (dict): Mapping von Klassennamen zu Indizes (z.B. {'no': 0, 'yes': 1}).
        output_dir_session (Path): Basisverzeichnis, in dem die Fehleranalyse-Ordner erstellt werden.
        experiment_tag (str): Ein Tag zur Benennung der Ausgabe (z.B. Modellname oder Experiment-ID).
    """
    print(f"  Speichere Fehleranalyse-Bilder für '{experiment_tag}' nach '{output_dir_session.resolve()}'...")
    fn_dir = output_dir_session / "false_negatives" # Bilder, die 'yes' waren, aber als 'no' vorhergesagt wurden
    fp_dir = output_dir_session / "false_positives" # Bilder, die 'no' waren, aber als 'yes' vorhergesagt wurden
    fn_dir.mkdir(parents=True, exist_ok=True)
    fp_dir.mkdir(parents=True, exist_ok=True)

    model.eval() # Modell in den Evaluationsmodus setzen
    yes_idx = class_to_idx.get('yes', 1) # Index für die positive Klasse 'yes'
    no_idx = class_to_idx.get('no', 0)   # Index für die negative Klasse 'no'
    fn_count, fp_count = 0, 0

    # Dateipfade und wahre Labels aus dem val_loader.dataset extrahieren
    if not hasattr(val_loader.dataset, 'samples') or not val_loader.dataset.samples:
        print("  WARNUNG: 'samples'-Attribut im val_loader.dataset nicht gefunden oder leer. Fehleranalyse übersprungen.")
        return

    filepaths = [s[0] for s in val_loader.dataset.samples] # s[0] ist der Pfad
    true_labels_indices = [s[1] for s in val_loader.dataset.samples] # s[1] ist der Klassenindex
    all_predicted_indices = []
    use_amp_for_eval = (device_obj.type == 'cuda') # Mixed Precision auch für Inferenz, falls auf GPU

    # Vorhersagen für das gesamte Validierungsset in einem Rutsch (Batch-weise)
    with torch.no_grad(): # Keine Gradientenberechnung für Inferenz
        for inputs_batch, _ in val_loader: # Die Labels aus dem Loader werden hier nicht direkt für die Vorhersage benötigt
            inputs_batch = inputs_batch.to(device_obj, non_blocking=True)
            with torch.amp.autocast(device_type=device_obj.type, enabled=use_amp_for_eval):
                outputs_batch = model(inputs_batch)
            # Umwandlung in binäre Vorhersagen (0 oder 1)
            preds_batch = (torch.sigmoid(outputs_batch) > 0.5).int().cpu().flatten().tolist()
            all_predicted_indices.extend(preds_batch)

    if len(filepaths) != len(all_predicted_indices):
        print(f"  WARNUNG: Anzahl der Dateipfade ({len(filepaths)}) stimmt nicht mit der Anzahl der "
              f"Vorhersagen ({len(all_predicted_indices)}) überein. Fehleranalyse möglicherweise unvollständig.")
        return

    # Vergleiche wahre Labels mit Vorhersagen und kopiere falsch klassifizierte Bilder
    for i, img_filepath_str in enumerate(filepaths):
        img_filepath = Path(img_filepath_str)
        true_label_idx = true_labels_indices[i]
        pred_label_idx = all_predicted_indices[i]

        try:
            if true_label_idx == yes_idx and pred_label_idx == no_idx: # False Negative (FN)
                if img_filepath.exists(): shutil.copy(img_filepath, fn_dir / img_filepath.name)
                fn_count += 1
            elif true_label_idx == no_idx and pred_label_idx == yes_idx: # False Positive (FP)
                if img_filepath.exists(): shutil.copy(img_filepath, fp_dir / img_filepath.name)
                fp_count += 1
        except Exception as e_copy:
            print(f"    FEHLER beim Kopieren des Fehlerbildes '{img_filepath.name}': {e_copy}")

    print(f"    Fehleranalyse abgeschlossen. Gespeichert: {fn_count} False Negatives, {fp_count} False Positives.")

def plot_training_history(history: dict, output_dir_session: Path, experiment_tag: str) -> None:
    """
    Plottet die Trainings- und Validierungshistorie (Loss, Accuracy, F1, etc.)
    und speichert den Plot als PNG-Datei.

    Args:
        history (dict): Dictionary mit Listen von Metriken pro Epoche.
                        Erwartet Keys wie 'epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', etc.
        output_dir_session (Path): Verzeichnis, in dem der Plot gespeichert wird.
        experiment_tag (str): Ein Tag zur Benennung der Plot-Datei (z.B. Modellname).
    """
    print(f"  Plotte Trainingshistorie für '{experiment_tag}' nach '{output_dir_session.resolve()}'...")

    # Überprüfe, ob genügend Daten zum Plotten vorhanden sind
    # Mindestens 'val_acc' oder eine andere Haupt-Validierungsmetrik sollte vorhanden sein.
    main_metric_to_check = 'val_acc' # Kann an die primär beobachtete Metrik angepasst werden
    if not history or main_metric_to_check not in history or not history[main_metric_to_check]:
        print(f"    WARNUNG: Nicht genügend Daten (insbesondere '{main_metric_to_check}') in der History vorhanden, um zu plotten.")
        return

    epochs_ran = history.get('epoch', [])
    # Falls 'epoch' nicht da ist, versuche die Länge einer anderen Metrik als Basis zu nehmen
    if not epochs_ran and history.get(main_metric_to_check):
        epochs_ran = range(1, len(history[main_metric_to_check]) + 1)

    if not epochs_ran or not isinstance(epochs_ran, (list, range)) or len(epochs_ran) == 0:
        print("    Epoch-Daten fehlen oder sind ungültig in der History. Plot wird abgebrochen.")
        return

    plt.style.use('seaborn-v0_8-whitegrid') # Ein etwas ansprechenderer Plot-Stil
    fig, axs = plt.subplots(2, 2, figsize=(20, 15)) # Erstellt Figure und Subplots
    fig.suptitle(f'Trainingsverlauf: {experiment_tag}', fontsize=16)

    # Plot 1: Accuracy (Training & Validierung)
    if 'train_acc' in history and history['train_acc']:
        axs[0, 0].plot(epochs_ran, history['train_acc'], 'r.-', label='Train Accuracy')
    if 'val_acc' in history and history['val_acc']:
        axs[0, 0].plot(epochs_ran, history['val_acc'], 'b.-', label='Val Accuracy')
    axs[0, 0].set_title('Accuracy vs. Epoche')
    axs[0, 0].set_xlabel('Epoche'); axs[0, 0].set_ylabel('Accuracy'); axs[0, 0].legend(); axs[0, 0].grid(True)

    # Plot 2: Loss (Training & Validierung)
    if 'train_loss' in history and history['train_loss']:
        axs[0, 1].plot(epochs_ran, history['train_loss'], 'r.-', label='Train Loss')
    if 'val_loss' in history and history['val_loss']:
        axs[0, 1].plot(epochs_ran, history['val_loss'], 'b.-', label='Val Loss')
    axs[0, 1].set_title('Loss vs. Epoche')
    axs[0, 1].set_xlabel('Epoche'); axs[0, 1].set_ylabel('Loss'); axs[0, 1].legend(); axs[0, 1].grid(True)

    # Plot 3: F1-Score, Precision, Recall (Validierung - falls vorhanden)
    plot3_has_data = False
    if 'val_f1' in history and history['val_f1']:
        axs[1, 0].plot(epochs_ran, history['val_f1'], 'g.-', label='Val F1-Score')
        plot3_has_data = True
    if 'val_precision' in history and history['val_precision']:
        axs[1, 0].plot(epochs_ran, history['val_precision'], 'c.-', label='Val Precision')
        plot3_has_data = True
    if 'val_recall' in history and history['val_recall']:
        axs[1, 0].plot(epochs_ran, history['val_recall'], 'm.-', label='Val Recall')
        plot3_has_data = True
    if plot3_has_data:
        axs[1, 0].set_title('Validierungs F1/Precision/Recall vs. Epoche')
        axs[1, 0].set_xlabel('Epoche'); axs[1, 0].set_ylabel('Score'); axs[1, 0].legend(); axs[1, 0].grid(True)
    else:
        axs[1, 0].text(0.5, 0.5, 'Keine F1/Precision/Recall Daten verfügbar', horizontalalignment='center', verticalalignment='center', transform=axs[1,0].transAxes)
        axs[1, 0].set_title('Validierungs F1/Precision/Recall')


    # Plot 4: Lernrate (falls vorhanden)
    if 'lr' in history and history['lr']:
        axs[1, 1].plot(epochs_ran, history['lr'], 'k.-', label='Lernrate')
        axs[1, 1].set_title('Lernrate vs. Epoche')
        axs[1, 1].set_xlabel('Epoche'); axs[1, 1].set_ylabel('Lernrate'); axs[1, 1].legend(); axs[1, 1].grid(True)
        axs[1, 1].set_yscale('log') # Logarithmische Skala ist oft sinnvoll für Lernraten
    else:
        axs[1, 1].text(0.5, 0.5, 'Keine Lernraten-Daten verfügbar', horizontalalignment='center', verticalalignment='center', transform=axs[1,1].transAxes)
        axs[1, 1].set_title('Lernrate')


    plt.tight_layout(rect=[0, 0, 1, 0.96]) # rect passt den Platz für suptitle an
    plot_save_path = output_dir_session / f"{experiment_tag}_training_history.png"
    try:
        plt.savefig(plot_save_path)
        print(f"    Trainingshistorie-Plot erfolgreich gespeichert: {plot_save_path.resolve()}")
    except Exception as e:
        print(f"    FEHLER beim Speichern des Trainingshistorie-Plots: {e}")
    plt.close(fig) # Schließe die Figur explizit, um Speicher freizugeben