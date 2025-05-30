# src/config.py
# Dieses Modul zentralisiert alle Konfigurationsparameter für das Trainingsexperiment.
# Es definiert Pfade, Trainingshyperparameter, Optuna-Einstellungen und andere globale Variablen.

from pathlib import Path
import torch

# --- 0. Bestimmung des Projekt-Root-Verzeichnisses ---
# Annahme: Dieses Skript (config.py) befindet sich in Hackatron/model_training/src/
# und main_training.py befindet sich in Hackatron/model_training/.
# PROJECT_ROOT soll auf den Hauptordner "Hackatron" zeigen.
try:
    # Geht drei Ebenen vom aktuellen Dateiort (__file__) nach oben: src -> model_training -> Hackatron
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
except NameError:
    # Fallback, falls __file__ nicht definiert ist (z.B. bei interaktiver Ausführung ohne Datei-Kontext)
    # Dies nimmt das aktuelle Arbeitsverzeichnis an, was angepasst werden muss, wenn es nicht dem Projekt-Root entspricht.
    PROJECT_ROOT = Path(".").resolve()
    print(f"WARNUNG: __file__ nicht definiert, PROJECT_ROOT auf aktuelles Arbeitsverzeichnis gesetzt: {PROJECT_ROOT}")
    print("         Bitte sicherstellen, dass dies dem korrekten Projekt-Root entspricht oder das Skript normal ausführen.")

print(f"Project Root (definiert in config.py): {PROJECT_ROOT.resolve()}")

# --- Experiment-spezifische Konfiguration ---
# WICHTIG: Dieser Name sollte für jeden neuen, vollständigen Experimentlauf geändert werden,
# um Ergebnisse sauber zu trennen und nachvollziehbar zu machen.
EXPERIMENT_NAME = "retrain_v3" # Beispiel: v3 des Retrainings, Fokus auf Accuracy, mit Early Stopping

# Haupt-Ausgabeordner für alle Ergebnisse dieses spezifischen Experiments.
# Alle Ausgaben (Daten-Splits, Modelle, Logs, Plots) werden hierunter gespeichert.
EXPERIMENT_OUTPUT_DIR = PROJECT_ROOT / "experiment_runs" / EXPERIMENT_NAME

# --- 1. Pfaddefinitionen ---
# Quellordner der kombinierten, annotierten Daten für den initialen Train/Val-Split.
# Dieser Ordner sollte die Unterordner 'yes' und 'no' mit den entsprechenden Bildern enthalten.
ANNOTATED_DATA_SOURCE_DIR = PROJECT_ROOT / "data_processed" / "data_annotated_kombiniert"

# Basisordner für den aufgeteilten Datensatz (Train/Validation) dieses Experiments.
# Wird innerhalb des EXPERIMENT_OUTPUT_DIR erstellt.
BASE_DATA_SPLIT_DIR = EXPERIMENT_OUTPUT_DIR / "data_split"
TRAIN_DIR = BASE_DATA_SPLIT_DIR / "train"
VAL_DIR = BASE_DATA_SPLIT_DIR / "validation"

# Speicherort für das finale, beste Modell dieses Experiments.
# Wird innerhalb des EXPERIMENT_OUTPUT_DIR in einem Unterordner gespeichert.
FINAL_MODEL_OUTPUT_SUBDIR = EXPERIMENT_OUTPUT_DIR / "final_trained_model"
FINAL_MODEL_SAVE_PATH = FINAL_MODEL_OUTPUT_SUBDIR / f"best_model_{EXPERIMENT_NAME}.pth"

# Optuna-spezifische Pfade und Namen, ebenfalls im Experiment-Ordner.
OPTUNA_OUTPUT_SUBDIR = EXPERIMENT_OUTPUT_DIR / "optuna_study"
OPTUNA_DB_PATH = OPTUNA_OUTPUT_SUBDIR / f"study_data_{EXPERIMENT_NAME}.db" # Datenbankdatei
OPTUNA_STUDY_NAME = f"{EXPERIMENT_NAME}_search" # Name der Studie innerhalb der Datenbank

# Ordner für von Optuna generierte Visualisierungen und temporär gespeicherte Modelle einzelner Trials.
OPTUNA_VISUALS_DIR = OPTUNA_OUTPUT_SUBDIR / "visualizations"
OPTUNA_TEMP_MODELS_DIR = OPTUNA_OUTPUT_SUBDIR / "trial_temp_models" # Für Modelle aus Optuna-Trials
ERROR_ANALYSIS_DIR = EXPERIMENT_OUTPUT_DIR / "error_analysis" # Für False Positives/Negatives

# --- 2. Trainingsparameter ---
BATCH_SIZE = 128     # Batch-Größe für Training und Validierung. Ggf. an GPU-Speicher anpassen.
NUM_WORKERS = 6     # Anzahl der CPU-Worker für den DataLoader. Empfehlung: Anzahl CPU-Kerne - 2.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Nutzt GPU, falls verfügbar.

# Optuna Hyperparameter-Suche Konfiguration
N_TRIALS = 100       # Anzahl der verschiedenen Hyperparameter-Kombinationen, die Optuna testen soll.
                     # Für eine gründliche Suche sind Werte von 100-200+ üblich.
EPOCHS_PER_TRIAL_MAX = 50 # Maximale Anzahl von Trainings-Epochen für jeden einzelnen Optuna-Trial.
                          # Early Stopping kann einen Trial früher beenden.
OPTUNA_EARLY_STOPPING_PATIENCE_TRIAL = 8 # Anzahl der Epochen ohne Verbesserung der Zielmetrik in einem
                                         # Optuna-Trial, bevor dieser Trial abgebrochen wird.

# Finales Training Konfiguration (nachdem Optuna die besten Parameter gefunden hat)
FINAL_TRAINING_EPOCHS_MAX = 150 # Maximale Anzahl von Trainings-Epochen für das finale Modell.
FINAL_EARLY_STOPPING_PATIENCE = 15 # Anzahl der Epochen ohne Verbesserung der Zielmetrik im
                                   # finalen Training, bevor es gestoppt wird.

# --- 3. Bildparameter und globale Seeds ---
IMG_SIZE = 250      # Zielgröße der Bilder (in Pixeln) nach der Transformation.
RANDOM_SEED = 42    # Seed für Zufallsgeneratoren zur Gewährleistung der Reproduzierbarkeit.
TRAIN_VAL_SPLIT = 0.8 # Anteil der Daten für das Trainingsset (z.B. 0.8 = 80% Training, 20% Validierung).

# Standard-Normalisierungsparameter (basierend auf ImageNet-Statistiken).
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# --- 4. Metrik für Optimierung und Modellauswahl ---
# Die Validierungsmetrik, die Optuna optimieren soll und nach der das beste Modell
# während des Trainings (Optuna-Trials und finales Training) ausgewählt wird.
# Mögliche Optionen: 'val_acc' (Accuracy), 'val_f1' (F1-Score), 'val_loss' (Loss).
METRIC_TO_OPTIMIZE = 'val_acc'

# Gibt an, ob ein höherer Wert der METRIC_TO_OPTIMIZE besser ist.
# True für Accuracy, F1-Score, Precision, Recall.
# False für Loss.
HIGHER_IS_BETTER_FOR_METRIC = True # Da Accuracy maximiert werden soll.

# --- 5. Sicherstellen, dass wichtige Ausgabeordner existieren ---
# Dies geschieht beim Start des Hauptskripts, um Fehler zu vermeiden.
# Die spezifischen Unterordner wie TRAIN_DIR, VAL_DIR werden von der split_data Funktion erstellt.
EXPERIMENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_MODEL_OUTPUT_SUBDIR.mkdir(parents=True, exist_ok=True)
OPTUNA_OUTPUT_SUBDIR.mkdir(parents=True, exist_ok=True) # Stellt sicher, dass der Oberordner für DB, Visuals etc. da ist
OPTUNA_VISUALS_DIR.mkdir(parents=True, exist_ok=True)
OPTUNA_TEMP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
ERROR_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# Dieser globale Ordner für trainierte Modelle ist optional, wenn alle Modelle
# pro Experiment gespeichert werden. Kann aber nützlich sein für eine zentrale Ablage.
# (PROJECT_ROOT / "trained_models").mkdir(parents=True, exist_ok=True) #AUSKOMMENTIERT/ENTFERNT

print(f"Konfiguration für Experiment '{EXPERIMENT_NAME}' geladen.")
print(f"  Datenquelle für Split: {ANNOTATED_DATA_SOURCE_DIR.resolve()}")
print(f"  Ausgaben werden gespeichert in: {EXPERIMENT_OUTPUT_DIR.resolve()}")
print(f"  Optimiere für Metrik: {METRIC_TO_OPTIMIZE} (Höher ist besser: {HIGHER_IS_BETTER_FOR_METRIC})")