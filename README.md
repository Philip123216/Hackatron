# Hackatron: Deep Learning Pipeline für Bildklassifikation

Dieses Projekt bietet eine vollständige Pipeline für die binäre Bildklassifikation (z.B. Erkennung von Ladestationen auf Luftbildern) mit PyTorch. Es umfasst Datenbeschaffung, Annotation, Vorverarbeitung, Training (inkl. Hyperparameter-Optimierung), Evaluation, Fehleranalyse und Inferenz.

---

## Inhaltsverzeichnis
- [Projektüberblick](#projektüberblick)
- [Verzeichnisstruktur](#verzeichnisstruktur)
- [Ablauf & Datenfluss](#ablauf--datenfluss)
- [Wichtige Dateien & Skripte](#wichtige-dateien--skripte)
- [Python-Dateien im Detail](#python-dateien-im-detail)
- [Installation & Nutzung](#installation--nutzung)
- [Tipps & Hinweise](#tipps--hinweise)

---

## Projektüberblick

- **Ziel:** Automatisierte Klassifikation von Bildern (z.B. Luftbilder, Satellitenbilder) in zwei Klassen ("yes"/"no").
- **Technologien:** PyTorch, Optuna, OpenCV, PIL, torchvision, pandas, requests, pyproj, imagehash.
- **Features:** 
  - Datenbeschaffung via WMS
  - Manuelle und halbautomatische Annotation
  - Duplikaterkennung
  - Flexible, hyperparametrisierbare CNN-Architektur
  - Automatische Hyperparameter-Optimierung (Optuna)
  - Fehleranalyse (False Positives/Negatives)
  - Inferenz auf neuen Daten

---

## Verzeichnisstruktur

```
Hackatron/
│
├── get_and_process_data/         # Datenbeschaffung, -annotation, -vorverarbeitung
│   ├── annotate_images_keyboard.py
│   ├── annotate_images_keyboard_reevaluate_yes.py
│   ├── annotate_images_keyboard_sfoe.py
│   ├── download_ocm_images_wms.py
│   ├── download_ocm_images_wms_sfoe.py
│   ├── find_duplicate_images.py
│   ├── process_ch_files.py
│   └── using_trained_models/
│       └── predict_on_new_images.py
│
├── model_training/               # Trainingspipeline, Modell, Hilfsfunktionen
│   ├── main_training.py
│   └── src/
│       ├── config.py
│       ├── data_utils.py
│       ├── model.py
│       ├── trainer.py
│       └── utils.py
│
├── data_unprocessed/             # Rohdaten (CSV, Originalbilder)
├── data_processed/               # Vorverarbeitete/annotierte Daten
├── experiment_runs/              # Ergebnisse, Modelle, Fehleranalysen
├── requirements.txt              # Python-Abhängigkeiten
├── README.md                     # (Diese Datei)
├── evaluate_hackathon_dataset.py # Evaluation
├── trained_models/
│   └── best_model_retrain_v3.pth  <-- Vortrainiertes Modell für Evaluation
├── eparking/                      <-- HIER DIE EVALUATIONSDATEN DER DOZIERENDEN 
```

*_archive erhält nur alte versionen
---

## Ablauf & Datenfluss

### OCM-Datenpipeline
1. **Datenbeschaffung:**
   - `data_unprocessed/` enthält die heruntergeladenen Rohdaten aus OpenChargeMap (OCM) als CSV und ggf. weitere Formate.
2. **Koordinaten-Extraktion:**
   - `process_ch_files.py` (Skript 1): Extrahiert Längen- und Breitengrade (Lat/Lon) aus den Schweizer OCM-Daten. Ergebnis: `ocm_coords.csv` mit Kandidatenkoordinaten für Ladestationen.
3. **Bild-Download:**
   - `download_ocm_images_wms.py` (Skript 3): Lädt für jede OCM-Koordinate ein Luftbild (25x25m) via Swisstopo WMS herunter. Ergebnis: Ein Bild pro Koordinate im Ordner `ocm_images_wms`.
4. **Manuelle Annotation:**
   - `annotate_images_keyboard.py` (Skript 4): Tastaturgesteuerte Annotation der heruntergeladenen Bilder (OpenCV). Bilder werden per Tastendruck als "yes", "no" oder "skip" klassifiziert und automatisch in `data_annotated/` verschoben.

### SFOE-Datenpipeline
1. **Datenbeschaffung:**
   - `data_unprocessed/` enthält die heruntergeladenen Rohdaten aus SFOE als CSV und ggf. weitere Formate.
2. **Koordinaten-Extraktion:**
   - `process_sfoe_data.py` (analog zu OCM, falls vorhanden): Extrahiert Koordinaten aus SFOE-Daten. Ergebnis: `ocm_coords_sfoe.csv`.
3. **Bild-Download:**
   - `download_ocm_images_wms_sfoe.py`: Lädt für jede SFOE-Koordinate ein Luftbild via Swisstopo WMS herunter. Ergebnis: Bilder im Ordner `unlabeled_sfoe_images`.
4. **Modellbasierte Vorauswahl:**
   - `using_trained_models/predict_on_new_images.py`: Lädt ein trainiertes Modell und filtert aus den SFOE-Bildern nur die, die als positiv erkannt werden. Diese werden in `model_labeled_sfoe_images` gespeichert, um den manuellen Aufwand zu reduzieren.
5. **Manuelle Annotation:**
   - `annotate_images_keyboard_sfoe.py`: Tastaturgesteuerte manuelle Verifizierung der vom Modell als "yes" klassifizierten SFOE-Bilder. Bilder werden als "wirklich ja", "falsch positiv" oder "skip" sortiert.

### Weitere wichtige Schritte
- **Re-Annotation:**
  - `annotate_images_keyboard_reevaluate_yes.py`: Überprüfung der bereits als "yes" markierten Bilder, um die Qualität der Annotation zu erhöhen.
- **Duplikaterkennung:**
  - `find_duplicate_images.py`: Findet und verschiebt Duplikate beim Kombinieren der OCM- und SFOE-Datenquellen.

---

## Wichtige Dateien & Skripte

### Daten & Vorverarbeitung
- **data_unprocessed/**
  - Enthält die Rohdaten aus OCM und SFOE (CSV, ggf. weitere Formate).
  - Hier werden auch die extrahierten Koordinaten als CSV abgelegt (`ocm_coords.csv`, `ocm_coords_sfoe.csv`).

- **process_ch_files.py**
  - Extrahiert Lat/Lon aus OCM-Daten, erzeugt `ocm_coords.csv`.
- **process_sfoe_data.py**
  - (Analog zu oben, für SFOE-Daten, falls vorhanden.)

### Bild-Download
- **download_ocm_images_wms.py**
  - Lädt für jede OCM-Koordinate ein Luftbild via Swisstopo WMS herunter.
- **download_ocm_images_wms_sfoe.py**
  - Lädt für jede SFOE-Koordinate ein Luftbild via Swisstopo WMS herunter.

### Annotation
- **annotate_images_keyboard.py**
  - Tastaturgesteuerte Annotation der OCM-Bilder (OpenCV, 'y', 'n', 's').
- **annotate_images_keyboard_sfoe.py**
  - Tastaturgesteuerte Annotation der SFOE-Bilder.
- **annotate_images_keyboard_reevaluate_yes.py**
  - Überprüfung der als "yes" markierten Bilder.

### Duplikaterkennung
- **find_duplicate_images.py**
  - Findet und verschiebt Duplikate beim Kombinieren der OCM- und SFOE-Datenquellen.

### Modellbasierte Vorauswahl
- **using_trained_models/predict_on_new_images.py**
  - Lädt ein trainiertes Modell und filtert aus den SFOE-Bildern nur die, die als positiv erkannt werden.

### Training & Modell
- **model_training/main_training.py**
  - Zentrales Skript für die gesamte Trainingspipeline: Daten splitten, Optuna-Suche, finales Training, Evaluation, Fehleranalyse.
- **model_training/src/config.py**
  - Zentrale Konfigurationsdatei für Pfade, Hyperparameter, Metriken, Seeds, etc.
- **model_training/src/data_utils.py**
  - Hilfsfunktionen für Datenaufteilung, Transformationen, DataLoader.
- **model_training/src/model.py**
  - Definition des flexiblen CustomCNN-Modells.
- **model_training/src/trainer.py**
  - Trainingslogik, Optuna-Objective, Early Stopping, Modell-Checkpointing.
- **model_training/src/utils.py**
  - Hilfsfunktionen für Metriken, Fehleranalyse, Plots.

---

## Installation & Nutzung

### 0. Python-Umgebung:
Es wird empfohlen, ein virtuelles Environment (venv) zu verwenden, um Abhängigkeitskonflikte zu vermeiden.
Navigieren Sie im Terminal zum Hauptverzeichnis dieses Projekts (Hackatron/).
Erstellen Sie ein venv (z.B. mit dem Namen venv):

```bash
python -m venv venv
```
Aktivieren Sie das venv:
Windows (PowerShell): .\venv\Scripts\activate
macOS/Linux (bash/zsh): source venv/bin/activate

### 1. Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

### Nutzung für die Bonus-Aufgabe (Evaluation)
Dieses Projekt enthält ein vortrainiertes Modell und ein Skript (evaluate_hackathon_dataset.py) zur Evaluation auf dem von den Dozierenden bereitgestellten Datensatz.
Vorbereitung des Evaluationsdatensatzes:

-Kopieren Sie die 50 positiven Beispielbilder (mit Ladestationen) in den Ordner Hackatron/eparking/y/.
-Kopieren Sie die 50 negativen Beispielbilder (ohne Ladestationen) in den Ordner Hackatron/eparking/n/.
Ausführen des Evaluationsskripts:
Stellen Sie sicher, dass Ihr virtuelles Environment (falls erstellt) aktiviert ist.
Navigieren Sie im Terminal zum Hauptverzeichnis dieses Projekts (Hackatron/).
Führen Sie das Skript aus:
```bash
python evaluate_hackathon_dataset.py
```

Ergebnisse:
Das Skript gibt die Performance-Metriken (insbesondere die Accuracy) auf der Konsole aus.
Das verwendete vortrainierte Modell (best_model_retrain_v3.pth) befindet sich im Ordner trained_models/. Die Architekturparameter dieses Modells sind im Evaluationsskript hinterlegt.

### 2. Daten vorbereiten

- Rohdaten (CSV, Bilder) in `data_unprocessed/` ablegen.
- Mit den Skripten in `get_and_process_data/` herunterladen, annotieren und verarbeiten.
- Sicherstellen, dass in `data_processed/data_annotated_kombiniert/` die Unterordner `yes/` und `no/` mit Bildern existieren.

### 3. Training starten

```bash
cd model_training
python main_training.py
```

- Das Skript übernimmt die komplette Pipeline: Split, Optuna-Suche, finales Training, Evaluation.
- Ergebnisse erscheinen in `experiment_runs/<experiment_name>/`.


---

## Tipps & Hinweise

- **Konfiguration:** Alle wichtigen Parameter sind in `model_training/src/config.py` zentral definiert.
- **Modellarchitektur:** Anpassbar in `model_training/src/model.py`.
- **Datenaugmentation:** Anpassbar über Optuna und in den Transformationsfunktionen.
- **Fehleranalyse:** Falsch klassifizierte Bilder werden automatisch gespeichert.
- **.gitignore:** Große Daten- und Ergebnisordner sind ausgeschlossen.

---
