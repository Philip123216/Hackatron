import os
import json
import pandas as pd
from pathlib import Path

# --- Konfiguration ---
# Pfad zum Ordner, der die einzelnen OCM JSON-Dateien für die Schweiz enthält
# Passe dies an, wenn du den Ordner woanders erstellt hast
INPUT_CH_DATA_FOLDER = Path("./Annotator/ocm_ch_data/data/CH")
# Name der CSV-Datei, die erstellt werden soll
OUTPUT_CSV_FILE = Path("./ocm_coords.csv")
# --- Ende Konfiguration ---

print(f"Suche nach JSON-Dateien in: {INPUT_CH_DATA_FOLDER}")

if not INPUT_CH_DATA_FOLDER.is_dir():
    print(f"FEHLER: Eingabeordner '{INPUT_CH_DATA_FOLDER}' nicht gefunden oder ist kein Ordner.")
    print("Stelle sicher, dass du Schritt 1-6 (git sparse checkout) erfolgreich ausgeführt hast.")
    exit()

all_coordinates = []
file_count = 0
error_count = 0

# Gehe durch alle Dateien im CH-Ordner
for filename in os.listdir(INPUT_CH_DATA_FOLDER):
    if filename.lower().endswith(".json"):
        file_count += 1
        filepath = INPUT_CH_DATA_FOLDER / filename
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                poi_data = json.load(f)

            # Extrahiere Koordinaten, wenn vorhanden
            if isinstance(poi_data, dict) and \
               'AddressInfo' in poi_data and \
               isinstance(poi_data.get('AddressInfo'), dict) and \
               poi_data['AddressInfo'].get('Latitude') is not None and \
               poi_data['AddressInfo'].get('Longitude') is not None:

                latitude = poi_data['AddressInfo']['Latitude']
                longitude = poi_data['AddressInfo']['Longitude']
                all_coordinates.append({'latitude': latitude, 'longitude': longitude})
            # else:
                # Optional: Protokollieren, wenn keine Koordinaten gefunden wurden
                # print(f"Warnung: Keine Koordinaten in Datei {filename} gefunden.")

        except json.JSONDecodeError:
            print(f"FEHLER: Ungültiges JSON in Datei: {filename}")
            error_count += 1
        except Exception as e:
            print(f"FEHLER beim Verarbeiten der Datei {filename}: {e}")
            error_count += 1

print(f"\nVerarbeitung abgeschlossen.")
print(f"Dateien gefunden: {file_count}")
print(f"Koordinaten extrahiert: {len(all_coordinates)}")
print(f"Fehler beim Verarbeiten: {error_count}")

if not all_coordinates:
    print("\nKeine Koordinaten zum Speichern gefunden.")
else:
    # Erstelle einen Pandas DataFrame
    coords_df = pd.DataFrame(all_coordinates)

    # Speichere den DataFrame als CSV-Datei
    try:
        coords_df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8')
        print(f"\nKoordinaten erfolgreich in '{OUTPUT_CSV_FILE}' gespeichert.")
    except Exception as e:
        print(f"\nFEHLER beim Speichern der CSV-Datei '{OUTPUT_CSV_FILE}': {e}")