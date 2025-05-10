import pandas as pd
from pyproj import Transformer
from pyproj.exceptions import ProjError
from pathlib import Path
import time # Importieren für eine kleine Pause

# --- Konfiguration ---
# Die CSV-Datei mit den ursprünglichen Lat/Lon-Koordinaten
# Nimm die gefilterte oder die ungefilterte, je nachdem, was du willst
INPUT_COORDS_FILE = Path("ocm_coords_filtered.csv") # Oder "./ocm_coords.csv"
# Name der Ausgabedatei für die umgewandelten Koordinaten
OUTPUT_CSV_FILE_WITH_CH = Path("./ocm_coords_with_ch.csv")
# --- Ende Konfiguration ---

# --- PyProj Transformer initialisieren ---
try:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    print("PyProj Transformer für WGS84 -> CH1903+/LV95 initialisiert.")
except ProjError as e:
    print(f"FEHLER: Konnte PyProj Transformer nicht initialisieren: {e}")
    print("Stelle sicher, dass 'pyproj' korrekt installiert ist (`pip install pyproj`).")
    exit()
# -------------------------------------

# --- Hauptskript ---
try:
    coords_df = pd.read_csv(INPUT_COORDS_FILE)
    print(f"{len(coords_df)} Lat/Lon Koordinaten aus {INPUT_COORDS_FILE} geladen.")
except FileNotFoundError:
    print(f"FEHLER: Eingabedatei {INPUT_COORDS_FILE} nicht gefunden.")
    exit()
except Exception as e:
    print(f"FEHLER beim Lesen der CSV-Datei {INPUT_COORDS_FILE}: {e}")
    exit()

print("Starte Koordinatenumwandlung...")
start_time = time.time() # Zeitmessung starten

coordinates_to_save = []
processed_count = 0
error_count = 0

for index, row in coords_df.iterrows():
    processed_count += 1
    if pd.isna(row['latitude']) or pd.isna(row['longitude']):
        print(f"Überspringe Zeile {index+1}: Fehlende Lat/Lon Koordinaten.")
        error_count += 1
        continue

    lat = row['latitude']
    lon = row['longitude']

    try:
        # --- Koordinaten umwandeln ---
        ch_x, ch_y = transformer.transform(lon, lat)
        # --- Ende Umwandlung ---

        coordinates_to_save.append({'latitude': lat, 'longitude': lon, 'ch_x': ch_x, 'ch_y': ch_y})

        # Fortschrittsanzeige alle 100 Einträge
        if processed_count % 100 == 0:
             print(f"Verarbeitet: {processed_count}/{len(coords_df)}")

    except Exception as e:
        print(f"FEHLER bei der Umwandlung für Zeile {index+1} (Lat: {lat}, Lon: {lon}): {e}")
        error_count += 1
        # Füge trotzdem einen Eintrag hinzu, um die Zeilenzahl zu halten, aber ohne CH-Koordinaten
        coordinates_to_save.append({'latitude': lat, 'longitude': lon, 'ch_x': None, 'ch_y': None})


end_time = time.time() # Zeitmessung stoppen
duration = end_time - start_time
print(f"\nUmwandlung abgeschlossen in {duration:.2f} Sekunden.")
print(f"Erfolgreich umgewandelt: {len(coordinates_to_save) - error_count}")
print(f"Fehler bei Umwandlung/Übersprungen: {error_count}")


# Speichere die umgewandelten Koordinaten
if coordinates_to_save:
    try:
        save_df = pd.DataFrame(coordinates_to_save)
        save_df.to_csv(OUTPUT_CSV_FILE_WITH_CH, index=False, encoding='utf-8')
        print(f"\nUmgewandelte Koordinaten erfolgreich gespeichert in '{OUTPUT_CSV_FILE_WITH_CH}'")
    except Exception as e:
        print(f"\nFEHLER beim Speichern der CSV-Datei '{OUTPUT_CSV_FILE_WITH_CH}': {e}")
else:
    print("\nKeine Koordinaten zum Speichern vorhanden.")