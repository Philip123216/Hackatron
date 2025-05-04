import requests
import pandas as pd
import time
import json
from pyproj import Transformer # Importieren für die Umwandlung
from pathlib import Path

# --- Konfiguration ---
ANNOTATOR_URL = "http://127.0.0.1:5555" # Passe Port an, falls nötig
PROJECT_NAME = "ev_positive_filtered" # Passe an, falls nötig
COORDS_FILE = Path("./ocm_coords.csv") # Deine CSV mit Lat/Lon
OUTPUT_CSV_FILE_WITH_CH = Path("./ocm_coords_with_ch.csv") # Optional: Datei zum Speichern der umgewandelten Koordinaten
# --- Ende Konfiguration ---

# --- PyProj Transformer initialisieren ---
# Von WGS84 (EPSG:4326 - Lat/Lon) zu CH1903+/LV95 (EPSG:2056)
# always_xy=True bedeutet, die Eingabe ist (Longitude, Latitude)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
print("PyProj Transformer für WGS84 -> CH1903+/LV95 initialisiert.")
# -------------------------------------

def trigger_ch_download(ch_x, ch_y, project):
    """ Sendet eine Download-Anfrage mit CH1903+ Koordinaten an den Annotator. """
    download_endpoint = f"{ANNOTATOR_URL}/download"
    headers = {'Content-Type': 'application/json'}

    # Runden auf ganze Zahlen, wie im erfolgreichen Test
    target_x = round(ch_x)
    target_y = round(ch_y)

    payload = {
        "project_name": project,
        "map_type": "swisstopo", # Wichtig: Jetzt immer swisstopo verwenden!
        "x": target_x,
        "y": target_y
    }
    try:
        response = requests.post(download_endpoint, json=payload, headers=headers, timeout=180) # Timeout
        response.raise_for_status()
        result = response.json()
        print(f"Anfrage CH ({target_x}, {target_y}): {result.get('message', 'Keine Nachricht')}")
        return result.get('success', False)
    except requests.exceptions.Timeout:
        print(f"Fehler (Timeout) bei CH-Anfrage ({target_x}, {target_y}).")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Fehler (Request) bei CH-Anfrage ({target_x}, {target_y}): {e}")
        return False
    except Exception as e:
        print(f"Unerwarteter Fehler bei CH-Anfrage ({target_x}, {target_y}): {e}")
        return False

# --- Hauptskript ---
try:
    coords_df = pd.read_csv(COORDS_FILE)
    print(f"{len(coords_df)} Lat/Lon Koordinaten aus {COORDS_FILE} geladen.")
except FileNotFoundError:
    print(f"FEHLER: Datei {COORDS_FILE} nicht gefunden.")
    exit()
except Exception as e:
    print(f"FEHLER beim Lesen der CSV-Datei: {e}")
    exit()

# Stelle sicher, dass der Annotator-Server läuft!
print("\n--- WICHTIG: Stelle sicher, dass der Annotator-Server läuft! ---")
print(f"Ziel-URL: {ANNOTATOR_URL}")
print(f"Ziel-Projekt: {PROJECT_NAME}")
print("Starte Download in 5 Sekunden...")
time.sleep(5)

print(f"Starte Download für Projekt '{PROJECT_NAME}' mit Koordinatenumwandlung...")
successful_requests = 0
failed_requests = 0
coordinates_to_save = []

for index, row in coords_df.iterrows():
    if pd.isna(row['latitude']) or pd.isna(row['longitude']):
        print(f"Überspringe Zeile {index+1}: Fehlende Lat/Lon Koordinaten.")
        failed_requests += 1
        continue

    lat = row['latitude']
    lon = row['longitude']

    try:
        # --- Koordinaten umwandeln ---
        ch_x, ch_y = transformer.transform(lon, lat) # Beachte Reihenfolge: Lon, Lat!
        # -----------------------------

        # Umgewandelte Koordinaten zur optionalen Speicherliste hinzufügen
        coordinates_to_save.append({'latitude': lat, 'longitude': lon, 'ch_x': ch_x, 'ch_y': ch_y})

        # Download mit den neuen CH-Koordinaten anstoßen
        if trigger_ch_download(ch_x, ch_y, PROJECT_NAME):
            successful_requests += 1
        else:
            failed_requests += 1

    except Exception as e:
        print(f"FEHLER bei der Umwandlung oder Anfrage für Zeile {index+1} (Lat: {lat}, Lon: {lon}): {e}")
        failed_requests += 1

    # Pause
    time.sleep(1) # Kannst du ggf. wieder auf 3 setzen, wenn es Probleme gibt

print("\nDownload-Zusammenfassung:")
print(f"Erfolgreich verarbeitete Anfragen: {successful_requests}")
print(f"Fehlgeschlagene/Übersprungene Anfragen: {failed_requests}")
print("Downloads abgeschlossen (im Hintergrund läuft ggf. noch das Cropping im Annotator).")

# Optional: Die umgewandelten Koordinaten auch speichern
if coordinates_to_save:
    try:
        save_df = pd.DataFrame(coordinates_to_save)
        save_df.to_csv(OUTPUT_CSV_FILE_WITH_CH, index=False, encoding='utf-8')
        print(f"\nUmgewandelte Koordinaten zusätzlich gespeichert in '{OUTPUT_CSV_FILE_WITH_CH}'")
    except Exception as e:
        print(f"\nFEHLER beim Speichern der umgewandelten Koordinaten: {e}")