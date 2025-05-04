import requests
import pandas as pd # Installieren mit: pip install pandas
import time

# --- Konfiguration ---
# Stelle sicher, dass diese URL genau der entspricht, auf der dein Annotator läuft
ANNOTATOR_URL = "http://127.0.0.1:5555"
# Stelle sicher, dass dieser Name exakt dem entspricht, den du im Annotator-Browserfenster erstellt hast!
PROJECT_NAME = "ev_positive_candidates" # ÄNDERE DAS, wenn du einen anderen Namen gewählt hast!
# Der Name der CSV-Datei mit den Koordinaten
COORDS_FILE = "ocm_coords.csv"
# --- Ende Konfiguration ---

def trigger_download(latitude, longitude, project):
    """ Sendet eine Download-Anfrage an den Annotator für eine Koordinate. """
    download_endpoint = f"{ANNOTATOR_URL}/download"
    headers = {'Content-Type': 'application/json'}
    # Wir simulieren eine Anfrage von einer Kartenoberfläche (wie Leaflet/OSM).
    # Wir übergeben die einzelne Koordinate. Das Annotator-Tool sollte das verarbeiten können,
    # da es in downloader.py Koordinaten in eine Bounding Box umwandelt.
    payload = {
        "project_name": project,
        "map_type": "leaflet", # Oder "google", je nachdem, welches Interface im Annotator implementiert ist
        "north": latitude,
        "south": latitude,
        "east": longitude,
        "west": longitude
        # --- Falls der Annotator eine explizite Bounding Box braucht, könnte man das so machen: ---
        # delta = 0.0005 # Kleiner Wert für eine Bounding Box um den Punkt (ca. 50m)
        # "north": latitude + delta,
        # "south": latitude - delta,
        # "east": longitude + delta,
        # "west": longitude - delta,
    }
    try:
        # Timeout erhöhen, da Download+Cropping dauern kann
        response = requests.post(download_endpoint, json=payload, headers=headers, timeout=180)
        response.raise_for_status() # Wirft einen Fehler bei HTTP-Status 4xx oder 5xx
        result = response.json()
        print(f"Anfrage für ({latitude:.4f}, {longitude:.4f}): {result.get('message', 'Keine Nachricht')}")
        return result.get('success', False)
    except requests.exceptions.Timeout:
        print(f"Fehler (Timeout) bei Anfrage für ({latitude:.4f}, {longitude:.4f}).")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Fehler (Request) bei Anfrage für ({latitude:.4f}, {longitude:.4f}): {e}")
        return False
    except Exception as e:
        print(f"Unerwarteter Fehler bei ({latitude:.4f}, {longitude:.4f}): {e}")
        return False


# --- Hauptskript ---
try:
    coords_df = pd.read_csv(COORDS_FILE)
    print(f"{len(coords_df)} Koordinaten aus {COORDS_FILE} geladen.")
except FileNotFoundError:
    print(f"FEHLER: Datei {COORDS_FILE} nicht gefunden. Stelle sicher, dass sie im selben Ordner wie dieses Skript liegt.")
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

print(f"Starte Download für Projekt '{PROJECT_NAME}'...")
successful_downloads = 0
failed_downloads = 0


for index, row in coords_df.iterrows():
    # Überspringe Zeilen, wenn Latitude oder Longitude fehlen
    if pd.isna(row['latitude']) or pd.isna(row['longitude']):
        print(f"Überspringe Zeile {index+1}: Fehlende Koordinaten.")
        failed_downloads += 1
        continue

    lat = row['latitude']
    lon = row['longitude']

    if trigger_download(lat, lon, PROJECT_NAME):
        successful_downloads += 1
    else:
        failed_downloads += 1

    # Pause zwischen Anfragen, um den Server (und deinen PC) nicht zu überlasten
    # Ggf. anpassen, wenn du Fehler siehst oder es zu langsam ist
    time.sleep(3) # 3 Sekunden Pause

print("\nDownload-Zusammenfassung:")
print(f"Erfolgreich verarbeitete Anfragen: {successful_downloads}")
print(f"Fehlgeschlagene/Übersprungene Anfragen: {failed_downloads}")
print("Downloads abgeschlossen (im Hintergrund läuft ggf. noch das Cropping im Annotator).")