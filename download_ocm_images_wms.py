import requests
import pandas as pd
from pathlib import Path
import time
import math
from pyproj import Transformer, exceptions as ProjExceptions

# --- Konfiguration ---
# Eingabedatei: Enthält Lat/Lon Koordinaten von OCM
# Stelle sicher, dass die Spalten 'latitude' und 'longitude' heißen
INPUT_CSV_FILE = Path("./ocm_coords.csv")

# Ausgabeordner: Hier werden die einzelnen PNG-Bilder gespeichert
OUTPUT_IMAGE_DIR = Path("./ocm_images_wms")

# WMS Konfiguration
WMS_BASE_URL = "https://wms.geo.admin.ch/"
WMS_LAYER = 'ch.swisstopo.swissimage-product' # Aktueller Layer für SWISSIMAGE 10cm
TARGET_PIXEL_SIZE = 250 # Gewünschte Bildgröße in Pixeln (25m)
RESOLUTION_M_PER_PIXEL = 0.1 # Auflösung des Luftbilds (10cm)

# --- Ende Konfiguration ---

# --- PyProj Transformer initialisieren ---
try:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    print("PyProj Transformer für WGS84 -> CH1903+/LV95 initialisiert.")
except ProjExceptions.CRSError as e:
    print(f"FEHLER bei der Initialisierung von PyProj: {e}")
    print("Stelle sicher, dass PyProj korrekt installiert ist und die CRS-Definitionen verfügbar sind.")
    exit()
# -------------------------------------

def download_single_wms_tile(ch_x, ch_y, output_path):
    """ Lädt genau eine Kachel zentriert um ch_x, ch_y via WMS herunter. """
    try:
        half_width_pixels = TARGET_PIXEL_SIZE / 2
        half_width_meters = half_width_pixels * RESOLUTION_M_PER_PIXEL

        # Bounding Box in CH1903+ / LV95
        # Runden kann helfen, kleine Ungenauigkeiten zu vermeiden
        x_min = round(ch_x - half_width_meters, 2)
        x_max = round(ch_x + half_width_meters, 2)
        y_min = round(ch_y - half_width_meters, 2)
        y_max = round(ch_y + half_width_meters, 2)

        # WMS GetMap Parameter
        params = {
            'SERVICE': 'WMS',
            'VERSION': '1.3.0',
            'REQUEST': 'GetMap',
            'FORMAT': 'image/png',
            'TRANSPARENT': 'false',
            'LAYERS': WMS_LAYER,
            'STYLES': '',
            'WIDTH': str(TARGET_PIXEL_SIZE),
            'HEIGHT': str(TARGET_PIXEL_SIZE),
            'CRS': 'EPSG:2056',
            # BBOX Format: minx,miny,maxx,maxy (häufig für EPSG-Codes mit X zuerst)
            # Falls dies fehlschlägt, könnte es miny,minx,maxy,maxx sein.
            'BBOX': f'{x_min},{y_min},{x_max},{y_max}'
        }

        print(f"  -> Anfrage BBOX: {params['BBOX']}") # Zum Debuggen
        response = requests.get(WMS_BASE_URL, params=params, stream=True, timeout=45) # Timeout leicht erhöht
        response.raise_for_status() # Check for HTTP errors (4xx or 5xx)

        # Überprüfen, ob wirklich ein Bild zurückkam
        content_type = response.headers.get('Content-Type', '')
        if 'image' not in content_type.lower():
            print(f"  -> FEHLER: Server hat kein Bild zurückgegeben (Content-Type: {content_type}). Antworttext:")
            # Versuche, Text zu dekodieren, falls es eine Fehlermeldung vom Server ist
            try:
                print(response.text)
            except Exception:
                print("(Konnte Antworttext nicht dekodieren)")
            return False

        # Bild speichern
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        # print(f"  -> Bild erfolgreich gespeichert: {output_path.name}")
        return True

    except requests.exceptions.Timeout:
        print(f"  -> FEHLER (Timeout) beim WMS Download für CH ({round(ch_x)},{round(ch_y)}).")
        return False
    except requests.exceptions.RequestException as e:
        # Speziell für HTTP-Fehler den Statuscode und Text ausgeben
        status_code = e.response.status_code if e.response is not None else "N/A"
        error_text = e.response.text if e.response is not None else str(e)
        print(f"  -> FEHLER (Request {status_code}) beim WMS Download für CH ({round(ch_x)},{round(ch_y)}):")
        # Nur einen Teil des Fehlertextes ausgeben, falls er sehr lang ist
        print(error_text[:500] + ('...' if len(error_text) > 500 else ''))
        return False
    except Exception as e:
        print(f"  -> FEHLER (Allgemein) beim WMS Download für CH ({round(ch_x)},{round(ch_y)}): {e}")
        return False

# --- Hauptskript ---
# Ausgabeordner erstellen, falls nicht vorhanden
OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
print(f"Bilder werden in '{OUTPUT_IMAGE_DIR.resolve()}' gespeichert.")

try:
    coords_df = pd.read_csv(INPUT_CSV_FILE)
    print(f"{len(coords_df)} Lat/Lon Koordinaten aus '{INPUT_CSV_FILE}' geladen.")
except FileNotFoundError:
    print(f"FEHLER: Eingabedatei '{INPUT_CSV_FILE}' nicht gefunden.")
    exit()
except Exception as e:
    print(f"FEHLER beim Lesen der CSV-Datei: {e}")
    exit()

print(f"Starte WMS Download für {len(coords_df)} Koordinaten...")
successful_downloads = 0
failed_downloads = 0
already_exists = 0

# --- Schleife über die Koordinaten ---
for index, row in coords_df.iterrows():
    if pd.isna(row['latitude']) or pd.isna(row['longitude']):
        print(f"Zeile {index+1}/{len(coords_df)}: Übersprungen (fehlende Lat/Lon).")
        failed_downloads += 1
        continue

    lat = row['latitude']
    lon = row['longitude']
    print(f"Zeile {index+1}/{len(coords_df)}: Verarbeite Lat={lat:.5f}, Lon={lon:.5f}")

    try:
        # --- Koordinaten umwandeln ---
        ch_x, ch_y = transformer.transform(lon, lat) # Beachte Reihenfolge: Lon, Lat!
        # print(f"  -> Umgewandelt zu CH: X={ch_x:.2f}, Y={ch_y:.2f}")

        # --- Dateinamen definieren ---
        # Verwende gerundete CH-Koordinaten für einen eindeutigen Namen
        output_filename = f"ocm_img_{round(ch_x)}_{round(ch_y)}.png"
        output_filepath = OUTPUT_IMAGE_DIR / output_filename

        # --- Prüfen, ob Datei schon existiert ---
        if output_filepath.exists():
            print(f"  -> Existiert bereits: {output_filename}")
            already_exists += 1
            successful_downloads +=1 # Zählen wir als Erfolg, da Datei vorhanden
            continue # Nächste Koordinate

        # --- Download versuchen ---
        if download_single_wms_tile(ch_x, ch_y, output_filepath):
            successful_downloads += 1
        else:
            failed_downloads += 1

    except ProjExceptions.TransformError as e:
         print(f"  -> FEHLER bei der Koordinatenumwandlung für Lat={lat}, Lon={lon}: {e}")
         failed_downloads +=1
    except Exception as e:
        print(f"  -> FEHLER (Unerwartet) in Schleife für Lat={lat}, Lon={lon}: {e}")
        failed_downloads += 1

    # Kurze Pause, um den Swisstopo-Server nicht zu überlasten
    time.sleep(0.5) # 0.5 Sekunden sollten ok sein für WMS

# --- Ende der Schleife ---

print("\n--- Download abgeschlossen ---")
print(f"Gesamte Koordinaten verarbeitet: {len(coords_df)}")
print(f"Erfolgreich heruntergeladen/bereits vorhanden: {successful_downloads}")
print(f"Fehlgeschlagen/Übersprungen: {failed_downloads}")
print(f"Davon bereits vorhanden: {already_exists}")
print(f"Neue Bilder sollten sich im Ordner '{OUTPUT_IMAGE_DIR.resolve()}' befinden.")