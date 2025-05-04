import requests
import json

# --- Konfiguration ---
ANNOTATOR_URL = "http://127.0.0.1:5555" # Passe Port an, falls nötig
PROJECT_NAME = "ev_positive_filtered" # Passe an, falls du einen anderen Namen verwendest
# Die CH1903+ / LV95 Koordinaten von map.geo.admin.ch
TARGET_X = 2723599 # Runde auf ganze Zahl
TARGET_Y = 1077026 # Runde auf ganze Zahl
# --- Ende Konfiguration ---

download_endpoint = f"{ANNOTATOR_URL}/download"
headers = {'Content-Type': 'application/json'}

payload = {
    "project_name": PROJECT_NAME,
    "map_type": "swisstopo", # Wichtig: Hiermit sagen wir, dass x/y CH1903+ sind
    "x": TARGET_X,
    "y": TARGET_Y
}

print(f"Sende Test-Anfrage an {download_endpoint} mit Payload:")
print(json.dumps(payload, indent=2))

try:
    response = requests.post(download_endpoint, json=payload, headers=headers, timeout=180)
    response.raise_for_status()
    result = response.json()
    print("\nAntwort vom Server:")
    print(json.dumps(result, indent=2))

    if result.get('success'):
        print("\nDownload (und Crop) erfolgreich angestoßen!")
        print(f"Überprüfe jetzt die Bilder im Annotator unter: {ANNOTATOR_URL}/{PROJECT_NAME}/annotate")
    else:
        print("\nDownload fehlgeschlagen laut Server-Antwort.")

except requests.exceptions.RequestException as e:
    print(f"\nFehler bei der Anfrage: {e}")
except Exception as e:
    print(f"\nUnerwarteter Fehler: {e}")