import json
import pandas as pd
from pathlib import Path

# --- Konfiguration ---
# Pfad zur heruntergeladenen SFOE JSON-Datei
INPUT_SFOE_JSON_FILE = Path("../data_unprocessed/raw_SFOE/ch.bfe.ladestellen-elektromobilitaet.json")
# INPUT_SFOE_JSON_FILE = Path("C:/Users/pssol/PycharmProjects/Hackatron/data_unprocessed/raw_SFOE/ch.bfe.ladestellen-elektromobilitaet.json") # Alternativ: Absoluter Pfad

# Name der CSV-Datei, die erstellt werden soll
OUTPUT_CSV_FILE = Path("../ocm_coords_sfoe.csv") # Speichern wir es eine Ebene höher oder wo du deine CSVs hast
# OUTPUT_CSV_FILE = Path("C:/Users/pssol/PycharmProjects/Hackatron/ocm_coords_sfoe.csv")
# --- Ende Konfiguration ---

print(f"Lese SFOE JSON-Datei: {INPUT_SFOE_JSON_FILE}")

all_coordinates = []
error_count = 0
processed_records = 0

# Sicherstellen, dass der Ausgabeordner existiert (falls die CSV in einem Unterordner liegt)
OUTPUT_CSV_FILE.parent.mkdir(parents=True, exist_ok=True)


try:
    with open(INPUT_SFOE_JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Navigiere zur Liste der Ladestationen
    # Annahme: "EVSEData" ist eine Liste und wir nehmen das erste Element
    if "EVSEData" in data and isinstance(data["EVSEData"], list) and len(data["EVSEData"]) > 0:
        evse_data_records = data["EVSEData"][0].get("EVSEDataRecord", [])
        if not isinstance(evse_data_records, list):
            print("FEHLER: 'EVSEDataRecord' ist keine Liste oder nicht vorhanden im ersten Element von 'EVSEData'.")
            evse_data_records = [] # Leere Liste, um Fehler zu vermeiden
    else:
        print("FEHLER: Schlüssel 'EVSEData' nicht gefunden, ist keine Liste oder ist leer.")
        evse_data_records = []

    print(f"Anzahl der gefundenen EVSEDataRecord-Einträge: {len(evse_data_records)}")

    for record in evse_data_records:
        processed_records += 1
        try:
            geo_coords_dict = record.get("GeoCoordinates")
            if geo_coords_dict and isinstance(geo_coords_dict, dict):
                google_coords_str = geo_coords_dict.get("Google")
                if google_coords_str and isinstance(google_coords_str, str):
                    # Teile den String "Latitude Longitude"
                    parts = google_coords_str.split()
                    if len(parts) == 2:
                        lat_str, lon_str = parts
                        try:
                            latitude = float(lat_str)
                            longitude = float(lon_str)
                            all_coordinates.append({'latitude': latitude, 'longitude': longitude})
                        except ValueError:
                            print(f"  WARNUNG: Konnte Koordinaten nicht in Zahlen umwandeln für EvseID {record.get('EvseID', 'N/A')}: '{google_coords_str}'")
                            error_count += 1
                    else:
                        print(f"  WARNUNG: Unerwartetes Format für Google-Koordinaten für EvseID {record.get('EvseID', 'N/A')}: '{google_coords_str}'")
                        error_count += 1
                else:
                    # print(f"  INFO: Kein 'Google'-Schlüssel in GeoCoordinates für EvseID {record.get('EvseID', 'N/A')}")
                    error_count +=1 # Zählen wir als Fehler, da wir Koordinaten erwarten
            else:
                # print(f"  INFO: Keine 'GeoCoordinates' für EvseID {record.get('EvseID', 'N/A')}")
                error_count +=1 # Zählen wir als Fehler

        except Exception as e:
            print(f"FEHLER beim Verarbeiten eines Records (EvseID {record.get('EvseID', 'N/A')}): {e}")
            error_count += 1

except FileNotFoundError:
    print(f"FEHLER: Eingabedatei '{INPUT_SFOE_JSON_FILE}' nicht gefunden.")
    exit()
except json.JSONDecodeError:
    print(f"FEHLER: Die Datei '{INPUT_SFOE_JSON_FILE}' scheint keine gültige JSON-Datei zu sein.")
    exit()
except Exception as e:
    print(f"Ein unerwarteter Fehler beim Laden oder Verarbeiten der JSON-Datei ist aufgetreten: {e}")
    exit()

print(f"\nVerarbeitung abgeschlossen.")
print(f"Verarbeitete Records insgesamt: {processed_records}")
print(f"Koordinaten erfolgreich extrahiert: {len(all_coordinates)}")
print(f"Einträge ohne gültige Koordinaten / Verarbeitungsfehler: {error_count}")

if not all_coordinates:
    print("\nKeine Koordinaten zum Speichern gefunden.")
else:
    # Erstelle einen Pandas DataFrame
    coords_df = pd.DataFrame(all_coordinates)

    # Entferne Duplikate, falls vorhanden (dieselbe Koordinate könnte mehrmals vorkommen)
    coords_df.drop_duplicates(inplace=True)
    print(f"Koordinaten nach Entfernung von Duplikaten: {len(coords_df)}")


    # Speichere den DataFrame als CSV-Datei
    try:
        coords_df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8')
        print(f"\nKoordinaten erfolgreich in '{OUTPUT_CSV_FILE}' gespeichert.")
    except Exception as e:
        print(f"\nFEHLER beim Speichern der CSV-Datei '{OUTPUT_CSV_FILE}': {e}")