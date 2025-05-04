import cv2  # OpenCV zum Anzeigen von Bildern
import os
import shutil
import glob
from pathlib import Path
import random

# --- Konfiguration ---
# Ordner, in dem deine heruntergeladenen WMS-Bilder liegen
IMAGE_SOURCE_DIR = Path("./ocm_images_wms")

# Temporäre Ordner, in die wir zuerst klassifizieren
# Wir machen den Train/Val-Split später in einem separaten Schritt!
ANNOTATED_YES_DIR = Path("./data_annotated/yes")
ANNOTATED_NO_DIR = Path("./data_annotated/no")
ANNOTATED_SKIP_DIR = Path("./data_annotated/skip") # Optional: Für unklare Bilder

# Fenstername für die Anzeige
WINDOW_NAME = "Annotate Image (Y=Yes, N=No, S=Skip, Q=Quit)"
# --- Ende Konfiguration ---

def annotate_images():
    """ Zeigt Bilder an und verschiebt sie basierend auf Tastendruck. """

    # Sicherstellen, dass die Zielordner existieren
    ANNOTATED_YES_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_NO_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_SKIP_DIR.mkdir(parents=True, exist_ok=True)

    # Alle PNG-Bilder im Quellordner finden
    image_files = glob.glob(str(IMAGE_SOURCE_DIR / "*.png"))

    if not image_files:
        print(f"Keine PNG-Bilder im Ordner '{IMAGE_SOURCE_DIR}' gefunden.")
        return

    # Optional: Bilder mischen, um nicht immer ähnliche hintereinander zu sehen
    random.shuffle(image_files)

    print(f"\n--- Starte Annotation ---")
    print(f"Quellordner: {IMAGE_SOURCE_DIR}")
    print(f"Zielordner:")
    print(f"  'y' -> {ANNOTATED_YES_DIR}")
    print(f"  'n' -> {ANNOTATED_NO_DIR}")
    print(f"  's' -> {ANNOTATED_SKIP_DIR}")
    print(f"Drücke 'q', um zu beenden.")
    print(f"{len(image_files)} Bilder zu annotieren.")

    annotated_count = 0
    skipped_count = 0
    remaining_count = len(image_files)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL) # Erlaubt Größenänderung

    for img_path_str in image_files:
        img_path = Path(img_path_str)
        remaining_count -= 1
        print(f"\nZeige Bild: {img_path.name} (Verbleibend: {remaining_count})")

        try:
            # Bild laden
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  -> FEHLER: Konnte Bild nicht laden: {img_path.name}")
                target_dir = ANNOTATED_SKIP_DIR # Fehlerhafte Bilder überspringen
                shutil.move(str(img_path), str(target_dir / img_path.name))
                skipped_count += 1
                continue

            # Bild anzeigen
            cv2.imshow(WINDOW_NAME, img)

            # Auf Tastendruck warten (unendlich lange)
            key = cv2.waitKey(0) & 0xFF # Maske für 64-bit Systeme

            # Zielordner basierend auf Taste bestimmen
            target_dir = None
            action = "unbekannt"

            if key == ord('y'):
                target_dir = ANNOTATED_YES_DIR
                action = "Ja"
            elif key == ord('n'):
                target_dir = ANNOTATED_NO_DIR
                action = "Nein"
            elif key == ord('s'):
                target_dir = ANNOTATED_SKIP_DIR
                action = "Übersprungen"
            elif key == ord('q'):
                print("Annotation durch Benutzer beendet.")
                break # Schleife verlassen
            else:
                print("  -> Ungültige Taste gedrückt. Bild wird übersprungen.")
                target_dir = ANNOTATED_SKIP_DIR # Unbekannte Tasten -> überspringen
                action = "Übersprungen (ungültige Taste)"

            # Datei verschieben
            if target_dir:
                print(f"  -> Aktion: {action}. Verschiebe nach: {target_dir.name}")
                shutil.move(str(img_path), str(target_dir / img_path.name))
                if action not in ["Übersprungen", "Übersprungen (ungültige Taste)"]:
                    annotated_count += 1
                else:
                    skipped_count += 1

        except Exception as e:
            print(f"  -> FEHLER beim Verarbeiten von {img_path.name}: {e}")
            # Versuche trotzdem, das Bild in den Skip-Ordner zu verschieben
            try:
                 if img_path.exists(): # Nur verschieben, wenn es noch da ist
                     shutil.move(str(img_path), str(ANNOTATED_SKIP_DIR / img_path.name))
                 skipped_count += 1
            except Exception as move_e:
                 print(f"  -> FEHLER auch beim Verschieben in Skip-Ordner: {move_e}")

    # Nach der Schleife alle Fenster schließen
    cv2.destroyAllWindows()
    print("\n--- Annotation abgeschlossen ---")
    print(f"Annotierte Bilder (Ja/Nein): {annotated_count}")
    print(f"Übersprungene/Fehlerhafte Bilder: {skipped_count}")
    print(f"Annotierte Bilder befinden sich in: {ANNOTATED_YES_DIR.parent}")

if __name__ == "__main__":
    annotate_images()