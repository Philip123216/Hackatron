import cv2  # OpenCV zum Anzeigen von Bildern
import os
import shutil
import glob
from pathlib import Path
import random

# --- Konfiguration ---
# Ordner, in dem die vom Modell als "Ja" klassifizierten Kandidaten liegen (SFOE-Daten)
IMAGE_SOURCE_DIR = Path("../get_data_withmodel/model_labeled_sfoe_images") # Relativ zum Speicherort dieses Skripts

# Zielordner für die manuelle Verifizierung der SFOE-Daten
# Diese werden in einem eigenen Unterordner von "data_annotated" gespeichert
ANNOTATION_BASE_DIR = Path("../data_annotated_sfoe") # Hauptordner für diese Annotationsrunde
ANNOTATED_YES_DIR = ANNOTATION_BASE_DIR / "yes"
ANNOTATED_NO_DIR = ANNOTATION_BASE_DIR / "no" # Das werden die False Positives des Modells sein
ANNOTATED_SKIP_DIR = ANNOTATION_BASE_DIR / "skip"

# Fenstername für die Anzeige
WINDOW_NAME = "Manuelle Verifizierung SFOE (Y=Wirklich Ja, N=Falsch Positiv, S=Skip, Q=Quit)"
# --- Ende Konfiguration ---

def annotate_images():
    """ Zeigt Bilder an und verschiebt sie basierend auf Tastendruck. """

    # Sicherstellen, dass die Zielordner existieren
    ANNOTATED_YES_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_NO_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_SKIP_DIR.mkdir(parents=True, exist_ok=True)

    # Alle PNG-Bilder im Quellordner finden
    if not IMAGE_SOURCE_DIR.exists():
        print(f"FEHLER: Quellordner '{IMAGE_SOURCE_DIR.resolve()}' nicht gefunden.")
        print("Stelle sicher, dass das predict_on_new_images.py Skript erfolgreich gelaufen ist und Bilder dort abgelegt hat.")
        return

    image_files = glob.glob(str(IMAGE_SOURCE_DIR / "*.png"))

    if not image_files:
        print(f"Keine PNG-Bilder im Ordner '{IMAGE_SOURCE_DIR.resolve()}' gefunden.")
        return

    # Optional: Bilder mischen, um nicht immer ähnliche hintereinander zu sehen
    random.shuffle(image_files)

    print(f"\n--- Starte manuelle Verifizierung der SFOE-Kandidaten ---")
    print(f"Quellordner (vom Modell als 'Ja' klassifiziert): {IMAGE_SOURCE_DIR.resolve()}")
    print(f"Zielordner:")
    print(f"  'y' (Wirklich Ja) -> {ANNOTATED_YES_DIR.resolve()}")
    print(f"  'n' (Falsch Positiv) -> {ANNOTATED_NO_DIR.resolve()}")
    print(f"  's' (Unsicher/Überspringen) -> {ANNOTATED_SKIP_DIR.resolve()}")
    print(f"Drücke 'q', um zu beenden.")
    print(f"{len(image_files)} Bilder zu verifizieren.")

    verified_yes_count = 0
    verified_no_fp_count = 0 # False Positives des Modells
    skipped_count = 0
    remaining_count = len(image_files)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL) # Erlaubt Größenänderung
    # Versuche, das Fenster etwas größer zu machen, wenn es zu klein ist
    cv2.resizeWindow(WINDOW_NAME, 800, 800) # Breite, Höhe (kann angepasst werden)


    for img_path_str in image_files:
        img_path = Path(img_path_str)
        remaining_count -= 1
        print(f"\nZeige Bild: {img_path.name} (Verbleibend: {remaining_count})")

        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  -> FEHLER: Konnte Bild nicht laden: {img_path.name}")
                target_dir = ANNOTATED_SKIP_DIR
                shutil.move(str(img_path), str(target_dir / img_path.name))
                skipped_count += 1
                continue

            cv2.imshow(WINDOW_NAME, img)
            key = cv2.waitKey(0) & 0xFF

            target_dir = None
            action = "unbekannt"

            if key == ord('y'):
                target_dir = ANNOTATED_YES_DIR
                action = "Wirklich Ja"
                verified_yes_count +=1
            elif key == ord('n'):
                target_dir = ANNOTATED_NO_DIR
                action = "Falsch Positiv (Nein)"
                verified_no_fp_count +=1
            elif key == ord('s'):
                target_dir = ANNOTATED_SKIP_DIR
                action = "Übersprungen"
                skipped_count +=1
            elif key == ord('q'):
                print("Verifizierung durch Benutzer beendet.")
                break
            else:
                print("  -> Ungültige Taste gedrückt. Bild wird übersprungen.")
                target_dir = ANNOTATED_SKIP_DIR
                action = "Übersprungen (ungültige Taste)"
                skipped_count +=1

            if target_dir and img_path.exists(): # Nur verschieben, wenn Datei noch da ist
                print(f"  -> Aktion: {action}. Verschiebe nach: {target_dir.name}")
                shutil.move(str(img_path), str(target_dir / img_path.name))

        except Exception as e:
            print(f"  -> FEHLER beim Verarbeiten von {img_path.name}: {e}")
            try:
                 if img_path.exists():
                     shutil.move(str(img_path), str(ANNOTATED_SKIP_DIR / img_path.name))
                 skipped_count += 1
            except Exception as move_e:
                 print(f"  -> FEHLER auch beim Verschieben in Skip-Ordner: {move_e}")

    cv2.destroyAllWindows()
    print("\n--- Manuelle Verifizierung abgeschlossen ---")
    print(f"Bilder als 'Wirklich Ja' bestätigt: {verified_yes_count}")
    print(f"Bilder als 'Falsch Positiv' (also 'Nein') klassifiziert: {verified_no_fp_count}")
    print(f"Übersprungene/Fehlerhafte Bilder: {skipped_count}")
    print(f"Verifizierte Bilder befinden sich in den Unterordnern von: {ANNOTATION_BASE_DIR.resolve()}")

if __name__ == "__main__":
    annotate_images()