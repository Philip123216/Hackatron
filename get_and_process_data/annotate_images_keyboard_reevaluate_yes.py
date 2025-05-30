import cv2  # OpenCV zum Anzeigen von Bildern
import os
import shutil
import glob
from pathlib import Path
import random

# --- Konfiguration für die Re-Annotation ---
# Ordner, der die zu überprüfenden "yes"-Bilder enthält
IMAGE_SOURCE_DIR = Path(r"C:\Users\pssol\PycharmProjects\Hackatron\data_processed\data_annotated_kombiniert\yes")

# Zielordner für die Korrektur
# Wenn 'y' gedrückt wird, bleibt das Bild im IMAGE_SOURCE_DIR.
# Es wird also kein expliziter "CORRECTED_YES_DIR" benötigt, da es der Quellordner ist.
CORRECTED_NO_DIR = Path(r"C:\Users\pssol\PycharmProjects\Hackatron\data_processed\data_annotated_kombiniert\no")
# Optional: Für unklare Bilder während der Re-Annotation
RECHECK_SKIP_DIR = Path(r"C:\Users\pssol\PycharmProjects\Hackatron\data_processed\data_annotated_kombiniert\skip_recheck")

# Fenstername für die Anzeige
WINDOW_NAME = "Re-Annotate (Y=Confirm Yes, N=Move to No, S=Skip, Q=Quit)"
# --- Ende Konfiguration ---

def re_annotate_images():
    """ Zeigt Bilder aus dem 'yes'-Ordner an und verschiebt sie bei Bedarf oder lässt sie. """

    # Sicherstellen, dass die notwendigen Zielordner existieren
    # Der IMAGE_SOURCE_DIR (wo 'yes' Bilder bleiben) existiert ja bereits.
    CORRECTED_NO_DIR.mkdir(parents=True, exist_ok=True)
    RECHECK_SKIP_DIR.mkdir(parents=True, exist_ok=True)

    # Alle PNG-Bilder im Quellordner finden
    image_files = glob.glob(str(IMAGE_SOURCE_DIR / "*.png"))

    if not image_files:
        print(f"Keine PNG-Bilder im Ordner '{IMAGE_SOURCE_DIR}' gefunden.")
        return

    # Optional: Bilder mischen
    random.shuffle(image_files)

    print(f"\n--- Starte Re-Annotation ---")
    print(f"Quellordner (Bilder aktuell als 'yes'): {IMAGE_SOURCE_DIR}")
    print(f"Aktionen:")
    print(f"  'y' -> Bild bleibt in {IMAGE_SOURCE_DIR.name} (Bestätigt als 'yes')")
    print(f"  'n' -> Bild wird verschoben nach {CORRECTED_NO_DIR.name} (Korrigiert zu 'no')")
    print(f"  's' -> Bild wird verschoben nach {RECHECK_SKIP_DIR.name}")
    print(f"Drücke 'q', um zu beenden.")
    print(f"{len(image_files)} Bilder zu überprüfen.")

    confirmed_yes_count = 0
    moved_to_no_count = 0
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
                # Versuche, fehlerhafte Bilder in den Skip-Ordner zu verschieben
                if img_path.exists():
                    shutil.move(str(img_path), str(RECHECK_SKIP_DIR / img_path.name))
                skipped_count += 1
                continue

            # Bild anzeigen
            cv2.imshow(WINDOW_NAME, img)

            # Auf Tastendruck warten (unendlich lange)
            key = cv2.waitKey(0) & 0xFF # Maske für 64-bit Systeme

            action_taken = False

            if key == ord('y'):
                print(f"  -> Aktion: Bestätigt als 'Ja'. Bild bleibt in: {IMAGE_SOURCE_DIR.name}")
                confirmed_yes_count += 1
                action_taken = True
                # Kein shutil.move() nötig, da es schon am richtigen Ort ist
            elif key == ord('n'):
                print(f"  -> Aktion: Korrigiert zu 'Nein'. Verschiebe nach: {CORRECTED_NO_DIR.name}")
                shutil.move(str(img_path), str(CORRECTED_NO_DIR / img_path.name))
                moved_to_no_count += 1
                action_taken = True
            elif key == ord('s'):
                print(f"  -> Aktion: Übersprungen. Verschiebe nach: {RECHECK_SKIP_DIR.name}")
                shutil.move(str(img_path), str(RECHECK_SKIP_DIR / img_path.name))
                skipped_count += 1
                action_taken = True
            elif key == ord('q'):
                print("Re-Annotation durch Benutzer beendet.")
                break # Schleife verlassen
            else:
                print(f"  -> Ungültige Taste gedrückt ('{chr(key)}'). Bild wird in Skip-Ordner verschoben.")
                shutil.move(str(img_path), str(RECHECK_SKIP_DIR / img_path.name))
                skipped_count += 1
                action_taken = True


        except Exception as e:
            print(f"  -> FEHLER beim Verarbeiten von {img_path.name}: {e}")
            # Versuche trotzdem, das Bild in den Skip-Ordner zu verschieben
            try:
                 if img_path.exists(): # Nur verschieben, wenn es noch da ist
                     shutil.move(str(img_path), str(RECHECK_SKIP_DIR / img_path.name))
                 skipped_count += 1
            except Exception as move_e:
                 print(f"  -> FEHLER auch beim Verschieben in Skip-Ordner: {move_e}")

    # Nach der Schleife alle Fenster schließen
    cv2.destroyAllWindows()
    print("\n--- Re-Annotation abgeschlossen ---")
    print(f"Als 'Ja' bestätigt (geblieben in {IMAGE_SOURCE_DIR.name}): {confirmed_yes_count}")
    print(f"Zu 'Nein' korrigiert (verschoben nach {CORRECTED_NO_DIR.name}): {moved_to_no_count}")
    print(f"Übersprungene/Fehlerhafte Bilder (verschoben nach {RECHECK_SKIP_DIR.name}): {skipped_count}")

if __name__ == "__main__":
    # Erstelle die Ordnerstruktur, falls sie noch nicht existiert, bevor du das Skript ausführst.
    # Beispiel:
    # data_annotated_kombiniert/
    # ├── yes/
    # │   ├── bild1.png
    # │   └── bild2.png
    # └── no/  (wird vom Skript erstellt, wenn es nicht existiert)
    #
    # Stellen Sie sicher, dass IMAGE_SOURCE_DIR korrekt ist und Bilder enthält.

    re_annotate_images()