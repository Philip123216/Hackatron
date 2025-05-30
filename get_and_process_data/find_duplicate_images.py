from PIL import Image
import imagehash # pip install imagehash Pillow
from pathlib import Path
import os
import shutil

# --- Konfiguration ---
# Pfade zu deinen beiden Annotations-Ordnern
DIR_A_BASE = Path(r"C:\Users\pssol\PycharmProjects\Hackatron\data_annotated")
DIR_B_BASE = Path(r"C:\Users\pssol\PycharmProjects\Hackatron\data_annotated_sfoe")

# Unterordner, die verglichen werden sollen (typischerweise die "yes"-Klassen)
SUBFOLDER_A = "yes"
SUBFOLDER_B = "yes"

# Wo sollen Duplikate aus Ordner B (oder A) gespeichert werden?
DUPLICATES_OUTPUT_DIR_FROM_B = Path("./duplicate_images_from_sfoe") # Für Duplikate aus SFOE, die in OCM schon sind
# DUPLICATES_OUTPUT_DIR_FROM_A = Path("./duplicate_images_from_ocm") # Falls du es andersrum machen willst

# Wie ähnlich müssen Hashes sein, um als Duplikat zu gelten?
# (Hamming-Distanz: 0 = identisch, kleine Zahl = sehr ähnlich)
# Für ahash/phash ist ein Wert von 0-5 oft gut. Experimentiere!
HASH_SIMILARITY_THRESHOLD = 12 # Niedriger Wert für hohe Ähnlichkeit

# Welchen Hashing-Algorithmus verwenden?
# 'ahash' (Average Hash) - schnell, gut für exakte/sehr nahe Duplikate
# 'phash' (Perceptual Hash) - robuster gegen kleine Änderungen (Größe, Kompression, leichte Farbänderung)
# 'dhash' (Difference Hash) - ähnlich phash
# 'whash' (Wavelet Hash) - oft noch robuster
HASH_METHOD = imagehash.phash # Empfehlung: phash oder dhash
# HASH_METHOD = imagehash.ahash
# HASH_METHOD = imagehash.dhash
# HASH_METHOD = imagehash.whash_haar # whash-db4 für Wavelet

# Sollte nur eine der Duplikat-Dateien verschoben/gelöscht werden oder beide?
# True: Verschiebt nur das Duplikat aus DIR_B_BASE
# False: Zeigt nur an (oder du könntest beide verschieben/löschen - Vorsicht!)
MOVE_DUPLICATES_FROM_B = True
# --- Ende Konfiguration ---

def get_image_hashes(directory, subfolder, hash_method):
    """ Erstellt eine Liste von Hashes für alle PNGs in einem Verzeichnis. """
    hashes = {}
    image_folder = directory / subfolder
    if not image_folder.exists():
        print(f"WARNUNG: Ordner {image_folder} nicht gefunden.")
        return hashes

    print(f"Hashing Bilder in {image_folder}...")
    image_files = list(image_folder.glob("*.png"))
    for i, img_path in enumerate(image_files):
        try:
            hash_val = hash_method(Image.open(img_path))
            hashes[img_path] = hash_val
            if (i + 1) % 50 == 0:
                print(f"  Gehasht: {i+1}/{len(image_files)}")
        except Exception as e:
            print(f"  Fehler beim Hashing von {img_path.name}: {e}")
    print(f"  Hashing für {image_folder} abgeschlossen. {len(hashes)} Hashes erstellt.")
    return hashes

def find_and_process_duplicates():
    """ Findet Duplikate zwischen zwei Ordnern basierend auf Image Hashing. """

    DUPLICATES_OUTPUT_DIR_FROM_B.mkdir(parents=True, exist_ok=True)

    hashes_a = get_image_hashes(DIR_A_BASE, SUBFOLDER_A, HASH_METHOD)
    hashes_b = get_image_hashes(DIR_B_BASE, SUBFOLDER_B, HASH_METHOD)

    if not hashes_a or not hashes_b:
        print("Konnte keine Hashes für einen oder beide Ordner erstellen. Breche ab.")
        return

    duplicate_count_from_b = 0
    processed_b_files = set() # Um Mehrfachvergleiche zu vermeiden, falls B Duplikate enthält

    print(f"\nVergleiche Hashes (Threshold: {HASH_SIMILARITY_THRESHOLD})...")

    # Iteriere durch die Hashes von Ordner B und vergleiche mit Ordner A
    for path_b, hash_b in hashes_b.items():
        if path_b in processed_b_files:
            continue
        processed_b_files.add(path_b)

        is_duplicate = False
        match_in_a = None

        for path_a, hash_a in hashes_a.items():
            # Berechne die Hamming-Distanz zwischen den Hashes
            distance = hash_b - hash_a
            if distance <= HASH_SIMILARITY_THRESHOLD:
                is_duplicate = True
                match_in_a = path_a
                break # Erstes gefundenes Duplikat reicht

        if is_duplicate:
            duplicate_count_from_b += 1
            print(f"  DUPLIKAT gefunden: '{path_b.name}' (aus {DIR_B_BASE.name}/{SUBFOLDER_B})")
            print(f"    ist ähnlich zu: '{match_in_a.name}' (aus {DIR_A_BASE.name}/{SUBFOLDER_A}) mit Distanz {distance}")

            if MOVE_DUPLICATES_FROM_B:
                try:
                    target_path = DUPLICATES_OUTPUT_DIR_FROM_B / path_b.name
                    # Um Überschreiben zu verhindern, falls Dateinamen identisch sind
                    # (unwahrscheinlich, wenn von verschiedenen Quellen, aber sicher ist sicher)
                    if target_path.exists():
                        base, ext = target_path.stem, target_path.suffix
                        target_path = DUPLICATES_OUTPUT_DIR_FROM_B / f"{base}_dup_{int(Path.cwd().stat().st_mtime)}{ext}"

                    shutil.move(str(path_b), str(target_path))
                    print(f"    -> '{path_b.name}' verschoben nach '{DUPLICATES_OUTPUT_DIR_FROM_B.name}'")
                except Exception as e:
                    print(f"    -> FEHLER beim Verschieben von '{path_b.name}': {e}")
            else:
                print(f"    -> (Aktion: Nur anzeigen, kein Verschieben)")

    print("\n--- Duplikatsuche abgeschlossen ---")
    print(f"Duplikate aus '{DIR_B_BASE.name}/{SUBFOLDER_B}' (die in '{DIR_A_BASE.name}/{SUBFOLDER_A}' existieren): {duplicate_count_from_b}")
    if MOVE_DUPLICATES_FROM_B:
        print(f"Diese wurden nach '{DUPLICATES_OUTPUT_DIR_FROM_B.resolve()}' verschoben.")

if __name__ == "__main__":
    find_and_process_duplicates()