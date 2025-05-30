# src/data_utils.py
# Dieses Modul enthält Hilfsfunktionen für die Datenvorbereitung:
# - Aufteilung des Datensatzes in Trainings- und Validierungssets.
# - Definition von Bildtransformationen (inkl. Augmentation für Trainingsdaten).
# - Erstellung von PyTorch DataLoaders.

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import glob
import random
import shutil
from pathlib import Path

def split_data(source_dir: Path, train_dir: Path, val_dir: Path,
               split_ratio: float, seed: int) -> None:
    """
    Teilt Bilddaten aus einem Quellverzeichnis in Trainings- und Validierungs-Unterverzeichnisse auf.
    Die Bilder werden pro Klasse aufgeteilt, um die Klassenverteilung beizubehalten.
    Der Quellordner muss Unterordner für jede Klasse enthalten (z.B. 'yes', 'no').

    Args:
        source_dir (Path): Pfad zum Hauptverzeichnis der annotierten Daten.
        train_dir (Path): Pfad zum Ziel-Trainingsverzeichnis.
        val_dir (Path): Pfad zum Ziel-Validierungsverzeichnis.
        split_ratio (float): Anteil der Daten für das Training (z.B. 0.8 für 80%).
        seed (int): Seed für den Zufallsgenerator zur Reproduzierbarkeit.
    """
    print(f"Starte Datenaufteilung von '{source_dir.resolve()}' nach '{train_dir.parent.resolve()}'...")
    random.seed(seed)

    if train_dir.parent.exists():
        print(f"  Entferne existierendes Datenaufteilungs-Verzeichnis: '{train_dir.parent.resolve()}'")
        shutil.rmtree(train_dir.parent)

    expected_classes = ['yes', 'no']
    for class_name in expected_classes:
        source_class_dir = source_dir / class_name
        train_class_dir = train_dir / class_name
        val_class_dir = val_dir / class_name

        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)

        if not source_class_dir.exists():
            print(f"  WARNUNG: Quellordner '{source_class_dir.resolve()}' für Klasse '{class_name}' nicht gefunden. Überspringe.")
            continue

        images = glob.glob(str(source_class_dir / "*.png"))
        if not images:
            print(f"  WARNUNG: Keine PNG-Bilder im Quellordner '{source_class_dir.resolve()}' für Klasse '{class_name}'. Überspringe.")
            continue

        random.shuffle(images)
        split_idx = int(len(images) * split_ratio)
        train_images, val_images = images[:split_idx], images[split_idx:]
        print(f"  Klasse '{class_name}': {len(images)} gesamt -> {len(train_images)} Training, {len(val_images)} Validierung.")

        for img_path_str in train_images:
            try: shutil.copy(Path(img_path_str), train_class_dir / Path(img_path_str).name)
            except Exception as e: print(f"    FEHLER beim Kopieren (Train) von '{img_path_str}': {e}")
        for img_path_str in val_images:
            try: shutil.copy(Path(img_path_str), val_class_dir / Path(img_path_str).name)
            except Exception as e: print(f"    FEHLER beim Kopieren (Val) von '{img_path_str}': {e}")
    print("Datenaufteilung abgeschlossen.")


def get_transforms_for_trial(img_size: int, normalize_mean: list, normalize_std: list,
                             rotation_degrees: int,
                             cj_brightness: float, cj_contrast: float,
                             cj_saturation: float, cj_hue: float,
                             hflip_p: float, vflip_p: float) -> dict:
    """
    Erstellt Trainings- und Validierungstransformationen basierend auf
    spezifischen (z.B. von Optuna vorgeschlagenen) Augmentationsparametern.
    """
    normalize = transforms.Normalize(mean=normalize_mean, std=normalize_std)
    train_transforms_list = [transforms.Resize((img_size, img_size))]
    if hflip_p > 0: train_transforms_list.append(transforms.RandomHorizontalFlip(p=hflip_p))
    if vflip_p > 0: train_transforms_list.append(transforms.RandomVerticalFlip(p=vflip_p))
    if rotation_degrees > 0: train_transforms_list.append(transforms.RandomRotation(rotation_degrees))
    # ColorJitter nur anwenden, wenn mindestens ein Parameter > 0 ist, um leere Transformation zu vermeiden
    if any(cj > 1e-6 for cj in [cj_brightness, cj_contrast, cj_saturation, cj_hue]): # Kleine Toleranz für float-Vergleich
        train_transforms_list.append(transforms.ColorJitter(
            brightness=cj_brightness, contrast=cj_contrast,
            saturation=cj_saturation, hue=cj_hue
        ))
    train_transforms_list.extend([transforms.ToTensor(), normalize])

    val_transforms_list = [transforms.Resize((img_size, img_size)), transforms.ToTensor(), normalize]
    return {'train': transforms.Compose(train_transforms_list), 'val': transforms.Compose(val_transforms_list)}


def get_fixed_transforms(img_size: int, normalize_mean: list, normalize_std: list) -> dict:
    """
    Erstellt Standard-Transformationen mit festen Augmentationswerten für das Training
    und Standard-Vorverarbeitung für die Validierung. Wird z.B. für das finale Training verwendet,
    wenn Augmentationsparameter nicht mehr getuned werden oder als Fallback.
    """
    normalize = transforms.Normalize(mean=normalize_mean, std=normalize_std)
    # Standard-Augmentationen für das finale Training (oder als Basis)
    train_transforms_obj = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20), # Fester Wert
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.08), # Feste Werte
        transforms.ToTensor(),
        normalize
    ])
    val_transforms_obj = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    return {'train': train_transforms_obj, 'val': val_transforms_obj}


def create_dataloaders(train_dir: Path, val_dir: Path, batch_size: int, num_workers: int,
                       train_transforms_obj: transforms.Compose,
                       val_transforms_obj: transforms.Compose,
                       device: torch.device) -> tuple:
    """
    Erstellt PyTorch DataLoaders für Trainings- und Validierungsdatensätze.
    Diese Funktion wird jetzt universell verwendet und bekommt die Transformationen übergeben.
    """
    try:
        train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transforms_obj)
        val_dataset = datasets.ImageFolder(str(val_dir), transform=val_transforms_obj)
    except FileNotFoundError as e:
        print(f"FEHLER: Datenordner für DataLoader nicht gefunden: {e}")
        print(f"  Überprüfte Pfade: Training='{train_dir.resolve()}', Validierung='{val_dir.resolve()}'")
        return None, None, None
    except Exception as e_general:
        print(f"FEHLER beim Erstellen der ImageFolder Datasets: {e_general}")
        print(f"  Überprüfte Pfade: Training='{train_dir.resolve()}', Validierung='{val_dir.resolve()}'")
        return None, None, None

    if not train_dataset.samples:
        print(f"FEHLER: Trainings-Datensatz in '{train_dir.resolve()}' ist leer oder enthält keine gültigen Bilder.")
        return None, None, None
    if not val_dataset.samples:
        print(f"FEHLER: Validierungs-Datensatz in '{val_dir.resolve()}' ist leer oder enthält keine gültigen Bilder.")
        return None, None, None

    use_persistent_workers = (num_workers > 0 and device.type == 'cuda')
    pin_memory_flag = (device.type == 'cuda')

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory_flag,
        persistent_workers=use_persistent_workers if use_persistent_workers else None,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory_flag,
        persistent_workers=use_persistent_workers if use_persistent_workers else None
    )

    print(f"\nDataLoaders erfolgreich erstellt:")
    print(f"  Training: {len(train_dataset)} Bilder in {len(train_loader)} Batches (Batch-Größe: {batch_size})")
    print(f"  Validierung: {len(val_dataset)} Bilder in {len(val_loader)} Batches (Batch-Größe: {batch_size})")
    print(f"  Gefundene Klassen: {train_dataset.classes}")
    print(f"  Klassen-Indizes-Mapping: {train_dataset.class_to_idx}")

    if train_dataset.class_to_idx.get('yes') != 1 or train_dataset.class_to_idx.get('no') != 0:
        print(f"\n*** ACHTUNG: Unerwartetes Klassen-Mapping! Erwartet {{'no': 0, 'yes': 1}}, aber erhalten: {train_dataset.class_to_idx} ***")
        print(f"  Dies beeinflusst die Metrikberechnung. Bitte Ordnernamen ('yes', 'no') im Trainingsdatensatz prüfen.")

    return train_loader, val_loader, train_dataset.class_to_idx