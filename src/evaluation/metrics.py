# -*- coding: utf-8 -*-
import torch
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from torch.cuda.amp import autocast

def calculate_metrics(outputs, labels):
    """
    Calculates metrics for binary classification.
    
    Args:
        outputs (torch.Tensor): Model outputs (logits)
        labels (torch.Tensor): Ground truth labels
        
    Returns:
        tuple: (accuracy, precision, recall, f1) - Evaluation metrics
    """
    probs = torch.sigmoid(outputs).detach().cpu().numpy()
    preds = (probs > 0.5).astype(int).flatten()
    labels = labels.detach().cpu().numpy().flatten()
    try:
        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='binary', pos_label=1, zero_division=0)
        recall = recall_score(labels, preds, average='binary', pos_label=1, zero_division=0)
        f1 = f1_score(labels, preds, average='binary', pos_label=1, zero_division=0)
    except Exception as e:
        print(f"  Error calculating metrics: {e}")
        acc, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
    return acc, precision, recall, f1

def find_and_show_false_negatives(model, val_loader, device, class_to_idx, num_to_show=10):
    """
    Finds False Negatives (actual: yes, predicted: no) in the validation set
    and displays some of them.
    
    Args:
        model (nn.Module): Trained model
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device to use for inference
        class_to_idx (dict): Class to index mapping
        num_to_show (int): Number of false negatives to display (default: 10)
    """
    print("\n--- Searching for False Negatives (Actual: yes, Predicted: no) ---")
    model.eval()
    false_negative_files = []
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    yes_label_index = class_to_idx.get('yes', -1)
    no_pred_index = 0
    if yes_label_index == -1: return
    if not hasattr(val_loader, 'dataset'): return

    filepaths = [s[0] for s in val_loader.dataset.samples]
    all_labels_list = [s[1] for s in val_loader.dataset.samples]
    all_preds_list = []
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            with autocast(enabled=(device.type == 'cuda')):
                outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int().cpu().flatten().tolist()
            all_preds_list.extend(preds)

    for i in range(len(filepaths)):
        if all_labels_list[i] == yes_label_index and all_preds_list[i] == no_pred_index:
            false_negative_files.append(filepaths[i])

    print(f"Found {len(false_negative_files)} False Negative images.")
    if not false_negative_files: return

    print(f"Showing up to {num_to_show} False Negative images...")
    num_files_to_show = min(num_to_show, len(false_negative_files))
    if num_files_to_show == 0: return
    num_cols = 5
    num_rows = math.ceil(num_files_to_show / num_cols)
    plt.figure(figsize=(15, 3 * num_rows))
    for i, img_path in enumerate(false_negative_files[:num_files_to_show]):
        try:
            img = Image.open(img_path)
            ax = plt.subplot(num_rows, num_cols, i + 1)
            ax.imshow(img)
            ax.set_title(f"FN: {Path(img_path).name}\nTrue: yes, Pred: no", fontsize=8)
            ax.axis("off")
        except Exception as e: print(f"Could not load/show image {img_path}: {e}")
    plt.tight_layout()
    plt.show()

def evaluate_model(model, val_loader, device, class_to_idx, show_false_negatives=True, num_to_show=15):
    """
    Evaluates a trained model on the validation set.
    
    Args:
        model (nn.Module): Trained model
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device to use for inference
        class_to_idx (dict): Class to index mapping
        show_false_negatives (bool): Whether to display false negative examples (default: True)
        num_to_show (int): Number of false negatives to display (default: 15)
        
    Returns:
        tuple: (accuracy, precision, recall, f1) - Evaluation metrics
    """
    print("\n--- Evaluating Model on Validation Set ---")
    model.eval()
    all_labels = []
    all_outputs = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            with autocast(enabled=(device.type == 'cuda')):
                outputs = model(inputs)
            all_labels.append(labels)
            all_outputs.append(outputs)
    
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    acc, prec, rec, f1 = calculate_metrics(all_outputs, all_labels)
    
    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation Precision: {prec:.4f}")
    print(f"Validation Recall: {rec:.4f}")
    print(f"Validation F1-Score: {f1:.4f}")
    
    # Confusion Matrix
    preds = (torch.sigmoid(all_outputs).detach().cpu().numpy() > 0.5).astype(int).flatten()
    labels_np = all_labels.detach().cpu().numpy().flatten()
    cm = confusion_matrix(labels_np, preds)
    print("\nConfusion Matrix (Validation Set):")
    print(f"Labels: {list(class_to_idx.keys())} (0: no, 1: yes)")
    print(cm)
    
    # Classification Report
    print("\nClassification Report (Validation Set):")
    print(classification_report(labels_np, preds, target_names=list(class_to_idx.keys()), zero_division=0))
    
    # Show false negatives if requested
    if show_false_negatives:
        find_and_show_false_negatives(model, val_loader, device, class_to_idx, num_to_show)
    
    return acc, prec, rec, f1