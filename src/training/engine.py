# -*- coding: utf-8 -*-
import torch
import copy
import time
from torch.cuda.amp import autocast, GradScaler
import optuna
from src.evaluation.metrics import calculate_metrics

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_save_path):
    """
    Trains and validates a model with Automatic Mixed Precision (AMP).

    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs (int): Number of epochs to train for
        device (torch.device): Device to use for training
        model_save_path (Path): Path to save the best model

    Returns:
        tuple: (trained_model, history) - Trained model and training history
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_metric = 0.0
    history = {
        'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [], 'val_precision': [], 'val_recall': []
    }

    # Initialize AMP GradScaler
    use_amp = (device.type == 'cuda')
    scaler = GradScaler(enabled=use_amp)
    if use_amp: 
        print("Automatic Mixed Precision (AMP) enabled.")

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        all_train_labels = []
        all_train_outputs = []

        batch_start_time = time.time()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            # Forward pass with autocast
            with autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            all_train_labels.append(labels.detach())
            all_train_outputs.append(outputs.detach())

            # Progress display
            if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                batch_time = time.time() - batch_start_time
                batches_left = len(train_loader) - (i + 1)
                est_time_left_epoch = batches_left * (batch_time / (i+1)) if (i+1) > 0 else 0
                print(f"  Batch {i+1}/{len(train_loader)}, Avg Loss: {running_loss / ((i+1)*train_loader.batch_size):.4f}, "
                      f"Time/Batch: {batch_time/(i+1):.2f}s, Est. Epoch Time Left: {est_time_left_epoch:.0f}s")

        epoch_train_loss = running_loss / len(train_loader.dataset)
        all_train_outputs_cpu = torch.cat(all_train_outputs)
        all_train_labels_cpu = torch.cat(all_train_labels)
        train_acc, train_prec, train_rec, train_f1 = calculate_metrics(all_train_outputs_cpu, all_train_labels_cpu)

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        print(f"  Train Loss: {epoch_train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f}")

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        all_val_labels = []
        all_val_outputs = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

                # Forward pass with autocast
                with autocast(enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                running_val_loss += loss.item() * inputs.size(0)
                all_val_labels.append(labels)
                all_val_outputs.append(outputs)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        all_val_outputs_cpu = torch.cat(all_val_outputs)
        all_val_labels_cpu = torch.cat(all_val_labels)
        val_acc, val_prec, val_rec, val_f1 = calculate_metrics(all_val_outputs_cpu, all_val_labels_cpu)

        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_rec)
        history['val_f1'].append(val_f1)
        print(f"  Val Loss: {epoch_val_loss:.4f} Acc: {val_acc:.4f} Precision: {val_prec:.4f} Recall: {val_rec:.4f} F1: {val_f1:.4f}")

        # Save best model based on F1 score
        current_val_metric = val_f1
        if current_val_metric > best_val_metric:
            print(f"  Validation F1 improved ({best_val_metric:.4f} --> {current_val_metric:.4f}). Saving model...")
            best_val_metric = current_val_metric
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, model_save_path)

    time_elapsed = time.time() - start_time
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Validation F1-Score: {best_val_metric:4f}")
    print(f"Best model saved to: {model_save_path}")

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

def train_trial(trial, model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    Trains a model for a single Optuna trial.

    Args:
        trial: Optuna trial object
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs (int): Number of epochs to train for
        device (torch.device): Device to use for training

    Returns:
        float: Best validation F1 score achieved during training
    """
    best_val_f1 = 0.0
    use_amp = (device.type == 'cuda')
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        all_val_labels = []
        all_val_outputs = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

                with autocast(enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                running_val_loss += loss.item() * inputs.size(0)
                all_val_labels.append(labels)
                all_val_outputs.append(outputs)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_acc, val_prec, val_rec, val_f1 = calculate_metrics(torch.cat(all_val_outputs), torch.cat(all_val_labels))

        print(f"  Trial {trial.number} Epoch {epoch+1}/{num_epochs} - Val Loss: {epoch_val_loss:.4f} Val F1: {val_f1:.4f}")

        best_val_f1 = max(best_val_f1, val_f1)

        # Report to Optuna for pruning
        trial.report(val_f1, epoch)
        if trial.should_prune():
            print(f"  Trial {trial.number} pruned.")
            raise optuna.exceptions.TrialPruned()

    return best_val_f1

def train_final_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path):
    """
    Trains the final model with the best hyperparameters.

    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs (int): Number of epochs to train for
        device (torch.device): Device to use for training
        save_path (Path): Path to save the best model

    Returns:
        nn.Module: Trained model
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_f1_final = 0.0
    use_amp = (device.type == 'cuda')
    scaler = GradScaler(enabled=use_amp)

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nFinal Epoch {epoch+1}/{num_epochs}")

        # Training phase
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)

        print(f"  Train Loss: {running_loss / len(train_loader.dataset):.4f}")

        # Validation phase
        model.eval()
        all_val_labels = []
        all_val_outputs = []
        running_val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                with autocast(enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                running_val_loss += loss.item() * inputs.size(0)
                all_val_labels.append(labels)
                all_val_outputs.append(outputs)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_acc, val_prec, val_rec, val_f1 = calculate_metrics(torch.cat(all_val_outputs), torch.cat(all_val_labels))

        print(f"  Val Loss: {epoch_val_loss:.4f} Acc: {val_acc:.4f} Precision: {val_prec:.4f} Recall: {val_rec:.4f} F1: {val_f1:.4f}")

        if val_f1 > best_val_f1_final:
            print(f"  Best Val F1 improved ({best_val_f1_final:.4f} -> {val_f1:.4f}). Saving final model...")
            best_val_f1_final = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, save_path)

    time_elapsed = time.time() - start_time
    print(f"\nFinal Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Val F1: {best_val_f1_final:4f}")
    print(f"Final model saved to: {save_path}")

    model.load_state_dict(best_model_wts)
    return model
