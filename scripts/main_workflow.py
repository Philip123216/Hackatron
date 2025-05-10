# -*- coding: utf-8 -*-
"""
Main workflow script for the Hackatron project.
This script orchestrates the entire pipeline from data preparation to model evaluation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from pathlib import Path
import glob
import optuna

# Import modules from src
from src.data.preparation import split_data, create_dataloaders
from src.training.model_arch import CustomCNN
from src.training.engine import train_model, train_trial, train_final_model
from src.evaluation.metrics import evaluate_model

# Default paths and parameters
ANNOTATED_DIR = Path("./data_annotated")
BASE_DATA_DIR = Path("./data")
TRAIN_DIR = BASE_DATA_DIR / "train"
VAL_DIR = BASE_DATA_DIR / "validation"
MODEL_SAVE_PATH = Path("./models/best_custom_cnn_model.pth")
TUNED_MODEL_SAVE_PATH = Path("./models/best_tuned_cnn_model_v2.pth")

# Training parameters
BATCH_SIZE = 128
NUM_WORKERS = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
TRAIN_VAL_SPLIT = 0.8
RANDOM_SEED = 42
IMG_SIZE = 250
N_TRIALS = 100
EPOCHS_PER_TRIAL = 15
FINAL_TRAINING_EPOCHS = 30

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Hackatron ML Pipeline")
    parser.add_argument("--mode", type=str, choices=["train", "tune", "evaluate"], 
                        default="train", help="Pipeline mode")
    parser.add_argument("--data_dir", type=str, default=str(ANNOTATED_DIR),
                        help="Directory with annotated data")
    parser.add_argument("--model_path", type=str, default=str(MODEL_SAVE_PATH),
                        help="Path to save/load model")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Batch size for training")
    parser.add_argument("--trials", type=int, default=N_TRIALS,
                        help="Number of Optuna trials for hyperparameter tuning")
    parser.add_argument("--no_split", action="store_true",
                        help="Skip data splitting (use existing split)")
    return parser.parse_args()

def setup_environment(seed=RANDOM_SEED):
    """Set up the environment for reproducibility."""
    import random
    import numpy as np
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def calculate_pos_weight(train_dir, device):
    """Calculate positive weight for weighted loss."""
    print("\nCalculating weight for positive class ('yes')...")
    try:
        num_no = len(glob.glob(str(train_dir / 'no' / '*.png')))
        num_yes = len(glob.glob(str(train_dir / 'yes' / '*.png')))
        
        if num_yes == 0:
            print("  Warning: No 'yes' samples found. Using default weight (1).")
            pos_weight_value = 1.0
        else:
            pos_weight_value = num_no / num_yes
            print(f"  Found {num_no} 'no' samples and {num_yes} 'yes' samples in training set.")
            print(f"  Calculated positive weight: {pos_weight_value:.2f}")
            
        pos_weight_tensor = torch.tensor([pos_weight_value], device=device)
    except Exception as e:
        print(f"  Error calculating weights: {e}. Using default weight (1).")
        pos_weight_tensor = torch.tensor([1.0], device=device)
        
    return pos_weight_tensor

def train_pipeline(args, device):
    """Run the training pipeline."""
    # 1. Split data if needed
    if not args.no_split and (not BASE_DATA_DIR.exists() or not (TRAIN_DIR / 'yes').exists()):
        print("Data split directory not found or incomplete, splitting data...")
        split_data(Path(args.data_dir), TRAIN_DIR, VAL_DIR, TRAIN_VAL_SPLIT, RANDOM_SEED)
    else:
        print("Using existing data split.")
        
    # 2. Create dataloaders
    train_loader, val_loader, class_to_idx = create_dataloaders(
        TRAIN_DIR, VAL_DIR, args.batch_size, NUM_WORKERS, IMG_SIZE, device.type
    )
    
    # 3. Calculate positive weight for loss function
    pos_weight_tensor = calculate_pos_weight(TRAIN_DIR, device)
    
    # 4. Initialize model
    model = CustomCNN(num_classes=1).to(device)
    
    # 5. Set up loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 6. Train model
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting training for {args.epochs} epochs...")
    trained_model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        args.epochs, device, model_path
    )
    
    # 7. Evaluate final model
    print("\nEvaluating final model...")
    evaluate_model(trained_model, val_loader, device, class_to_idx)
    
    return trained_model

def tune_pipeline(args, device):
    """Run the hyperparameter tuning pipeline."""
    # 1. Split data if needed
    if not args.no_split and (not BASE_DATA_DIR.exists() or not (TRAIN_DIR / 'yes').exists()):
        print("Data split directory not found or incomplete, splitting data...")
        split_data(Path(args.data_dir), TRAIN_DIR, VAL_DIR, TRAIN_VAL_SPLIT, RANDOM_SEED)
    else:
        print("Using existing data split.")
        
    # 2. Create dataloaders
    train_loader, val_loader, class_to_idx = create_dataloaders(
        TRAIN_DIR, VAL_DIR, args.batch_size, NUM_WORKERS, IMG_SIZE, device.type
    )
    
    # 3. Calculate positive weight for loss function
    pos_weight_tensor = calculate_pos_weight(TRAIN_DIR, device)
    
    # 4. Define Optuna objective function
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        dropout_rate = trial.suggest_float("dropout", 0.1, 0.6)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        
        model = CustomCNN(num_classes=1, dropout_rate=dropout_rate).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        print(f"\n--- Starting Trial {trial.number} ---")
        print(f"  Params: lr={lr:.6f}, dropout={dropout_rate:.3f}, weight_decay={weight_decay:.6f}")
        
        try:
            best_trial_f1 = train_trial(
                trial, model, train_loader, val_loader, criterion, 
                optimizer, EPOCHS_PER_TRIAL, device
            )
            return best_trial_f1
        except optuna.exceptions.TrialPruned:
            print(f"  Trial {trial.number} was pruned.")
            return 0.0
        except Exception as e:
            print(f"!! Trial {trial.number} failed with error: {e}")
            return 0.0
    
    # 5. Run Optuna study
    print(f"\n--- Starting Optuna Hyperparameter Search ({args.trials} Trials) ---")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=args.trials)
    
    # 6. Train final model with best hyperparameters
    print("\n--- Optuna Search Finished ---")
    print(f"Finished trials: {len(study.trials)}")
    
    if study.best_trial is None or study.best_value <= 0.0:
        print("\nERROR: Optuna search did not find any valid trials. Cannot proceed.")
        return None
        
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value (Best Val F1): {best_trial.value:.4f}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # 7. Train final model with best hyperparameters
    print("\n--- Starting Final Training with Best Hyperparameters ---")
    best_params = best_trial.params
    final_model = CustomCNN(num_classes=1, dropout_rate=best_params['dropout']).to(device)
    final_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    final_optimizer = optim.AdamW(
        final_model.parameters(), 
        lr=best_params['lr'], 
        weight_decay=best_params['weight_decay']
    )
    
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Training final model for {FINAL_TRAINING_EPOCHS} epochs...")
    trained_final_model = train_final_model(
        final_model, train_loader, val_loader, final_criterion, 
        final_optimizer, FINAL_TRAINING_EPOCHS, device, model_path
    )
    
    # 8. Evaluate final model
    print("\nEvaluating final tuned model...")
    evaluate_model(trained_final_model, val_loader, device, class_to_idx)
    
    return trained_final_model

def evaluate_pipeline(args, device):
    """Run the evaluation pipeline."""
    # 1. Check if validation data exists
    if not BASE_DATA_DIR.exists() or not (VAL_DIR / 'yes').exists():
        print(f"ERROR: Validation data not found under {VAL_DIR}.")
        print(f"Make sure the directory '{BASE_DATA_DIR.name}' exists and contains the split data.")
        return
    
    # 2. Create dataloaders (only need validation loader)
    _, val_loader, class_to_idx = create_dataloaders(
        TRAIN_DIR, VAL_DIR, args.batch_size, NUM_WORKERS, IMG_SIZE, device.type
    )
    
    # 3. Initialize model
    model = CustomCNN(num_classes=1).to(device)
    
    # 4. Load model weights
    model_path = Path(args.model_path)
    if model_path.exists():
        print(f"\nLoading model weights from {model_path}...")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Model weights loaded successfully.")
        except Exception as e:
            print(f"ERROR loading model file: {e}")
            return
    else:
        print(f"ERROR: Model file not found at {model_path}")
        return
    
    # 5. Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, val_loader, device, class_to_idx)

def main():
    """Main function to run the pipeline."""
    args = parse_args()
    device = setup_environment()
    
    if args.mode == "train":
        train_pipeline(args, device)
    elif args.mode == "tune":
        tune_pipeline(args, device)
    elif args.mode == "evaluate":
        evaluate_pipeline(args, device)
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()