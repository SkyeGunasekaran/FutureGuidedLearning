import argparse
import json
import os  # Added for file operations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import numpy as np

from utils.load_signals_student import PrepDataStudent
from utils.prep_data_student import train_val_test_split_continual_s
from models.models import CNN_LSTM_Model, MViT

# --- Helper Functions ---

def find_best_threshold(y_true, y_pred):
    """
    Determines the optimal classification threshold using the Youden index.
    (Function signature remains unchanged)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    if len(thresholds) == 0:
        return 0.5 # Return a default threshold if roc_curve is degenerate
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    return thresholds[optimal_idx]

def _prepare_data(target, device, val_ratio, test_ratio=0.35, batch_size=32):
    """Loads, splits, and prepares data into DataLoaders and test tensors."""
    with open('student_settings.json', 'r') as k:
        settings = json.load(k)
    
    ictal_X, ictal_y = PrepDataStudent(target, 'ictal', settings).apply()
    interictal_X, interictal_y = PrepDataStudent(target, 'interictal', settings).apply()

    # Use the updated split function to get train, val, and test sets
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split_continual_s(
        ictal_X, ictal_y, interictal_X, interictal_y, 
        test_ratio=test_ratio, 
        val_ratio=val_ratio
    )

    # Create DataLoader for training
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    # Shuffle training data for better generalization
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create DataLoader for validation
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create tensors for testing
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    
    return train_loader, val_loader, X_test_tensor, y_test_tensor, X_train_tensor.shape

def _build_model(model_type, input_shape, device):
    """Builds and returns the specified model."""
    if model_type == 'MViT':
        # MViT specific hyperparameters
        config = {
            "patch_size": (5, 10), "embed_dim": 128, "num_heads": 4,
            "hidden_dim": 256, "num_layers": 4, "dropout": 0.1
        }
        model = MViT(X_shape=input_shape, in_channels=input_shape[2], num_classes=2, **config).to(device)
    elif model_type == 'CNN_LSTM':
        model = CNN_LSTM_Model(input_shape).to(device)
    else:
        raise ValueError("Invalid model type specified.")
    return model

def _build_optimizer(model, optimizer_type, lr=5e-4):
    """Builds and returns the specified optimizer."""
    if optimizer_type == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    # Default to SGD
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

def _build_scheduler(optimizer):
    """Builds a learning rate scheduler."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.1, patience=3, verbose=True
    )

def _run_training_loop(model, train_loader, val_loader, epochs, optimizer, scheduler,
                        clip_value, patience, model_save_path, device, trial_desc):
    """Executes the training and validation process for a single trial."""
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    with tqdm(total=epochs, desc=trial_desc) as pbar:
        for epoch in range(epochs):
            # --- Training Phase ---
            model.train()
            running_train_loss = 0.0
            for X_batch, Y_batch in train_loader:
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, Y_batch)
                loss.backward()
                
                # --- Gradient Clipping ---
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                
                optimizer.step()
                running_train_loss += loss.item()
            
            avg_train_loss = running_train_loss / len(train_loader)
            
            # --- Validation Phase ---
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for X_val_batch, Y_val_batch in val_loader:
                    val_logits = model(X_val_batch)
                    val_loss = criterion(val_logits, Y_val_batch)
                    running_val_loss += val_loss.item()
            
            avg_val_loss = running_val_loss / len(val_loader)
            
            # Update progress bar
            pbar.set_postfix({'train_loss': f'{avg_train_loss:.4f}', 
                             'val_loss': f'{avg_val_loss:.4f}'})
            pbar.update(1)

            # --- Scheduler Step ---
            scheduler.step(avg_val_loss)
            
            # --- Early Stopping Check ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), model_save_path)
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs.')
                break

def _evaluate_model(model, X_test, y_test):
    """Evaluates the trained model and computes performance metrics."""
    model.eval()
    with torch.no_grad():
        predictions_raw = model(X_test)
    
    y_true = y_test.cpu().numpy()
    y_probs = F.softmax(predictions_raw, dim=1)[:, 1].cpu().numpy()
    
    # Calculate metrics
    threshold = find_best_threshold(y_true, y_probs)
    y_pred_binary = (y_probs >= threshold).astype(int)
    
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    except ValueError:
        # Handle cases where a class is not present
        if np.all(y_true == 0):
            tn, fp, fn, tp = len(y_true), 0, 0, 0
        else: # Assumes y_true has 1s or predictions are all 1s
            tn, fp, fn, tp = 0, 0, 0, len(y_true)
            
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    try:
        auc_roc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc_roc = 0.5 # Default if y_true is single-class
    
    return [fpr, sensitivity, auc_roc]

# --- Main Training Function ---

def train_and_evaluate(target, trials, model_type, epochs, 
                       val_ratio, patience, clip_value, optimizer_type):
    """
    Trains and evaluates a seizure prediction model for a given patient.
    """
    print(f'Training Model: {model_type} | Patient: {target} | Trials: {trials}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # Prepare data once for all trials
    train_loader, val_loader, X_test, y_test, input_shape = _prepare_data(
        target, device, val_ratio
    )
    
    student_results = []
    for trial in range(trials):
        print(f'\nStarting Trial {trial + 1}/{trials} for Patient {target}...')
        
        # Define a temporary path to save the best model for this trial
        model_save_path = f'./best_model_{target}_trial_{trial+1}.pth'
        
        student = _build_model(model_type, input_shape, device)
        optimizer = _build_optimizer(student, optimizer_type)
        scheduler = _build_scheduler(optimizer)
        
        trial_desc = f"Training {model_type} for Patient {target}, Trial {trial + 1}"
        _run_training_loop(
            student, train_loader, val_loader, epochs, optimizer, scheduler,
            clip_value, patience, model_save_path, device, trial_desc
        )
        
        # --- Load Best Model and Evaluate ---
        try:
            student.load_state_dict(torch.load(model_save_path))
            
            metrics = _evaluate_model(student, X_test, y_test)
            fpr, sensitivity, auc_roc = metrics
            
            print(f'Patient {target}, Trial {trial + 1} Best Model Metrics:')
            print(f'  False Positive Rate (FPR): {fpr:.4f}')
            print(f'  Sensitivity: {sensitivity:.4f}')
            print(f'  AUCROC: {auc_roc:.4f}')
            student_results.append(metrics)
            
        except FileNotFoundError:
            print(f"Warning: Model file not found for trial {trial + 1}. Skipping evaluation.")
            student_results.append([np.nan, np.nan, np.nan]) # Append NaNs
        
        # Clean up the temporary model file
        if os.path.exists(model_save_path):
            os.remove(model_save_path)
            
    return student_results

# --- Execution Block ---

def main():
    """Main execution function to parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Seizure Prediction Model Training")
    parser.add_argument("--patient", type=str, required=True, help="Target patient ID")
    parser.add_argument("--trials", type=int, default=3, help="Number of training trials")
    parser.add_argument("--model", type=str, choices=['CNN_LSTM', 'MViT'], default='CNN_LSTM', help="Model architecture")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--optimizer", type=str, choices=['SGD', 'Adam'], default='Adam', help="Optimizer type")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Percentage of data to use for validation (e.g., 0.2 for 20%)")
    parser.add_argument("--patience", type=int, default=5, help="Epochs to wait for val_loss improvement before early stopping")
    parser.add_argument("--clip_value", type=float, default=1.0, help="Maximum norm for gradient clipping")
    
    args = parser.parse_args()

    results = train_and_evaluate(
        args.patient, args.trials, args.model, args.epochs,
        args.val_ratio, args.patience, args.clip_value, args.optimizer
    )

    # Save results to a structured JSON file for easier parsing
    try:
        with open("Prediction_results.json", 'r') as f:
            existing_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_results = {}
        
    existing_results[args.patient] = results
    
    with open("Prediction_results.json", 'w') as f:
        json.dump(existing_results, f, indent=4)
        
    print(f"\nTraining complete. Results for Patient {args.patient} saved to Prediction_results.json")

if __name__ == "__main__":
    main()