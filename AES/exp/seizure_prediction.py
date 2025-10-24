import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

from utils.load_signals_student import PrepDataStudent
# Make sure to import the correct, updated split function
from utils.prep_data_student import train_val_test_split_continual_s
from models.models import CNN_LSTM_Model, MViT

# --- Helper Functions ---

def makedirs(dir_path):
    """Creates a directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        pass

def find_best_threshold(y_true, y_pred):
    """Determines the optimal classification threshold using the Youden index."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    if len(thresholds) == 0:
        return 0.5 # Default threshold
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    return thresholds[optimal_idx]

def _prepare_data(target, device, val_ratio, test_ratio=0.35, batch_size=32):
    """Loads, splits, and prepares data into DataLoaders and test tensors."""
    with open('student_settings.json', 'r') as k:
        settings = json.load(k)
    makedirs(str(settings['cachedir']))
    
    ictal_X, ictal_y = PrepDataStudent(target, 'ictal', settings).apply()
    interictal_X, interictal_y = PrepDataStudent(target, 'interictal', settings).apply()

    # Use the updated split function to get train, val, and test sets
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split_continual_s(
        ictal_X, ictal_y, interictal_X, interictal_y, 
        test_ratio=test_ratio, 
        val_ratio=val_ratio
    )

    # Create Train DataLoader
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    # Shuffle training data for better generalization
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    
    # Create Validation DataLoader
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    # No need to shuffle validation data
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create Test Tensors (no loader needed for final evaluation)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    
    return train_loader, val_loader, X_test_tensor, y_test_tensor, X_train_tensor.shape

def _build_model(model_type, input_shape, device):
    """Builds and returns the specified model."""
    if model_type == 'MViT':
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
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

def _build_scheduler(optimizer):
    """Builds a learning rate scheduler."""
    # Reduces LR when validation loss plateaus
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
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
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
                    val_outputs = model(X_val_batch)
                    val_loss = criterion(val_outputs, Y_val_batch)
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
                # Save the best model
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
    
    threshold = find_best_threshold(y_true, y_probs)
    y_pred_binary = (y_probs >= threshold).astype(int)
    
    # Handle cases where a class is not present in the test set
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    except ValueError:
        # This can happen if y_true or y_pred_binary has only one class
        if np.all(y_true == 0):
            tn = len(y_true)
            fp = 0
            fn = 0
            tp = 0
        elif np.all(y_true == 1):
            tn = 0
            fp = 0
            fn = 0
            tp = len(y_true)
        else:
            # Fallback if confusion matrix fails unexpectedly
            tn, fp, fn, tp = 0, 0, 0, 0 
            
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    auc_roc = roc_auc_score(y_true, y_probs)
    
    return [fpr, sensitivity, auc_roc]

# --- Main Training Function ---

def train_student_model(target, epochs, trials, model_type, optimizer_type, 
                        val_ratio, clip_value, patience):
    """
    Trains and evaluates a seizure prediction student model.
    """
    print(f'\nTraining Student Model: Target {target} | Model: {model_type} | Epochs: {epochs} | Trials: {trials} | Optimizer: {optimizer_type}')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # Pass val_ratio to data preparation
    train_loader, val_loader, X_test, y_test, input_shape = _prepare_data(
        target, device, val_ratio
    )
    
    student_results = []

    for trial in range(trials):
        print(f'\nPatient {target} | Trial {trial + 1}/{trials}')
        
        # Define a temporary path to save the best model for this trial
        model_save_path = f'./best_model_{target}_trial_{trial+1}.pth'
        
        student = _build_model(model_type, input_shape, device)
        optimizer = _build_optimizer(student, optimizer_type)
        scheduler = _build_scheduler(optimizer) # Create scheduler
        
        trial_desc = f"Training {model_type} for {target}, Trial {trial + 1}"
        
        # Run the updated training loop
        _run_training_loop(
            student, train_loader, val_loader, epochs, optimizer, scheduler,
            clip_value, patience, model_save_path, device, trial_desc
        )
        
        # --- Load Best Model and Evaluate ---
        # Load the best weights saved during training
        try:
            student.load_state_dict(torch.load(model_save_path))
            
            metrics = _evaluate_model(student, X_test, y_test)
            fpr, sensitivity, auc_roc = metrics
            print(f'Patient {target} | Trial {trial + 1} Best Model Metrics:')
            print(f'  FPR: {fpr:.4f} | Sensitivity: {sensitivity:.4f} | AUCROC: {auc_roc:.4f}')
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
    parser = argparse.ArgumentParser(description="Seizure Prediction Student Model Training")
    parser.add_argument("--subject", type=str, required=True, help="Target subject ID (use 'all' for default list)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--model", type=str, choices=['CNN_LSTM', 'MViT'], default='MViT', help="Model architecture")
    parser.add_argument("--optimizer", type=str, choices=['SGD', 'Adam'], default='Adam', help="Optimizer type")
    parser.add_argument("--trials", type=int, default=3, help="Number of training trials")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Percentage of data to use for validation (e.g., 0.2 for 20%)")
    parser.add_argument("--patience", type=int, default=5, help="Epochs to wait for val_loss improvement before early stopping")
    parser.add_argument("--clip_value", type=float, default=1.0, help="Maximum norm for gradient clipping")
    
    args = parser.parse_args()

    default_subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    subjects_to_train = default_subjects if args.subject == 'all' else [args.subject]

    all_results = {}
    for subject in subjects_to_train:
        all_results[subject] = train_student_model(
            subject, args.epochs, args.trials, args.model, args.optimizer,
            args.val_ratio, args.clip_value, args.patience # Pass new args
        )

    # Save results to a structured JSON file for easier parsing
    try:
        with open("Prediction_results.json", 'r') as f:
            existing_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_results = {}
    
    existing_results.update(all_results)

    with open("Prediction_results.json", 'w') as f:
        json.dump(existing_results, f, indent=4)
    
    print("\nTraining complete. Results saved to Prediction_results.json")

if __name__ == '__main__':
    main()