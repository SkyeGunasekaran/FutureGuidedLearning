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

def _prepare_data(target, device):
    """Loads, splits, and prepares data into a DataLoader and test tensors."""
    with open('student_settings.json', 'r') as k:
        settings = json.load(k)
    makedirs(str(settings['cachedir']))
    
    ictal_X, ictal_y = PrepDataStudent(target, 'ictal', settings).apply()
    interictal_X, interictal_y = PrepDataStudent(target, 'interictal', settings).apply()

    X_train, y_train, X_test, y_test = train_val_test_split_continual_s(
        ictal_X, ictal_y, interictal_X, interictal_y, 0.35
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    
    return train_loader, X_test_tensor, y_test_tensor, X_train_tensor.shape

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

def _build_optimizer(model, optimizer_type):
    """Builds and returns the specified optimizer."""
    if optimizer_type == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-8)
    return torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9)

def _run_training_loop(model, train_loader, epochs, optimizer, device, trial_desc):
    """Executes the training process for a single trial."""
    criterion = nn.CrossEntropyLoss()
    with tqdm(total=epochs, desc=trial_desc) as pbar:
        for epoch in range(epochs):
            model.train()
            for X_batch, Y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                loss.backward()
                optimizer.step()
            pbar.update(1)

def _evaluate_model(model, X_test, y_test):
    """Evaluates the trained model and computes performance metrics."""
    model.eval()
    with torch.no_grad():
        predictions_raw = model(X_test)
    
    y_true = y_test.cpu().numpy()
    y_probs = F.softmax(predictions_raw, dim=1)[:, 1].cpu().numpy()
    
    threshold = find_best_threshold(y_true, y_probs)
    y_pred_binary = (y_probs >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    auc_roc = roc_auc_score(y_true, y_probs)
    
    return [fpr, sensitivity, auc_roc]

# --- Main Training Function ---

def train_student_model(target, epochs, trials, model, optimizer_type):
    """
    Trains and evaluates a seizure prediction student model.
    (Function signature remains unchanged)
    """
    print(f'\nTraining Student Model: Target {target} | Model: {model} | Epochs: {epochs} | Trials: {trials} | Optimizer: {optimizer_type}')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    train_loader, X_test, y_test, input_shape = _prepare_data(target, device)
    student_results = []

    for trial in range(trials):
        print(f'\nPatient {target} | Trial {trial + 1}/{trials}')
        student = _build_model(model, input_shape, device)
        optimizer = _build_optimizer(student, optimizer_type)
        
        trial_desc = f"Training {model} for {target}, Trial {trial + 1}"
        _run_training_loop(student, train_loader, epochs, optimizer, device, trial_desc)
        
        metrics = _evaluate_model(student, X_test, y_test)
        fpr, sensitivity, auc_roc = metrics
        print(f'Patient {target} | FPR: {fpr:.4f} | Sensitivity: {sensitivity:.4f} | AUCROC: {auc_roc:.4f}')
        student_results.append(metrics)
        
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
    args = parser.parse_args()

    default_subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    subjects_to_train = default_subjects if args.subject == 'all' else [args.subject]

    all_results = {}
    for subject in subjects_to_train:
        all_results[subject] = train_student_model(
            subject, args.epochs, args.trials, args.model, args.optimizer
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