import argparse
import json
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

def _prepare_data(target, device):
    """Loads, splits, and prepares data into a DataLoader and test tensors."""
    with open('student_settings.json', 'r') as k:
        settings = json.load(k)
    
    ictal_X, ictal_y = PrepDataStudent(target, 'ictal', settings).apply()
    interictal_X, interictal_y = PrepDataStudent(target, 'interictal', settings).apply()

    X_train, y_train, X_test, y_test = train_val_test_split_continual_s(
        ictal_X, ictal_y, interictal_X, interictal_y, 0.35
    )

    # Create DataLoader for training
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)
    
    # Create tensors for testing
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    
    return train_loader, X_test_tensor, y_test_tensor, X_train_tensor.shape

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

def _run_training_loop(model, train_loader, epochs, device, trial_desc):
    """Executes the training process for a single trial."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-8)
    
    with tqdm(total=epochs, desc=trial_desc) as pbar:
        for epoch in range(epochs):
            model.train()
            for X_batch, Y_batch in train_loader:
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, Y_batch)
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
    
    # Calculate metrics
    threshold = find_best_threshold(y_true, y_probs)
    y_pred_binary = (y_probs >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    auc_roc = roc_auc_score(y_true, y_probs)
    
    return [fpr, sensitivity, auc_roc]

# --- Main Training Function ---

def train_and_evaluate(target, trials, model_type, epochs=25):
    """
    Trains and evaluates a seizure prediction model for a given patient.
    (Function signature remains unchanged)
    """
    print(f'Training Model: {model_type} | Patient: {target} | Trials: {trials}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # Prepare data once for all trials
    train_loader, X_test, y_test, input_shape = _prepare_data(target, device)
    
    student_results = []
    for trial in range(trials):
        print(f'\nStarting Trial {trial + 1}/{trials} for Patient {target}...')
        student = _build_model(model_type, input_shape, device)
        
        trial_desc = f"Training {model_type} for Patient {target}, Trial {trial + 1}"
        _run_training_loop(student, train_loader, epochs, device, trial_desc)
        
        metrics = _evaluate_model(student, X_test, y_test)
        fpr, sensitivity, auc_roc = metrics
        
        print(f'Patient {target}, Trial {trial + 1}:')
        print(f'  False Positive Rate (FPR): {fpr:.4f}')
        print(f'  Sensitivity: {sensitivity:.4f}')
        print(f'  AUCROC: {auc_roc:.4f}')
        student_results.append(metrics)
        
    return student_results

# --- Execution Block ---

def main():
    """Main execution function to parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Seizure Prediction Model Training")
    parser.add_argument("--patient", type=str, required=True, help="Target patient ID")
    parser.add_argument("--trials", type=int, default=3, help="Number of training trials")
    parser.add_argument("--model", type=str, choices=['CNN_LSTM', 'MViT'], default='CNN_LSTM', help="Model architecture")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    args = parser.parse_args()

    results = train_and_evaluate(args.patient, args.trials, args.model, args.epochs)

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