import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from utils.model_stuff import EarlyStopping
from utils.load_signals_teacher import PrepDataTeacher
from utils.prep_data_teacher import train_val_test_split_continual_t
from models.models import CNN_LSTM_Model

# --- Helper Functions ---

def _prepare_data(target, device):
    """Loads, splits, and prepares data into a DataLoader and test tensors."""
    with open('teacher_settings.json', 'r') as k:
        settings = json.load(k)
    
    ictal_X, ictal_y = PrepDataTeacher(target, 'ictal', settings).apply()
    interictal_X, interictal_y = PrepDataTeacher(target, 'interictal', settings).apply()

    X_train, y_train, X_test, y_test = train_val_test_split_continual_t(
        ictal_X, ictal_y, interictal_X, interictal_y, 0.35
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    
    return train_loader, X_test_tensor, y_test_tensor, X_train_tensor.shape

def _build_model_and_optimizer(input_shape, optimizer_type, device):
    """Initializes the model and its optimizer."""
    model = CNN_LSTM_Model(input_shape).to(device)
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-8)
    else:  # Default to SGD
        optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9)
    return model, optimizer

def _train_epoch(model, data_loader, criterion, optimizer, device):
    """Runs a single training epoch and returns the average loss."""
    model.train()
    total_loss = 0
    for X_batch, Y_batch in data_loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, Y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def _run_training_loop(model, loader, optimizer, epochs, patience, device, pbar):
    """Executes the training process with early stopping."""
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=patience, mode='min')
    losses = []

    for epoch in range(epochs):
        avg_loss = _train_epoch(model, loader, criterion, optimizer, device)
        losses.append(avg_loss)
        pbar.update(1)
        
        early_stopping.step(avg_loss, epoch)
        if early_stopping.is_stopped():
            print(f"\nEarly stopping at epoch {epoch+1} with best loss {early_stopping.best_loss:.4f}")
            break
    return losses

def _evaluate_model(model, X_test, y_test):
    """Evaluates the trained model and returns the AUC score."""
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
    
    y_probs = F.softmax(predictions, dim=1)[:, 1].cpu().numpy()
    y_true = y_test.cpu().numpy()
    return roc_auc_score(y_true, y_probs)

def _save_and_plot(model, target, losses):
    """Saves the model and plots the training loss."""
    model_path = f'pytorch_models/Patient_{target}_detection'
    torch.save(model, model_path)
    
    plt.figure()
    plt.plot(losses, label=f'Patient {target} Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss for Patient {target}')
    # plt.show() # Uncomment to display plots during execution

# --- Main Training Function ---

def train_teacher_model(target, epochs, optimizer_type, patience):
    """
    Trains and evaluates a seizure detection model for a given patient.
    (Function signature remains unchanged)
    """
    print(f'\nTraining Teacher Model: Patient {target} | Epochs: {epochs} | Optimizer: {optimizer_type} | Patience: {patience}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    loader, X_test, y_test, shape = _prepare_data(target, device)
    teacher, optimizer = _build_model_and_optimizer(shape, optimizer_type, device)

    with tqdm(total=epochs, desc=f"Training Teacher for Patient {target}") as pbar:
        losses = _run_training_loop(teacher, loader, optimizer, epochs, patience, device, pbar)
        
    auc_test = _evaluate_model(teacher, X_test, y_test)
    print(f'Patient {target} - Test AUC: {auc_test:.4f}')

    _save_and_plot(teacher, target, losses)
    
    return auc_test

# --- Execution Block ---

def main():
    """Main execution function to parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Seizure Detection Model Training")
    parser.add_argument("--patient", type=str, required=True, help="Patient ID (or 'all')")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs")
    parser.add_argument("--optimizer", type=str, choices=['SGD', 'Adam'], default='SGD', help="Optimizer")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    args = parser.parse_args()

    default_patients = ['1', '2', '3', '5', '9', '10', '13', '18', '19', '20', '21', '23']
    patients_to_run = default_patients if args.patient == 'all' else [args.patient]
    
    results = {}
    for patient in patients_to_run:
        results[patient] = train_teacher_model(patient, args.epochs, args.optimizer, args.patience)

    # Save results to a structured JSON file
    try:
        with open("Detection_results.json", 'r') as f:
            existing_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_results = {}
    
    existing_results.update(results)
    with open("Detection_results.json", 'w') as f:
        json.dump(existing_results, f, indent=4)
        
    print("\nTraining complete. Results saved to Detection_results.json")

if __name__ == "__main__":
    main()