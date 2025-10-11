import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

from models.models import MViT, CNN_LSTM_Model
from utils.load_signals_teacher import PrepDataTeacher, makedirs
from utils.prep_data_teacher import train_val_test_split_continual_t

# --- Helper Functions ---

def _prepare_data(target, freq, teacher_channels, device):
    """Loads, processes, and prepares the data into a DataLoader."""
    with open('teacher_settings.json', 'r') as f:
        settings = json.load(f)
    makedirs(str(settings['cachedir']))
    
    # Load ictal and interictal data
    ictal_X, ictal_y = PrepDataTeacher(target, 'ictal', settings, freq, teacher_channels).apply()
    interictal_X, interictal_y = PrepDataTeacher(target, 'interictal', settings, freq, teacher_channels).apply()

    # Create training set
    X_train, y_train = train_val_test_split_continual_t(
        ictal_X, ictal_y, interictal_X, interictal_y, 0, no_test=True
    )

    # Create DataLoader
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)
    
    return train_loader, X_train_tensor.shape

def _build_model(model_type, input_shape, device):
    """Builds and returns the specified model."""
    if model_type == 'MViT':
        # MViT specific hyperparameters
        config = {
            "patch_size": (5, 10), "embed_dim": 128, "num_heads": 4,
            "hidden_dim": 256, "num_layers": 4, "dropout": 0.1
        }
        model = MViT(
            X_shape=input_shape, in_channels=input_shape[2], num_classes=2, **config
        ).to(device)
    elif model_type == 'CNN_LSTM':
        model = CNN_LSTM_Model(input_shape).to(device)
    else:
        raise ValueError("Invalid model type specified.")
        
    return model

def _build_optimizer(model, optimizer_type):
    """Builds and returns the specified optimizer."""
    if optimizer_type == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-8)
    # Default to SGD
    return torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9)

def _train_epoch(model, data_loader, criterion, optimizer, device):
    """Runs a single training epoch and returns the average loss."""
    model.train()
    total_loss = 0
    for X_batch, Y_batch in data_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(data_loader)

# --- Main Training Function ---

def train_teacher_model(target, epochs, optimizer_type, freq, teacher_channels, model):
    """
    Trains a model for seizure detection.
    (Function signature remains unchanged)
    """
    print(f'\nTraining Detector: Target {target} | Model: {model} | Epochs: {epochs} | Optimizer: {optimizer_type}')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # Prepare data, model, and optimizer
    train_loader, input_shape = _prepare_data(target, freq, teacher_channels, device)
    teacher = _build_model(model, input_shape, device)
    optimizer = _build_optimizer(teacher, optimizer_type)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    teacher_losses = []
    with tqdm(total=epochs, desc=f"Training {model} for {target}") as pbar:
        for epoch in range(epochs):
            avg_loss = _train_epoch(teacher, train_loader, criterion, optimizer, device)
            teacher_losses.append(avg_loss)
            pbar.set_postfix(loss=f'{avg_loss:.4f}')
            pbar.update(1)
            
    return teacher_losses

# --- Execution Block ---

def main():
    """Main execution function to parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Teacher Model Training for Seizure Detection")
    parser.add_argument("--subject", type=str, required=True, help="Target subject ID (use 'all' for default list)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--optimizer", type=str, choices=['SGD', 'Adam'], default='Adam', help="Optimizer type")
    parser.add_argument("--freq", type=int, default=1000, help="Sampling frequency")
    parser.add_argument("--channels", type=int, default=15, help="Number of EEG channels")
    parser.add_argument("--model", type=str, choices=['CNN_LSTM', 'MViT'], default='CNN_LSTM', help="Model architecture")
    args = parser.parse_args()

    default_subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Patient_3', 'Patient_5', 'Patient_6', 'Patient_7']
    subjects_to_train = default_subjects if args.subject == 'all' else [args.subject]

    all_results = {}
    for subject in subjects_to_train:
        all_results[subject] = train_teacher_model(
            subject, args.epochs, args.optimizer, args.freq, args.channels, args.model
        )

    # Save results to a structured JSON file
    try:
        with open("Detection_results.json", 'r') as f:
            existing_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_results = {}
    
    existing_results.update(all_results)

    with open("Detection_results.json", 'w') as f:
        json.dump(existing_results, f, indent=4)
    
    print("\nTraining complete. Results saved to Detection_results.json")

if __name__ == '__main__':
    main()