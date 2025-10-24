import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import os # Import os for file operations

from models.models import MViT, CNN_LSTM_Model
from utils.load_signals_teacher import PrepDataTeacher, makedirs
# Import the updated split function
from utils.prep_data_teacher import train_val_test_split_continual_t

# --- Helper Functions ---

def _prepare_data(target, freq, teacher_channels, device, val_ratio, batch_size=32):
    """Loads, processes, and prepares the data into DataLoaders."""
    with open('teacher_settings.json', 'r') as f:
        settings = json.load(f)
    makedirs(str(settings['cachedir']))
    
    # Load ictal and interictal data
    ictal_X, ictal_y = PrepDataTeacher(target, 'ictal', settings, freq, teacher_channels).apply()
    interictal_X, interictal_y = PrepDataTeacher(target, 'interictal', settings, freq, teacher_channels).apply()

    # Create training and validation sets
    # We set test_ratio=0 and no_test=True, and use val_ratio
    X_train, y_train, X_val, y_val, _, _ = train_val_test_split_continual_t(
        ictal_X, ictal_y, interictal_X, interictal_y, 
        test_ratio=0, 
        val_ratio=val_ratio, 
        no_test=True
    )

    # Create Train DataLoader
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    # Shuffle training data
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create Validation DataLoader
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    # No shuffle needed for validation
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, X_train_tensor.shape

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

def _build_optimizer(model, optimizer_type, lr=5e-4):
    """Builds and returns the specified optimizer."""
    if optimizer_type == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    # Default to SGD
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

def _build_scheduler(optimizer):
    """Builds a learning rate scheduler."""
    # Reduces LR when validation loss plateaus
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.1, patience=3, verbose=True
    )

def _train_epoch(model, data_loader, criterion, optimizer, device, clip_value):
    """Runs a single training epoch and returns the average loss."""
    model.train()
    total_loss = 0
    for X_batch, Y_batch in data_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        loss.backward()
        
        # --- Gradient Clipping ---
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(data_loader)

def _validate_epoch(model, data_loader, criterion, device):
    """Runs a single validation epoch and returns the average loss."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, Y_batch in data_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            total_loss += loss.item()
    return total_loss / len(data_loader)

# --- Main Training Function ---

def train_teacher_model(target, epochs, optimizer_type, freq, teacher_channels, 
                        model_type, val_ratio, patience, clip_value):
    """
    Trains a model for seizure detection with validation and early stopping.
    """
    print(f'\nTraining Detector: Target {target} | Model: {model_type} | Epochs: {epochs} | Optimizer: {optimizer_type}')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # Prepare data, model, and optimizer
    train_loader, val_loader, input_shape = _prepare_data(
        target, freq, teacher_channels, device, val_ratio
    )
    
    teacher = _build_model(model_type, input_shape, device)
    optimizer = _build_optimizer(teacher, optimizer_type)
    scheduler = _build_scheduler(optimizer) # Create scheduler
    criterion = nn.CrossEntropyLoss()
    
    # Path to save the best model
    model_save_dir = "./saved_teacher_models"
    makedirs(model_save_dir)
    model_save_path = os.path.join(model_save_dir, f'best_teacher_{target}_{model_type}.pth')

    # Training loop state
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    with tqdm(total=epochs, desc=f"Training {model_type} for {target}") as pbar:
        for epoch in range(epochs):
            # --- Training ---
            avg_train_loss = _train_epoch(
                teacher, train_loader, criterion, optimizer, device, clip_value
            )
            train_losses.append(avg_train_loss)
            
            # --- Validation ---
            avg_val_loss = _validate_epoch(
                teacher, val_loader, criterion, device
            )
            val_losses.append(avg_val_loss)
            
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
                torch.save(teacher.state_dict(), model_save_path)
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs.')
                break
                
    print(f"Best model for {target} saved to {model_save_path}")
    
    # Return history of losses
    return {"train_loss": train_losses, "val_loss": val_losses}

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
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Percentage of data to use for validation (e.g., 0.2 for 20%)")
    parser.add_argument("--patience", type=int, default=5, help="Epochs to wait for val_loss improvement before early stopping")
    parser.add_argument("--clip_value", type=float, default=1.0, help="Maximum norm for gradient clipping")
    
    args = parser.parse_args()

    default_subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Patient_3', 'Patient_5', 'Patient_6', 'Patient_7']
    subjects_to_train = default_subjects if args.subject == 'all' else [args.subject]

    all_results = {}
    for subject in subjects_to_train:
        all_results[subject] = train_teacher_model(
            subject, args.epochs, args.optimizer, args.freq, args.channels, 
            args.model, args.val_ratio, args.patience, args.clip_value # Pass new args
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