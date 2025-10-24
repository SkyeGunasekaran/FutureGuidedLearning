import argparse
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.models import CNN_LSTM_Model
from utils.prep_data_teacher import train_val_test_split_continual_t
from utils.load_signals_teacher import make_teacher, makedirs

def _get_config(target):
    """Determines the mode and model save path based on the target."""
    if target == 'Dog':
        mode = 'Dog'
        file_name = 'teacher_dog.pth'
    else:
        # For Patient_1 and Patient_2, the mode is the same as the target
        mode = target
        file_name = f'{target}.pth'
        
    # Ensure models are saved in a specific directory
    save_dir = "./saved_teacher_models"
    makedirs(save_dir)
    return mode, os.path.join(save_dir, file_name)

def _load_settings_and_create_dirs():
    """Loads settings from JSON files and creates necessary directories."""
    with open('teacher_settings.json', 'r') as f:
        teacher_settings = json.load(f)
    with open('student_settings.json', 'r') as k:
        student_settings = json.load(k)
    
    makedirs(str(teacher_settings['cachedir']))
    makedirs(str(student_settings['cachedir']))
    
    return teacher_settings

def _prepare_data(mode, settings, batch_size, device, val_ratio):
    """Prepares and loads data into a DataLoader."""
    ictal_X, ictal_y, interictal_X, interictal_y = make_teacher(mode=mode, teacher_settings=settings)
    
    # Split into train and validation sets
    X_train, y_train, X_val, y_val, _, _ = train_val_test_split_continual_t(
        ictal_X, ictal_y, interictal_X, interictal_y, 
        test_ratio=0.0, 
        val_ratio=val_ratio, 
        no_test=True
    )

    # Create Train DataLoader
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create Validation DataLoader
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, X_train_tensor.shape

def _build_model_and_optimizer(input_shape, device, optimizer_type, lr=5e-4):
    """Initializes the model, loss function, and optimizer."""
    model = CNN_LSTM_Model(input_shape).to(device)
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    else: # Default to SGD
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        
    return model, criterion, optimizer

def _build_scheduler(optimizer):
    """Builds a learning rate scheduler."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.1, patience=3, verbose=True
    )

def _train_epoch(model, data_loader, criterion, optimizer, device, clip_value):
    """Runs a single training epoch."""
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

def train_teacher_model(target, epochs, val_ratio, patience, clip_value, optimizer_type):
    """
    Trains a seizure detection teacher model for a given subject.
    """
    print(f'\nTraining Teacher Model: Target {target} | Epochs: {epochs} | Optimizer: {optimizer_type}')
    
    # Setup device, configuration, and data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.cuda.empty_cache()

    mode, save_path = _get_config(target)
    teacher_settings = _load_settings_and_create_dirs()
    
    # Hyperparameters
    BATCH_SIZE = 32

    # Data Loading
    train_loader, val_loader, input_shape = _prepare_data(
        mode, teacher_settings, BATCH_SIZE, device, val_ratio
    )
    
    # Model Initialization
    teacher, criterion, optimizer = _build_model_and_optimizer(
        input_shape, device, optimizer_type
    )
    scheduler = _build_scheduler(optimizer)
    
    # Training Loop State
    best_val_loss = float('inf')
    epochs_no_improve = 0

    with tqdm(total=epochs, desc=f"Training Teacher Model for {target}") as pbar:
        for epoch in range(epochs):
            # --- Training ---
            avg_train_loss = _train_epoch(
                teacher, train_loader, criterion, optimizer, device, clip_value
            )
            
            # --- Validation ---
            avg_val_loss = _validate_epoch(
                teacher, val_loader, criterion, device
            )
            
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
                # Save the best model (as a whole object for KD script)
                torch.save(teacher, save_path)
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs.')
                break

    print(f"Best model saved to {save_path}")


if __name__ == '__main__':
    """
    Main execution loop that handles command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Seizure Detection Teacher Model Training")
    parser.add_argument("--subject", type=str, choices=['Dog', 'Patient_1', 'Patient_2'], required=True,
                        help="Target subject: 'Dog', 'Patient_1', or 'Patient_2'")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--optimizer", type=str, choices=['SGD', 'Adam'], default='SGD', help="Optimizer type")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Percentage of data to use for validation (e.g., 0.2 for 20%)")
    parser.add_argument("--patience", type=int, default=5, help="Epochs to wait for val_loss improvement before early stopping")
    parser.add_argument("--clip_value", type=float, default=1.0, help="Maximum norm for gradient clipping")

    args = parser.parse_args()
    
    train_teacher_model(
        args.subject, args.epochs, args.val_ratio, 
        args.patience, args.clip_value, args.optimizer
    )