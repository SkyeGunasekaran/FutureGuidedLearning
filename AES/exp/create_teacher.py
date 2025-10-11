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
        return 'Dog', 'teacher_dog.pth'
    # For Patient_1 and Patient_2, the mode is the same as the target
    return target, f'{target}.pth'

def _load_settings_and_create_dirs():
    """Loads settings from JSON files and creates necessary directories."""
    with open('teacher_settings.json', 'r') as f:
        teacher_settings = json.load(f)
    with open('student_settings.json', 'r') as k:
        student_settings = json.load(k)
    
    makedirs(str(teacher_settings['cachedir']))
    makedirs(str(student_settings['cachedir']))
    
    return teacher_settings

def _prepare_data(mode, settings, batch_size, device):
    """Prepares and loads data into a DataLoader."""
    ictal_X, ictal_y, interictal_X, interictal_y = make_teacher(mode=mode, teacher_settings=settings)
    X_train, y_train = train_val_test_split_continual_t(
        ictal_X, ictal_y, interictal_X, interictal_y, 0.0, no_test=True
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, X_train_tensor.shape

def _build_model_and_optimizer(input_shape, device):
    """Initializes the model, loss function, and optimizer."""
    model = CNN_LSTM_Model(input_shape).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9)
    return model, criterion, optimizer

def _train_epoch(model, data_loader, criterion, optimizer, device):
    """Runs a single training epoch."""
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

def train_teacher_model(target, epochs):
    """
    Trains a seizure detection teacher model for a given subject.

    Args:
        target (str): Subject identifier ('Dog', 'Patient_1', or 'Patient_2').
        epochs (int): Number of training epochs.

    Returns:
        None: The trained model is saved to disk.
    """
    print(f'\nTraining Teacher Model: Target {target} | Epochs: {epochs}')
    
    # Setup device, configuration, and data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.cuda.empty_cache()

    mode, save_path = _get_config(target)
    teacher_settings = _load_settings_and_create_dirs()
    
    # Hyperparameters
    BATCH_SIZE = 32

    # Data Loading
    train_loader, input_shape = _prepare_data(mode, teacher_settings, BATCH_SIZE, device)
    
    # Model Initialization
    teacher, criterion, optimizer = _build_model_and_optimizer(input_shape, device)

    # Training Loop
    with tqdm(total=epochs, desc=f"Training Teacher Model for {target}") as pbar:
        for epoch in range(epochs):
            avg_loss = _train_epoch(teacher, train_loader, criterion, optimizer, device)
            pbar.set_postfix(loss=f'{avg_loss:.4f}')
            pbar.update(1)

    # Save the trained model
    torch.save(teacher, save_path)
    print(f"Model saved to {save_path}")


if __name__ == '__main__':
    """
    Main execution loop that takes command-line arguments for:
    - Subject ('Dog', 'Patient_1', or 'Patient_2')
    - Number of epochs
    """
    parser = argparse.ArgumentParser(description="Seizure Detection Teacher Model Training")
    parser.add_argument("--subject", type=str, choices=['Dog', 'Patient_1', 'Patient_2'], required=True,
                        help="Target subject: 'Dog', 'Patient_1', or 'Patient_2'")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")

    args = parser.parse_args()
    train_teacher_model(args.subject, args.epochs)