import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from models import MViT
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample
import stft

from utils.load_signals_teacher import PrepDataTeacher, makedirs
from utils.prep_data_teacher import train_val_test_split_continual_t
from models.models import CNN_LSTM_Model


def train_teacher_model(target, epochs, optimizer_type, freq, teacher_channels, model):
    """
    Trains a model for seizure detection.

    Args:
        target (str): Subject identifier (e.g., 'Dog_1', 'Patient_3').
        epochs (int): Number of training epochs.
        optimizer_type (str): Optimizer type ('SGD' or 'Adam').
        freq (int): Sampling frequency for EEG data.
        teacher_channels (int): Number of EEG channels to use.

    Returns:
        list: Training loss per epoch.
    """

    print(f'\nTraining Detector: Target {target} | Epochs: {epochs} | Optimizer: {optimizer_type}')
    torch.cuda.empty_cache()

    # Load teacher model settings
    with open('teacher_settings.json') as f:
        teacher_settings = json.load(f)

    makedirs(str(teacher_settings['cachedir']))

    # Load ictal and interictal data
    ictal_X, ictal_y = PrepDataTeacher(
        target, type='ictal', settings=teacher_settings, freq=freq, teacher_channels=teacher_channels
    ).apply()
    interictal_X, interictal_y = PrepDataTeacher(
        target, type='interictal', settings=teacher_settings, freq=freq, teacher_channels=teacher_channels
    ).apply()

    # Prepare training data
    X_train, y_train = train_val_test_split_continual_t(
        ictal_X, ictal_y, interictal_X, interictal_y, 0, no_test=True
    )

    # Initialize teacher model
    if model == 'MViT':
        teacher = MViT(
                X_shape=X_train.shape,
                in_channels=X_train.shape[2],
                num_classes=2,
                patch_size=(5, 10),
                embed_dim=128,
                num_heads=4,
                hidden_dim=256,
                num_layers=4,
                dropout=0.1
            ).to('cuda')
    else:
        teacher = CNN_LSTM_Model(X_train.shape).to('cuda')

    # Convert training data to PyTorch tensors
    Y_train = torch.tensor(y_train).long().to('cuda')
    X_train = torch.tensor(X_train).float().to('cuda')

    # Create DataLoader for training
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader_teacher = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)

    # Define loss function
    ce_loss = nn.CrossEntropyLoss()

    # Select optimizer
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(teacher.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-8)
    else:  # Default to SGD
        optimizer = torch.optim.SGD(teacher.parameters(), lr=5e-4, momentum=0.9)

    pbar = tqdm(total=epochs, desc=f"Training Teacher Model for {target}")

    teacher_losses = []
    for epoch in range(epochs):
        teacher.train()
        total_loss = 0

        for X_batch, Y_batch in train_loader_teacher:
            X_batch, Y_batch = X_batch.to("cuda"), Y_batch.to("cuda")

            # Forward pass
            outputs = teacher(X_batch)
            loss = ce_loss(outputs, Y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader_teacher)
        teacher_losses.append(avg_loss)
        print(f'Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}')
        pbar.update(1)

    pbar.close()

    return teacher_losses


if __name__ == '__main__':
    """
    Main execution loop that takes command-line arguments for:
    - Subject (dog or patient)
    - Number of epochs
    - Optimizer type (SGD or Adam)
    - Frequency setting
    - Number of EEG channels
    - Model type (CNN_LSTM or MViT)
    """

    parser = argparse.ArgumentParser(description="Teacher Model Training for Seizure Detection")
    parser.add_argument("--subject", type=str, required=True,
                        help="Target subject ID (use 'all' to train for all subjects)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs (default: 25)")
    parser.add_argument("--optimizer", type=str, choices=['SGD', 'Adam'], default='Adam',
                        help="Optimizer type: 'SGD' or 'Adam' (default: Adam)")
    parser.add_argument("--freq", type=int, default=1000, help="Sampling frequency (default: 1000Hz)")
    parser.add_argument("--channels", type=int, default=15, help="Number of EEG channels to use (default: 15)")
    parser.add_argument("--model", type=str, choices=['CNN_LSTM', 'MViT'], default='CNN_LSTM',
                        help="Model type: 'CNN_LSTM' or 'MViT' (default: CNN_LSTM)")
    args = parser.parse_args()

    # List of default subjects (4 dogs, 4 patients)
    subject_list = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Patient_3', 'Patient_5', 'Patient_6', 'Patient_7']
    if args.subject != 'all':
        subject_list = [args.subject]

    results = {}

    for subject in subject_list:
        results[subject] = train_teacher_model(
            subject, args.epochs, args.optimizer, args.freq, args.channels
        )

    # Save results to file
    with open("Detection_results.txt", 'a') as f:
        f.write(f'{str(results)}\n')