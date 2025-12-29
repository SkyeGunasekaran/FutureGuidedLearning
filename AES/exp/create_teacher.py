import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample
import stft

from utils.save_load import save_pickle_file, load_pickle_file, save_hickle_file, load_hickle_file
from models.models import CNN_LSTM_Model
from utils.prep_data_teacher import train_val_test_split_continual_t
from utils.load_signals_teacher import PrepDataTeacher, make_teacher, makedirs


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
    torch.cuda.empty_cache()

    # Load settings
    with open('teacher_settings.json') as f:
        teacher_settings = json.load(f)
    with open('student_settings.json') as k:
        student_settings = json.load(k)

    makedirs(str(teacher_settings['cachedir']))
    makedirs(str(student_settings['cachedir']))

    # Define mode and save path
    if target == 'Dog':
        mode = 'Dog'
        path = 'teacher_dog.pth'
    elif target == 'Patient_1':
        mode = 'Patient_1'
        path = f'{target}.pth'
    else:
        mode = 'Patient_2'
        path = f'{target}.pth'

    # Preprocessing
    ictal_X, ictal_y, interictal_X, interictal_y = make_teacher(mode=mode, teacher_settings=teacher_settings)
    X_train, y_train = train_val_test_split_continual_t(
        ictal_X, ictal_y, interictal_X, interictal_y, 0.0, no_test=True
    )

    # Convert training data to PyTorch tensors and move to GPU
    Y_train = torch.tensor(y_train).long().to('cuda')
    X_train = torch.tensor(X_train).float().to('cuda')

    # Create DataLoader for training
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader_teacher = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)

    # Model creation
    teacher = CNN_LSTM_Model(X_train.shape).to('cuda')
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(teacher.parameters(), lr=5e-4, momentum=0.9)

    pbar = tqdm(total=epochs, desc=f"Training Teacher Model for {target}")

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

        print(f'Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(train_loader_teacher):.4f}')
        pbar.update(1)

    pbar.close()

    # Save the trained model
    torch.save(teacher, path)


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