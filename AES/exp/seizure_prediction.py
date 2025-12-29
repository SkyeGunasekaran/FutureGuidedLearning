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
from models.models import CNN_LSTM_Model
from models.MViT import MViT


def makedirs(dir_path):
    """Creates a directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        pass


def find_best_threshold(y_true, y_pred):
    """
    Determines the optimal classification threshold using the Youden index.

    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: Optimal threshold value.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    return thresholds[optimal_idx]


def train_student_model(target, epochs, trials, model, optimizer_type):
    """
    Trains and evaluates a seizure prediction student model for a given subject.

    Args:
        target (str): Subject identifier (e.g., 'Dog_1', 'Patient_1').
        epochs (int): Number of training epochs.
        trials (int): Number of training trials.
        model (str): Model type ('CNN_LSTM' or 'MViT').
        optimizer_type (str): Optimizer type ('SGD' or 'Adam').

    Returns:
        list: A list of results containing FPR, Sensitivity, and AUC-ROC for each trial.
    """
    print(f'\nTraining Student Model: Target {target} | Model: {model} | Epochs: {epochs} | Trials: {trials} | Optimizer: {optimizer_type}')
    torch.cuda.empty_cache()

    # Load student model settings
    with open('student_settings.json') as k:
        student_settings = json.load(k)

    makedirs(str(student_settings['cachedir']))

    # Load ictal and interictal data
    ictal_X, ictal_y = PrepDataStudent(target, type='ictal', settings=student_settings).apply()
    interictal_X, interictal_y = PrepDataStudent(target, type='interictal', settings=student_settings).apply()

    # Split into training and testing sets
    X_train, y_train, X_test, y_test = train_val_test_split_continual_s(
        ictal_X, ictal_y, interictal_X, interictal_y, 0.35
    )

    # Convert training data to PyTorch tensors and move to GPU
    Y_train = torch.tensor(y_train).long().to('cuda')
    X_train = torch.tensor(X_train).float().to('cuda')

    # Create DataLoader for training
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    student_results = []

    for trial in range(trials):
        print(f'\nPatient {target} | Trial {trial + 1}/{trials}')

        # Initialize model
        if model == 'MViT':
            student = MViT(
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
            student = CNN_LSTM_Model(X_train.shape).to('cuda')

        # Define loss function
        ce_loss = nn.CrossEntropyLoss()

        # Select optimizer
        if optimizer_type == 'Adam':
            optimizer = torch.optim.Adam(student.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-8)
        else:  # Default to SGD
            optimizer = torch.optim.SGD(student.parameters(), lr=5e-4, momentum=0.9)

        pbar = tqdm(total=epochs, desc=f"Training {model} for {target}, Trial {trial + 1}")

        # Training loop
        for epoch in range(epochs):
            student.train()
            total_loss = 0

            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to("cuda"), Y_batch.to("cuda")

                # Forward pass
                student_logits = student(X_batch)
                loss = ce_loss(student_logits, Y_batch)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            pbar.update(1)

        pbar.close()

        # Evaluate the trained model
        student.eval()
        X_tensor = torch.tensor(X_test).float().to('cuda')
        y_tensor = torch.tensor(y_test).long().to('cuda')

        with torch.no_grad():
            predictions = student(X_tensor)

        predictions = F.softmax(predictions, dim=1)[:, 1].cpu().numpy()
        threshold = find_best_threshold(y_tensor.cpu(), predictions)
        binary_predictions = (predictions >= threshold).astype(int)

        # Compute confusion matrix and metrics
        cm = confusion_matrix(y_tensor.cpu(), binary_predictions)
        tn, fp, fn, tp = cm.ravel()

        fpr = fp / (fp + tn)
        sensitivity = tp / (tp + fn)
        auc_roc = roc_auc_score(y_tensor.cpu(), predictions)

        print(f'Patient {target} | FPR: {fpr:.4f} | Sensitivity: {sensitivity:.4f} | AUCROC: {auc_roc:.4f}')

        student_results.append([fpr, sensitivity, auc_roc])

    return student_results


if __name__ == '__main__':
    """
    Main execution loop that takes command-line arguments for:
    - Subject (dog or patient)
    - Number of epochs
    - Model type (CNN_LSTM or MViT)
    - Optimizer type (SGD or Adam)
    - Number of trials
    """

    parser = argparse.ArgumentParser(description="Seizure Prediction Student Model Training")
    parser.add_argument("--subject", type=str, required=True,
                        help="Target subject ID (use 'all' to train for all subjects)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs (default: 25)")
    parser.add_argument("--model", type=str, choices=['CNN_LSTM', 'MViT'], default='MViT',
                        help="Model type: 'CNN_LSTM' or 'MViT' (default: MViT)")
    parser.add_argument("--optimizer", type=str, choices=['SGD', 'Adam'], default='Adam',
                        help="Optimizer type: 'SGD' or 'Adam' (default: Adam)")
    parser.add_argument("--trials", type=int, default=3, help="Number of training trials (default: 3)")

    args = parser.parse_args()

    # List of default subjects (Dogs & Patients)
    subject_list = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    if args.subject != 'all':
        subject_list = [args.subject]

    results = {}

    for subject in subject_list:
        results[subject] = train_student_model(
            subject, args.epochs, args.trials, args.model, args.optimizer
        )

    # Save results to file
    with open("Prediction_results.txt", 'a') as f:
        f.write(f'{str(results)}\n')