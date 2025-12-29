import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

from utils.load_signals_student import PrepDataStudent
from utils.prep_data_student import train_val_test_split_continual_s
from models.model import CNN_LSTM_Model
from models.MViT import MViT


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


def train_and_evaluate(target, trials, model_type, epochs=25):
    """
    Trains and evaluates a seizure prediction model for a given patient.

    Args:
        target (str): Patient identifier.
        trials (int): Number of training trials.
        model_type (str): Model type, either 'CNN_LSTM' or 'MViT'.
        epochs (int): Number of training epochs.

    Returns:
        list: A list of results containing FPR, Sensitivity, and AUC-ROC for each trial.
    """
    print(f'Training Model: {model_type} | Patient: {target} | Trials: {trials}')
    torch.cuda.empty_cache()

    # Load student model settings
    with open('student_settings.json') as k:
        student_settings = json.load(k)

    student_results = []

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

    # Create a DataLoader for the training set
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)

    # Perform multiple training trials
    for trial in range(trials):
        print(f'\nStarting Trial {trial + 1}/{trials} for Patient {target}...')

        # Initialize model
        if model_type == 'MViT':
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

        # Define loss function and optimizer
        ce_loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(student.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-8)

        pbar = tqdm(total=epochs, desc=f"Training {model_type} for Patient {target}, Trial {trial + 1}")

        # Training loop
        for epoch in range(epochs):
            student.train()

            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to("cuda"), Y_batch.to("cuda")

                # Forward pass
                student_logits = student(X_batch)
                loss = ce_loss(student_logits, Y_batch)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            pbar.update(1)

        pbar.close()

        # Evaluate the trained model
        student.eval()
        X_tensor = torch.tensor(X_test).float().to('cuda')
        y_tensor = torch.tensor(y_test).long().to('cuda')

        with torch.no_grad():
            predictions = student(X_tensor)

        # Convert outputs to probabilities
        predictions = F.softmax(predictions, dim=1)[:, 1].cpu().numpy()

        # Find optimal threshold and generate binary predictions
        threshold = find_best_threshold(y_tensor.cpu(), predictions)
        binary_predictions = (predictions >= threshold).astype(int)

        # Compute confusion matrix and derived metrics
        cm = confusion_matrix(y_tensor.cpu(), binary_predictions)
        tn, fp, fn, tp = cm.ravel()

        fpr = fp / (fp + tn)
        sensitivity = tp / (tp + fn)
        auc_roc = roc_auc_score(y_tensor.cpu(), predictions)

        # Log results
        print(f'Patient {target}, Trial {trial + 1}:')
        print(f'  False Positive Rate (FPR): {fpr:.2f}')
        print(f'  Sensitivity: {sensitivity:.2f}')
        print(f'  AUCROC: {auc_roc:.2f}')

        student_results.append([fpr, sensitivity, auc_roc])

    return student_results


if __name__ == "__main__":
    """
    Main execution loop that takes command-line arguments for:
    - Patient ID
    - Number of trials
    - Model type (CNN_LSTM or MViT)
    """

    parser = argparse.ArgumentParser(description="Seizure Prediction Model Training")
    parser.add_argument("--patient", type=str, required=True, help="Target patient ID")
    parser.add_argument("--trials", type=int, default=3, help="Number of training trials (default: 3)")
    parser.add_argument("--model", type=str, choices=['CNN_LSTM', 'MViT'], default='CNN_LSTM',
                        help="Model type: 'CNN_LSTM' or 'MViT' (default: CNN_LSTM)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs (default: 25)")

    args = parser.parse_args()

    # Train and evaluate the model for the given patient
    results = train_and_evaluate(args.patient, args.trials, args.model, args.epochs)

    # Save results to a text file
    with open('Prediction_results.txt', 'a') as f:
        f.write(f'Patient {args.patient} Results: {str(results)}\n')