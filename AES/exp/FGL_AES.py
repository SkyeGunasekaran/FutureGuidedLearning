import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import numpy as np

from utils.load_signals_student import PrepDataStudent
from utils.prep_data_student import train_val_test_split_continual_s
from models.models import CNN_LSTM_Model


def ROC_threshold(y_true, y_probs):
    """
    Determines the optimal classification threshold using the Youden index.

    Args:
        y_true (array-like): Ground truth binary labels.
        y_probs (array-like): Predicted probabilities.

    Returns:
        float: Optimal threshold value.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    J = tpr - fpr  # Calculate Youden's J statistic
    ix = np.argmax(J)
    return thresholds[ix]


def train_student_model(target, student_model, epochs, temperature, optimizer_type, alpha):
    """
    Trains and evaluates a student model using knowledge distillation.

    Args:
        target (str): Subject identifier ('Dog_1', 'Patient_1', etc.).
        student_model (str): Model type ('CNN_LSTM').
        epochs (int): Number of training epochs.
        temperature (float): Temperature parameter for distillation.
        optimizer_type (str): Optimizer type ('SGD' or 'Adam').
        alpha (float): Weight for cross-entropy loss.

    Returns:
        list: A list of tuples containing Sensitivity, FPR, and AUC-ROC for each trial.
    """
    print(f'\nTraining Student Model: Target {target} | Model: {student_model} | Alpha: {alpha} | Epochs: {epochs} | Optimizer: {optimizer_type} | Temperature: {temperature}')
    torch.cuda.empty_cache()

    trials = 3

    # Load student settings
    with open('student_settings.json') as k:
        student_settings = json.load(k)

    # Determine teacher model path
    if 'Dog_' in target:
        model_path = 'teacher_dog.pth'
    elif target in ['Patient_1', 'Patient_2']:
        model_path = f'{target}.pth'
    else:
        raise ValueError("Invalid target. Must be 'Dog_1' to 'Dog_5', 'Patient_1', or 'Patient_2'.")

    # Check if teacher model exists
    if not os.path.exists(model_path):
        print(f"Error: Teacher model '{model_path}' not found. Please train the universal teacher model first.")
        return []

    # Load teacher model
    teacher = torch.load(model_path)
    teacher.eval()

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

    results = []

    for trial in range(trials):
        print(f'\nPatient {target} | Trial {trial + 1}/{trials}')

        # Initialize student model
        if student_model == 'CNN_LSTM':
            student = CNN_LSTM_Model(X_train.shape).to('cuda')
        else:
            raise ValueError("Invalid student model. Only 'CNN_LSTM' is supported.")

        ce_loss = nn.CrossEntropyLoss()

        # Select optimizer
        if optimizer_type == 'Adam':
            optimizer = torch.optim.Adam(student.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-8)
        else:  # Default to SGD
            optimizer = torch.optim.SGD(student.parameters(), lr=5e-4, momentum=0.9)

        pbar = tqdm(total=epochs, desc=f"Training {student_model} for {target}, Trial {trial + 1}")

        # Training loop
        for epoch in range(epochs):
            student.train()
            total_loss = 0

            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to("cuda"), Y_batch.to("cuda")

                with torch.no_grad():
                    teacher_logits = teacher(X_batch)

                student_logits = student(X_batch)

                # Compute soft targets using temperature scaling
                soft_targets = F.softmax(teacher_logits / temperature, dim=1)
                soft_prob = F.log_softmax(student_logits / temperature, dim=1)
                soft_targets_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)

                # Compute total loss
                loss = alpha * ce_loss(student_logits, Y_batch) + (1 - alpha) * soft_targets_loss

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

        threshold = ROC_threshold(y_tensor.cpu(), predictions)
        binary_predictions = (predictions >= threshold).astype(int)

        cm = confusion_matrix(y_tensor.cpu(), binary_predictions)
        tn, fp, fn, tp = cm.ravel()

        # Calculate FPR, Sensitivity, and AUC-ROC
        fpr = fp / (fp + tn)
        sensitivity = tp / (tp + fn)
        auc_roc = roc_auc_score(y_tensor.cpu(), predictions)

        print(f'Patient {target} | Sensitivity: {sensitivity:.4f} | FPR: {fpr:.4f} | AUCROC: {auc_roc:.4f}')
        results.append((sensitivity, fpr, auc_roc))

    return results


if __name__ == '__main__':
    """
    Main execution loop that takes command-line arguments for:
    - Subject ('Dog_1' to 'Dog_5', 'Patient_1', or 'Patient_2')
    - Student model ('CNN_LSTM')
    - Number of epochs
    - Temperature for distillation
    - Optimizer type ('SGD' or 'Adam')
    - Alpha value for loss balancing
    """

    parser = argparse.ArgumentParser(description="FGL on AES")
    parser.add_argument("--subject", type=str, choices=['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2'], required=True,
                        help="Target subject: 'Dog_1' to 'Dog_5', 'Patient_1', or 'Patient_2'")
    parser.add_argument("--model", type=str, choices=['CNN_LSTM'], default='CNN_LSTM',
                        help="Student model type (default: CNN_LSTM)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs (default: 25)")
    parser.add_argument("--temperature", type=float, default=4, help="Temperature for distillation (default: 4)")
    parser.add_argument("--optimizer", type=str, choices=['SGD', 'Adam'], default='SGD',
                        help="Optimizer type (default: SGD)")
    parser.add_argument("--alpha", type=float, required=True, 
                        help="Alpha value for loss weighting (cross-entropy weight)")

    args = parser.parse_args()

    results = train_student_model(args.subject, args.model, args.epochs, args.temperature, args.optimizer, args.alpha)

    # Save results
    with open("FGL_AES.txt", 'a') as f:
        f.write(f'{args.subject}_results={str(results)}\n')