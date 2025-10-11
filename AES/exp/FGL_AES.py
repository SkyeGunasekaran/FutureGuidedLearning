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

# --- Helper Functions ---

def ROC_threshold(y_true, y_probs):
    """Determines the optimal classification threshold using the Youden index."""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    if len(thresholds) == 0:
        return 0.5 # Default threshold
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    return thresholds[best_index]

def _get_teacher_path(target):
    """Determines the file path for the pre-trained teacher model."""
    if 'Dog_' in target:
        return 'teacher_dog.pth'
    elif target in ['Patient_1', 'Patient_2']:
        return f'{target}.pth'
    raise ValueError("Invalid target. Teacher not specified for this target.")

def _prepare_data(target, settings, device):
    """Loads, splits, and prepares data for training and testing."""
    # Load raw data
    ictal_X, ictal_y = PrepDataStudent(target, type='ictal', settings=settings).apply()
    interictal_X, interictal_y = PrepDataStudent(target, type='interictal', settings=settings).apply()

    # Split into training and testing sets
    X_train, y_train, X_test, y_test = train_val_test_split_continual_s(
        ictal_X, ictal_y, interictal_X, interictal_y, 0.35
    )

    # Create training DataLoader
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    
    # Prepare test tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    
    return train_loader, X_test_tensor, y_test_tensor, X_train_tensor.shape

def _build_student_and_optimizer(model_type, input_shape, optimizer_type, device):
    """Initializes the student model and its optimizer."""
    if model_type == 'CNN_LSTM':
        student = CNN_LSTM_Model(input_shape).to(device)
    else:
        raise ValueError("Invalid student model. Only 'CNN_LSTM' is supported.")

    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(student.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-8)
    else:  # Default to SGD
        optimizer = torch.optim.SGD(student.parameters(), lr=5e-4, momentum=0.9)
        
    return student, optimizer

def _compute_distillation_loss(student_logits, teacher_logits, temperature):
    """Calculates the Kullback-Leibler divergence loss for knowledge distillation."""
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    soft_prob = F.log_softmax(student_logits / temperature, dim=1)
    # Scale loss by T^2 as proposed in the Hinton et al. paper
    distillation_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)
    return distillation_loss

def _run_training_loop(student, teacher, train_loader, epochs, optimizer, alpha, temperature, device, trial_desc):
    """Executes the training process over all epochs for a single trial."""
    criterion = nn.CrossEntropyLoss()
    with tqdm(total=epochs, desc=trial_desc) as pbar:
        for epoch in range(epochs):
            student.train()
            for X_batch, Y_batch in train_loader:
                with torch.no_grad():
                    teacher_logits = teacher(X_batch)
                
                student_logits = student(X_batch)
                
                # Calculate losses
                distill_loss = _compute_distillation_loss(student_logits, teacher_logits, temperature)
                student_loss = criterion(student_logits, Y_batch)
                
                loss = alpha * student_loss + (1 - alpha) * distill_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            pbar.update(1)

def _evaluate_model(student, X_test, y_test):
    """Evaluates the trained student model and computes performance metrics."""
    student.eval()
    with torch.no_grad():
        predictions_raw = student(X_test)
    
    y_true = y_test.cpu().numpy()
    y_probs = F.softmax(predictions_raw, dim=1)[:, 1].cpu().numpy()
    
    # Calculate metrics
    threshold = ROC_threshold(y_true, y_probs)
    y_pred_binary = (y_probs >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    auc_roc = roc_auc_score(y_true, y_probs)
    
    return sensitivity, fpr, auc_roc

# --- Main Function ---

def train_student_model(target, student_model, epochs, temperature, optimizer_type, alpha):
    """
    Trains and evaluates a student model using knowledge distillation.
    (Function signature remains unchanged)
    """
    print(f'\nTraining Student Model: Target {target} | Model: {student_model} | Alpha: {alpha} | Epochs: {epochs} | Optimizer: {optimizer_type} | Temperature: {temperature}')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    
    # Load teacher model
    teacher_path = _get_teacher_path(target)
    if not os.path.exists(teacher_path):
        print(f"Error: Teacher model '{teacher_path}' not found.")
        return []
    teacher = torch.load(teacher_path).to(device)
    teacher.eval()
    
    # Load data
    with open('student_settings.json') as k:
        student_settings = json.load(k)
    train_loader, X_test, y_test, input_shape = _prepare_data(target, student_settings, device)

    results = []
    num_trials = 3
    for trial in range(num_trials):
        print(f'\nPatient {target} | Trial {trial + 1}/{num_trials}')
        
        # Initialize model and optimizer for the new trial
        student, optimizer = _build_student_and_optimizer(student_model, input_shape, optimizer_type, device)
        
        # Train the model
        trial_desc = f"Training {student_model} for {target}, Trial {trial + 1}"
        _run_training_loop(student, teacher, train_loader, epochs, optimizer, alpha, temperature, device, trial_desc)
        
        # Evaluate and store results
        sensitivity, fpr, auc_roc = _evaluate_model(student, X_test, y_test)
        print(f'Patient {target} | Sensitivity: {sensitivity:.4f} | FPR: {fpr:.4f} | AUCROC: {auc_roc:.4f}')
        results.append((sensitivity, fpr, auc_roc))
        
    return results


if __name__ == '__main__':
    """
    Main execution loop that handles command-line arguments.
    (Functionality remains unchanged)
    """
    parser = argparse.ArgumentParser(description="FGL on AES")
    parser.add_argument("--subject", type=str, choices=['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2'], required=True,
                        help="Target subject")
    parser.add_argument("--model", type=str, choices=['CNN_LSTM'], default='CNN_LSTM', help="Student model type")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--temperature", type=float, default=4, help="Temperature for distillation")
    parser.add_argument("--optimizer", type=str, choices=['SGD', 'Adam'], default='SGD', help="Optimizer type")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha weight for cross-entropy loss")

    args = parser.parse_args()
    
    results = train_student_model(args.subject, args.model, args.epochs, args.temperature, args.optimizer, args.alpha)

    # Save results
    with open("FGL_AES.txt", 'a') as f:
        f.write(f'{args.subject}_results={str(results)}\n')