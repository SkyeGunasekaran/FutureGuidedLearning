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

def _prepare_data(target, settings, device, val_ratio, test_ratio=0.35, batch_size=32):
    """Loads, splits, and prepares data for training, validation, and testing."""
    # Load raw data
    ictal_X, ictal_y = PrepDataStudent(target, type='ictal', settings=settings).apply()
    interictal_X, interictal_y = PrepDataStudent(target, type='interictal', settings=settings).apply()

    # Split into training, validation, and testing sets
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split_continual_s(
        ictal_X, ictal_y, interictal_X, interictal_y, 
        test_ratio=test_ratio, 
        val_ratio=val_ratio
    )

    # Create training DataLoader
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Shuffle=True for training
    
    # Create validation DataLoader
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Prepare test tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    
    return train_loader, val_loader, X_test_tensor, y_test_tensor, X_train_tensor.shape

def _build_student_and_optimizer(model_type, input_shape, optimizer_type, device, lr=5e-4):
    """Initializes the student model and its optimizer."""
    if model_type == 'CNN_LSTM':
        student = CNN_LSTM_Model(input_shape).to(device)
    else:
        raise ValueError("Invalid student model. Only 'CNN_LSTM' is supported.")

    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(student.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    else:  # Default to SGD
        optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9)
        
    return student, optimizer

def _build_scheduler(optimizer):
    """Builds a learning rate scheduler."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.1, patience=3, verbose=True
    )

def _compute_distillation_loss(student_logits, teacher_logits, temperature):
    """Calculates the Kullback-Leibler divergence loss for knowledge distillation."""
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    soft_prob = F.log_softmax(student_logits / temperature, dim=1)
    # Scale loss by T^2 as proposed in the Hinton et al. paper
    distillation_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)
    return distillation_loss

def _run_training_loop(student, teacher, train_loader, val_loader, epochs, 
                       optimizer, scheduler, alpha, temperature, clip_value, 
                       patience, model_save_path, device, trial_desc):
    """Executes the training and validation process for a single trial."""
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    with tqdm(total=epochs, desc=trial_desc) as pbar:
        for epoch in range(epochs):
            # --- Training Phase ---
            student.train()
            running_train_loss = 0.0
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
                
                # --- Gradient Clipping ---
                torch.nn.utils.clip_grad_norm_(student.parameters(), clip_value)
                
                optimizer.step()
                running_train_loss += loss.item()
                
            avg_train_loss = running_train_loss / len(train_loader)
            
            # --- Validation Phase ---
            student.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for X_val_batch, Y_val_batch in val_loader:
                    val_student_logits = student(X_val_batch)
                    # Validate on the student's performance on ground truth
                    val_loss = criterion(val_student_logits, Y_val_batch)
                    running_val_loss += val_loss.item()
            
            avg_val_loss = running_val_loss / len(val_loader)
            
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
                torch.save(student.state_dict(), model_save_path)
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs.')
                break

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
    
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    except ValueError:
        # Handle single-class case
        if np.all(y_pred_binary == 0):
            tn, fp, fn, tp = len(y_true), 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, len(y_true)
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # AUC ROC should be robust to single-class predictions, but not single-class true labels
    try:
        auc_roc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc_roc = 0.5 # Default value if y_true is single-class
    
    return sensitivity, fpr, auc_roc

# --- Main Function ---

def train_student_model(target, student_model, epochs, temperature, optimizer_type, 
                        alpha, val_ratio, patience, clip_value):
    """
    Trains and evaluates a student model using knowledge distillation.
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
        
    train_loader, val_loader, X_test, y_test, input_shape = _prepare_data(
        target, student_settings, device, val_ratio
    )

    results = []
    num_trials = 3
    for trial in range(num_trials):
        print(f'\nPatient {target} | Trial {trial + 1}/{num_trials}')
        
        # Define a temporary path to save the best model for this trial
        model_save_path = f'./best_student_kd_{target}_trial_{trial+1}.pth'
        
        # Initialize model and optimizer for the new trial
        student, optimizer = _build_student_and_optimizer(
            student_model, input_shape, optimizer_type, device
        )
        scheduler = _build_scheduler(optimizer)
        
        # Train the model
        trial_desc = f"Training {student_model} for {target}, Trial {trial + 1}"
        _run_training_loop(
            student, teacher, train_loader, val_loader, epochs, optimizer, scheduler,
            alpha, temperature, clip_value, patience, model_save_path, device, trial_desc
        )
        
        # --- Load Best Model and Evaluate ---
        try:
            student.load_state_dict(torch.load(model_save_path))
            
            sensitivity, fpr, auc_roc = _evaluate_model(student, X_test, y_test)
            print(f'Patient {target} | Sensitivity: {sensitivity:.4f} | FPR: {fpr:.4f} | AUCROC: {auc_roc:.4f}')
            results.append((sensitivity, fpr, auc_roc))
            
        except FileNotFoundError:
            print(f"Warning: Model file not found for trial {trial + 1}. Skipping evaluation.")
            results.append((np.nan, np.nan, np.nan)) # Append NaNs
        
        # Clean up the temporary model file
        if os.path.exists(model_save_path):
            os.remove(model_save_path)
            
    return results


if __name__ == '__main__':
    """
    Main execution loop that handles command-line arguments.
    """
    parser = argparse.ArgumentParser(description="FGL on AES")
    parser.add_argument("--subject", type=str, choices=['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2'], required=True,
                        help="Target subject")
    parser.add_argument("--model", type=str, choices=['CNN_LSTM'], default='CNN_LSTM', help="Student model type")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--temperature", type=float, default=4, help="Temperature for distillation")
    parser.add_argument("--optimizer", type=str, choices=['SGD', 'Adam'], default='SGD', help="Optimizer type")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha weight for cross-entropy loss")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Percentage of data to use for validation (e.g., 0.2 for 20%)")
    parser.add_argument("--patience", type=int, default=5, help="Epochs to wait for val_loss improvement before early stopping")
    parser.add_argument("--clip_value", type=float, default=1.0, help="Maximum norm for gradient clipping")
    
    args = parser.parse_args()
    
    results = train_student_model(
        args.subject, args.model, args.epochs, args.temperature, 
        args.optimizer, args.alpha, args.val_ratio, args.patience, args.clip_value
    )

    # Save results
    with open("FGL_AES.txt", 'a') as f:
        f.write(f'{args.subject}_results={str(results)}\n')