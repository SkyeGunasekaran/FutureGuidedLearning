import argparse
import json
import os  # Added
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np  # Added for np.nan

from utils.load_signals_student import PrepDataStudent
from utils.prep_data_student import train_val_test_split_continual_s
from models.models import CNN_LSTM_Model

# --- Helper Functions ---

def _prepare_data(target, settings, device, val_ratio, test_ratio=0.35, batch_size=32):
    """Loads, splits, and prepares data into DataLoaders."""
    ictal_X, ictal_y = PrepDataStudent(target, 'ictal', settings).apply()
    interictal_X, interictal_y = PrepDataStudent(target, 'interictal', settings).apply()

    # Use the updated split function to get train, val, and test sets
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split_continual_s(
        ictal_X, ictal_y, interictal_X, interictal_y, 
        test_ratio=test_ratio, 
        val_ratio=val_ratio
    )

    # Create training DataLoader
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create validation DataLoader
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create test tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    
    return train_loader, val_loader, X_test_tensor, y_test_tensor, X_train_tensor.shape

def _build_student_and_optimizer(input_shape, optimizer_type, device, lr=5e-4):
    """Initializes the student model and its optimizer."""
    student = CNN_LSTM_Model(input_shape).to(device)
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

def _compute_distillation_loss(student_logits, teacher_logits, temp):
    """Calculates the Kullback-Leibler divergence loss for knowledge distillation."""
    soft_targets = F.softmax(teacher_logits / temp, dim=1)
    soft_prob = F.log_softmax(student_logits / temp, dim=1)
    return F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temp ** 2)

def _train_epoch(student, teacher, loader, optimizer, alpha, temp, device, clip_value):
    """Runs a single training epoch for knowledge distillation and returns avg loss."""
    student.train()
    teacher.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    
    for X_batch, Y_batch in loader:
        student_logits = student(X_batch)
        with torch.no_grad():
            teacher_logits = teacher(X_batch)
            
        student_loss = criterion(student_logits, Y_batch)
        distill_loss = _compute_distillation_loss(student_logits, teacher_logits, temp)
        
        loss = alpha * student_loss + (1 - alpha) * distill_loss
        
        optimizer.zero_grad()
        loss.backward()
        
        # --- Gradient Clipping ---
        torch.nn.utils.clip_grad_norm_(student.parameters(), clip_value)
        
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)

def _validate_epoch(student, val_loader, device):
    """Runs a validation epoch and returns avg student loss."""
    student.eval()
    criterion = nn.CrossEntropyLoss()
    total_val_loss = 0
    with torch.no_grad():
        for X_val_batch, Y_val_batch in val_loader:
            val_student_logits = student(X_val_batch)
            # Validate on the student's performance on ground truth
            val_loss = criterion(val_student_logits, Y_val_batch)
            total_val_loss += val_loss.item()
    return total_val_loss / len(val_loader)


def _evaluate_student(student, X_test, y_test):
    """Evaluates the student model and returns the AUC score."""
    student.eval()
    with torch.no_grad():
        predictions = student(X_test)
    
    y_probs = F.softmax(predictions, dim=1)[:, 1].cpu().numpy()
    y_true = y_test.cpu().numpy()
    
    try:
        return roc_auc_score(y_true, y_probs)
    except ValueError:
        print("Warning: Could not compute AUC. Test set might be single-class.")
        return 0.5 # Default AUC

# --- Main Distillation Function ---

def distill_student_model(target, epochs, trials, optimizer_type, alpha, temperature,
                          val_ratio, patience, clip_value):
    """
    Performs knowledge distillation for a given patient.
    """
    print(f'\nKnowledge Distillation: Patient {target} | Alpha: {alpha:.2f} | Epochs: {epochs} | Trials: {trials} | Optimizer: {optimizer_type}')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    with open('student_settings.json') as k:
        student_settings = json.load(k)
        
    teacher_path = f'pytorch_models/Patient_{target}_detection'
    if not os.path.exists(teacher_path):
        print(f"Error: Teacher model '{teacher_path}' not found. Skipping patient {target}.")
        return []
        
    teacher = torch.load(teacher_path).to(device)
    
    # Prepare data, including validation loader
    train_loader, val_loader, X_test, y_test, shape = _prepare_data(
        target, student_settings, device, val_ratio
    )

    auc_list = []
    for trial in range(trials):
        print(f'\nPatient {target} | Alpha: {alpha:.2f} | Trial {trial + 1}/{trials}')
        
        student, optimizer = _build_student_and_optimizer(shape, optimizer_type, device)
        scheduler = _build_scheduler(optimizer)
        
        # Define a temporary path to save the best model for this trial
        model_save_path = f'./best_student_kd_{target}_alpha{alpha}_trial_{trial+1}.pth'
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        desc = f"Trial {trial+1} (Alpha: {alpha:.2f}) for Patient {target}"
        with tqdm(total=epochs, desc=desc) as pbar:
            for epoch in range(epochs):
                # --- Training ---
                avg_train_loss = _train_epoch(
                    student, teacher, train_loader, optimizer, alpha, temperature, device, clip_value
                )
                
                # --- Validation ---
                avg_val_loss = _validate_epoch(student, val_loader, device)
                
                pbar.set_postfix({'train_loss': f'{avg_train_loss:.4f}', 
                                 'val_loss': f'{avg_val_loss:.4f}'})
                pbar.update(1)
                
                # --- Scheduler Step ---
                scheduler.step(avg_val_loss)
                
                # --- Early Stopping Check & Model Saving ---
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    torch.save(student.state_dict(), model_save_path)
                else:
                    epochs_no_improve += 1
                
                if epochs_no_improve >= patience:
                    print(f'\nEarly stopping triggered after {epoch + 1} epochs.')
                    break
        
        # --- Load Best Model and Evaluate ---
        try:
            student.load_state_dict(torch.load(model_save_path))
            auc_test = _evaluate_student(student, X_test, y_test)
            print(f'Patient {target}, Alpha {alpha:.2f} | Best Model Test AUC: {auc_test:.4f}')
            auc_list.append(auc_test)
            
        except FileNotFoundError:
            print(f"Warning: Model file not found for trial {trial + 1}. Skipping evaluation.")
            auc_list.append(np.nan) # Append NaN
        
        # Clean up the temporary model file
        if os.path.exists(model_save_path):
            os.remove(model_save_path)

    # Save intermediate results for the current patient
    with open("FGL_results.txt", 'a') as f:
        f.write(f'Patient_{target}_Alpha_{alpha:.2f}_Results= {str(auc_list)}\n')
        
    return auc_list

# --- Execution Block ---

def main():
    """Main execution function to parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Knowledge Distillation for Seizure Prediction")
    parser.add_argument("--patient", type=str, required=True, help="Patient ID (or 'all')")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--optimizer", type=str, choices=['SGD', 'Adam'], default='SGD', help="Optimizer type")
    parser.add_argument("--trials", type=int, default=3, help="Number of training trials")
    parser.add_argument("--alpha", type=float, required=True, help="Weight for cross-entropy loss")
    parser.add_argument("--temperature", type=float, default=4, help="Temperature for distillation")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Percentage of data for validation")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--clip_value", type=float, default=1.0, help="Gradient clipping norm")
    
    args = parser.parse_args()

    default_patients = ['1', '2', '3', '5', '9', '10', '13', '18', '19', '20', '21', '23']
    patients_to_run = default_patients if args.patient == 'all' else [args.patient]
    
    all_results = {}
    for patient in patients_to_run:
        all_results[patient] = distill_student_model(
            patient, args.epochs, args.trials, args.optimizer, args.alpha, args.temperature,
            args.val_ratio, args.patience, args.clip_value
        )

    # Save final aggregated results to a structured JSON file
    try:
        with open("FGL_results.json", 'r') as f:
            existing_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_results = {}

    existing_results[f"alpha_{args.alpha:.2f}"] = all_results
    with open("FGL_results.json", 'w') as f:
        json.dump(existing_results, f, indent=4)
        
    print("\nDistillation complete. Aggregated results saved to FGL_results.json")

if __name__ == "__main__":
    main()