import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils.load_signals_student import PrepDataStudent
from utils.prep_data_student import train_val_test_split_continual_s
from models.models import CNN_LSTM_Model

# --- Helper Functions ---

def _prepare_data(target, settings, device):
    """Loads, splits, and prepares data into a DataLoader and test tensors."""
    ictal_X, ictal_y = PrepDataStudent(target, 'ictal', settings).apply()
    interictal_X, interictal_y = PrepDataStudent(target, 'interictal', settings).apply()

    X_train, y_train, X_test, y_test = train_val_test_split_continual_s(
        ictal_X, ictal_y, interictal_X, interictal_y, 0.35
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    
    return train_loader, X_test_tensor, y_test_tensor, X_train_tensor.shape

def _build_student_and_optimizer(input_shape, optimizer_type, device):
    """Initializes the student model and its optimizer."""
    student = CNN_LSTM_Model(input_shape).to(device)
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(student.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-8)
    else:  # Default to SGD
        optimizer = torch.optim.SGD(student.parameters(), lr=5e-4, momentum=0.9)
    return student, optimizer

def _compute_distillation_loss(student_logits, teacher_logits, temp):
    """Calculates the Kullback-Leibler divergence loss for knowledge distillation."""
    soft_targets = F.softmax(teacher_logits / temp, dim=1)
    soft_prob = F.log_softmax(student_logits / temp, dim=1)
    return F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temp ** 2)

def _train_epoch(student, teacher, loader, optimizer, alpha, temp, device):
    """Runs a single training epoch for knowledge distillation."""
    student.train()
    teacher.eval()
    criterion = nn.CrossEntropyLoss()
    
    for X_batch, Y_batch in loader:
        student_logits = student(X_batch)
        with torch.no_grad():
            teacher_logits = teacher(X_batch)
            
        student_loss = criterion(student_logits, Y_batch)
        distill_loss = _compute_distillation_loss(student_logits, teacher_logits, temp)
        
        loss = alpha * student_loss + (1 - alpha) * distill_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def _evaluate_student(student, X_test, y_test):
    """Evaluates the student model and returns the AUC score."""
    student.eval()
    with torch.no_grad():
        predictions = student(X_test)
    
    y_probs = F.softmax(predictions, dim=1)[:, 1].cpu().numpy()
    y_true = y_test.cpu().numpy()
    
    return roc_auc_score(y_true, y_probs)

# --- Main Distillation Function ---

def distill_student_model(target, epochs, trials, optimizer_type, alpha, temperature):
    """
    Performs knowledge distillation for a given patient.
    (Function signature remains unchanged)
    """
    print(f'\nKnowledge Distillation: Patient {target} | Alpha: {alpha:.2f} | Epochs: {epochs} | Trials: {trials} | Optimizer: {optimizer_type}')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    with open('student_settings.json') as k:
        student_settings = json.load(k)

    teacher = torch.load(f'pytorch_models/Patient_{target}_detection').to(device)
    loader, X_test, y_test, shape = _prepare_data(target, student_settings, device)

    auc_list = []
    for trial in range(trials):
        print(f'\nPatient {target} | Alpha: {alpha:.2f} | Trial {trial + 1}/{trials}')
        student, optimizer = _build_student_and_optimizer(shape, optimizer_type, device)
        
        desc = f"Training Trial {trial+1} (Alpha: {alpha:.2f}) for Patient {target}"
        with tqdm(total=epochs, desc=desc) as pbar:
            for epoch in range(epochs):
                _train_epoch(student, teacher, loader, optimizer, alpha, temperature, device)
                pbar.update(1)
        
        auc_test = _evaluate_student(student, X_test, y_test)
        print(f'Patient {target}, Alpha {alpha:.2f} | Test AUC: {auc_test:.4f}')
        auc_list.append(auc_test)

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
    args = parser.parse_args()

    default_patients = ['1', '2', '3', '5', '9', '10', '13', '18', '19', '20', '21', '23']
    patients_to_run = default_patients if args.patient == 'all' else [args.patient]
    
    all_results = {}
    for patient in patients_to_run:
        all_results[patient] = distill_student_model(
            patient, args.epochs, args.trials, args.optimizer, args.alpha, args.temperature
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