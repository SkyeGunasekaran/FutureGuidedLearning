import argparse
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from utils.utils import RNN, create_time_series_dataset, KL

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EarlyStopper:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Saves the best model state and restores it upon completion.
    """
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_state = None

    def step(self, current_loss, model):
        """Checks if training should stop."""
        if current_loss + self.min_delta < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
            self.best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            return False  # Do not stop
        else:
            self.counter += 1
            return self.counter >= self.patience # Stop if patience is exceeded

    def restore(self, model):
        """Loads the best model state found during training."""
        if self.best_state:
            model.load_state_dict(self.best_state)

def _run_training_loop(model, train_loader, val_loader, optimizer, epochs, patience, 
                       lookback_window, model_name, is_distillation=False, **kwargs):
    """
    A generic training and validation loop with early stopping.
    Handles both standard training and knowledge distillation.
    """
    stopper = EarlyStopper(patience=patience)
    celoss = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        # --- Training Step ---
        model.train()
        for batch_data in train_loader:
            if is_distillation:
                (x_s, y_s), (x_t, y_t) = batch_data 
                
                # Move to device
                targets = y_s.long().to(device)
                x_s = x_s.float().to(device)
                x_t = x_t.float().to(device)
                
                with torch.no_grad():
                    teacher_logits = kwargs['teacher'](x_t)
                
                outputs = model(x_s)
                loss = kwargs['alpha'] * celoss(outputs, targets) + KL(outputs, teacher_logits, kwargs['temp'], kwargs['alpha'])
            
            else:
                # Standard training
                x, y = batch_data
                
                x = x.float().to(device)
                y = y.long().to(device)
                
                outputs = model(x)
                loss = celoss(outputs, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- Validation Step ---
        model.eval()
        val_loss = 0.
        with torch.no_grad():
            for x, y in val_loader:
                
                x = x.float().to(device)
                y = y.long().to(device)
                
                outputs = model(x)
                val_loss += celoss(outputs, y).item()
                
        val_loss /= len(val_loader)
        
        if stopper.step(val_loss, model):
            print(f"[{model_name}] Early stopping at epoch {epoch+1}")
            break
            
    stopper.restore(model)
    print(f"[{model_name}] Best Val Loss = {stopper.best_loss:.4f}")
    return model

def _evaluate_on_test_set(model, loader, lookback_window):
    """Evaluates a trained model on the test set and returns the MSE."""
    model.eval()
    total_mse = 0.
    with torch.no_grad():
        for x, y in loader:
            
            x = x.float().to(device)
            y = y.float().to(device) 
            
            pred = model(x).argmax(dim=1).float()
            total_mse += F.mse_loss(pred, y).item()
            
    return total_mse / len(loader)

def train_student_model(student_horizon, alpha, num_bins, val_size, test_size, epochs,
                        temperature, lookback_window, batch_size, patience=5):
    """
    Trains and compares Teacher, Baseline, and Student models for time-series forecasting.
    (Function signature remains unchanged)
    """
    torch.manual_seed(42)
    print(f"\nTraining | Horizon={student_horizon} Alpha={alpha} Num bins={num_bins}")

    # Hyperparameters
    hyperparams = {'hidden_size': 128, 'output_size': num_bins, 'num_layers': 2, 'lr': 1e-4}
    with open("data.pkl", "rb") as f:
        data = pickle.load(f)

    # Create datasets for teacher (1-step ahead) and student (H-step ahead)
    common_args = {'data': data, 'lookback_window': lookback_window, 'num_bins': num_bins,
                   'val_size': val_size, 'test_size': test_size, 'batch_size': batch_size}
    teacher_train, teacher_val, teacher_test, _, _ = create_time_series_dataset(
        **common_args, forecasting_horizon=1, offset=student_horizon - 1
    )
    student_train, student_val, student_test, _, _ = create_time_series_dataset(
        **common_args, forecasting_horizon=student_horizon, offset=0
    )

    # --- Train Teacher Model ---
    teacher = RNN(lookback_window, **hyperparams).to(device)
    optimizer_t = torch.optim.Adam(teacher.parameters(), lr=hyperparams['lr'])
    teacher = _run_training_loop(
        teacher, teacher_train, teacher_val, optimizer_t, epochs, patience,
        lookback_window, "Teacher"
    )

    # --- Train Baseline Model ---
    baseline = RNN(lookback_window, **hyperparams).to(device)
    optimizer_b = torch.optim.Adam(baseline.parameters(), lr=hyperparams['lr'])
    baseline = _run_training_loop(
        baseline, student_train, student_val, optimizer_b, epochs, patience,
        lookback_window, "Baseline"
    )

    # --- Train Student Model (with Distillation) ---
    student = RNN(lookback_window, **hyperparams).to(device)
    optimizer_s = torch.optim.Adam(student.parameters(), lr=hyperparams['lr'])
    distillation_params = {'teacher': teacher, 'alpha': alpha, 'temp': temperature}
    student = _run_training_loop(
        student, zip(student_train, teacher_train), student_val, optimizer_s,
        epochs, patience, lookback_window, "Student", is_distillation=True, **distillation_params
    )

    # --- Final Evaluation ---
    Tmse = _evaluate_on_test_set(teacher, teacher_test, lookback_window)
    Bmse = _evaluate_on_test_set(baseline, student_test, lookback_window)
    Smse = _evaluate_on_test_set(student, student_test, lookback_window)
    print(f"\nFinal Test MSE:\n Teacher:  {Tmse:.4f}\n Baseline: {Bmse:.4f}\n Student:  {Smse:.4f}")
    
    return None

def main():
    """Parses arguments and runs the training process."""
    parser = argparse.ArgumentParser(description="Future-Guided Learning for Time-Series Forecasting")
    parser.add_argument("--horizon", type=int, required=True, help="Student horizon (H)")
    parser.add_argument("--alpha", type=float, required=True, help="Loss weight α")
    parser.add_argument("--num_bins", type=int, default=50, help="Number of bins")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--temperature", type=float, default=4, help='Softness of teacher logits')
    parser.add_argument("--lookback_window", type=int, default=1, help="Length of history fed to RNN")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--val_size", type=float, default=0.2, help="Fraction for validation")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction for testing")
    args = parser.parse_args()

    train_student_model(
        student_horizon=args.horizon, alpha=args.alpha, num_bins=args.num_bins,
        val_size=args.val_size, test_size=args.test_size, epochs=args.epochs,
        temperature=args.temperature, lookback_window=args.lookback_window,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
