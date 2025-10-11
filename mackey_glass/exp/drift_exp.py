#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore", message="PyTorch is not compiled with NCCL support")

import os
import argparse
import pickle
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from utils import RNN, create_time_series_dataset, KL

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# --- Utility Classes and Functions ---

class EarlyStopper:
    """Early stops training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_state = None

    def step(self, current_loss, model):
        """Checks if training should stop based on validation loss."""
        if current_loss + self.min_delta < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
            self.best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore(self, model):
        """Loads the best model state found during training."""
        if self.best_state:
            model.load_state_dict(self.best_state)

def page_hinkley_update(error, state, delta):
    """Performs a single update step for the Page-Hinkley test."""
    state['t'] += 1
    m_prev = state['m']
    state['m'] += (error - state['m']) / state['t']
    state['PH'] = max(0.0, state['PH'] + (error - m_prev - delta))
    return state

def _evaluate_with_page_hinkley(model, loader, args):
    """Evaluates a model using the Page-Hinkley drift detection and retraining method."""
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    state = {'m': 0.0, 'PH': 0.0, 't': 0}
    window = deque(maxlen=args['window_size'])
    errors = []

    for _, x, y in loader:
        x = x.float().to(device).view(-1, 1, args['lookback_window'])
        y_int, y_float = y.long().to(device).squeeze(-1), y.float().to(device).squeeze(-1)

        with torch.no_grad():
            pred_class = model(x).argmax(dim=1)
            err = F.mse_loss(pred_class.float(), y_float).item()
        
        errors.append(err)
        state = page_hinkley_update(err, state, args['delta'])
        window.append((x.cpu(), y_int.cpu()))

        # If drift is detected, retrain on the current window
        if state['PH'] > args['lambda_thr'] and len(window) == args['window_size']:
            model.train()
            for _ in range(args['retrain_epochs']):
                for wx, wy in window:
                    optimizer.zero_grad()
                    loss = F.cross_entropy(model(wx.to(device)), wy.to(device))
                    loss.backward()
                    optimizer.step()
            model.eval()
            state = {'m': 0.0, 'PH': 0.0, 't': 0} # Reset state
            window.clear()
            
    return sum(errors) / len(errors)

def evaluate(model, loader, use_ph=False, **kwargs):
    """Dispatches to the correct evaluation method (standard or Page-Hinkley)."""
    model.eval()
    if not use_ph:
        total_mse = 0.0
        with torch.no_grad():
            for _, x, y in loader:
                x = x.float().to(device).view(-1, 1, kwargs['lookback_window'])
                y_float = y.float().to(device).squeeze(-1)
                pred = model(x).argmax(dim=1).float()
                total_mse += F.mse_loss(pred, y_float).item()
        return total_mse / len(loader)
    else:
        return _evaluate_with_page_hinkley(model, loader, kwargs)

def _run_training_loop(model, train_loader, val_loader, optimizer, epochs, patience, lookback_window, model_name, is_distillation=False, **kwargs):
    """A generic training and validation loop with early stopping."""
    stopper = EarlyStopper(patience=patience)
    for epoch in range(epochs):
        model.train()
        # --- Training Step ---
        for batch_data in train_loader:
            if is_distillation:
                (_, x_s, y_s), (_, x_t, _) = batch_data
                targets = y_s.long().to(device).squeeze(-1)
                with torch.no_grad():
                    teacher_logits = kwargs['teacher'](x_t.float().to(device).view(-1, 1, lookback_window))
                outputs = model(x_s.float().to(device).view(-1, 1, lookback_window))
                loss = kwargs['alpha'] * F.cross_entropy(outputs, targets) + KL(outputs, teacher_logits, kwargs['temp'], kwargs['alpha'])
            else:
                _, x, y = batch_data
                x, y = x.float().to(device).view(-1, 1, lookback_window), y.long().to(device).squeeze(-1)
                loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        # --- Validation Step ---
        val_loss = evaluate(model, val_loader, lookback_window=lookback_window)
        if stopper.step(val_loss, model):
            print(f"[{model_name}] Early stopping at epoch {epoch+1}")
            break
    
    stopper.restore(model)
    return model

def train_student_model(student_horizon, alpha, num_bins, val_size, test_size, epochs, temperature, lookback_window, batch_size, use_ph=False, ph_delta=0.005, ph_lambda=1.0, ph_window=50, ph_retrain_epochs=3, patience=5):
    """Trains and compares Teacher, Baseline, and Student models."""
    torch.manual_seed(42)
    print(f"\nTraining | Horizon={student_horizon}, Alpha={alpha}, Use_PH={use_ph}")

    hyperparams = {'hidden_size': 128, 'output_size': num_bins, 'num_layers': 2, 'lr': 1e-4}
    with open("data.pkl", "rb") as f: data = pickle.load(f)

    common_args = {'data': data, 'lookback_window': lookback_window, 'num_bins': num_bins, 'batch_size': batch_size, 'val_size': val_size, 'test_size': test_size}
    teacher_train, teacher_val, teacher_test, _, _ = create_time_series_dataset(**common_args, forecasting_horizon=1, offset=student_horizon - 1)
    student_train, student_val, student_test, _, _ = create_time_series_dataset(**common_args, forecasting_horizon=student_horizon, offset=0)
    
    # --- Train Models ---
    teacher = RNN(lookback_window, **hyperparams).to(device)
    teacher = _run_training_loop(teacher, teacher_train, teacher_val, optim.Adam(teacher.parameters(), lr=hyperparams['lr']), epochs, patience, lookback_window, "Teacher")
    
    baseline = RNN(lookback_window, **hyperparams).to(device)
    baseline = _run_training_loop(baseline, student_train, student_val, optim.Adam(baseline.parameters(), lr=hyperparams['lr']), epochs, patience, lookback_window, "Baseline")

    student = RNN(lookback_window, **hyperparams).to(device)
    distill_args = {'teacher': teacher, 'alpha': alpha, 'temp': temperature}
    student = _run_training_loop(student, zip(student_train, teacher_train), student_val, optim.Adam(student.parameters(), lr=hyperparams['lr']), epochs, patience, lookback_window, "Student", is_distillation=True, **distill_args)

    # --- Final Evaluation ---
    ph_args = {'delta': ph_delta, 'lambda_thr': ph_lambda, 'window_size': ph_window, 'retrain_epochs': ph_retrain_epochs, 'lr': hyperparams['lr'], 'lookback_window': lookback_window}
    teacher_mse = evaluate(teacher, teacher_test, use_ph, **ph_args)
    baseline_mse = evaluate(baseline, student_test, use_ph, **ph_args)
    student_mse = evaluate(student, student_test, use_ph, **ph_args)

    print(f"\nFinal Test MSE:\n Teacher:  {teacher_mse:.4f}\n Baseline: {baseline_mse:.4f}\n Student:  {student_mse:.4f}")

def main():
    """Parses arguments and runs the training and evaluation process."""
    parser = argparse.ArgumentParser(description="Future-Guided Learning for Time-Series Forecasting")
    parser.add_argument("--horizon", type=int, required=True, help="Student horizon (H)")
    parser.add_argument("--alpha", type=float, required=True, help="Loss weight α")
    # Add other arguments...
    parser.add_argument("--num_bins", type=int, default=50, help="Number of bins")
    parser.add_argument("--val_size", type=float, default=0.2, help="Fraction for validation")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction for testing")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--temperature", type=float, default=4, help="Softness of teacher logits")
    parser.add_argument("--lookback_window", type=int, default=1, help="Length of history fed to RNN")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--use_ph", action="store_true", help="Enable Page-Hinkley test-time retraining")
    parser.add_argument("--ph_delta", type=float, default=0.005, help="PH delta threshold")
    parser.add_argument("--ph_lambda", type=float, default=1.0, help="PH retrain threshold")
    parser.add_argument("--ph_window", type=int, default=50, help="PH retraining window size")
    parser.add_argument("--ph_retrain_epochs", type=int, default=3, help="PH retrain epochs per drift")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    args = parser.parse_args()

    train_student_model(**vars(args))

if __name__ == "__main__":
    main()