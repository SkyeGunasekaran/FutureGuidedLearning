#!/usr/bin/env python
import warnings
warnings.filterwarnings(
    "ignore",
    message="PyTorch is not compiled with NCCL support"
)

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

from utils.utils import RNN, create_time_series_dataset, KL

# —— Device setup ——
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

class EarlyStopper:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float('inf')
        self.counter    = 0
        self.best_state = None

    def step(self, current_loss, model):
        if current_loss + self.min_delta < self.best_loss:
            self.best_loss  = current_loss
            self.counter    = 0
            # always store CPU copy of best weights
            self.best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            return False
        else:
            self.counter += 1
            return (self.counter >= self.patience)

    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)

def page_hinkley_update(error, state, delta):
    state['t'] += 1
    m_prev = state['m']
    state['m'] += (error - state['m']) / state['t']
    state['PH'] = max(0.0, state['PH'] + (error - m_prev - delta))
    return state


def evaluate(model, loader, use_ph=False,
             delta=0.005, lambda_thr=1.0,
             window_size=50, retrain_epochs=3,
             lr=1e-4, lookback_window=1):
    """
    Standard or Page–Hinkley–driven evaluation loop.
    If use_ph is False, returns plain MSE.
    If use_ph is True, applies PH drift detection and retraining.
    """
    mse_loss = nn.MSELoss()
    celoss   = nn.CrossEntropyLoss()
    model.eval()

    if not use_ph:
        total = 0.0
        with torch.no_grad():
            for _, x, y in loader:
                x = x.float().to(device).view(-1,1,lookback_window)
                y_int = y.long().to(device).squeeze(-1)
                pred = model(x).argmax(dim=1).float()
                total += mse_loss(pred, y_int.float()).item()
        return total / len(loader)

    # Page–Hinkley–driven evaluation
    optimizer = optim.Adam(model.parameters(), lr=lr)
    state = {'m':0.0, 'PH':0.0, 't':0}
    window = deque(maxlen=window_size)
    errors = []

    for _, x, y in loader:
        x = x.float().to(device).view(-1,1,lookback_window)
        # keep both int and float targets
        y_int   = y.long().to(device).squeeze(-1)
        y_float = y_int.float()

        with torch.no_grad():
            pred_class = model(x).argmax(dim=1)
        err = mse_loss(pred_class.float(), y_float).item()
        errors.append(err)
        state = page_hinkley_update(err, state, delta)
        # store int targets for retraining
        window.append((x.cpu(), y_int.cpu()))

        if state['PH'] > lambda_thr and len(window) == window_size:
            model.train()
            for _ in range(retrain_epochs):
                for wx, wy_int in window:
                    wx = wx.to(device)
                    wy = wy_int.to(device)
                    out = model(wx)
                    loss = celoss(out, wy)
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
            model.eval()
            state = {'m':0.0, 'PH':0.0, 't':0}
            window.clear()

    return sum(errors) / len(errors)

def train_student_model(student_horizon, alpha, num_bins, val_size, test_size, 
                        epochs, temperature, lookback_window, batch_size,
                        use_ph=False, ph_delta=0.005, ph_lambda=1.0, ph_window=50, 
                        ph_retrain_epochs=3, patience=5):
    torch.manual_seed(42)
    print(f"\nTraining | Horizon={student_horizon} Alpha={alpha} "
          f"Epochs={epochs} Temp={temperature} "
          f"Use_PH={use_ph} Delta={ph_delta} Lambda={ph_lambda}")

    # hyperparams
    hidden_size = 128
    output_size = num_bins
    num_layers  = 2
    lr          = 1e-4

    # load data
    with open("data.pkl", "rb") as f:
        data = pickle.load(f)

    # prepare datasets
    # teacher (1-step, offset=H-1)
    teacher_train, teacher_val, teacher_test, _, _ = create_time_series_dataset(
        data=data, 
        lookback_window=lookback_window, 
        forecasting_horizon=1,
        num_bins=num_bins, 
        val_size=val_size,
        test_size=test_size,
        offset=student_horizon-1, 
        batch_size=batch_size
    )
    # student (H-step, offset=0)
    student_train, student_val, student_test, _, _ = create_time_series_dataset(
        data=data, 
        lookback_window=lookback_window, 
        forecasting_horizon=student_horizon,
        num_bins=num_bins, 
        val_size=val_size,
        test_size=test_size,
        offset=0, 
        batch_size=batch_size
    )

    mse    = nn.MSELoss()
    celoss = nn.CrossEntropyLoss()

    # —— Teacher training ——
    teacher = RNN(lookback_window, hidden_size, output_size, num_layers).to(device)
    opt_t   = optim.Adam(teacher.parameters(), lr=lr)
    stop_t  = EarlyStopper(patience=patience)

    for epoch in range(epochs):
        teacher.train()
        for _, x, y in teacher_train:
            x = x.float().to(device).view(-1,1,lookback_window)
            y = y.long().to(device)
            loss = celoss(teacher(x), y)
            opt_t.zero_grad(); loss.backward(); opt_t.step()

        # validation
        teacher.eval()
        val_loss = 0.0
        with torch.no_grad():
            for _, x, y in teacher_val:
                x = x.float().to(device).view(-1,1,lookback_window)
                y = y.long().to(device)
                pred = teacher(x)
                val_loss += celoss(pred, y).item()
        val_loss /= len(teacher_val)

        if stop_t.step(val_loss, teacher):
            print(f"[Teacher] Early stopping at epoch {epoch+1}")
            break

    stop_t.restore(teacher)
    #print(f"[Teacher] Best Val MSE = {stop_t.best_loss:.4f}")

    # —— Baseline training ——
    baseline = RNN(lookback_window, hidden_size, output_size, num_layers).to(device)
    opt_b    = optim.Adam(baseline.parameters(), lr=lr)
    stop_b   = EarlyStopper(patience=patience)

    for epoch in range(epochs):
        baseline.train()
        for _, x, y in student_train:
            x = x.float().to(device).view(-1,1,lookback_window)
            y = y.long().to(device)
            out = baseline(x)
            loss = celoss(out, y)
            opt_b.zero_grad(); loss.backward(); opt_b.step()

        baseline.eval()
        val_loss = 0.0
        with torch.no_grad():
            for _, x, y in student_val:
                x = x.float().to(device).view(-1,1,lookback_window)
                y = y.long().to(device)
                pred = baseline(x)
                val_loss += celoss(pred, y).item()
        val_loss /= len(student_val)

        if stop_b.step(val_loss, baseline):
            print(f"[Baseline] Early stopping at epoch {epoch+1}")
            break

    stop_b.restore(baseline)
    #print(f"[Baseline] Best Val MSE = {stop_b.best_loss:.4f}")

    # —— Student training ——
    student = RNN(lookback_window, hidden_size, output_size, num_layers).to(device)
    opt_s    = optim.Adam(student.parameters(), lr=lr)
    stop_s   = EarlyStopper(patience=patience)

    for epoch in range(epochs):
        student.train()
        for (i_s, x_s, y_s), (_, x_t, _) in zip(student_train, teacher_train):
            x_s = x_s.float().to(device).view(-1,1,lookback_window)
            y_s = y_s.long().to(device)
            with torch.no_grad():
                logits_t = teacher(x_t.float().to(device).view(-1,1,lookback_window))
            loss = alpha * celoss(student(x_s), y_s) + KL(student(x_s), logits_t, temperature, alpha)
            opt_s.zero_grad(); loss.backward(); opt_s.step()

        student.eval()
        val_loss = 0.0
        with torch.no_grad():
            for _, x, y in student_val:
                x = x.float().to(device).view(-1,1,lookback_window)
                y = y.long().to(device)
                pred = student(x)
                val_loss += celoss(pred, y).item()
        val_loss /= len(student_val)

        if stop_s.step(val_loss, student):
            print(f"[Student] Early stopping at epoch {epoch+1}")
            break

    stop_s.restore(student)
    #print(f"[Student] Best Val MSE = {stop_s.best_loss:.4f}")

    # —— Final evaluation ——
    teacher_mse  = evaluate(teacher, teacher_test, use_ph,
                             ph_delta, ph_lambda, ph_window,
                             ph_retrain_epochs, lr, lookback_window)
    baseline_mse = evaluate(baseline, student_test, use_ph,
                             ph_delta, ph_lambda, ph_window,
                             ph_retrain_epochs, lr, lookback_window)
    student_mse  = evaluate(student, student_test, use_ph,
                             ph_delta, ph_lambda, ph_window,
                             ph_retrain_epochs, lr, lookback_window)

    print(f"Teacher Test MSE:  {teacher_mse:.4f}")
    print(f"Baseline Test MSE: {baseline_mse:.4f}")
    print(f"Student Test MSE:  {student_mse:.4f}")


if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser(
        description="Future-Guided Learning for Time-Series Forecasting"
    )
    parser.add_argument("--horizon", type=int, required=True, 
                        help="Student horizon (N)")
    parser.add_argument("--alpha",   type=float, required=True, 
                        help="Loss weight α")
    parser.add_argument("--num_bins",type=int, default=50,   
                        help="Number of bins")
    parser.add_argument("--val_size",  type=float, default=0.2,
                        help="Fraction of data for validation")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of data for testing")
    parser.add_argument("--epochs",  type=int, default=10,   
                        help="Training epochs")
    parser.add_argument("--temperature", type=float, default=4,
                        help="Controls softness of teacher logits")
    parser.add_argument("--lookback_window", type=int, default=1,
                        help="Length of history fed to RNN")
    parser.add_argument("--batch_size",      type=int, default=1,
                        help="Batch size for all loaders")
    parser.add_argument("--use_ph", action="store_true",
                        help="Enable Page–Hinkley–driven test-time retraining")
    parser.add_argument("--ph_delta", type=float, default=0.005,
                        help="PH delta threshold")
    parser.add_argument("--ph_lambda", type=float, default=1.0,
                        help="PH retrain threshold")
    parser.add_argument("--ph_window", type=int, default=50,
                        help="PH retraining window size")
    parser.add_argument("--ph_retrain_epochs", type=int, default=3,
                        help="PH retrain epochs per drift")
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    train_student_model(
        student_horizon    = args.horizon,
        alpha              = args.alpha,
        num_bins           = args.num_bins,
        val_size           = args.val_size,
        test_size          = args.test_size,
        epochs             = args.epochs,
        temperature        = args.temperature,
        lookback_window    = args.lookback_window,
        batch_size         = args.batch_size,
        use_ph             = args.use_ph,
        ph_delta           = args.ph_delta,
        ph_lambda          = args.ph_lambda,
        ph_window          = args.ph_window,
        ph_retrain_epochs  = args.ph_retrain_epochs,
        patience           = args.patience
    )
    
    PH Params
    Bins=25
    delta=0.130033
    lambda_thr=0.647096
    Bins=50
    delta=5.775345
    lambda_thr=7.836623
    '''
    for i in range(2, 16):
        train_student_model(student_horizon=i,
                            alpha=0.0,
                            num_bins=25,
                            val_size=0.2,
                            test_size=0.2,
                            epochs=50,
                            temperature=4,
                            lookback_window=8,
                            batch_size=128,
                            use_ph=True,
                            ph_delta=0.130033,
                            ph_lambda=0.647096,
                            ph_window=3,
                            ph_retrain_epochs=3,
                            patience=5)
    
