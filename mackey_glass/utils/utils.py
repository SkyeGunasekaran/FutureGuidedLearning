import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, Dataset, DataLoader
from jitcdde import jitcdde_lyap, y, t
import matplotlib.pyplot as plt

# --- Dataset and Model Definitions ---

class MackeyGlass(Dataset):
    """
    Generates and serves the Mackey-Glass time-series dataset using jitcdde.
    """
    def __init__(self, tau=17, constant_past=0.9, nmg=10, beta=0.2, gamma=0.1, 
                 dt=1.0, splits=(8000., 2000.), start_offset=0., seed_id=0):
        super().__init__()
        self.params = {
            'tau': tau, 'constant_past': constant_past, 'nmg': nmg,
            'beta': beta, 'gamma': gamma, 'dt': dt, 'seed_id': seed_id,
            'start_offset': start_offset
        }
        
        # Calculate time points
        traintime, testtime = splits
        maxtime = traintime + testtime + dt
        self.traintime_pts = round(traintime / dt)
        self.maxtime_pts = round(maxtime / dt)

        self._generate_data()
        self._split_data()

    def _generate_data(self):
        """Generates the time-series using the jitcdde solver."""
        np.random.seed(self.params['seed_id'])
        
        spec = [
            self.params['beta'] * y(0, t - self.params['tau']) /
            (1 + y(0, t - self.params['tau'])**self.params['nmg']) -
            self.params['gamma'] * y(0)
        ]
        DDE = jitcdde_lyap(spec)
        DDE.constant_past([self.params['constant_past']])
        DDE.step_on_discontinuities()

        self.mackeyglass_soln = torch.zeros((self.maxtime_pts, 1), dtype=torch.float64)
        times = torch.arange(
            DDE.t + self.params['start_offset'],
            DDE.t + self.params['start_offset'] + self.maxtime_pts * self.params['dt'],
            self.params['dt'],
            dtype=torch.float64
        )
        
        for i, time_val in enumerate(times):
            value, _, _ = DDE.integrate(time_val.item())
            self.mackeyglass_soln[i, 0] = value[0]

    def _split_data(self):
        """Generates training and testing indices."""
        self.ind_train = torch.arange(0, self.traintime_pts)
        self.ind_test = torch.arange(self.traintime_pts, self.maxtime_pts - 1)

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.mackeyglass_soln) - 1

    def __getitem__(self, idx):
        """Returns a single sample and its corresponding target."""
        sample = self.mackeyglass_soln[idx, :].unsqueeze(0)
        target = self.mackeyglass_soln[idx + 1, :]
        return sample, target

class RNN(nn.Module):
    """A simple multi-layer RNN with a linear output layer for regression."""
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, lr=None):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        # Pass the output of the last time step to the classifier
        out = self.fc(out[:, -1, :])
        return out

# --- Data Preparation ---

def _create_sliding_windows_from_series(data, lookback_window, forecasting_horizon):
    """
    Creates sliding windows from a raw time-series.
    
    Args:
        data (np.array): The raw time-series, shape (N,).
        lookback_window (int): Number of past steps to use as input (X).
        forecasting_horizon (int): The future step to predict (y). 
                                 e.g., horizon=1 means predict t+1.
    
    Returns:
        (np.array, np.array): X, y
    """
    X, y = [], []
    N = len(data)
    
    # End point for window creation
    # We need 'lookback_window' steps for X and 'forecasting_horizon' steps to find y
    end_idx = N - lookback_window - forecasting_horizon + 1
    
    for i in range(end_idx):
        X.append(data[i : i + lookback_window])
        y.append(data[i + lookback_window + forecasting_horizon - 1]) # Predict the single point at the horizon
    
    return np.array(X), np.array(y)


def create_time_series_dataset(data, lookback_window, forecasting_horizon, num_bins,
                               val_size, test_size, offset=0, MSE=False, batch_size=1):
    """
    Generates train/val/test DataLoaders from a raw time-series tensor.
    
    Args:
        data (torch.Tensor or np.array): The raw time-series data, 
                                         shape (N, 1) or (N,).
    """
    
    # 1. Ensure data is a 1D numpy array
    if hasattr(data, 'numpy'): # Check if it's a torch tensor
        data = data.numpy()
    data = data.squeeze() # Convert from (N, 1) to (N,)
    if data.ndim != 1:
        raise ValueError(f"Input data must be a 1D time-series, but got shape {data.shape}")

    # 2. Create sliding windows from the raw series
    X, y = _create_sliding_windows_from_series(data, lookback_window, forecasting_horizon)
        
    # 3. Chronological split
    N = X.shape[0]
    if N == 0:
        raise ValueError("Not enough data to create any sliding windows. "
                         "Check data length, lookback, and horizon.")
        
    n_test = int(N * test_size)
    n_val = int(N * val_size)
    n_train = N - n_val - n_test
    
    if n_train <= 0:
        raise ValueError("Not enough data for a training split. "
                         "Adjust sizes or increase dataset length.")

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[-n_test:], y[-n_test:]

    original_data_val, original_data_test = y_val.copy(), y_test.copy()

    # 4. Discretize targets
    if not MSE:
        bin_edges = np.linspace(y_train.min(), y_train.max(), num_bins)
        y_train = np.digitize(y_train, bin_edges)
        y_val = np.digitize(y_val, bin_edges)
        y_test = np.digitize(y_test, bin_edges)

    # 5. Create DataLoaders
    def create_loader(X_arr, y_arr):
        if offset > 0:
            X_arr, y_arr = X_arr[offset:], y_arr[offset:]
        
        # Convert to Tensors for DataLoader
        # We add a channel/feature dimension to X, as most models (RNN, CNN)
        # expect input shape (batch, sequence_length, features)
        X_tensor = torch.from_numpy(X_arr).float().unsqueeze(-1) # Shape: (batch, lookback, 1)
        y_tensor = torch.from_numpy(y_arr).float()               # Shape: (batch,)
        
        if not MSE:
            y_tensor = y_tensor.long() # Use Long for classification labels
            
        # Using TensorDataset is much more efficient than list(zip(...))
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    train_loader = create_loader(X_train, y_train)
    val_loader = create_loader(X_val, y_val)
    test_loader = create_loader(X_test, y_test)

    return train_loader, val_loader, test_loader, original_data_val, original_data_test

# --- Utilities ---

def KL(student_logits, teacher_logits, temperature, alpha):
    """
    Computes the knowledge distillation loss component.
    Returns: (1–alpha) * T^2 * KL( softmax(teacher/T) || log_softmax(student/T) )
    """
    log_p_student = F.log_softmax(student_logits / temperature, dim=1)
    p_teacher = F.softmax(teacher_logits / temperature, dim=1)
    
    # Batchmean KL divergence, scaled by T^2 as in Hinton's paper
    kd_loss = F.kl_div(log_p_student, p_teacher, reduction='batchmean') * (temperature ** 2)
    return (1.0 - alpha) * kd_loss
