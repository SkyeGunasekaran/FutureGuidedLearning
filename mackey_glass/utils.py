import numpy as np
import torch
import math
from jitcdde import jitcdde, y, t, jitcdde_lyap
from torch.utils.data import Dataset
from torch.utils.data import Subset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import random
import numpy as np
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

class MackeyGlass(Dataset):
    """ Dataset for the Mackey-Glass task.
    """
    def __init__(self,
                 tau,
                 constant_past,
                 nmg = 10,
                 beta = 0.2,
                 gamma = 0.1,
                 dt=1.0,
                 splits=(8000., 2000.),
                 start_offset=0.,
                 seed_id=0,
    ):
        """
        Initializes the Mackey-Glass dataset.

        Args:
            tau (float): parameter of the Mackey-Glass equation
            constant_past (float): initial condition for the solver
            nmg (float): parameter of the Mackey-Glass equation
            beta (float): parameter of the Mackey-Glass equation
            gamma (float): parameter of the Mackey-Glass equation
            dt (float): time step length for sampling data
            splits (tuple): data split in time units for training and testing data, respectively
            start_offset (float): added offset of the starting point of the time-series, in case of repeating using same function values
            seed_id (int): seed for generating function solution
        """

        super().__init__()

        # Parameters
        self.tau = tau
        self.constant_past = constant_past
        self.nmg = nmg
        self.beta = beta
        self.gamma = gamma
        self.dt = dt

        # Time units for train (user should split out the warmup or validation)
        self.traintime = splits[0]
        # Time units to forecast
        self.testtime = splits[1]

        self.start_offset = start_offset
        self.seed_id = seed_id

        # Total time to simulate the system
        self.maxtime = self.traintime + self.testtime + self.dt

        # Discrete-time versions of the continuous times specified above
        self.traintime_pts = round(self.traintime/self.dt)
        self.testtime_pts = round(self.testtime/self.dt)
        self.maxtime_pts = self.traintime_pts + self.testtime_pts + 1 # eval one past the end

        # Specify the system using the provided parameters
        self.mackeyglass_specification = [ self.beta * y(0,t-self.tau) / (1 + y(0,t-self.tau)**self.nmg) - self.gamma*y(0) ]

        # Generate time-series
        self.generate_data()

        # Generate train/test indices
        self.split_data()


    def generate_data(self):
        """ Generate time-series using the provided parameters of the equation.
        """
        np.random.seed(self.seed_id)

        # Create the equation object based on the settings
        self.DDE = jitcdde_lyap(self.mackeyglass_specification)
        self.DDE.constant_past([self.constant_past])
        self.DDE.step_on_discontinuities()

        ##
        ## Generate data from the Mackey-Glass system
        ##
        self.mackeyglass_soln = torch.zeros((self.maxtime_pts,1),dtype=torch.float64)
        lyaps = torch.zeros((self.maxtime_pts,1),dtype=torch.float64)
        lyaps_weights = torch.zeros((self.maxtime_pts,1),dtype=torch.float64)
        count = 0
        for time in torch.arange(self.DDE.t+self.start_offset, self.DDE.t+self.start_offset+self.maxtime, self.dt,dtype=torch.float64):
            value, lyap, weight = self.DDE.integrate(time.item())
            self.mackeyglass_soln[count,0] = value[0]
            lyaps[count,0] = lyap[0]
            lyaps_weights[count,0] = weight
            count += 1

        # Total variance of the generated Mackey-Glass time-series
        self.total_var=torch.var(self.mackeyglass_soln[:,0], True)

        # Estimate Lyapunov exponent
        self.lyap_exp = ((lyaps.T@lyaps_weights)/lyaps_weights.sum()).item()


    def split_data(self):
        """ Generate training and testing indices.
        """
        self.ind_train = torch.arange(0, self.traintime_pts)
        self.ind_test = torch.arange(self.traintime_pts, self.maxtime_pts-1)

    def __len__(self):
        """ Returns number of samples in dataset.

        Returns:
            int: number of samples in dataset
        """
        return len(self.mackeyglass_soln)-1

    def __getitem__(self, idx):
        """ Getter method for dataset.

        Args:
            idx (int): index of sample to return

        Returns:
            sample (tensor): individual data sample, shape=(timestamps, features)=(1,1)
            target (tensor): corresponding next state of the system, shape=(label,)=(1,)
        """
        sample = torch.unsqueeze(self.mackeyglass_soln[idx, :], dim=0)
        target = self.mackeyglass_soln[idx+1, :]

        return sample, target
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        out, hidden = self.rnn(x, h0)
        out = self.fc1(out[:, -1, :])
        out = nn.functional.relu(out)
        out = self.fc2(out)
        return out

def create_time_series_dataset(data, lookback_window, forecasting_horizon, num_bins, test_size, offset=0, MSE=False):
    x = np.array([point[0] for point in data])
    y = np.array([point[1] for point in data])
    x_processed = []
    y_processed = []

    for i in range(len(x) - lookback_window - forecasting_horizon + 1):
        x_window = x[i:i + lookback_window]
        y_value = y[i + lookback_window + forecasting_horizon - 1]
        x_processed.append(x_window)
        y_processed.append(y_value)

    X_train, X_test, y_train, y_test = train_test_split(x_processed, y_processed, test_size=test_size, shuffle=False)
    original_data_test = y_test.copy()
    bin_edges = np.linspace(np.min(y), np.max(y), num_bins - 1)
    if not MSE:
        X_train = np.digitize(X_train, bin_edges)
        X_test = np.digitize(X_test, bin_edges)
        y_train = np.digitize(y_train, bin_edges)
        y_test = np.digitize(y_test, bin_edges)

    train  = [(X_train[i].squeeze(-1), y_train[i]) for i in range(offset, len(X_train))]
    test = [(X_test[i].squeeze(-1), y_test[i]) for i in range(offset, len(X_test))]

    # Create DataLoaders
    train_loader = DataLoader(train, batch_size=1, shuffle=False)
    test_loader = DataLoader(test, batch_size=1, shuffle=False)

    return train_loader, test_loader, original_data_test, y_test

def KL(outputs, logits, alpha, kl, T):
    return (1-alpha)*kl(F.log_softmax(outputs / T, dim=1), F.softmax(logits / T, dim=1)) * T**2

def plot_predictions(predictions_teacher, true_values_teacher, predictions_baseline, true_values_baseline, predictions_student, true_values_student, original_data_train, y_train):
    # Configure font
    font_manager.fontManager.addfont('C:/Users/gunak/Downloads/helvetica-255/Helvetica.ttf')
    prop = font_manager.FontProperties(fname='C:/Users/gunak/Downloads/helvetica-255/Helvetica.ttf')
    plt.rcParams['font.family'] = 'Helvetica'
    plt.rcParams['font.sans-serif'] = prop.get_name()
    plt.rcParams['font.size'] = 18  # Set general font size

    # Create a figure with 4 subplots for Discretization, Teacher, Baseline, and Student models
    fig, ax = plt.subplots(1, 4, figsize=(24, 6))

    # Plot Discretization Process (Original vs Discretized y_train) using twinx() for y-axes
    ax_left = ax[0]  # Left axis for original data
    ax_right = ax_left.twinx()  # Right axis for discretized data

    # Plot the original data on the left y-axis
    original_line, = ax_left.plot(original_data_train, label="Original data", color='blue')
    ax_left.set_ylabel("Amplitude", fontsize=18)
    ax_left.tick_params(axis='y')

    # Plot the discretized data on the right y-axis
    discretized_line, = ax_right.plot(y_train, label="Bin Number", color='green')
    ax_right.set_ylabel("Discretized Values", fontsize=18)
    ax_right.tick_params(axis='y')

    # Set the title and labels for the x-axis
    ax_left.set_title("Discretization Process", fontsize=18)
    ax_left.set_xlabel("Time", fontsize=18)

    # Combine the two legends into one and position it at the bottom right
    ax_left.legend(handles=[original_line, discretized_line], loc="lower right")

    # Plot Teacher predictions
    ax[1].plot(true_values_teacher, label="True Values", color='blue')
    ax[1].plot(predictions_teacher, label="Predictions", color='red')
    ax[1].set_title("Teacher", fontsize=18)
    ax[1].set_xlabel("Time", fontsize=18)
    ax[1].set_ylabel("Amplitude", fontsize=18)
    ax[1].legend(loc="lower right")

    # Plot Baseline predictions
    ax[2].plot(true_values_baseline, label="True Values", color='blue')
    ax[2].plot(predictions_baseline, label="Predictions", color='red')
    ax[2].set_title("Baseline", fontsize=18)
    ax[2].set_xlabel("Time", fontsize=18)
    ax[2].set_ylabel("Amplitude", fontsize=18)
    ax[2].legend(loc="lower right")

    # Plot Student predictions
    ax[3].plot(true_values_student, label="True Values", color='blue')
    ax[3].plot(predictions_student, label="Predictions", color='red')
    ax[3].set_title("Student (FGL)", fontsize=18)
    ax[3].set_xlabel("Time", fontsize=18)
    ax[3].set_ylabel("Amplitude", fontsize=18)
    ax[3].legend(loc="lower right")
    
    fig.text(0.1285, 0.02, '(a)', ha='center', fontsize=18)  # Under subplot 1 (Discretization)
    fig.text(0.385, 0.02, '(b)', ha='center', fontsize=18)   # Under subplot 2 (Teacher)
    fig.text(0.64, 0.02, '(c)', ha='center', fontsize=18)   # Under subplot 3 (Baseline)
    fig.text(0.89, 0.02, '(d)', ha='center', fontsize=18)   # Under subplot 4 (Student)


    # Adjust layout and show the plot

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust rect to fit (a), (b), (c), (d) labels
    plt.savefig('nature-fig_4.pdf', dpi=400)
    plt.show()

# Example usage:
# plot_predictions(predictions_teacher, true_values_teacher, predictions_baseline, true_values_baseline, predictions_student, true_values_student, original_data_train, y_train)
