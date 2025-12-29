import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

def training_interpretability(model, loader):
    for X_batch, y_batch in loader:
        X_tensor, y_tensor = X_batch.to("cuda"), y_batch.to("cuda")
    with torch.no_grad():
        predictions = model(X_tensor)
    predictions = F.softmax(predictions)
    predictions = predictions[:, 1].cpu().numpy()
    auc_test = roc_auc_score(y_tensor.cpu(), predictions)
    binary = np.where(predictions > 0.5, 1, 0)
    cm = confusion_matrix(y_tensor.cpu(), binary)
    print('Train AUC is:', auc_test)
    print('confusion matrix', cm)

class EarlyStopping:
    def __init__(self, patience=5, mode='min'):
        """
        Initialize the early stopping mechanism.

        :param patience: Number of epochs with no improvement to wait before stopping training.
        :param mode: 'min' to stop when the monitored loss doesn't decrease, 'max' if it doesn't increase.
        """
        self.patience = patience
        self.mode = mode
        self.best_loss = float('inf') if mode == 'min' else -float('inf')
        self.best_epoch = 0
        self.wait = 0
        self.stopped = False

    def step(self, loss, epoch):
        """
        Update the early stopping mechanism with the current loss and epoch.

        :param loss: The current loss value.
        :param epoch: The current epoch number.
        """
        if (self.mode == 'min' and loss < self.best_loss) or (self.mode == 'max' and loss > self.best_loss):
            self.best_loss = loss
            self.best_epoch = epoch
            self.wait = 0  # Reset wait counter as the loss improved
        else:
            self.wait += 1  # Increment wait counter

        # Check if we need to stop training
        if self.wait >= self.patience:
            self.stopped = True

    def is_stopped(self):
        """
        Return whether training should be stopped.

        :return: True if training should be stopped, otherwise False.
        """
        return self.stopped

def split_arrays_ictal(x_arr, y_arr, ratio):
    # Ensure y_arr is sorted
    assert np.all(y_arr[:-1] <= y_arr[1:]), "y_arr must be sorted"
    
    # Count the number of 1s and 2s in y_arr
    num_ones = np.sum(y_arr == 1)
    num_twos = np.sum(y_arr == 2)
    
    # Calculate the split index for 1s and 2s
    split_index_ones = int(num_ones * (1 - ratio))
    split_index_twos = int(num_twos * (1 - ratio))
    
    # Split y_arr
    y_part1 = np.concatenate((y_arr[:split_index_ones], y_arr[num_ones:num_ones + split_index_twos]))
    y_part2 = np.concatenate((y_arr[split_index_ones:num_ones], y_arr[num_ones + split_index_twos:]))
    
    # Split x_arr using the same indices
    x_part1 = np.concatenate((x_arr[:split_index_ones], x_arr[num_ones:num_ones + split_index_twos]))
    x_part2 = np.concatenate((x_arr[split_index_ones:num_ones], x_arr[num_ones + split_index_twos:]))
    
    return x_part1, y_part1, x_part2, y_part2