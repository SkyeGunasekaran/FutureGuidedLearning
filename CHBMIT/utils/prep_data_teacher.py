import numpy as np
from sklearn.utils import shuffle
from utils.model_stuff import split_arrays_ictal

# --- Helper Functions ---

def _flatten_interictal_data(interictal_X, interictal_y):
    """Checks if interictal data is a list of arrays and concatenates if so."""
    if isinstance(interictal_y, list):
        interictal_X = np.concatenate(interictal_X, axis=0)
        interictal_y = np.concatenate(interictal_y, axis=0)
    return interictal_X, interictal_y

def _finalize_split_arrays(train_lists, val_lists, test_lists):
    """Concatenates lists of arrays into final training, validation, and testing sets."""
    X_train = np.concatenate(train_lists[0], axis=0) if train_lists[0] else np.array([])
    y_train = np.concatenate(train_lists[1], axis=0) if train_lists[1] else np.array([])
    
    X_val = np.concatenate(val_lists[0], axis=0) if val_lists[0] else np.array([])
    y_val = np.concatenate(val_lists[1], axis=0) if val_lists[1] else np.array([])
    
    X_test = np.concatenate(test_lists[0], axis=0) if test_lists[0] else np.array([])
    y_test = np.concatenate(test_lists[1], axis=0) if test_lists[1] else np.array([])
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# --- Main Splitting Functions ---

def train_val_test_split_continual_t(ictal_X, ictal_y, interictal_X, interictal_y, test_ratio, val_ratio, no_test=False):
    """
    Splits data chronologically: earlier seizures for training, middle for validation,
    and later seizures for testing.
    """
    num_sz = len(ictal_y)
    
    if no_test:
        num_sz_test = 0
    else:
        num_sz_test = int(test_ratio * num_sz)
        
    # Calculate validation seizures from the remaining portion
    num_sz_val = int(val_ratio * num_sz)
    
    # Calculate training seizures
    num_sz_train = num_sz - num_sz_test - num_sz_val

    if num_sz_train <= 0:
        raise ValueError("test_ratio and val_ratio sum is too large, or ratios are invalid. No training data left.")

    # Define split indices
    val_start_index = num_sz_train
    test_start_index = num_sz_train + num_sz_val

    print(f'Total {num_sz} seizures, split as:')
    print(f'  {num_sz_train} train (Seizures 0 to {val_start_index - 1})')
    print(f'  {num_sz_val} val   (Seizures {val_start_index} to {test_start_index - 1})')
    print(f'  {num_sz_test} test  (Seizures {test_start_index} to {num_sz - 1})')

    interictal_X, interictal_y = _flatten_interictal_data(interictal_X, interictal_y)
    interictal_fold_len = int(round(interictal_y.shape[0] / num_sz))
    print(f'Length of each interictal segment: {interictal_fold_len}')

    # Initialize lists for all splits
    X_train_all, y_train_all = [], []
    X_val_all, y_val_all = [], []
    X_test_all, y_test_all = [], []

    for i in range(num_sz):
        # Combine the i-th ictal seizure with its corresponding interictal segment
        start, end = i * interictal_fold_len, (i + 1) * interictal_fold_len
        X_combined = np.concatenate((interictal_X[start:end], ictal_X[i]), axis=0)
        y_combined = np.concatenate((interictal_y[start:end], ictal_y[i]), axis=0)

        # Assign to train, val, or test set based on chronological order
        if i < val_start_index:
            # --- TRAINING DATA ---
            # map oversampled labels (2, -1) to respective classes (1, 0)
            y_combined[y_combined == 2] = 1
            y_combined[y_combined == -1] = 0
            X_train_all.append(X_combined)
            y_train_all.append(y_combined)
            
        elif i < test_start_index:
            # --- VALIDATION DATA ---
            # Process like test data: remove all oversampled samples (labels 2, -1)
            mask = (y_combined != 2) & (y_combined != -1)
            X_val_all.append(X_combined[mask])
            y_val_all.append(y_combined[mask])
            
        else:
            # --- TESTING DATA ---
            # remove all oversampled samples (labels 2, -1)
            mask = (y_combined != 2) & (y_combined != -1)
            X_test_all.append(X_combined[mask])
            y_test_all.append(y_combined[mask])
            
    return _finalize_split_arrays(
        (X_train_all, y_train_all),
        (X_val_all, y_val_all),
        (X_test_all, y_test_all)
    )