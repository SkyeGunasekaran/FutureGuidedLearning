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

def _finalize_split_arrays(train_lists, test_lists, no_test=False):
    """Concatenates lists of arrays into final training and testing sets."""
    X_train = np.concatenate(train_lists[0], axis=0) if train_lists[0] else np.array([])
    y_train = np.concatenate(train_lists[1], axis=0) if train_lists[1] else np.array([])
    
    if no_test:
        return X_train, y_train
        
    X_test = np.concatenate(test_lists[0], axis=0) if test_lists[0] else np.array([])
    y_test = np.concatenate(test_lists[1], axis=0) if test_lists[1] else np.array([])
    
    return X_train, y_train, X_test, y_test

# --- Main Splitting Functions ---

def train_val_test_split_continual_t(ictal_X, ictal_y, interictal_X, interictal_y, test_ratio, no_test=False):
    """
    Splits data chronologically: earlier seizures for training, later seizures for testing.
    """
    num_sz = len(ictal_y)
    
    if no_test:
        num_sz_test = 0
        print(f'Total {num_sz} seizures, all are used for training.')
    else:
        num_sz_test = max(1, int(test_ratio * num_sz))
        print(f'Total {num_sz} seizures, last {num_sz_test} is used for testing.')

    interictal_X, interictal_y = _flatten_interictal_data(interictal_X, interictal_y)
    interictal_fold_len = int(round(interictal_y.shape[0] / num_sz))
    print(f'Length of each interictal segment: {interictal_fold_len}')

    X_train_all, y_train_all, X_test_all, y_test_all = [], [], [], []

    for i in range(num_sz):
        # Combine the i-th ictal seizure with its corresponding interictal segment
        start, end = i * interictal_fold_len, (i + 1) * interictal_fold_len
        X_combined = np.concatenate((interictal_X[start:end], ictal_X[i]), axis=0)
        y_combined = np.concatenate((interictal_y[start:end], ictal_y[i]), axis=0)

        # Assign to train or test set based on chronological order
        if i < num_sz - num_sz_test:
            # Training data: map oversampled labels (2, -1) to respective classes (1, 0)
            y_combined[y_combined == 2] = 1
            y_combined[y_combined == -1] = 0
            X_train_all.append(X_combined)
            y_train_all.append(y_combined)
        else:
            # Testing data: remove all oversampled samples (labels 2, -1)
            mask = (y_combined != 2) & (y_combined != -1)
            X_test_all.append(X_combined[mask])
            y_test_all.append(y_combined[mask])
            
    return _finalize_split_arrays(
        (X_train_all, y_train_all), 
        (X_test_all, y_test_all), 
        no_test=no_test
    )

def train_test_split_t(ictal_X, ictal_y, interictal_X, interictal_y, test_ratio):
    """
    Splits each seizure individually into training and testing sets.
    Note: `test_ratio` here defines the proportion used for the *training* set.
    """
    num_sz = len(ictal_y)
    print(f'Total {num_sz} seizures, {test_ratio:.2%} of each seizure used for training.')

    interictal_X, interictal_y = _flatten_interictal_data(interictal_X, interictal_y)
    interictal_fold_len = int(round(interictal_y.shape[0] / num_sz))
    print(f'Length of each interictal segment: {interictal_fold_len}')

    X_train_all, y_train_all, X_test_all, y_test_all = [], [], [], []

    for i in range(num_sz):
        # Get the i-th ictal seizure and its corresponding interictal data
        start, end = i * interictal_fold_len, (i + 1) * interictal_fold_len
        X_interictal_seg, y_interictal_seg = interictal_X[start:end], interictal_y[start:end]
        X_ictal_seg, y_ictal_seg = ictal_X[i], ictal_y[i]

        # Split interictal data
        split_idx = int(X_interictal_seg.shape[0] * test_ratio)
        X_train_inter, X_test_inter = X_interictal_seg[:split_idx], X_interictal_seg[split_idx:]
        y_train_inter, y_test_inter = y_interictal_seg[:split_idx], y_interictal_seg[split_idx:]
        
        # Split ictal data (handles oversampled data internally)
        X_train_ictal, y_train_ictal, X_test_ictal, y_test_ictal = split_arrays_ictal(
            X_ictal_seg, y_ictal_seg, test_ratio
        )

        # Process labels for training and testing sets
        y_train_ictal[y_train_ictal == 2] = 1 # Map oversampled to positive class for training
        test_ictal_mask = y_test_ictal != 2 # Remove oversampled from testing
        
        # Combine and append results for this seizure
        X_train_all.append(np.concatenate((X_train_inter, X_train_ictal), axis=0))
        y_train_all.append(np.concatenate((y_train_inter, y_train_ictal), axis=0))
        X_test_all.append(np.concatenate((X_test_inter, X_test_ictal[test_ictal_mask]), axis=0))
        y_test_all.append(np.concatenate((y_test_inter, y_test_ictal[test_ictal_mask]), axis=0))
            
    return _finalize_split_arrays(
        (X_train_all, y_train_all), 
        (X_test_all, y_test_all)
    )