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

def _finalize_split_arrays(X_train_list, y_train_list, X_test_list, y_test_list):
    """Concatenates lists of arrays into final training and testing sets."""
    X_train = np.concatenate(X_train_list, axis=0) if X_train_list else np.array([])
    y_train = np.concatenate(y_train_list, axis=0) if y_train_list else np.array([])
    X_test = np.concatenate(X_test_list, axis=0) if X_test_list else np.array([])
    y_test = np.concatenate(y_test_list, axis=0) if y_test_list else np.array([])
    return X_train, y_train, X_test, y_test

# --- Main Splitting Functions ---

def train_val_test_split_continual_s(ictal_X, ictal_y, interictal_X, interictal_y, test_ratio):
    """
    Splits data chronologically: earlier seizures for training, later seizures for testing.
    """
    num_sz = len(ictal_y)
    num_sz_test = max(1, int(test_ratio * num_sz))
    print(f'Total {num_sz} seizures, last {num_sz_test} used for testing.')

    interictal_X, interictal_y = _flatten_interictal_data(interictal_X, interictal_y)
    interictal_fold_len = int(round(interictal_y.shape[0] / num_sz))
    print(f'Length of each interictal segment: {interictal_fold_len}')

    X_train_all, y_train_all, X_test_all, y_test_all = [], [], [], []

    for i in range(num_sz):
        # Combine the i-th ictal seizure with its corresponding interictal segment
        X_ictal_seg = ictal_X[i]
        y_ictal_seg = ictal_y[i]
        
        start, end = i * interictal_fold_len, (i + 1) * interictal_fold_len
        X_interictal_seg = interictal_X[start:end]
        y_interictal_seg = interictal_y[start:end]
        
        X_combined = np.concatenate((X_interictal_seg, X_ictal_seg), axis=0)
        y_combined = np.concatenate((y_interictal_seg, y_ictal_seg), axis=0)

        # Assign to train or test set based on chronological order
        if i < num_sz - num_sz_test:
            # Training data: map oversampled label (2) to positive class (1)
            y_combined[y_combined == 2] = 1
            X_train_all.append(X_combined)
            y_train_all.append(y_combined)
        else:
            # Testing data: remove oversampled samples (label 2) entirely
            mask = y_combined != 2
            X_test_all.append(X_combined[mask])
            y_test_all.append(y_combined[mask])
            
    return _finalize_split_arrays(X_train_all, y_train_all, X_test_all, y_test_all)

def train_test_split_s(ictal_X, ictal_y, interictal_X, interictal_y, test_ratio):
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
        X_ictal_seg, y_ictal_seg = ictal_X[i], ictal_y[i]
        
        start, end = i * interictal_fold_len, (i + 1) * interictal_fold_len
        X_interictal_seg, y_interictal_seg = interictal_X[start:end], interictal_y[start:end]

        # Split interictal data for this seizure
        split_idx = int(X_interictal_seg.shape[0] * test_ratio)
        X_train_interictal, X_test_interictal = X_interictal_seg[:split_idx], X_interictal_seg[split_idx:]
        y_train_interictal, y_test_interictal = y_interictal_seg[:split_idx], y_interictal_seg[split_idx:]
        
        # Split ictal data for this seizure (handles oversampled data internally)
        X_train_ictal, y_train_ictal, X_test_ictal, y_test_ictal = split_arrays_ictal(
            X_ictal_seg, y_ictal_seg, test_ratio
        )

        # Process labels: map oversampled (2) to positive (1) for training
        y_train_ictal[y_train_ictal == 2] = 1
        
        # Process labels: remove oversampled data (2) from testing sets
        test_ictal_mask = y_test_ictal != 2
        X_test_ictal, y_test_ictal = X_test_ictal[test_ictal_mask], y_test_ictal[test_ictal_mask]

        # Combine and append results for this seizure
        X_train_all.append(np.concatenate((X_train_interictal, X_train_ictal), axis=0))
        y_train_all.append(np.concatenate((y_train_interictal, y_train_ictal), axis=0))
        X_test_all.append(np.concatenate((X_test_interictal, X_test_ictal), axis=0))
        y_test_all.append(np.concatenate((y_test_interictal, y_test_ictal), axis=0))
            
    return _finalize_split_arrays(X_train_all, y_train_all, X_test_all, y_test_all)