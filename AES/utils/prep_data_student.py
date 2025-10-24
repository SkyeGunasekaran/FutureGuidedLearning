import numpy as np

def train_val_test_split_continual_s(ictal_X, ictal_y, interictal_X, interictal_y, test_ratio, val_ratio, no_test=False, balancing=False):
    """
    Splits data into train, validation, and test sets based on a continual (chronological) split of seizures.
    
    The split is done as:
    - First N seizures: Training
    - Next M seizures: Validation
    - Last K seizures: Testing
    """
    num_sz = len(ictal_y)
    
    if not no_test:
        num_sz_test = int(test_ratio * num_sz)
    else:
        num_sz_test = 0
        
    # Calculate validation seizures from the remaining portion
    num_sz_val = int(val_ratio * num_sz)
    
    # Calculate training seizures
    num_sz_train = num_sz - num_sz_test - num_sz_val

    if num_sz_train <= 0:
        raise ValueError("test_ratio and val_ratio sum is too large, or ratios are invalid. No training data left.")

    # Define the split indices
    val_start_index = num_sz_train
    test_start_index = num_sz_train + num_sz_val

    if not no_test:
        print(f'Total {num_sz} seizures:')
        print(f'  {num_sz_train} used for training (Seizures 0 to {val_start_index - 1})')
        print(f'  {num_sz_val} used for validation (Seizures {val_start_index} to {test_start_index - 1})')
        print(f'  {num_sz_test} used for testing (Seizures {test_start_index} to {num_sz - 1})')
    else:
        print(f'Total {num_sz} seizures (No Test Set):')
        print(f'  {num_sz_train} used for training (Seizures 0 to {val_start_index - 1})')
        print(f'  {num_sz_val} used for validation (Seizures {val_start_index} to {num_sz - 1})')

    if isinstance(interictal_y, list):
        interictal_X = np.concatenate(interictal_X, axis=0)
        interictal_y = np.concatenate(interictal_y, axis=0)
    interictal_fold_len = int(round(1.0*interictal_y.shape[0]/num_sz))

    print ('length of each interical segment', interictal_fold_len)

    # Initialize lists for all splits
    X_train_all, y_train_all = [], []
    X_val_all, y_val_all = [], []
    X_test_all, y_test_all = [], []

    for i in range(num_sz):
        X_temp_ictal = ictal_X[i]
        y_temp_ictal = ictal_y[i]

        X_temp_interictal = interictal_X[i*interictal_fold_len:(i+1)*interictal_fold_len]
        y_temp_interictal = interictal_y[i*interictal_fold_len:(i+1)*interictal_fold_len]
        
        '''
        Downsampling interictal training set so that the 2 classes
        are balanced
        '''
        if balancing:
            print ('Balancing:', y_temp_ictal.shape,y_temp_interictal.shape)

            down_spl = int(np.floor(y_temp_interictal.shape[0]/y_temp_ictal.shape[0]))
            if down_spl > 1:
                X_temp_interictal = X_temp_interictal[::down_spl]
                y_temp_interictal = y_temp_interictal[::down_spl]
            elif down_spl == 1:
                X_temp_interictal = X_temp_interictal[:X_temp_ictal.shape[0]]
                y_temp_interictal = y_temp_interictal[:X_temp_ictal.shape[0]]

        X_temp = np.concatenate((X_temp_interictal, X_temp_ictal), axis=0)
        y_temp = np.concatenate((y_temp_interictal, y_temp_ictal), axis=0)

        if i < val_start_index:
            # --- TRAINING SAMPLE ---
            # We treat this as a training sample
            y_temp[y_temp==2] = 1
            X_train_all.append(X_temp)
            y_train_all.append(y_temp)
        
        elif i < test_start_index:
            # --- VALIDATION SAMPLE ---
            # We treat this like a testing sample (remove pre-ictal '2' labels)
            X_temp = X_temp[y_temp != 2]
            y_temp = y_temp[y_temp != 2]
            X_val_all.append(X_temp)
            y_val_all.append(y_temp)
            
        else:
            # --- TESTING SAMPLE ---
            # we treat this as a testing sample
            X_temp = X_temp[y_temp != 2]
            y_temp = y_temp[y_temp != 2]
            X_test_all.append(X_temp)
            y_test_all.append(y_temp)

    # Concatenate all arrays, handling cases where a list might be empty
    X_train_all = np.concatenate(X_train_all, axis=0) if X_train_all else np.array([])
    y_train_all = np.concatenate(y_train_all, axis=0) if y_train_all else np.array([])
    
    X_val_all = np.concatenate(X_val_all, axis=0) if X_val_all else np.array([])
    y_val_all = np.concatenate(y_val_all, axis=0) if y_val_all else np.array([])
    
    X_test_all = np.concatenate(X_test_all, axis=0) if X_test_all else np.array([])
    y_test_all = np.concatenate(y_test_all, axis=0) if y_test_all else np.array([])

    # Return all 6 arrays for a consistent API
    return X_train_all, y_train_all, X_val_all, y_val_all, X_test_all, y_test_all