import numpy as np
from sklearn.utils import shuffle
from utils.model_stuff import split_arrays_ictal


def train_val_test_split_continual_t(ictal_X, ictal_y, interictal_X, interictal_y, test_ratio, no_test=False):
    num_sz = len(ictal_y)
    if not no_test:
        num_sz_test = int(test_ratio*num_sz)
        if num_sz_test < 1: num_sz_test = 1
        print ('Total %d seizures, last %d is used for testing.' %(num_sz, num_sz_test))
    else:
        num_sz_test = 0
        print ('Total %d seizures, all are used for training.' %(num_sz))

    if isinstance(interictal_y, list):
        interictal_X = np.concatenate(interictal_X, axis=0)
        interictal_y = np.concatenate(interictal_y, axis=0)
    interictal_fold_len = int(round(1.0*interictal_y.shape[0]/num_sz))

    print ('length of each interical segment',interictal_fold_len)

    X_train_all, y_train_all, X_test_all, y_test_all, = [], [], [], []

    for i in range(num_sz):
        X_temp_ictal = ictal_X[i]
        y_temp_ictal = ictal_y[i]

        X_temp_interictal = interictal_X[i*interictal_fold_len:(i+1)*interictal_fold_len]
        y_temp_interictal = interictal_y[i*interictal_fold_len:(i+1)*interictal_fold_len]
        
        #Downsampling interictal training set so that the 2 classes
        #are balanced
        '''
        print ('Balancing:', y_temp_ictal.shape,y_temp_interictal.shape)
        down_spl = int(np.floor(y_temp_interictal.shape[0]/y_temp_ictal.shape[0]))
        if down_spl > 1:
            X_temp_interictal = X_temp_interictal[::down_spl]
            y_temp_interictal = y_temp_interictal[::down_spl]
        elif down_spl == 1:
            X_temp_interictal = X_temp_interictal[:X_temp_ictal.shape[0]]
            y_temp_interictal = y_temp_interictal[:X_temp_ictal.shape[0]]
        '''
        X_temp = np.concatenate((X_temp_interictal, X_temp_ictal), axis=0)
        y_temp = np.concatenate((y_temp_interictal, y_temp_ictal), axis=0)


        if i < num_sz - num_sz_test:
            #We treat this as a training sample, mask overlapped samples
            y_temp[y_temp==2] = 1
            y_temp[y_temp==-1] = 0
            X_train_all.append(X_temp)
            y_train_all.append(y_temp)
        else:
            #we treat this as a testing sample, eliminate overlapped samples
            X_temp = X_temp[y_temp != 2]
            y_temp = y_temp[y_temp != 2]
            X_temp = X_temp[y_temp != -1]
            y_temp = y_temp[y_temp != -1]
            X_test_all.append(X_temp)
            y_test_all.append(y_temp)

    X_train_all = np.concatenate(X_train_all, axis=0)
    y_train_all = np.concatenate(y_train_all, axis=0)
    if not no_test:
        X_test_all = np.concatenate(X_test_all, axis=0)
        y_test_all = np.concatenate(y_test_all, axis=0)
        return X_train_all, y_train_all, X_test_all, y_test_all
    else:
        return X_train_all, y_train_all
    
def train_test_split_t(ictal_X, ictal_y, interictal_X, interictal_y, test_ratio):
    num_sz = len(ictal_y)
    
    print ('Total %d seizures, %f percent of each seizure is used for testing.' %(num_sz, test_ratio))

    if isinstance(interictal_y, list):
        interictal_X = np.concatenate(interictal_X, axis=0)
        interictal_y = np.concatenate(interictal_y, axis=0)
    interictal_fold_len = int(round(1.0*interictal_y.shape[0]/num_sz))

    print ('length of each interical segment',interictal_fold_len)

    X_train_all, y_train_all, X_test_all, y_test_all, = [], [], [], []

    for i in range(num_sz):
        X_temp_ictal = ictal_X[i]
        y_temp_ictal = ictal_y[i]

        #process interictal data by dividing by fold length
        X_temp_interictal = interictal_X[i*interictal_fold_len:(i+1)*interictal_fold_len]
        y_temp_interictal = interictal_y[i*interictal_fold_len:(i+1)*interictal_fold_len]
        X_train_interictal, X_test_interictal = X_temp_interictal[:int(X_temp_interictal.shape[0]*test_ratio)], X_temp_interictal[int(X_temp_interictal.shape[0]*test_ratio):]
        y_train_interictal, y_test_interictal = y_temp_interictal[:int(y_temp_interictal.shape[0]*test_ratio)], y_temp_interictal[int(y_temp_interictal.shape[0]*test_ratio):]
    
        #process ictal data by separating by 1s/2s and splitting 
        X_train_ictal, y_train_ictal, X_test_ictal, y_test_ictal = split_arrays_ictal(X_temp_ictal, y_temp_ictal, test_ratio)
        
        #oversampled data is treated as normal data for training
        y_train_ictal[y_train_ictal==2] = 1
        y_train_interictal[y_train_interictal==2] = 1

        #remove oversampled data in testing
        X_test_ictal = X_test_ictal[y_test_ictal != 2]
        X_test_interictal = X_test_interictal[y_test_interictal != 2]
        y_test_ictal = y_test_ictal[y_test_ictal != 2]
        y_test_interictal = y_test_interictal[y_test_interictal != 2]


        #append data to total list 
        X_train_all.append(np.concatenate((X_train_interictal, X_train_ictal), axis=0))
        X_test_all.append(np.concatenate((X_test_interictal, X_test_ictal), axis=0))
        y_train_all.append(np.concatenate((y_train_interictal, y_train_ictal), axis=0))
        y_test_all.append(np.concatenate((y_test_interictal, y_test_ictal), axis=0))
        
    X_train_all = np.concatenate(X_train_all, axis=0)
    y_train_all = np.concatenate(y_train_all, axis=0)
    X_test_all = np.concatenate(X_test_all, axis=0)
    y_test_all = np.concatenate(y_test_all, axis=0)
    return X_train_all, y_train_all, X_test_all, y_test_all