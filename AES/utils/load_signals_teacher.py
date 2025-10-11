import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample
import stft
import json
from itertools import count
from utils.group_seizure_teacher import group_seizure
from utils.log import log
from utils.save_load import save_pickle_file, load_pickle_file, save_hickle_file, load_hickle_file
import random

def makedirs(dir):
    """Creates a directory if it does not already exist."""
    try:
        os.makedirs(dir)
    except FileExistsError:
        pass

# Channel significance order of the teacher model for each patient
global significance
significance = {
    'Patient_1': [17, 46, 8, 28, 25, 16, 18, 9, 24, 19, 12, 42, 11, 7, 23, 0, 32, 45, 3, 2, 4, 14, 21, 27, 34],
    'Patient_2': None,
    'Patient_3': [3, 4, 43, 24, 0, 18, 10, 5, 6, 46, 25, 44, 9, 1, 8, 23, 17, 15, 11, 38, 35, 2, 22, 42, 14],
    'Patient_4': [35, 43, 42, 44, 34, 40, 2, 32, 0, 36, 1, 45, 31, 28, 12, 20, 37, 33, 3, 14, 21, 23, 46, 39, 7],
    'Patient_5': [0, 16, 8, 7, 23, 13, 5, 4, 15, 22, 1, 12, 30, 6, 21, 28, 17, 3, 39, 9, 19, 26, 29, 33, 14],
    'Patient_6': [13, 22, 21, 14, 23, 5, 15, 6, 12, 28, 3, 2, 20, 9, 27, 19, 4, 1, 16, 11, 24, 10, 17, 8, 7],
    'Patient_7': [26, 7, 34, 10, 8, 24, 9, 29, 28, 30, 25, 31, 33, 32, 27, 19, 23, 4, 22, 13, 17, 11, 18, 6, 21],
    'Patient_8': None
}

class PrepDataTeacher():
    def __init__(self, target, type, settings, freq, teacher_channels=None):
        self.target = target
        self.settings = settings
        self.type = type
        self.freq = freq
        self.teacher_channels = teacher_channels

    def most_significant_channels(self, data, channels, num_channels):
        """Selects the most significant channels from the data."""
        return data[channels[:num_channels], :]

    def load_signals_Kaggle2014Det(self):
        """Loads and preprocesses seizure detection data."""
        data_dir = self.settings['datadir']
        print(f'Seizure Detection - Loading {self.type} data for patient {self.target}')
        dir_path = os.path.join(data_dir, self.target)
        result, latencies = [], [0]
        prev_latency = -1

        for i in count(1):
            filename = f'{dir_path}/{self.target}_{self.type}_segment_{i}.mat'
            if not os.path.exists(filename):
                break

            data = loadmat(filename)
            temp_data = data['data']
            if "Patient_" in self.target:
                channels = significance.get(self.target)
                if channels:
                    temp_data = self.most_significant_channels(temp_data, channels, self.teacher_channels)
            
            if temp_data.shape[-1] > self.freq:
                temp_data = resample(temp_data, self.freq, axis=-1)

            if self.type == 'ictal':
                latency = data['latency'][0]
                if latency < prev_latency:
                    latencies.append(i * self.freq)
                prev_latency = latency
            
            result.append(temp_data)

        latencies.append(len(result) * self.freq)
        print(latencies)
        return result, latencies

    @staticmethod
    def combine_matrices(matrix_list):
        """Combines a list of matrices into a single matrix."""
        if not matrix_list:
            raise ValueError("Matrix list is empty.")
        if not all(matrix.shape[0] == matrix_list[0].shape[0] for matrix in matrix_list):
            raise ValueError("All matrices must have the same number of rows.")
        return np.transpose(np.concatenate(matrix_list, axis=1))

    def _process_sliding_window(self, combination, ovl_len, window_len, y_val, non_ovl_y_val, divisor):
        """Helper to process data using a sliding window."""
        X_data, y_data = [], []
        i = 0
        while (window_len + i * ovl_len <= combination.shape[0]):
            a = i * ovl_len
            b = a + window_len
            s = combination[a:b, :]

            DataSampleSize = self.freq if 'Dog_' in self.target else int(self.freq / 5)
            stft_data = stft.spectrogram(s, framelength=DataSampleSize, centered=False)
            stft_data = np.log10(np.abs(stft_data[1:, :, :]) + 1e-6)
            stft_data[stft_data <= 0] = 0
            stft_data = np.transpose(stft_data, (2, 1, 0))
            stft_data = stft_data.reshape(-1, 1, *stft_data.shape)

            X_data.append(stft_data)
            y_data.append(non_ovl_y_val if i % divisor == 0 or i == 0 else y_val)
            i += 1
        return X_data, y_data

    def process_raw_data(self):
        """Processes raw data by applying STFT and creating labeled windows."""
        result, latencies = self.load_signals_Kaggle2014Det()
        combination = self.combine_matrices(result)

        df_sampling = pd.read_csv('sampling_Kaggle2014Det.csv')
        sampling_info = df_sampling[df_sampling.Subject == self.target]
        numts = 30
        DataSampleSize = self.freq if 'Dog_' in self.target else int(self.freq / 5)
        window_len = int(DataSampleSize * numts)

        if self.type == 'ictal':
            ovl_pt = sampling_info.ictal_ovl.values[0]
            ovl_len = int(self.freq * ovl_pt)
            divisor = window_len / ovl_len
            X_data, y_data = self._process_sliding_window(combination, ovl_len, window_len, 2, 1, divisor)

            onset_indices = [0] + [i for i, _ in enumerate(X_data) if (i * ovl_len + window_len) in latencies] + [len(X_data)]
            Xg, yg = group_seizure(X=X_data, y=y_data, onset_indices=onset_indices)
            print(f'Number of seizures {len(Xg)}', Xg[0].shape, yg[0].shape)
            return Xg, yg
        
        elif self.type == 'interictal':
            ovl_pt = sampling_info.interictal_ovl.values[0]
            ovl_len = int(self.freq * ovl_pt)
            divisor = window_len / ovl_len
            X_data, y_data = self._process_sliding_window(combination, ovl_len, window_len, -1, 0, divisor)
            
            X = np.concatenate(X_data) if X_data else np.array([])
            y = np.array(y_data)
            print('X', X.shape, 'y', y.shape)
            return X, y

    def apply(self):
        """Applies preprocessing, using a cache if available."""
        filename = f'{self.type}_{self.target}'
        cache_path = os.path.join(self.settings['cachedir'], filename)
        
        cache = load_hickle_file(cache_path)
        if cache is not None:
            return cache
        
        X, y = self.process_raw_data()
        # save_hickle_file(cache_path, [X, y]) # Caching is disabled
        return X, y

def make_teacher(mode, teacher_settings, shuffle=False):
    """Creates teacher data by processing multiple targets."""
    
    def shuffle_lists(list1, list2):
        combined = list(zip(list1, list2))
        random.shuffle(combined)
        shuffled_list1, shuffled_list2 = zip(*combined)
        return list(shuffled_list1), list(shuffled_list2)

    config = {
        'Dog': {'freq': 200, 'targets': ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4'], 'channels': None},
        'Patient_1': {'freq': 1000, 'targets': ['Patient_3', 'Patient_5', 'Patient_6', 'Patient_7'], 'channels': 15},
        'Patient_Default': {'freq': 1000, 'targets': ['Patient_3', 'Patient_5', 'Patient_6', 'Patient_7'], 'channels': 24}
    }

    mode_config = config.get(mode, config['Patient_Default'])
    freq = mode_config['freq']
    targets = mode_config['targets']
    teacher_channels = mode_config['channels']

    ictal_data_X, ictal_data_y = [], []
    interictal_data_X, interictal_data_y = [], []

    for target in targets:
        prep_teacher = PrepDataTeacher(target, 'ictal', teacher_settings, freq, teacher_channels)
        ictal_X, ictal_y = prep_teacher.apply()
        
        prep_teacher.type = 'interictal'
        interictal_X, interictal_y = prep_teacher.apply()

        ictal_data_X.extend(ictal_X)
        ictal_data_y.extend(ictal_y)
        interictal_data_X.append(interictal_X)
        interictal_data_y.append(interictal_y)

    if shuffle:
        ictal_data_X, ictal_data_y = shuffle_lists(ictal_data_X, ictal_data_y)
        
    return ictal_data_X, ictal_data_y, interictal_data_X, interictal_data_y