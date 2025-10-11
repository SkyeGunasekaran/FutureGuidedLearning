import os
import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import resample
import stft
from utils.save_load import save_hickle_file, load_hickle_file
from utils.group_seizure_student import group_seizure

def makedirs(dir):
    """Creates a directory if it does not already exist."""
    try:
        os.makedirs(dir)
    except FileExistsError:
        pass

def _get_segment_fpath(data_dir, target, data_type, segment_num):
    """Formats the file path for a given data segment."""
    dir_path = os.path.join(data_dir, target)
    fname = f"{target}_{data_type}_segment_{str(segment_num).zfill(4)}.mat"
    return os.path.join(dir_path, target, fname)

def calculate_interictal_hours(data_dir, target):
    """Calculates the total hours of interictal data for a patient."""
    print(f'Calculating interictal hours for patient {target}')
    total_length = 0
    freq = 0
    for i in range(1, 10000):  # A safe upper limit for segment numbers
        filename = _get_segment_fpath(data_dir, target, 'interictal', i)
        if not os.path.exists(filename):
            break
        data = scipy.io.loadmat(filename)
        segment_key = f'interictal_segment_{i}'
        total_length += data[segment_key][0, 0]['data'].shape[1]
        if freq == 0:
            freq = int(data[segment_key][0, 0]['sampling_frequency'][0, 0])
    
    if freq == 0:
        print(f"No interictal data found for patient {target}")
        return

    total_hours = total_length / (60 * 60 * freq)
    print(f'Total hours of interictal data for patient {target}: {total_hours}')

def load_signals_Kaggle2014Pred(data_dir, target, data_type):
    """Loads seizure prediction data segments for a given patient and data type."""
    print(f'Seizure Prediction - Loading {data_type} data for patient {target}')
    for i in range(1, 10000):
        filename = _get_segment_fpath(data_dir, target, data_type, i)
        if not os.path.exists(filename):
            if i == 1:
                raise FileNotFoundError(f"File {filename} not found")
            break

        data = scipy.io.loadmat(filename)
        if data_type == 'preictal':
            mykey = next((key for key in data if "_segment_" in key.lower()), None)
            if mykey and data[mykey][0, 0]['sequence'][0, 0] <= 3:
                print(f'Skipping {filename}....')
                continue
        
        yield data

class PrepDataStudent():
    def __init__(self, target, type, settings):
        self.target = target
        self.settings = settings
        self.type = type

    def read_raw_signal(self):
        """Reads raw signal data based on the dataset specified in settings."""
        if self.settings['dataset'] == 'Kaggle2014':
            data_type = 'preictal' if self.type == 'ictal' else self.type
            return load_signals_Kaggle2014Pred(self.settings['datadir'], self.target, data_type)
        return 'array, freq, misc'

    def _process_single_segment(self, segment, y_value, DataSampleSize, numts, ictal_ovl_len=0):
        """Helper function to process a single raw data segment."""
        X, y, sequences = [], [], []
        
        mykey = next((key for key in segment if "_segment_" in key.lower()), None)
        if not mykey:
            return X, y, sequences

        data = segment[mykey][0, 0]['data']
        sampleFrequency = segment[mykey][0, 0]['sampling_frequency'][0, 0]
        sequence = segment[mykey][0, 0]['sequence'][0, 0] if 'sequence' in segment[mykey][0, 0].dtype.names else None

        targetFrequency = 200 if 'Dog_' in self.target else 1000
        if sampleFrequency > targetFrequency:
            data = resample(data, int(targetFrequency * (data.shape[1] / sampleFrequency)), axis=-1)
        
        data = data.transpose()
        window_len = int(DataSampleSize * numts)

        for i in range(int(data.shape[0] / window_len)):
            s = data[i * window_len:(i + 1) * window_len, :]
            stft_data = self._compute_stft(s, DataSampleSize)
            X.append(stft_data)
            y.append(y_value)
            if sequence is not None:
                sequences.append(sequence)

        if ictal_ovl_len > 0:
            i = 1
            while (window_len + (i + 1) * ictal_ovl_len <= data.shape[0]):
                s = data[i * ictal_ovl_len : i * ictal_ovl_len + window_len, :]
                stft_data = self._compute_stft(s, DataSampleSize)
                X.append(stft_data)
                y.append(2) # Special label for overlapped preictal data
                sequences.append(sequence)
                i += 1
        
        return X, y, sequences

    def _compute_stft(self, s, DataSampleSize):
        """Computes the Short-Time Fourier Transform of a signal."""
        stft_data = stft.spectrogram(s, framelength=DataSampleSize, centered=False)
        stft_data = np.log10(np.abs(stft_data[1:, :, :]) + 1e-6)
        stft_data[stft_data <= 0] = 0
        stft_data = np.transpose(stft_data, (2, 1, 0))
        return stft_data.reshape(-1, 1, *stft_data.shape)

    def preprocess_Kaggle(self, data_):
        """Processes Kaggle competition data, including STFT computation."""
        ictal = self.type == 'ictal'
        interictal = self.type == 'interictal'

        targetFrequency = 200 if 'Dog_' in self.target else 1000
        DataSampleSize = targetFrequency if 'Dog_' in self.target else int(targetFrequency / 5)
        numts = 30
        
        df_sampling = pd.read_csv('sampling_Kaggle2014Pred.csv')
        ictal_ovl_pt = df_sampling[df_sampling.Subject == self.target].ictal_ovl.values[0]
        ictal_ovl_len = int(targetFrequency * ictal_ovl_pt * numts)

        X_all, y_all, sequences_all = [], [], []
        y_value = 1 if ictal else 0
        
        for segment in data_:
            X_seg, y_seg, sequences_seg = self._process_single_segment(segment, y_value, DataSampleSize, numts, ictal_ovl_len if ictal else 0)
            X_all.extend(X_seg)
            y_all.extend(y_seg)
            sequences_all.extend(sequences_seg)

        if ictal:
            X_all, y_all = group_seizure(X_all, y_all, sequences_all)
            print('X', len(X_all), X_all[0].shape)
            return X_all, y_all
        else:
            X_all = np.concatenate(X_all) if X_all else np.array([])
            y_all = np.array(y_all)
            print('X', X_all.shape, 'y', y_all.shape)
            return X_all, y_all if interictal else None

    def apply(self):
        """Applies preprocessing to the data, using cached results if available."""
        filename = f'{self.type}_{self.target}'
        cache_path = os.path.join(self.settings['cachedir'], filename)
        
        cache = load_hickle_file(cache_path)
        if cache is not None:
            return cache

        data = self.read_raw_signal()
        
        if self.settings['dataset'] == 'Kaggle2014':
            X, y = self.preprocess_Kaggle(data)
        else:
            X, y = self.preprocess(data) # Assuming a generic preprocess method exists

        save_hickle_file(cache_path, [X, y])
        return X, y