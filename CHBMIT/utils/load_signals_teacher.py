import os
import numpy as np
import pandas as pd
from mne.io import read_raw_edf
import stft
from utils.save_load import save_hickle_file, load_hickle_file

# --- Configuration ---

# Channel configurations for different CHB-MIT patients
CHBMIT_CHANNEL_CONFIG = {
    'default': [
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 
        'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8'
    ],
    'subset_13_16': [ # For patients 13, 16
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'FZ-CZ', 'CZ-PZ'
    ],
    'subset_4': { # For patient 4
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'P8-O2', 'FZ-CZ', 
        'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT10-T8'
    }
}
# Frequency bands to keep from STFT (removes powerline noise frequencies)
STFT_BANDS_TO_KEEP = [slice(1, 57), slice(64, 117), slice(124, None)]
SAMPLING_RATE = 256

# --- Helper Functions ---

def makedirs(dir_path):
    """Creates a directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        pass

def _load_chbmit_metadata(data_dir):
    """Loads and processes all metadata CSVs for the CHB-MIT dataset."""
    seizure_summary = pd.read_csv(os.path.join(data_dir, 'seizure_summary.csv'))
    segmentation = pd.read_csv(os.path.join(data_dir, 'segmentation.csv'), header=None)
    special_interictal = pd.read_csv(os.path.join(data_dir, 'special_interictal.csv'), header=None)

    ns_filenames = set(segmentation[segmentation[1] == 0][0])
    ns_dict = {
        str(t): {fn for fn in ns_filenames if f'chb{str(t).zfill(2)}' in fn}
        for t in range(1, 24)
    }
    return seizure_summary, ns_dict, special_interictal

def load_signals_CHBMIT(data_dir, target, data_type):
    """Generator to load and yield specified data segments for a CHB-MIT patient."""
    print(f'load_signals_CHBMIT for Patient {target}')
    
    # Load metadata once
    seizure_summary, ns_dict, special_interictal = _load_chbmit_metadata(data_dir)
    patient_dir = os.path.join(data_dir, f'chb{target.zfill(2)}')
    
    # Determine which files to load and their type
    all_edf_files = {f for f in os.listdir(patient_dir) if f.endswith('.edf')}
    seizure_files = set(seizure_summary['File_name'])
    
    filenames = []
    if data_type == 'ictal':
        filenames = sorted([f for f in all_edf_files if f in seizure_files])
    elif data_type == 'interictal':
        filenames = sorted([f for f in all_edf_files if f in ns_dict[target]])

    # Select appropriate channel configuration
    if target in ['13', '16']: chs = CHBMIT_CHANNEL_CONFIG['subset_13_16']
    elif target == '4': chs = CHBMIT_CHANNEL_CONFIG['subset_4']
    else: chs = CHBMIT_CHANNEL_CONFIG['default']
    
    for filename in filenames:
        rawEEG = read_raw_edf(os.path.join(patient_dir, filename), preload=True, verbose=0)
        rawEEG.pick_channels(chs, ordered=False)
        if target == '13' and 'T8-P8' in rawEEG.ch_names:
            rawEEG.drop_channels('T8-P8')
        
        data = rawEEG.to_data_frame().values

        if data_type == 'ictal':
            seizures_in_file = seizure_summary[seizure_summary['File_name'] == filename]
            for _, seizure in seizures_in_file.iterrows():
                st = int(seizure['Seizure_start'] * SAMPLING_RATE)
                sp = int(seizure['Seizure_stop'] * SAMPLING_RATE)
                yield data[st:sp]
        
        elif data_type == 'interictal':
            if filename in list(special_interictal[0]):
                idx = list(special_interictal[0]).index(filename)
                st = int(special_interictal[1][idx] * SAMPLING_RATE)
                sp = int(special_interictal[2][idx] * SAMPLING_RATE)
                yield data[st:sp if sp > 0 else None]
            else:
                yield data

class PrepDataTeacher():
    def __init__(self, target, type, settings):
        self.target = target
        self.settings = settings
        self.type = type
        self.freq = SAMPLING_RATE

    def read_raw_signal(self):
        """Reads raw signals using the CHB-MIT data loader."""
        return load_signals_CHBMIT(self.settings['datadir'], self.target, self.type)

    def _compute_stft_features(self, segment):
        """Computes STFT features for a given data segment."""
        stft_data = stft.spectrogram(segment, framelength=self.freq, centered=False)
        stft_data = np.log10(np.abs(stft_data) + 1e-6)
        stft_data[stft_data <= 0] = 0
        stft_data = np.transpose(stft_data, (2, 1, 0))
        
        stft_data = np.concatenate([stft_data[:, :, band] for band in STFT_BANDS_TO_KEEP], axis=-1)
        return stft_data.reshape(-1, 1, *stft_data.shape)

    def preprocess(self, data_generator):
        """Processes raw data into windowed, oversampled STFT features."""
        print(f'Preprocessing {self.type} data for patient {self.target}...')
        X_processed, y_processed = [], []
        
        df_sampling = pd.read_csv('sampling_CHBMIT.csv')
        ictal_ovl_len = int(self.freq * df_sampling[df_sampling.Subject == int(self.target)].ictal_ovl.values[0])
        
        for data_segment in data_generator:
            is_ictal = self.type == 'ictal'
            y_value = 1 if is_ictal else 0
            window_len = self.freq * 30  # 30-second windows

            X_temp, y_temp = [], []
            
            # Non-overlapped windows
            for i in range(data_segment.shape[0] // window_len):
                window = data_segment[i * window_len : (i + 1) * window_len, :]
                X_temp.append(self._compute_stft_features(window))
                y_temp.append(y_value)

            # Overlapped windows for ictal data augmentation
            if is_ictal:
                i = 1
                while (i * ictal_ovl_len + window_len) <= data_segment.shape[0]:
                    window = data_segment[i * ictal_ovl_len : i * ictal_ovl_len + window_len, :]
                    X_temp.append(self._compute_stft_features(window))
                    y_temp.append(2) # Label for overlapped samples
                    i += 1
            
            if not X_temp:
                print('Warning: Seizure too short to create a window.')
                continue

            X_processed.append(np.concatenate(X_temp, axis=0))
            y_processed.append(np.array(y_temp))
            
        print(f'X: {len(X_processed)} segments, y: {len(y_processed)} segments')
        return X_processed, y_processed

    def apply(self):
        """Applies the full preprocessing pipeline, using cache if available."""
        filename = f'{self.type}_{self.target}'
        cache_path = os.path.join(self.settings['cachedir'], filename)
        
        cached_data = load_hickle_file(cache_path)
        if cached_data is not None:
            return cached_data

        raw_data = self.read_raw_signal()
        X, y = self.preprocess(raw_data)
        
        save_hickle_file(cache_path, [X, y])
        return X, y