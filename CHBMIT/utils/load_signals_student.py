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
    'subset_1': [ # For patients 13, 16
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'FZ-CZ', 'CZ-PZ'
    ],
    'subset_2': { # For patient 4
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'P8-O2', 'FZ-CZ', 
        'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT10-T8'
    }
}
# Frequency bands to keep from STFT (removes powerline noise frequencies)
STFT_BANDS_TO_KEEP = [slice(1, 57), slice(64, 117), slice(124, None)]
SAMPLING_RATE = 256
PREICTAL_SOP_SECONDS = 30 * 60 # Seizure Occurrence Period
PREICTAL_SPH_SECONDS = 5 * 60   # Seizure Prediction Horizon

# --- Metadata Loading ---

def _load_chbmit_metadata(data_dir):
    """Loads all necessary metadata CSV files for the CHB-MIT dataset."""
    summary_path = os.path.join(data_dir, 'seizure_summary.csv')
    segment_path = os.path.join(data_dir, 'segmentation.csv')
    special_path = os.path.join(data_dir, 'special_interictal.csv')

    seizure_summary = pd.read_csv(summary_path)
    segmentation = pd.read_csv(segment_path, header=None)
    special_interictal = pd.read_csv(special_path, header=None)
    
    # Create a dictionary mapping patient ID to their non-seizure files
    ns_filenames = list(segmentation[segmentation[1] == 0][0])
    ns_dict = {
        str(t): [
            fn for fn in ns_filenames if f'chb{str(t).zfill(2)}' in fn or f'chb{t}_' in fn
        ] for t in range(1, 24)
    }
    return seizure_summary, ns_dict, special_interictal

# --- Main Functions ---

def calculate_interictal_hours():
    """Calculates and prints the total hours of interictal data for each patient."""
    data_dir = 'Dataset'
    _, ns_dict, _ = _load_chbmit_metadata(data_dir)
    
    hours = {}
    for target_id in range(1, 24):
        patient_id_str = str(target_id).zfill(2)
        patient_dir = os.path.join(data_dir, f'chb{patient_id_str}')
        
        if not os.path.isdir(patient_dir):
            continue
            
        print(f'Calculating interictal hours for patient {target_id}')
        edf_files = [f for f in os.listdir(patient_dir) if f.endswith('.edf') and f in ns_dict[str(target_id)]]
        
        total_duration_seconds = 0.0
        for fname in edf_files:
            file_path = os.path.join(patient_dir, fname)
            raw = read_raw_edf(file_path, preload=False, verbose=False)
            total_duration_seconds += raw.n_times / raw.info['sfreq']
            
        hours[target_id] = total_duration_seconds / 3600
        print(f'Total duration for patient {target_id}: {hours[target_id]:.2f} hours')
    print("\nSummary of interictal hours:", hours)

def load_signals_CHBMIT(data_dir, target, data_type):
    """Generator function to load and yield EEG data for the CHB-MIT dataset."""
    print(f'load_signals_CHBMIT for Patient {target}')
    seizure_summary, ns_dict, special_interictal = _load_chbmit_metadata(data_dir)
    patient_dir = os.path.join(data_dir, f'chb{str(target).zfill(2)}')
    
    # Determine which files to load
    all_edf_files = {f for f in os.listdir(patient_dir) if f.endswith('.edf')}
    if data_type == 'ictal':
        filenames = [fn for fn in all_edf_files if fn in set(seizure_summary['File_name'])]
    elif data_type == 'interictal':
        filenames = [fn for fn in all_edf_files if fn in ns_dict[target]]
    else:
        return
        
    print(f'Total {data_type} files: {len(filenames)}')
    
    for filename in filenames:
        # Complex logic for yielding specific segments of data would go here,
        # tailored to ictal (pre-seizure) and interictal periods.
        # This simplified version yields the full file data as a placeholder.
        raw = read_raw_edf(os.path.join(patient_dir, filename), preload=True, verbose=False)
        yield raw.get_data().T

class PrepDataStudent():
    def __init__(self, target, type, settings):
        self.target = target
        self.settings = settings
        self.type = type
        self.samp_freq = SAMPLING_RATE

    def read_raw_signal(self):
        """Reads raw EEG signals using the CHB-MIT loader."""
        return load_signals_CHBMIT(self.settings['datadir'], self.target, self.type)

    def _compute_stft_features(self, segment):
        """Computes STFT features for a given data segment."""
        stft_data = stft.spectrogram(segment, framelength=self.samp_freq, centered=False)
        stft_data = np.log10(np.abs(stft_data) + 1e-6)
        stft_data[stft_data <= 0] = 0
        stft_data = np.transpose(stft_data, (2, 1, 0))
        
        # Concatenate specified frequency bands
        stft_data = np.concatenate([stft_data[:, :, band] for band in STFT_BANDS_TO_KEEP], axis=-1)
        
        return stft_data.reshape(-1, 1, *stft_data.shape)

    def preprocess(self, raw_data_generator):
        """Processes raw data into windowed STFT features."""
        print('Loading and processing data...')
        X, y = [], []
        y_value = 1 if self.type == 'ictal' else 0
        window_len = self.samp_freq * 30 # 30-second windows

        for data_segment in raw_data_generator:
            X_temp, y_temp = [], []
            
            # Create non-overlapping windows
            num_windows = data_segment.shape[0] // window_len
            for i in range(num_windows):
                window = data_segment[i * window_len : (i + 1) * window_len, :]
                stft_features = self._compute_stft_features(window)
                X_temp.append(stft_features)
                y_temp.append(y_value)
            
            if not X_temp:
                continue

            X.append(np.concatenate(X_temp, axis=0))
            y.append(np.array(y_temp))
        
        print(f'Processed {len(X)} segments.')
        return X, y

    def apply(self):
        """Applies the full preprocessing pipeline, using cached data if available."""
        filename = f'{self.type}_{self.target}'
        cache_path = os.path.join(self.settings['cachedir'], filename)
        
        cache = load_hickle_file(cache_path)
        if cache is not None:
            return cache

        raw_data_gen = self.read_raw_signal()
        X, y = self.preprocess(raw_data_gen)

        save_hickle_file(cache_path, [X, y])
        return X, y