import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample
import stft
import json
from utils.group_seizure_teacher import group_seizure
from utils.log import log
from utils.save_load import save_pickle_file, load_pickle_file, save_hickle_file, load_hickle_file
import random

def makedirs(dir):
    try:
        os.makedirs(dir)
    except:
        pass

#channel significance order of the teacher model for each patient
global significance
significance = {
    'Patient_1':[17, 46,  8, 28, 25, 16, 18, 9, 24, 19, 12, 42, 11, 7, 23, 0, 32, 45, 3, 2, 4, 14, 21, 27,34],
    'Patient_2':None,
    'Patient_3':[ 3,  4, 43, 24,  0, 18, 10,  5,  6, 46, 25, 44,  9,  1,  8, 23, 17, 15, 11, 38, 35,  2, 22, 42, 14],
    'Patient_4':[35, 43, 42, 44, 34, 40,  2, 32,  0, 36,  1, 45, 31, 28, 12, 20, 37, 33,  3, 14, 21, 23, 46, 39, 7],
    'Patient_5':[ 0, 16,  8,  7, 23, 13,  5,  4, 15, 22,  1, 12, 30,  6, 21, 28, 17,  3, 39,  9, 19, 26, 29, 33, 14],
    'Patient_6':[13, 22, 21, 14, 23,  5, 15,  6, 12, 28,  3,  2, 20,  9, 27, 19,  4,  1, 16, 11, 24, 10, 17,  8, 7],
    'Patient_7':[26,  7, 34, 10,  8, 24,  9, 29, 28, 30, 25, 31, 33, 32, 27, 19, 23,  4, 22, 13, 17, 11, 18,  6, 21],
    'Patient_8':None 
}

class PrepDataTeacher():
    def __init__(self, target, type, settings, freq, teacher_channels=None):
        self.target = target
        self.settings = settings
        self.type = type
        self.freq = freq
        self.teacher_channels = teacher_channels
    
    def most_significant_channels(self, data, channels, num_channels):
        channels = channels[0:num_channels]
        result_matrix = data[channels, :]
        return result_matrix
    
    def load_signals_Kaggle2014Det(self):
        data_dir = self.settings['datadir']
        print (f'Seizure Detection - Loading {self.type} data for patient {self.target}')
        dir = os.path.join(data_dir, self.target)
        done = False
        i = 0
        result = []
        latencies = [0]
        prev_latency = -1
        targetFrequency = self.freq
        while not done:
            i+=1
            filename = '%s/%s_%s_segment_%s.mat' % (dir, self.target, self.type, i)
            if os.path.exists(filename):
                data = loadmat(filename)
                if "Patient_" in self.target:
                    channels = significance[self.target]
                    temp = self.most_significant_channels(data['data'], channels, self.teacher_channels)
                else:
                    temp = data['data']
                #print(f'this is the frequency of patient {self.target}', temp.shape[-1])
                if temp.shape[-1] > targetFrequency:
                    temp = resample(temp, targetFrequency, axis=temp.ndim - 1)
                if self.type=='ictal':
                    latency = data['latency'][0]
                    if latency < prev_latency:
                        latencies.append(i*targetFrequency)
                    prev_latency = latency
                result.append(temp)
            else:
                done = True
        latencies.append(i*targetFrequency)
        print(latencies)
        return result, latencies

    def combine_matrices(matrix_list):
        if not matrix_list:
            raise ValueError("Matrix list is empty.")
        num_rows = matrix_list[0].shape[0]
        if not all(matrix.shape[0] == num_rows for matrix in matrix_list):
            raise ValueError("Number of rows in all matrices must be the same.")
        combined_matrix = np.concatenate(matrix_list, axis=1)
        return np.transpose(combined_matrix)

    def process_raw_data(self):

        result, latencies = PrepDataTeacher.load_signals_Kaggle2014Det(self)
        combination = PrepDataTeacher.combine_matrices(result)
        X_data = []
        y_data = []

        #preprocessinbg parameters
        targetFrequency = self.freq
        if 'Patient_' in self.target:
            DataSampleSize = int(targetFrequency/5)
        else:
            DataSampleSize = targetFrequency
        numts = 30
            
        df_sampling = pd.read_csv(
                'sampling_Kaggle2014Det.csv',
                header=0,index_col=None)
        onset_indices = [0]

        if self.type=='ictal':
            ictal_ovl_pt = \
                df_sampling[df_sampling.Subject==self.target].ictal_ovl.values[0]
            ictal_ovl_len = int(targetFrequency*ictal_ovl_pt)
            window_len = int(DataSampleSize*numts)
            divisor = window_len/ictal_ovl_len
            i = 0
            while (window_len + (i)*ictal_ovl_len <= combination.shape[0]):
                a = i*ictal_ovl_len
                b = i*ictal_ovl_len + window_len
                #print("there are the window indices: ", a, b)
                s = combination[a:b, :]

                stft_data = stft.spectrogram(s, framelength=DataSampleSize,centered=False)
                stft_data = stft_data[1:,:,:]
                stft_data = np.log10(stft_data)
                indices = np.where(stft_data <= 0)
                stft_data[indices] = 0
                stft_data = np.transpose(stft_data,(2,1,0))
                stft_data = np.abs(stft_data)+1e-6
                stft_data = stft_data.reshape(-1, 1, stft_data.shape[0],stft_data.shape[1],stft_data.shape[2])
                #print(stft_data.shape)
                X_data.append(stft_data)
                if i % divisor == 0 or i == 0:
                    y_data.append(1)
                else:
                    y_data.append(2)
                if b in latencies:
                    onset_indices.append(i)
                i+=1
            onset_indices.append(i)

        elif self.type=='interictal':
            interictal_ovl_pt = \
                df_sampling[df_sampling.Subject==self.target].interictal_ovl.values[0]
            interictal_ovl_len = int(targetFrequency*interictal_ovl_pt)
            window_len = int(DataSampleSize*numts)
            divisor = window_len/interictal_ovl_len
            i = 0
            while (window_len + (i)*interictal_ovl_len <= combination.shape[0]):
                a = i*interictal_ovl_len
                b = i*interictal_ovl_len + window_len
                #print("these are the window indices: ", a, b, i)
                s = combination[a:b, :]

                stft_data = stft.spectrogram(s, framelength=DataSampleSize,centered=False)
                stft_data = stft_data[1:,:,:]
                stft_data = np.log10(stft_data)
                indices = np.where(stft_data <= 0)
                stft_data[indices] = 0
                stft_data = np.transpose(stft_data,(2,1,0))
                stft_data = np.abs(stft_data)+1e-6
                stft_data = stft_data.reshape(-1, 1, stft_data.shape[0],stft_data.shape[1],stft_data.shape[2])
                #print(stft_data.shape)
                X_data.append(stft_data)
                if i % divisor == 0 or i == 0:
                    y_data.append(0)
                else:
                    y_data.append(-1)
                i+=1

        if self.type=='ictal':
            Xg, yg = group_seizure(X=X_data, y=y_data, onset_indices=onset_indices)
            print ('Number of seizures %d' %len(Xg), Xg[0].shape, yg[0].shape)
            return Xg, yg
        elif self.type=='interictal':
            X = np.concatenate(X_data)
            y = np.array(y_data)
            print ('X', X.shape, 'y', y.shape)
            return X, y
        
    def apply(self):
        filename = '%s_%s' % (self.type, self.target)
        cache = load_hickle_file(
            os.path.join(self.settings['cachedir'], filename))
        if cache is not None:
            return cache
        X, y = self.process_raw_data()
        #save_hickle_file(
        #    os.path.join(self.settings['cachedir'], filename),
        #    [X, y])
        return X, y
    
def make_teacher(mode, teacher_settings, shuffle = False):
    def shuffle_lists(list1, list2):
        combined = list(zip(list1, list2))
        random.shuffle(combined)
        shuffled_list1, shuffled_list2 = zip(*combined)
        return list(shuffled_list1), list(shuffled_list2)

    dog_targets =  ['Dog_1', 'Dog_2','Dog_3','Dog_4']
    human_targets = ['Patient_3','Patient_5','Patient_6','Patient_7']
    ictal_data_X, ictal_data_y = [], []
    interictal_data_X, interictal_data_y = [], []

    if mode == 'Dog':
        freq = 200
        targets = dog_targets
        teacher_channels=None
    elif mode == 'Patient_1':
        freq = 1000
        targets = human_targets
        teacher_channels=15
    else:
        freq = 1000
        targets = human_targets
        teacher_channels=24

    for target in targets:
        ictal_X, ictal_y = PrepDataTeacher(target, type='ictal', settings=teacher_settings, freq=freq, teacher_channels=teacher_channels).apply()
        interictal_X, interictal_y = PrepDataTeacher(target, type='interictal', settings=teacher_settings, freq=freq, teacher_channels=teacher_channels).apply()
        ictal_data_X.extend(ictal_X)
        ictal_data_y.extend(ictal_y)
        interictal_data_X.append(interictal_X)
        interictal_data_y.append(interictal_y)
    if shuffle:
        ictal_data_X, ictal_data_y = shuffle_lists(ictal_data_X, ictal_data_y)
    return ictal_data_X, ictal_data_y, interictal_data_X, interictal_data_y
