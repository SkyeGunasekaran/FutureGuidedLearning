import os
import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import resample
import stft
from utils.save_load import save_pickle_file, load_pickle_file, save_hickle_file, load_hickle_file
from utils.group_seizure_student import group_seizure

def makedirs(dir):
    try:
        os.makedirs(dir)
    except:
        pass

def calculate_interictal_hours(data_dir, target):
    print (f'Calculating interictal hours for patient {target}')
    dir = os.path.join(data_dir, target)
    done = False
    i = 0
    total_length = 0
    while not done:
        i += 1
        if i < 10:
            nstr = '000%d' %i
        elif i < 100:
            nstr = '00%d' %i
        elif i < 1000:
            nstr = '0%d' %i
        else:
            nstr = '%d' %i

        filename = '%s/%s/%s_interictal_segment_%s.mat' % (dir, target, target, nstr)
        if os.path.exists(filename):
            data = scipy.io.loadmat(filename)
            freq = int(data[f'interictal_segment_{i}'][0][0][2][0][0])+1
            total_length += data[f'interictal_segment_{i}'][0][0][0].shape[1]
        else:
            done = True
    total_hours = total_length/60/60/freq
    print(f'total hours of interictal data for patient {target}: ', total_hours)
def load_signals_Kaggle2014Pred(data_dir, target, data_type):
    print (f'Seizure Prediction - Loading {data_type} data for patient {target}')
    dir = os.path.join(data_dir, target)
    done = False
    i = 0
    while not done:
        i += 1
        if i < 10:
            nstr = '000%d' %i
        elif i < 100:
            nstr = '00%d' %i
        elif i < 1000:
            nstr = '0%d' %i
        else:
            nstr = '%d' %i

        filename = '%s/%s/%s_%s_segment_%s.mat' % (dir, target, target, data_type, nstr)
        if os.path.exists(filename):
            data = scipy.io.loadmat(filename)
            # discard preictal segments from 66 to 35 min prior to seizure
            if data_type == 'preictal':
                for skey in data.keys():
                    if "_segment_" in skey.lower():
                        mykey = skey
                sequence = data[mykey][0][0][4][0][0]
                if (sequence <= 3):
                    print ('Skipping %s....' %filename)
                    continue
            yield(data)
        else:
            if i == 1:
                raise Exception("file %s not found" % filename)
            done = True


class PrepDataStudent():
    def __init__(self, target, type, settings):
        self.target = target
        self.settings = settings
        self.type = type
        self.global_proj = np.array([0.0]*114)

    def read_raw_signal(self):
        if self.settings['dataset'] == 'Kaggle2014':
            if self.type == 'ictal':
                data_type = 'preictal'
            else:
                data_type = self.type
            return load_signals_Kaggle2014Pred(self.settings['datadir'], self.target, data_type)

        return 'array, freq, misc'


    def preprocess_Kaggle(self, data_):
        ictal = self.type == 'ictal'
        interictal = self.type == 'interictal'
       
        if 'Dog_' in self.target:
            targetFrequency = 200   #re-sample to target frequency
            DataSampleSize = targetFrequency
            numts = 30
        else:
            targetFrequency = 1000
            DataSampleSize = int(targetFrequency/5)
            numts = 30
    
        sampleSizeinSecond = 600

        df_sampling = pd.read_csv(
            'sampling_Kaggle2014Pred.csv',
            header=0,index_col=None)
        trg = self.target
        print (df_sampling)
        print (df_sampling[df_sampling.Subject==trg].ictal_ovl.values)
        ictal_ovl_pt = \
            df_sampling[df_sampling.Subject==trg].ictal_ovl.values[0]
        ictal_ovl_len = int(targetFrequency*ictal_ovl_pt*numts)

        def process_raw_data(mat_data):
            print ('Loading data')
            X = []
            y = []
            sequences = []
            #scale_ = scale_coef[target]
            for segment in mat_data:
                for skey in segment.keys():
                    if "_segment_" in skey.lower():
                        mykey = skey
                if ictal:
                    y_value=1
                    sequence = segment[mykey][0][0][4][0][0]
                else:
                    y_value=0

                data = segment[mykey][0][0][0]

                '''
                if self.target == 'Dog_5':
                    m, n = data.shape
                    new_channel = np.copy(data[-1, :])
                    new_channel = new_channel.reshape((1, n))
                    data = np.vstack((data, new_channel))
                '''
                sampleFrequency = segment[mykey][0][0][2][0][0]

                if sampleFrequency > targetFrequency:   #resample to target frequency
                    data = resample(data, targetFrequency*sampleSizeinSecond, axis=-1)

                data = data.transpose()

                from mne.filter import notch_filter

                totalSample = int(data.shape[0]/DataSampleSize/numts) + 1
                window_len = int(DataSampleSize*numts)
                #print ('DEBUG: window_len, totalSample', window_len, totalSample)
                for i in range(totalSample):

                    if (i+1)*window_len <= data.shape[0]:
                        s = data[i*window_len:(i+1)*window_len,:]
                        stft_data = stft.spectrogram(s,framelength=DataSampleSize,centered=False)
                        stft_data = stft_data[1:,:,:]
                        stft_data = np.log10(stft_data)
                        indices = np.where(stft_data <= 0)
                        stft_data[indices] = 0
                        stft_data = np.transpose(stft_data,(2,1,0))
                        stft_data = np.abs(stft_data)+1e-6


                        stft_data = stft_data.reshape(-1, 1, stft_data.shape[0],stft_data.shape[1],stft_data.shape[2])

                        X.append(stft_data)
                        y.append(y_value)
                        if ictal:
                            sequences.append(sequence)

                if ictal:
                    #print ('Generating more preictal samples....')
                    #overlapped window
                    i=1
                    while (window_len + (i + 1)*ictal_ovl_len <= data.shape[0]):
                        s = data[i*ictal_ovl_len:i*ictal_ovl_len + window_len, :]

                        stft_data = stft.spectrogram(s, framelength=DataSampleSize,centered=False)

                        stft_data = stft_data[1:,:,:]
                        stft_data = np.log10(stft_data)
                        indices = np.where(stft_data <= 0)
                        stft_data[indices] = 0
                        stft_data = np.transpose(stft_data, (2, 1, 0))
                        stft_data = np.abs(stft_data)+1e-6

                        stft_data = stft_data.reshape(-1, 1, stft_data.shape[0], stft_data.shape[1], stft_data.shape[2])

                        X.append(stft_data)
                        y.append(2)
                        sequences.append(sequence)
                        i += 1

            if ictal:
                assert len(X) == len(y)
                assert len(X) == len(sequences)
                X, y = group_seizure(X, y, sequences)
                print ('X', len(X), X[0].shape)
                return X, y
            elif interictal:
                X = np.concatenate(X)
                y = np.array(y)
                print ('X', X.shape, 'y', y.shape)
                return X, y
            else:
                X = np.concatenate(X)
                print ('X', X.shape)
                return X, None

        data = process_raw_data(data_)
        return data

    def apply(self):
        filename = '%s_%s' % (self.type, self.target)
        cache = load_hickle_file(
            os.path.join(self.settings['cachedir'], filename))
        if cache is not None:
            return cache

        data = self.read_raw_signal()
        if self.settings['dataset']=='Kaggle2014':
            X, y = self.preprocess_Kaggle(data)
        else:
            X, y = self.preprocess(data)
        save_hickle_file(
            os.path.join(self.settings['cachedir'], filename),
            [X, y])
        return X, y
