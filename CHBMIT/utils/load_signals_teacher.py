import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample
import stft
import json
from utils.log import log
from utils.save_load import save_pickle_file, load_pickle_file, save_hickle_file, load_hickle_file
from mne.io import read_raw_edf

def load_signals_CHBMIT(data_dir, target, data_type):
    print ('load_signals_CHBMIT for Patient', target)

    onset = pd.read_csv(os.path.join(data_dir, 'seizure_summary.csv'),header=0)
    #print (onset)
    osfilenames,sstart,sstop = onset['File_name'],onset['Seizure_start'],onset['Seizure_stop']
    osfilenames = list(osfilenames)
    #print ('Seizure files:', osfilenames)

    segment = pd.read_csv(os.path.join(data_dir, 'segmentation.csv'),header=None)
    nsfilenames = list(segment[segment[1]==0][0])

    nsdict = {
            '0':[]
    }
    targets = [
        '1',
        '2',
        '3',
        '4',
        '5',
        '6',
        '7',
        '8',
        '9',
        '10',
        '11',
        '12',
        '13',
        '14',
        '15',
        '16',
        '17',
        '18',
        '19',
        '20',
        '21',
        '22',
        '23'
    ]
    for t in targets:
        nslist = [elem for elem in nsfilenames if elem.find('chb%s_' %t)!= -1 or elem.find('chb0%s_' %t)!= -1 or elem.find('chb%sa_' %t)!= -1 or elem.find('chb%sb_' %t)!= -1 or elem.find('chb%sc_' %t)!= -1] # could be done much better, I am so lazy
        nsdict[t] = nslist
 
    special_interictal = pd.read_csv(os.path.join(data_dir, 'special_interictal.csv'),header=None)
    sifilenames,sistart,sistop = special_interictal[0],special_interictal[1],special_interictal[2]
    sifilenames = list(sifilenames)

    def strcv(i):
        if i < 10:
            return '0' + str(i)
        elif i < 100:
            return str(i)

    strtrg = 'chb' + strcv(int(target))
    dir = os.path.join(data_dir, strtrg)
    text_files = [f for f in os.listdir(dir) if f.endswith('.edf')]

    if data_type == 'ictal':
        filenames = [filename for filename in text_files if filename in osfilenames]
    elif data_type == 'interictal':
        filenames = [filename for filename in text_files if filename in nsdict[target]]

    totalfiles = len(filenames)
    #print ('Total %s files %d' % (data_type,totalfiles))
    for filename in filenames:
        if target in ['13','16']:
            chs = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3', u'F3-C3', u'C3-P3', u'P3-O1', u'FP2-F4', u'F4-C4', u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'T8-P8', u'FZ-CZ', u'CZ-PZ']
        elif target in ['4']:
            chs = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3', u'F3-C3', u'C3-P3', u'P3-O1', u'FP2-F4', u'F4-C4', u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'P8-O2', u'FZ-CZ', u'CZ-PZ', u'P7-T7', u'T7-FT9', u'FT10-T8']
        else:
            chs = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3', u'F3-C3', u'C3-P3', u'P3-O1', u'FP2-F4', u'F4-C4', u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'T8-P8', u'P8-O2', u'FZ-CZ', u'CZ-PZ', u'P7-T7', u'T7-FT9', u'FT9-FT10', u'FT10-T8']

        rawEEG = read_raw_edf('%s/%s' % (dir, filename), verbose=0,preload=True)
        rawEEG.pick_channels(chs, ordered=False)
    
        if target == '13' and 'T8-P8' in rawEEG.ch_names: rawEEG.drop_channels('T8-P8')

        tmp = rawEEG.to_data_frame()
        tmp = tmp.values

        if data_type == 'ictal':
            # get onset information
            indices = [ind for ind,x in enumerate(osfilenames) if x==filename]
            if len(indices) > 0: #multiple seizures in one file
                print ('%d seizures in the file %s' % (len(indices),filename))
                for i in range(len(indices)):
                    st = sstart[indices[i]]*256
                    sp = sstop[indices[i]]*256
                    print ('%s: Seizure %d starts at %d stops at %d' % (filename, i, st,sp))
                    data = tmp[st:sp]
                    print ('data shape', data.shape)
                    yield(data)


        elif data_type == 'interictal':
            if filename in sifilenames:
                print ('Special interictal %s' % filename)
                st = sistart[sifilenames.index(filename)] * 256
                sp = sistop[sifilenames.index(filename)] * 256
                if sp < 0:
                    data = tmp[st:]
                else:
                    data = tmp[st:sp]
            else:
                data = tmp
            print ('data shape', data.shape)
            yield(data)

class PrepDataTeacher():
    def __init__(self, target, type, settings):
        self.target = target
        self.settings = settings
        self.type = type


    def read_raw_signal(self):
        self.freq = 256
        self.global_proj = np.array([0.0]*114)
        return load_signals_CHBMIT(self.settings['datadir'], self.target, self.type)

    def preprocess(self, data_):
        ictal = self.type == 'ictal'
        interictal = self.type == 'interictal'
        targetFrequency = self.freq
        numts = 30
        
        df_sampling = pd.read_csv('sampling_CHBMIT.csv')
        trg = int(self.target)
        #print (df_sampling)
        #print (df_sampling[df_sampling.Subject==trg].ictal_ovl.values)
        ictal_ovl_pt = \
            df_sampling[df_sampling.Subject==trg].ictal_ovl.values[0]
        ictal_ovl_len = int(targetFrequency*ictal_ovl_pt)

        def process_raw_data(mat_data):            
            print ('Loading data')
    
            X = []
            y = []
    
            for data in mat_data:
                if ictal:
                    y_value=1
                else:
                    y_value=0
    
                X_temp = []
                y_temp = []
    
                totalSample = int(data.shape[0]/targetFrequency/numts) + 1
                window_len = int(targetFrequency*numts)
                for i in range(totalSample):
                    if (i+1)*window_len <= data.shape[0]:
                        s = data[i*window_len:(i+1)*window_len,:]
    
                        stft_data = stft.spectrogram(s, framelength=targetFrequency,centered=False)
                        stft_data = np.abs(stft_data)+1e-6
                        stft_data = np.log10(stft_data)
                        indices = np.where(stft_data <= 0)
                        stft_data[indices] = 0

                        stft_data = np.transpose(stft_data, (2,1,0))
                        stft_data = np.concatenate((stft_data[:,:,1:57],
                                                        stft_data[:,:,64:117],
                                                        stft_data[:,:,124:]),
                                                       axis=-1)
    
                        stft_data = stft_data.reshape(-1, 1, stft_data.shape[0],
                                                      stft_data.shape[1],
                                                      stft_data.shape[2])
    
                        X_temp.append(stft_data)
                        y_temp.append(y_value)
    
                #overlapped window
                if ictal:
                    i = 1
                    print ('ictal oversampling ratio =', ictal_ovl_len)
                    while (window_len + (i + 1)*ictal_ovl_len <= data.shape[0]):
                        s = data[i*ictal_ovl_len:i*ictal_ovl_len + window_len, :]
    
                        stft_data = stft.spectrogram(s, framelength=targetFrequency,centered=False)
                        stft_data = np.abs(stft_data)+1e-6
                        stft_data = np.log10(stft_data)
                        indices = np.where(stft_data <= 0)
                        stft_data[indices] = 0

                        stft_data = np.transpose(stft_data, (2,1,0))
                        stft_data = np.concatenate((stft_data[:,:,1:57],
                                                    stft_data[:,:,64:117],
                                                    stft_data[:,:,124:]),
                                                    axis=-1)
    
                        stft_data = stft_data.reshape(-1, 1, stft_data.shape[0],
                                                      stft_data.shape[1],
                                                      stft_data.shape[2])
    
                        X_temp.append(stft_data)
                        # to differentiate between non overlapped and overlapped
                        # samples. Testing only uses non overlapped ones.
                        y_temp.append(2)
                        i += 1
    
                try:
                    X_temp = np.concatenate(X_temp, axis=0)
                    y_temp = np.array(y_temp)
                    X.append(X_temp)
                    y.append(y_temp)
                except:
                    print('seizure too short')
    

            #y = np.array(y)
            print ('X', len(X), X[0].shape, 'y', len(y), y[0].shape)
            return X, y
        
        data = process_raw_data(data_)
        return  data

    def apply(self):
        filename = '%s_%s' % (self.type, self.target)
        cache = load_hickle_file(
            os.path.join(self.settings['cachedir'], filename))
        if cache is not None:
            return cache

        data = self.read_raw_signal()
        X, y = self.preprocess(data)
        save_hickle_file(
            os.path.join(self.settings['cachedir'], filename),
            [X, y])
        return X, y

def makedirs(dir):
    try:
        os.makedirs(dir)
    except:
        pass