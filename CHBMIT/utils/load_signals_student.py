import os
import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import resample
import stft
import matplotlib.pyplot as plt
from mne.io import read_raw_edf
from utils.save_load import save_pickle_file, load_pickle_file, save_hickle_file, load_hickle_file

def calculate_interictal_hours():
    hours = dict()
    targets = [i for i in range(1, 24)]
    for target in  targets:
        if target < 10:
            thingy = '0' + str(target)
        elif target < 100:
            thingy = str(target)
        data_dir = f'Dataset/chb{thingy}'
        print ('Calculating interictal hours for patient', target)
        onset = pd.read_csv(f'Dataset/seizure_summary.csv', header=0)
        #print (onset)
        osfilenames,szstart,szstop = onset['File_name'],onset['Seizure_start'],onset['Seizure_stop']
        osfilenames = list(osfilenames)
        #print ('Seizure files:', osfilenames)

        segment = pd.read_csv('Dataset/segmentation.csv', header=None)
        nsfilenames = list(segment[segment[1]==0][0])

        nsdict = {
                '0':[]
        }
        targets = [str(i) for i in range(1, 24)]
        for t in targets:
            nslist = [elem for elem in nsfilenames if
                      elem.find('chb%s_' %t)!= -1 or
                      elem.find('chb0%s_' %t)!= -1 or
                      elem.find('chb%sa_' %t)!= -1 or
                      elem.find('chb%sb_' %t)!= -1 or
                      elem.find('chb%sc_' %t)!= -1]
            nsdict[t] = nslist

        special_interictal = pd.read_csv('Dataset/special_interictal.csv',header=None)
        sifilenames,sistart,sistop = special_interictal[0],special_interictal[1],special_interictal[2]
        sifilenames = list(sifilenames)

        def strcv(i):
            if i < 10:
                return '0' + str(i)
            elif i < 100:
                return str(i)

        strtrg = 'chb' + strcv(int(target))    
        dir = data_dir
        text_files = [f for f in os.listdir(dir) if f.endswith('.edf')]
    
        edf_files = [filename for filename in text_files if filename in nsdict[str(target)]]
        sampling_rate = 256  # Hz
        total_duration = 0.0

        for file in edf_files:
            file = f'Dataset/chb{thingy}/{file}'
            raw = read_raw_edf(file, preload=True)
            total_samples = raw.n_times
            duration = total_samples / sampling_rate
            total_duration += duration
        print('total duration in hours: ', total_duration/60/60)
        hours[target] = total_duration/60/60
    print(hours)
def load_signals_CHBMIT(data_dir, target, data_type):
    print ('load_signals_CHBMIT for Patient', target)

    onset = pd.read_csv(os.path.join(data_dir, 'seizure_summary.csv'),header=0)
    #print (onset)
    osfilenames,szstart,szstop = onset['File_name'],onset['Seizure_start'],onset['Seizure_stop']
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
        nslist = [elem for elem in nsfilenames if
                  elem.find('chb%s_' %t)!= -1 or
                  elem.find('chb0%s_' %t)!= -1 or
                  elem.find('chb%sa_' %t)!= -1 or
                  elem.find('chb%sb_' %t)!= -1 or
                  elem.find('chb%sc_' %t)!= -1]
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
    #print (text_files)

    if data_type == 'ictal':
        filenames = [filename for filename in text_files if filename in osfilenames]
    elif data_type == 'interictal':
        filenames = [filename for filename in text_files if filename in nsdict[target]]

    totalfiles = len(filenames)
    print ('Total %s files %d' % (data_type,totalfiles))
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
            SOP = 30 * 60 * 256
            # get seizure onset information
            indices = [ind for ind,x in enumerate(osfilenames) if x==filename]
            if len(indices) > 0:
                print('%d seizures in the file %s' % (len(indices),filename))
                prev_sp = -1e6
                for i in range(len(indices)):
                    st = szstart[indices[i]]*256 - 5 * 60 * 256 #SPH=5min
                    sp = szstop[indices[i]]*256
                    #print ('Seizure %s %d starts at %d stops at %d last sz stop is %d' % (filename, i, (st+5*60*256),sp,prev_sp))

                    # take care of some special filenames
                    if filename[6]=='_':
                        seq = int(filename[7:9])
                    else:
                        seq = int(filename[6:8])
                    if filename == 'chb02_16+.edf':
                        prevfile = 'chb02_16.edf'
                    else:
                        if filename[6]=='_':
                            prevfile = '%s_%s.edf' %(filename[:6],strcv(seq-1))
                        else:
                            prevfile = '%s_%s.edf' %(filename[:5],strcv(seq-1))

                    if st - SOP > prev_sp:
                        prev_sp = sp
                        if st - SOP >= 0:
                            data = tmp[st - SOP : st]
                        else:
                            if os.path.exists('%s/%s' % (dir, prevfile)):
                                rawEEG = read_raw_edf('%s/%s' % (dir, prevfile), preload=True,verbose=0)
                                rawEEG.pick_channels(chs)
                                if target == '13' and 'T8-P8' in rawEEG.ch_names: rawEEG.drop_channels('T8-P8')
                                prevtmp = rawEEG.to_data_frame()
                                prevtmp = prevtmp.values
                                if st > 0:
                                    data = np.concatenate((prevtmp[st - SOP:], tmp[:st]))
                                else:
                                    data = prevtmp[st - SOP:st]

                            else:
                                if st > 0:
                                    data = tmp[:st]
                                else:
                                    #raise Exception("file %s does not contain useful info" % filename)
                                    print("WARNING: file %s does not contain useful info" % filename)
                                    continue
                    else:
                        prev_sp = sp
                        continue

                    print ('data shape', data.shape)
                    if data.shape[0] == SOP:
                        yield(data)
                    else:
                        continue

        elif data_type == 'interictal':
            if filename in sifilenames:
                st = sistart[sifilenames.index(filename)]
                sp = sistop[sifilenames.index(filename)]
                if sp < 0:
                    data = tmp[st*256:]
                else:
                    data = tmp[st*256:sp*256]
            else:
                data = tmp
            print ('data shape', data.shape)
            yield(data)

class PrepDataStudent():
    def __init__(self, target, type, settings):
        self.target = target
        self.settings = settings
        self.type = type

    def read_raw_signal(self):
        self.samp_freq = 256
        self.freq = 256
        return load_signals_CHBMIT(self.settings['datadir'], self.target, self.type)

    def preprocess(self, data_):
        ictal = self.type == 'ictal'
        interictal = self.type == 'interictal'
        targetFrequency = self.freq  # re-sample to target frequency
        numts = 30
        
        df_sampling = pd.read_csv(
            'sampling_%s.csv' % self.settings['dataset'],
            header=0,index_col=None)
        trg = int(self.target)
        #print (df_sampling)
        #print (df_sampling[df_sampling.Subject==trg].ictal_ovl.values)
        ictal_ovl_pt = \
            df_sampling[df_sampling.Subject==trg].ictal_ovl.values[0]
        ictal_ovl_len = int(targetFrequency*ictal_ovl_pt*numts)

        def process_raw_data(mat_data):            
            print ('Loading data')
            X = []
            y = []
            #scale_ = scale_coef[target]
            for data in mat_data:
                if self.settings['dataset'] == 'FB':
                    data = data.transpose()
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

                #overdsampling
                if ictal:
                    i = 1
                    print ('ictal_ovl_len =', ictal_ovl_len)
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
                        # differentiate between non-overlapped and overlapped
                        # samples. Testing only uses non-overlapped ones.
                        y_temp.append(2)
                        i += 1

                X_temp = np.concatenate(X_temp, axis=0)
                y_temp = np.array(y_temp)
                X.append(X_temp)
                y.append(y_temp)

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
