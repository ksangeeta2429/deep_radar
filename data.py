import os
import math
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

def lbls_for_cls(labels, lbls_list=None):
    labels_copy = labels.copy()
    new_labels = [i for i in range(len(lbls_list))]
    for i, lbl in enumerate(lbls_list):
        labels[labels == lbl] = new_labels[i]

    return labels_copy, labels

def get_time_data(data_dir,
                  sel_cls,
                  win_len=None,
                  task_type='cls'):
    
    data_path = os.path.join(data_dir, 'Data_all.npy')
    labels_path = os.path.join(data_dir, 'label_all.npy')
    data = np.load(data_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)
    
    win_smpls = win_len * 2 if win_len else None
    labels_idx = []

    if len(sel_cls) < 6:
        for cls in sel_cls:
            labels_idx += np.argwhere(labels == cls).flatten().tolist() 
        sel_labels = [labels[idx] for idx in labels_idx]
        sel_data = [data[idx] for idx in labels_idx]
        labels = np.array(sel_labels)
        data = np.array(sel_data)

    if win_len:
        n_idx = 0
        n_win = [int(d.shape[0]/win_smpls) for d in data]
        smpls = sum(n_win)
        n_data = np.empty((smpls, win_len*2)) #np.empty((smpls, win_smpls)) 
        n_labels = [labels[idx] for idx in range(labels.shape[0]) for i in range(n_win[idx])]
        for idx in range(labels.shape[0]):
            for i in range(n_win[idx]):
                n_data[n_idx, :] = data[idx][win_smpls*i:win_smpls*(i+1)] #.reshape(-1, 2)
                n_idx += 1

        data = n_data        
        labels = np.array(n_labels)
    else: 
        seqs = [int(x.shape[0]/2) for x in data]
        seqs.sort()
        data = [d[:seqs[0]*2].reshape(-1, 2) for d in data]
        win_len = seqs[0]
        data = np.array(data)

    if task_type == 'reg':
        labels = labels.reshape(-1, 1) 
    else:
        _, labels = lbls_for_cls(labels, lbls_list=sel_cls)
        labels = labels.reshape(-1, 1)
    data = data.astype(np.float32)    

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = X_train.reshape(X_train.shape[0], -1, 2)
    X_test = X_test.reshape(X_test.shape[0], -1, 2)
    
    return X_train, X_test, y_train, y_test

def get_fft_data(data_dir,
                 sel_cls,
                 data_mode='amp',
                 task_type='cls',
                 scaling=True):
    
    data_path = os.path.join(data_dir, 'Data_win_fft.npy')
    labels_path = os.path.join(data_dir, 'label_win_fft.npy')
    seqs_path = os.path.join(data_dir, 'seqs_win_fft.npy')
    data = np.load(data_path) #shape: (18642, 256, 5)
    labels = np.load(labels_path) #shape: (18642,)
    seqs = np.load(seqs_path) #shape: (18642,)

    n_data = data.swapaxes(1, 2)
    amp_data = np.absolute(n_data)
    phase_data = np.angle(n_data)
    power_data = np.absolute(n_data)**2
    real_data = np.real(n_data)
    imag_data = np.imag(n_data)
    
    if data_mode == 'amp':
        data = amp_data
    elif data_mode == 'phase':
        data = phase_data
    elif data_mode == 'power':
        data == power_data

    labels_idx = []

    if len(sel_cls) < 6:
        for cls in sel_cls:
            labels_idx += np.argwhere(labels == cls).flatten().tolist()
        #sel_data = np.zeros((len(labels_idx), n_data.shape[1], n_data.shape[2]))
        sel_labels = [labels[idx] for idx in labels_idx]
        sel_data = [data[idx] for idx in labels_idx]
        sel_seqs = [seqs[idx] for idx in labels_idx]
        labels = np.array(sel_labels)
        data = np.array(sel_data)
        seqs = np.array(sel_seqs)

    if task_type == 'reg':
        labels = labels.reshape(-1, 1) 
    else:
        old_labels, labels = lbls_for_cls(labels, lbls_list=sel_cls)
        old_labels = old_labels.reshape(-1, 1)
        labels = labels.reshape(-1, 1)
    data = data.astype(np.float32)    

    X_train, X_test, y_train, y_test, old_y_train, old_y_test, seqs_train, seqs_test = train_test_split(data,
                                                                                                        labels,
                                                                                                        old_labels,
                                                                                                        seqs,
                                                                                                        test_size=0.1,
                                                                                                        random_state=42)
    if scaling:
        scaler = preprocessing.StandardScaler().fit(X_train.reshape(X_train.shape[0], -1))
        X_train = scaler.transform(X_train.reshape(X_train.shape[0], -1))
        X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1))
        X_train = X_train.reshape(X_train.shape[0], n_data.shape[1], n_data.shape[2])
        X_test = X_test.reshape(X_test.shape[0], n_data.shape[1], n_data.shape[2])
            
    return X_train, X_test, y_train, y_test, old_y_train, old_y_test, seqs_train, seqs_test

def histedges_equalN(seq_lengths, n_bins):
    npt = len(seq_lengths)
    return np.interp(np.linspace(0, npt, n_bins + 1),
                     np.arange(npt),
                     np.sort(seq_lengths))

def element_to_bucket_id(x, buckets_min, buckets_max):
    seq_length = x.shape[0]
    conditions_c = np.logical_and(np.less_equal(buckets_min, seq_length),
                                  np.less(seq_length, buckets_max))
    bucket_id = np.min(np.where(conditions_c))
    return bucket_id

def pad_sequence(x, max_len=None, padding_value=0):
    orig_length = x.shape[0]
    new_x = np.zeros((max_len, 512), dtype=np.float64)
    new_x[0:orig_length,:] = x
    return new_x

def broadcasting_app(x, window, offset):  # Window len = window, stride len/stepsize/offset = offset
    if len(x.shape) > 1:
        x = x.reshape(-1) #flatten the array for windowing

    nrows = ((x.size-window)//offset)+1
    return x[offset*np.arange(nrows)[:,None] + np.arange(window)]

def get_windowed_data(data, window, offset):
    wdata = []
    wseqs = []
    [wdata.append(broadcasting_app(x, window, offset)) for x in data]
    [wseqs.append(len(x)) for x in wdata]
    return np.asarray(wdata), np.asarray(wseqs)

def get_data(fileloc, window=512, offset=None):
    x_train = np.load(os.path.join(fileloc, "train.npy"))
    x_val = np.load(os.path.join(fileloc, "val.npy"))
    x_test = np.load(os.path.join(fileloc, "test.npy"))

    y_train = np.load(os.path.join(fileloc, "train_lbls.npy"))
    y_val = np.load(os.path.join(fileloc, "val_lbls.npy"))
    y_test = np.load(os.path.join(fileloc, "test_lbls.npy"))
        
    if offset is None:
        seqs_train = np.load(os.path.join(fileloc, "train_seqs.npy"))
        seqs_val = np.load(os.path.join(fileloc, "val_seqs.npy"))
        seqs_test = np.load(os.path.join(fileloc, "test_seqs.npy"))    
    else:
        x_train, seqs_train = get_windowed_data(x_train, window, offset)
        x_val, seqs_val = get_windowed_data(x_val, window, offset)
        x_test, seqs_test = get_windowed_data(x_test, window, offset)
    
    return x_train, x_val, x_test, y_train.astype(np.int16), y_val.astype(np.int16),\
            y_test.astype(np.int16), seqs_train, seqs_val, seqs_test

class train_generator(keras.utils.Sequence):            
    def _permute(self):
        #Shuffle the buckets
        self.b_ids = np.random.permutation(self.n_bins)
        
        # Shuffle bucket contents
        for key in self.b_ids:
            xbin = np.array(self.buckets[key]['x'])
            ybin = np.array(self.buckets[key]['y'])
            #print(xbin.shape)
            index_array = np.random.permutation(len(self.buckets[key]['x']))
            self.buckets[key]['x'] = xbin[index_array]
            self.buckets[key]['y'] = ybin[index_array]

    def on_epoch_end(self):
        self._permute()
    
    def __len__(self):
        """Denotes the number of batches per epoch which is equivalent to steps_per_batch"""
        return self.n_bins
    
    def __init__(self, n_bins, data, labels, seq_lengths, padding=None, padding_value=None):
        bucket_sizes, bucket_boundaries = np.histogram(seq_lengths, bins = histedges_equalN(seq_lengths, n_bins))
        #print(bucket_sizes)
        #print(bucket_boundaries)

        data_buckets = dict()
        boundaries = list(bucket_boundaries)
        buckets_min = boundaries[:-1]
        buckets_max = boundaries[1:]
        buckets_max[n_bins-1] += 1
        #print(buckets_min)
        #print(buckets_max)
        
        for x, y in zip(data, labels):
            b_id = element_to_bucket_id(x, buckets_min, buckets_max)
            if padding:
                if x.shape[0] < buckets_max[b_id]:
                    max_len = int(math.ceil(buckets_max[b_id] - 1))
                    x = pad_sequence(x, max_len=max_len, padding_value=padding_value)
                    
            if b_id in data_buckets.keys():
                data_buckets[b_id]['x'].append(x)
                data_buckets[b_id]['y'].append(y)
            else:
                data_buckets[b_id] = {} 
                data_buckets[b_id]['x'] = [x]
                data_buckets[b_id]['y'] = [y]    
    
        self.n_bins = n_bins
        self.buckets = data_buckets
        self._permute()
        
    def __getitem__(self, idx):
        key = self.b_ids[idx]
        #print(self.buckets[key]['x'].shape)
        
        if np.any(np.isnan(self.buckets[key]['x'])):
            print('Nan value somewhere in dataset!')
            
        return self.buckets[key]['x'], self.buckets[key]['y']
    
class val_generator(keras.utils.Sequence): 
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.n_bins
    
    def __init__(self, data, labels): 
        self.x, self.y = data, labels
        self.n_bins = data.shape[0]
        
    def __getitem__(self, idx):
        x = self.x[idx].reshape(1, self.x[idx].shape[0], self.x[idx].shape[1])
        label = np.zeros((1, self.y.shape[-1]), dtype=np.int16)
        label[0] = self.y[idx]
        return x, label

class test_generator(keras.utils.Sequence): 
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.n_bins
    
    def __init__(self, data, labels): 
        self.x, self.y = data, labels
        self.n_bins = data.shape[0]
        
    def __getitem__(self, idx):
        x = self.x[idx].reshape(1, self.x[idx].shape[0], self.x[idx].shape[1])
        return x