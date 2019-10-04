import os
import math
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import keras

np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

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

def get_data(fileloc, window=256):
    x_train = np.load(os.path.join(fileloc, "train.npy"))
    x_val = np.load(os.path.join(fileloc, "val.npy"))
    x_test = np.load(os.path.join(fileloc, "test.npy"))

    y_train = np.load(os.path.join(fileloc, "train_lbls.npy"))
    y_val = np.load(os.path.join(fileloc, "val_lbls.npy"))
    y_test = np.load(os.path.join(fileloc, "test_lbls.npy"))

    seqs_train = np.load(os.path.join(fileloc, "train_seqs.npy"))
    seqs_val = np.load(os.path.join(fileloc, "val_seqs.npy"))
    seqs_test = np.load(os.path.join(fileloc, "test_seqs.npy"))
    
    return x_train, x_val, x_test, y_train, y_val, y_test, seqs_train, seqs_val, seqs_test

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
        label = np.zeros((1,1), dtype=np.int16)
        label[0] = self.y[idx]
        #print(x.shape)
        return x, label
