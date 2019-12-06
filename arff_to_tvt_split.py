from arff import dump
from scipy.io.arff import loadarff
import sys
import os
import numpy as np
import pandas as pd

# Load arff into dataframe
filename = '1006_f_465_humans_490_bikes.arff'
data = loadarff(filename)


df = pd.DataFrame(data[0]).astype({'Target Count_1005': 'int'})

# Convert label into string from bytestring
df['classLabel_1006'] = df['classLabel_1006'].str.decode("utf-8")

# Create train, val, test arffs
train_indices = np.loadtxt('train_indices.txt', dtype=int)-1  # 0-indexing
val_indices = np.loadtxt('val_indices.txt', dtype=int)-1      # 0-indexing
test_indices = np.loadtxt('test_indices.txt', dtype=int)-1    # 0-indexing

data_train = df.iloc[train_indices]
data_val = df.iloc[val_indices]
data_test = df.iloc[test_indices]

relname = data[1].name

dump(os.path.join(os.path.dirname(filename), 'train_' + filename)
      , data_train.values
      , relation=relname + '_train'
      , names=df.columns)

dump(os.path.join(os.path.dirname(filename), 'val_' + filename)
      , data_val.values
      , relation=relname + '_val'
      , names=df.columns)

dump(os.path.join(os.path.dirname(filename), 'test_' + filename)
      , data_test.values
      , relation=relname + '_test'
      , names=df.columns)