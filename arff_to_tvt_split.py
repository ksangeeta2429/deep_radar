from arff import dump
from scipy.io.arff import loadarff
import sys
import os
import numpy as np
import pandas as pd

# Load arff into dataframe
filename = sys.argv[1]
data = loadarff(filename)
df = pd.DataFrame(data[0])

# Create train, val, test arffs
train_indices = np.loadtxt('train_indices.txt', dtype=int)-1
val_indices = np.loadtxt('val_indices.txt', dtype=int)-1
test_indices = np.loadtxt('test_indices.txt', dtype=int)-1

data_train = df.iloc[train_indices]
data_val = df.iloc[val_indices]
data_test = df.iloc[test_indices]

relname = data[1].name

dump(os.path.join(os.path.dirname(filename), 'train.arff')
      , data_train.values
      , relation=relname + '_train'
      , names=df.columns)

dump(os.path.join(os.path.dirname(filename), 'val.arff')
      , data_val.values
      , relation=relname + '_val'
      , names=df.columns)

dump(os.path.join(os.path.dirname(filename), 'test.arff')
      , data_val.values
      , relation=relname + '_test'
      , names=df.columns)