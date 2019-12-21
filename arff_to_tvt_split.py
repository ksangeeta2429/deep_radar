from arff import dump
from scipy.io.arff import loadarff
import sys
import os
import numpy as np
import pandas as pd

def extract_list_to_file(in_list, line_indices, outfile):
      outlist = [ in_list[i] for i in line_indices ]

      with open(outfile, 'w') as filehandle:
            filehandle.writelines("%s" % y for y in outlist)

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

# Write train, valid, test files
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

# Write train, val, test file orders
fileorder = '1006_f_465_humans_490_bikes_file_order.txt'
with open(fileorder) as f:
      lines = f.readlines()

lines = [ x for x in lines if "Metadata" not in x] # Delete lines containing "Metadata"
assert len(lines)==len(df)

extract_list_to_file(lines, train_indices, os.path.join(os.path.dirname(fileorder), 'train_' + fileorder))
extract_list_to_file(lines, val_indices, os.path.join(os.path.dirname(fileorder), 'val_' + fileorder))
extract_list_to_file(lines, test_indices, os.path.join(os.path.dirname(fileorder), 'test_' + fileorder))