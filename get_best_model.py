import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import copy
import csv
import types
import random
import argparse
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import utils
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers import *
from data import *


np.random.seed(42)

parser = argparse.ArgumentParser(description='Arguments')
parser.add_argument('-datadir', type=str, default='/scratch/sk7898/pedbike/window_256',\
                                                help='Base location of data')

parser.add_argument('-datatype', type=str, default='stft', help='Data type: \'stft\' for spectrogram,' \
                                                                ' \'time\' for raw time series)')
parser.add_argument('-modeldir', type=str, default='/scratch/sk7898/radar_counting/models', help='Output folder for model to be saved')

#For stft count_dense1 = 128
#Best: -1.2366702235959561 using {'count_dense1': 128, 'count_dense2': 16, 'count_hidden1': 64, 'dropout': 0.2, 'epochs': 50, 'hidden1': 32, 'init': 'normal', 'lr': 0.0001, 'optimizer': 'adam'}
args=parser.parse_args()
data_type = args.datatype
data_base_path = args.datadir
model_dir = args.modeldir
loss_list = []

if data_type == 'stft':
    model_base_path = os.path.join(model_dir, 'lstm_stft')
    fileloc = os.path.join(data_base_path, 'downstream_stft')
elif data_type == 'time':
    model_base_path = os.path.join(model_dir, 'lstm_time')
    fileloc = os.path.join(data_base_path, 'downstream_time')
else:
    raise ValueError('Data type not supported!')
    
x_train, x_val, x_test, y_train, y_val, y_test, seqs_train, seqs_val, seqs_test = get_data(fileloc)

for file in os.listdir(model_base_path):
    if 'csv' in file or 'count_dense1=32|count_dense2=32|count_hidden1=32' not in file:
        continue
    print('.............................................')
    model_path = os.path.join(model_base_path, file, 'best_val_loss_model.h5')
    lr_idx = file.rindex('lr')
    learning_rate = float(file[lr_idx+3:file.rindex('|')])
    model = load_model(model_path)
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=learning_rate), metrics=['mae'])
    test_gen = val_generator(x_test, y_test)
    loss = model.evaluate_generator(test_gen, steps=len(test_gen))
    print('File:', file, ' Loss:', loss[0], ' MAE:', loss[1])
    loss_list.append({'file': file, 'loss': loss[0], 'mae': loss[1]})
    

csv_columns = ['file', 'loss', 'mae']
csv_file = os.path.join(model_base_path, 'output.csv')
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in loss_list:
            writer.writerow(data)
except IOError:
    print("I/O error") 
