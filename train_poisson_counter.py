import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import copy
import types
import random
import argparse
import datetime
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import utils
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers import *
from sklearn.metrics import mean_absolute_error
from data import *

def lstm_embedding_model(hidden_1, hidden_2,\
                         num_layers=1, reshape=None):
    global input_shape
    model = Sequential()
    if reshape:
        model.add(Reshape(reshape, input_shape=input_shape))
        model.add(LSTM(hidden_1, return_sequences=True))
    else:
        model.add(LSTM(hidden_1, return_sequences=True, input_shape=input_shape))
    return model

def lstm_counting_model(model, counting_hidden_1, counting_dense_1,\
                        counting_dense_2, kernel_initializer='normal',\
                        optimizer=None, learning_rate=0.001, dropout=None):
    
    if optimizer == 'adam' or optimizer is None:
        optimizer = keras.optimizers.Adam(lr=learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(lr=learning_rate)
    elif optimizer == 'sgd':
        optimizer = keras.optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        
    #model.add(Masking(mask_value=0.0, name='mask'))
    model.add(LSTM(counting_hidden_1, return_sequences=False, name='counting_lstm_1'))
    #model.add(Dense(counting_dense_1, activation='relu', kernel_initializer=kernel_initializer, name='counting_dense_1'))
    #model.add(Dense(counting_dense_2, activation='relu', kernel_initializer=kernel_initializer, name='counting_dense_2'))
    model.add(Dense(1, name='output'))
    model.add(Activation('softplus'))
    model.compile(loss='poisson', optimizer=optimizer)
    return model

def build_lstm_time_model(hidden_1, hidden_2, counting_hidden_1,\
                          counting_dense_1, counting_dense_2,\
                          kernel_initializer='normal',\
                          learning_rate=1e-2, optimizer='adam', dropout=None):
    
    model = lstm_embedding_model(hidden_1, hidden_2, reshape=(-1, 2))
    counting_model = lstm_counting_model(model, counting_hidden_1,\
                                         counting_dense_1, counting_dense_2,\
                                         kernel_initializer=kernel_initializer,\
                                         optimizer=optimizer, learning_rate=learning_rate, dropout=dropout)
    return counting_model

if True:
    data_type = 'time'
    window = 256
    loss = 'poisson'
    model_type = 'lstm_time'
    base_path = '/scratch/sk7898/pedbike/window_256'
    batch_size = 64
    scaling = False

if data_type == 'stft':
    fileloc = os.path.join(base_path, 'downstream_stft')
elif data_type == 'time':
    fileloc = os.path.join(base_path, 'downstream_time')
else:
    raise ValueError('Only stft/time are valid data types')
    
x_train, x_val, x_test, y_train, y_val, y_test, seqs_train, seqs_val, seqs_test = get_data(fileloc, window=512, offset=64)
n_bins = int(len(seqs_train)/batch_size)

assert x_train.shape[0] == y_train.shape[0] == seqs_train.shape[0]    

n_timesteps, n_features = None, window*2
input_shape=(n_timesteps, n_features)

train_gen = train_generator(n_bins, x_train, y_train, seq_lengths=seqs_train, padding=True, padding_value=0.0)
val_gen = val_generator(x_val, y_val)

epochs = 50
lr = 1e-4
optimizer = 'adam'
dropout = 0.2
hidden_1 = 64
hidden_2 = 32
counting_hidden_1 = 32
counting_dense_1 = 64
counting_dense_2 = 16

output_path = os.path.join('/scratch/sk7898/radar_counting/models/' + loss, model_type, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
os.makedirs(output_path, exist_ok=True)

#Callbacks for the training
early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=1e-4, verbose=5, mode="auto")
reduce_LR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)        
model_checkpoint = ModelCheckpoint(os.path.join(output_path, 'best_val_loss_model.h5'),\
                                   monitor='val_loss', verbose=5, save_best_only=True, mode="auto")
callbacks = [early_stopping, reduce_LR, model_checkpoint]
        
model = build_lstm_time_model(hidden_1, hidden_2, counting_hidden_1,\
                              counting_dense_1, counting_dense_2,\
                              learning_rate=lr, optimizer=optimizer, dropout=dropout)

H_train = model.fit_generator(train_gen, validation_data=val_gen, validation_steps=1,\
                              steps_per_epoch=n_bins, epochs=epochs, callbacks=callbacks)

test_gen = test_generator(x_test, y_test)
test_score = model.predict_generator(test_gen, steps=len(seqs_test))

if scaling:
    test_score = scaler.inverse_transform(test_score.reshape(-1, 1))
    
df_result = pd.DataFrame({'Actual' : [], 'Prediction' : []})
for i in range(len(x_test)):
    df_result = df_result.append({'Actual': y_test[i], 'Prediction': test_score[i][0]}, ignore_index=True)
    
print(df_result)
print(mean_absolute_error(y_test.astype(np.int8), test_score.astype(np.int8)))
