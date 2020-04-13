import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import copy
import types
import datetime
import random
import argparse
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import utils
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import *
from sklearn import preprocessing
from sklearn.utils import class_weight
from data import *

def lstm_embedding_model(hidden_1, hidden_2, reshape=None):
    
    global input_shape
    model = Sequential()
    if reshape:
        model.add(Reshape(reshape, input_shape=input_shape))
        model.add(LSTM(hidden_1, return_sequences=True))
    else:
        model.add(LSTM(hidden_1, return_sequences=True, input_shape=input_shape))
    return model

def lstm_counting_model(model, counting_hidden_1, counting_dense_1, counting_dense_2,\
                        kernel_initializer='normal', dropout=None):
        
    model.add(Masking(mask_value=0.0, name='mask'))
    model.add(LSTM(counting_hidden_1, return_sequences=False, name='counting_lstm_1'))
    model.add(Dense(counting_dense_1, activation='relu', kernel_initializer=kernel_initializer, name='counting_dense_1'))
    model.add(Dense(counting_dense_2, activation='relu', kernel_initializer=kernel_initializer, name='counting_dense_2'))
    model.add(Dense(1, activation='linear', name='output'))
    return model

def build_lstm_time_model(hidden_1, hidden_2, counting_hidden_1,\
                          counting_dense_1, counting_dense_2,\
                          kernel_initializer='normal', dropout=None):
    
    model = lstm_embedding_model(hidden_1, hidden_2, reshape=(-1, 2))
    counting_model = lstm_counting_model(model, counting_hidden_1,\
                                         counting_dense_1, counting_dense_2,\
                                         kernel_initializer=kernel_initializer,\
                                         dropout=dropout)
    return counting_model
    
def get_weights(y_train):
    classes = np.unique(y_train)
    weights = {}
    for cls in classes:
        x = np.where(y_train == cls)
        weights[str(cls - 1)] = len(x[0])/len(y_train)
    return weights

def train_model(model, output_dir, train_gen, val_gen, n_bins, class_weights=None,
                epochs=30, optimizer=None, learning_rate=1e-4):
    
    if optimizer == 'adam' or optimizer is None:
        optimizer = keras.optimizers.Adam(lr=learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(lr=learning_rate)
    elif optimizer == 'sgd':
        optimizer = keras.optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        
    output_path = os.path.join(output_dir, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    os.makedirs(output_path, exist_ok=True)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=1e-4, verbose=5, mode='auto')
    reduce_LR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)        
    model_checkpoint = ModelCheckpoint(os.path.join(output_path, 'best_val_loss_model.h5'),\
                                       monitor='val_loss', verbose=5, save_best_only=True, mode='auto')
    callbacks = [early_stopping, reduce_LR, model_checkpoint]

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    H_train = model.fit_generator(train_gen, validation_data=val_gen, validation_steps=1, class_weight=class_weights,\
                                  steps_per_epoch=n_bins, epochs=epochs, callbacks=callbacks)
    return H_train
    
if __name__ == '__main__':

    data_type = 'time'
    epochs = 50
    lr = 1e-4
    optimizer = 'adam' #'adam', 'sgd'
    dropout = 0.2
    hidden_1 = 64
    hidden_2 = 32
    counting_hidden_1 = 32
    counting_dense_1 = 64
    counting_dense_2 = 16
    window = 256
    loss = 'mse'
    model_type = 'lstm_time'
    base_path = '/scratch/sk7898/pedbike/window_256'
    out_dir = '/scratch/sk7898/radar_counting/models/'
    batch_size = 64
    scaling = False
    fileloc = os.path.join(base_path, 'downstream_time')

    x_train, x_val, x_test, y_train, y_val, y_test, seqs_train, seqs_val, seqs_test = get_data(fileloc)

    n_bins = int(len(seqs_train)/batch_size)
    assert x_train.shape[0] == y_train.shape[0] == seqs_train.shape[0]    

    class_weights = class_weight.compute_class_weight('balanced', np.unique(list(y_train)), y_train)

    n_timesteps, n_features = None, window*2
    input_shape=(n_timesteps, n_features)

    y_val = np.array(y_val).reshape(-1, 1)
    train_gen = train_generator(n_bins, x_train, y_train, seq_lengths=seqs_train, padding=True, padding_value=0.0)
    val_gen = val_generator(x_val, y_val)

    output_dir = os.path.join(out_dir + loss, model_type)
    model = build_lstm_time_model(hidden_1, hidden_2, counting_hidden_1, counting_dense_1,\
                                  counting_dense_2, dropout=dropout)

    history = train_model(model, output_dir, train_gen, val_gen, n_bins, class_weights=class_weights,\
                          epochs=epochs, optimizer=optimizer, learning_rate=lr)

    # Testing
    y_test = np.array(y_test).reshape(-1, 1)
    test_gen = val_generator(x_test, y_test)
    test_score = model.evaluate_generator(test_gen, steps=len(seqs_test))
    print(test_score)