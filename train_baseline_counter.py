import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import copy
import types
import random
import argparse
import math
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
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
parser.add_argument('-base', type=str, default='/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/'\
                                                'Deep_Learning_Radar/Data/Counting',\
                                                help='Base location of data')

parser.add_argument('-datatype', type=str, default='stft', help='Data type: \'stft\' for spectrogram,' \
                                                                ' \'time\' for raw time series)')

parser.add_argument('-cv', type=str, default=None, help='Cross validation folds for GridSearch')
parser.add_argument('-mt', type=str, default='conv1d_stft', help='type of embedding model')
parser.add_argument('-window', type=int, default=256, help='Window length for embeddings to be extracted')
parser.add_argument('-outdir', type=str, default=None, help='Output folder for model to be saved')
parser.add_argument('-h1', type=int, default=None, help='Hidden units for dense_1 in lstm counting model')
parser.add_argument('-h2', type=int, default=None, help='Hidden units for dense_2 in lstm counting model')
#parser.add_argument('-epochs', type=int, default=10, help='Number of training epochs')
#parser.add_argument('-lr', type=float, default=1e-4, help='Optimization learning rate')
parser.add_argument('-tbs', type=int, default=64, help='Number of samples per training batch')


class KerasGeneratorRegressor(KerasRegressor):
    """
    Add fit_generator to KerasClassifier to get batches of sequences with similar length.
    """

    def fit(self, X, y, **kwargs):    
        # taken from keras.wrappers.scikit_learn.KerasClassifier.fit ###################################################
        if self.build_fn is None:
            self.model = self.__call__(**self.filter_sk_params(self.__call__))
        elif not isinstance(self.build_fn, types.FunctionType) and not isinstance(self.build_fn, types.MethodType):
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn.__call__))
        else:
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn))

        loss_name = self.model.loss
        if hasattr(loss_name, '__name__'):
            loss_name = loss_name.__name__
        if loss_name == 'categorical_crossentropy' and len(y.shape) != 2:
            y = to_categorical(y)
        
        ################################################################################################################
        model_dict = self.filter_sk_params(self.build_fn)
        model_str = '|'.join('{!s}={!s}'.format(key,val) for (key,val) in model_dict.items())
        
        n_bins = kwargs['n_bins']
        output_path = kwargs['output_path']
        epochs = self.sk_params['epochs']
        output_path = os.path.join(kwargs['output_path'], model_str)
        os.makedirs(output_path, exist_ok=True)
        
        #Callbacks for the training
        early_stopping = EarlyStopping(monitor="val_loss", patience=5, min_delta=1e-4, verbose=5, mode="auto")
        reduce_LR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)        
        #previous model naming convention: model_str+'.{epoch:02d}-{val_loss:.5f}.h5'
        model_checkpoint = ModelCheckpoint(os.path.join(output_path, 'best_val_loss_model.h5'),\
                                           monitor="val_loss", verbose=5, save_best_only=True, mode="auto")
        callbacks = [early_stopping, reduce_LR, model_checkpoint]
        
        if 'x_val' in kwargs and 'y_val' in kwargs:
            x_train = X
            y_train = y
            x_val = kwargs['x_val']
            y_val = kwargs['y_val']
            seqs_train = kwargs['seq_lengths']
        elif 'val_len' in kwargs:
            val_size = kwargs['val_len'] 
            seqs_train = kwargs['seq_lengths']
            x_train, x_val, y_train, y_val, seqs_train, seqs_val = train_test_split(X, y, seqs_train, test_size=val_size)
        else:
            raise ValueError('No validation data provided!')
            
        train_gen = train_generator(n_bins, x_train, y_train, seq_lengths=seqs_train, padding=True, padding_value=0.0)
        val_gen = val_generator(x_val, y_val)

        #for i, item in enumerate(train_gen):
        #    print(np.array(item[0]).shape)
        
        # Do not allocate all the memory for visible GPU
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)
        K.set_session(sess)
    
        self.__history = self.model.fit_generator(train_gen, validation_data=val_gen, \
                                                  validation_steps=len(val_gen),\
                                                  steps_per_epoch=len(train_gen),\
                                                  epochs=epochs,\
                                                  callbacks=callbacks)
        return self.__history

    """
    When score_func is a score function (default), higher value is good. 
    When it is a loss function, the lower the value, the better it is.
    In the latter case, the scorer object will sign-flip the outcome of the score_func.
    """
    def score(self, x, y, **kwargs):
        """Returns the mean loss on the given test data and labels.
        # Arguments
            x: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y: array-like, shape `(n_samples,)`
                True labels for `x`.
            **kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.evaluate`.
        # Returns
            score: float
                Mean accuracy of predictions on `x` wrt. `y`.
        """
        #kwargs = self.filter_sk_params(Sequential.evaluate_generator, kwargs)
        test_gen = val_generator(x, y)
        loss = self.model.evaluate_generator(test_gen, steps=len(test_gen))
        if isinstance(loss, list):
            return -loss[0]
        return -loss
    
    @property
    def history(self):
        return self.__history

def conv1d_embedding_model(filters1, filters2, conv_drop=None, reshape=None):
    global input_shape
    model = Sequential()
    if reshape:
        model.add(Reshape(reshape, input_shape=input_shape))
        model.add(Conv1D(filters=filters1, kernel_size=4, activation='relu'))
    else:
        model.add(Conv1D(filters=filters1, kernel_size=4, activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=filters2, kernel_size=2, activation='relu'))
    #model.add(MaxPooling1D(pool_size=2))
    return model

def lstm_embedding_model(hidden1, hidden2=None, num_layers=1, reshape=None):
    global input_shape
    model = Sequential()
    if reshape:
        model.add(Reshape(reshape, input_shape=input_shape))
        model.add(LSTM(hidden1, return_sequences=True, name='embedding_lstm_1'))
    else:
        model.add(LSTM(hidden1, return_sequences=True, input_shape=input_shape, name='embedding_lstm_1'))
    if num_layers == 2 and hidden2 is not None:
        model.add(LSTM(hidden2, return_sequences=True, name='embedding_lstm_2'))
    return model

def lstm_counting_model(model, count_hidden1, count_dense1, count_dense2,\
                        kernel_initializer='glorot_uniform', optimizer=None, learning_rate=0.0001, dropout=0.2):
    
    if optimizer == 'adam' or optimizer is None:
        adam = keras.optimizers.Adam(lr=learning_rate)
    
    model.add(Masking(mask_value=0.0, name='mask'))
    model.add(LSTM(count_hidden1, return_sequences=False, name='counting_lstm_1'))
    model.add(Dense(count_dense1, activation='relu', kernel_initializer=kernel_initializer, name='counting_dense_1'))
    model.add(Dropout(dropout))
    model.add(Dense(count_dense2, activation='relu', kernel_initializer=kernel_initializer, name='counting_dense_2'))
    model.add(Dense(1, kernel_initializer=kernel_initializer, name='output'))
    model.add(Activation('linear'))
    model.compile(loss='poisson', optimizer=adam, metrics=['mae'])
    return model
    
def conv1d_stft_test_keys():
    epochs = [1]
    lr = [1e-2]
    init = ['glorot_uniform']
    optimizers = ['adam'] 
    conv_drop= [0.2]
    dropout = [0.2]
    filters_1 = [64]
    filters_2 = [64]
    count_hidden1 = [32] 
    count_dense1 = [32]
    count_dense2 = [32]
    param_grid = dict(epochs=epochs, filters1=filters1, filters2=filters2, init=init,\
                      conv_drop=conv_drop, count_hidden1=count_hidden1,\
                      count_dense1=count_dense1, count_dense2=count_dense2,\
                      lr=lr, optimizer=optimizers, dropout=dropout)
    return param_grid
    
def lstm_stft_test_keys(args):
    epochs = [50] #[10, 20, 30]
    lr = [1e-4] #[1e-3, 1e-4, 1e-5]
    init = ['glorot_uniform']
    optimizers = ['adam']
    dropout = [0.2] #[0.2, 0.4]
    hidden1 = [64] #[32, 64]
    #hidden2 = [32] #[32, 64]
    count_hidden1 = [32] #[32, 64]
    count_dense1 = [64] #[32, 64, 128]
    count_dense2 = [16] #[32, 64]
    param_grid = dict(epochs=epochs, hidden1=hidden1,\
                      count_hidden1=count_hidden1, init=init,\
                      count_dense1=count_dense1, count_dense2=count_dense2,\
                      lr=lr, optimizer=optimizers, dropout=dropout)
    return param_grid

def lstm_stft_keys(args):
    epochs = [50]
    #lr = [1e-4, 1e-5]
    #init = ['glorot_uniform']
    #optimizers = ['adam']
    #dropout = [0.2, 0.4]
    hidden1 = [32, 64]
    count_hidden1 = [32, 64]
    if 'h1' in args and args.h1:
        count_dense1 = [ int(args.h1) ]
    else:
        count_dense1 = [32] #[32, 64]
    if 'h2' in args and args.h2:
        count_dense2 = [ int(args.h2) ]
    else:
        count_dense2 = [32] #[32, 16]
    
    print(count_dense1, count_dense2)
    param_grid = dict(epochs=epochs, hidden1=hidden1,\
                      count_hidden1=count_hidden1,\
                      count_dense1=count_dense1, count_dense2=count_dense2)
    return param_grid

def build_lstm_stft_model(hidden1, count_hidden1, count_dense1, count_dense2,\
                          init='normal', hidden2=None,\
                          lr=1e-4, optimizer='adam', dropout=None):
    
    model = lstm_embedding_model(hidden1, hidden2=hidden2)
    counting_model = lstm_counting_model(model, count_hidden1,\
                                         count_dense1, count_dense2,\
                                         kernel_initializer=init,\
                                         optimizer=optimizer, learning_rate=lr, dropout=dropout)
    return counting_model
    
def build_lstm_time_model(hidden1, count_hidden1, count_dense1, count_dense2,\
                          init='glorot_uniform', hidden2=None,\
                          lr=1e-4, optimizer='adam', dropout=0.2):
    
    model = lstm_embedding_model(hidden1, hidden2=hidden2, reshape=(-1, 2))
    counting_model = lstm_counting_model(model, count_hidden1,\
                                         count_dense1, count_dense2,\
                                         kernel_initializer=init,\
                                         optimizer=optimizer, learning_rate=lr, dropout=dropout)
    return counting_model

def build_conv1d_stft_model(filters1, filters2, count_hidden1, count_dense1, count_dense2,\
                            init='normal', conv_drop=None,\
                            lr=1e-2, optimizer='adam', dropout=None):
    
    model = conv1d_embedding_model(filters1, filters2, conv_drop=conv_drop)
    counting_model = lstm_counting_model(model, count_hidden1,\
                                         count_dense1, count_dense2,\
                                         kernel_initializer=init,\
                                         optimizer=optimizer, learning_rate=lr, dropout=dropout)
    return counting_model
    
def build_conv1d_time_model(filters1, filters2, count_hidden1, count_dense1, count_dense2,\
                            init='normal', conv_drop=None,\
                            lr=1e-2, optimizer='adam', dropout=None):
    
    model = conv1d_embedding_model(filters1, filters2, conv_drop=conv_drop, reshape=(-1, 2))
    counting_model = lstm_counting_model(model, count_hidden1, count_dense1, count_dense2,\
                                         kernel_initializer=init,\
                                         optimizer=optimizer, learning_rate=lr, dropout=dropout)
    return counting_model
    

args=parser.parse_args()
data_type = args.datatype
base_path = args.base
model_type = args.mt
out_dir = args.outdir
window = int(args.window)
cv = int(args.cv)
batch_size = int(args.tbs)

if data_type == 'stft':
    fileloc = os.path.join(base_path, 'downstream_stft')
elif data_type == 'time':
    fileloc = os.path.join(base_path, 'downstream_time')
else:
    raise ValueError('Only stft/time are valid data types')
    
x_train, x_val, x_test, y_train, y_val, y_test, seqs_train, seqs_val, seqs_test = get_data(fileloc)
n_bins = int(len(seqs_train)/batch_size)

if cv:
    x_train = np.array(x_train.tolist() + x_val.tolist())
    y_train = np.array(y_train.tolist() + y_val.tolist())
    seqs_train = np.array(seqs_train.tolist() + seqs_val.tolist())
    
assert x_train.shape[0] == y_train.shape[0] == seqs_train.shape[0]

n_timesteps, n_features = None, window*2
input_shape=(n_timesteps, n_features)

output_path = os.path.join(out_dir, model_type)
os.makedirs(output_path, exist_ok=True)

models = {
    'lstm_stft': build_lstm_stft_model,
    'lstm_time': build_lstm_time_model,
    'conv1d_stft': build_conv1d_stft_model,
    'conv1d_time': build_conv1d_time_model
}

params = {
    'lstm_stft': lstm_stft_keys(args),
    'lstm_time': lstm_stft_test_keys(args)
}

model = KerasGeneratorRegressor(build_fn=models[model_type], verbose=1)
if cv:
    grid = GridSearchCV(estimator=model, param_grid=params[model_type], n_jobs=1, cv=cv)
    grid_result = grid.fit(x_train, y_train, n_bins=n_bins, val_len=x_val.shape[0],\
                           seq_lengths=seqs_train, output_path=output_path)
else:
    grid = GridSearchCV(estimator=model, param_grid=params[model_type], n_jobs=1, cv=[(slice(None), slice(None))])
    grid_result = grid.fit(x_train, y_train, x_val=x_val, y_val=y_val, n_bins=n_bins,\
                           seq_lengths=seqs_train, output_path=output_path)
                      

print("Best: {} using {}".format(grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("{} {} with: {}".format(mean, stdev, param))

score = grid.score(x_test, y_test)
print('Loss on Test set: ', -score)

sort_by='mean_test_score'
unique_ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
out_file = 'summary_' + str(unique_ts) + '.csv'

frame = pd.DataFrame(grid_result.cv_results_)
df = frame.filter(regex='^(?!.*param_).*$')
df = df.sort_values([sort_by], ascending=False)
df = df.reset_index()
df = df.drop(['rank_test_score', 'index'], 1)
df.to_csv(os.path.join(output_path, out_file), index=False)
