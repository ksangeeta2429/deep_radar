import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import numpy as np
import keras
from keras.layers import LSTM, Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from data import get_fft_data

def lbls_for_cls(labels, lbls_list=None):
    new_labels = [i for i in range(len(lbls_list))]
    for i, lbl in enumerate(lbls_list):
        labels[labels == lbl] = new_labels[i]

    return labels

def get_model_str(sel_cls, hidden_1=64, data_mode='amp', win_len=512):
    model_str = ''
    for cls in sel_cls:
        model_str += str(cls) + '_'
    
    model_str += data_mode + '_' + str(win_len) + '_hidden_' + str(hidden_1) 
    return model_str

def get_data(sel_cls,
                 data_mode='amp', 
                 task_type='cls',
                 scaling=True):
    
    data_dir = '/scratch/sk7898/radar_data/pedbike/regression_fft_data'
    data_path = os.path.join(data_dir, 'Data_win_fft.npy')
    labels_path = os.path.join(data_dir, 'label_win_fft.npy')
    seqs_path = os.path.join(data_dir, 'seqs_fft.npy')
    data = np.load(data_path, allow_pickle=True) #shape: (18642, 256, 5)
    labels = np.load(labels_path, allow_pickle=True) #shape: (18642,)
    seqs = np.load(seqs_path, allow_pickle=True) #shape: (18642,)

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
        labels = lbls_for_cls(labels, lbls_list=sel_cls)
        labels = labels.reshape(-1, 1)
    data = data.astype(np.float32)    

    
    X_train, X_test, y_train, y_test, seqs_train, seqs_test = train_test_split(data, 
                                                                               labels,
                                                                               seqs,
                                                                               test_size=0.1,
                                                                               random_state=42)
    
    if scaling:
        scaler = preprocessing.StandardScaler().fit(X_train.reshape(X_train.shape[0], -1))
        X_train = scaler.transform(X_train.reshape(X_train.shape[0], -1))
        X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1))
        X_train = X_train.reshape(X_train.shape[0], n_data.shape[1], n_data.shape[2])
        X_test = X_test.reshape(X_test.shape[0], n_data.shape[1], n_data.shape[2])

    return X_train, X_test, y_train, y_test

def build_lstm_fft_model(hidden_1,
                         counting_dense_1,
                         counting_dense_2,
                         kernel_initializer='normal',
                         dropout_1=0.,
                         dropout_2=0.,
                         optimizer=None,
                         input_shape=(5, 256),
                         n_classes=2):
    
    model = Sequential()
    model.add(LSTM(hidden_1, return_sequences=False, input_shape=input_shape, name='lstm_1'))
    model.add(BatchNormalization())
    model.add(Dense(counting_dense_1, activation='relu', name='counting_dense_1'))
    model.add(Dropout(dropout_1))
    model.add(Dense(counting_dense_2, activation='relu', name='counting_dense_2'))
    model.add(Dropout(dropout_2))
    model.add(Dense(n_classes, activation='softmax', name='output'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['sparse_categorical_accuracy'])
        
    return model

def build_conv2d_fft_model(filters_1,
                           counting_dense_1,
                           counting_dense_2,
                           kernel_initializer='normal',
                           dropout=None,
                           optimizer=None,
                           input_shape=(5, 256, 1),
                           task_type=None,
                           n_classes=2):
    
    model = Sequential()
    model.add(Conv2D(filters_1, kernel_size=(2, 8), strides=(1, 8), data_format='channels_last', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(counting_dense_1, activation='relu', name='counting_dense_1'))
    model.add(Dropout(dropout))
    model.add(Dense(counting_dense_2, activation='relu', name='counting_dense_2'))
    model.add(Dense(n_classes, activation='softmax', name='output'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['sparse_categorical_accuracy'])
        
    return model

if __name__ == '__main__':
    
    epochs = 60
    batch_size = 64
    learning_rate = 1e-4
    dropout_1 = 0.4
    hidden_1 = 128
    filters_1 = 64
    counting_dense_1 = 256
    counting_dense_2 = 64

    cls_list = [[1, 2, 3, 4]] #[2, 4], [1, 2, 3], [1, 2, 3, 4], [1, 2], [1, 3], [1, 4], [2, 3], [1, 2, 3]
    model_type = 'lstm'
    data_mode = 'amp'
    radar_dir = '/scratch/sk7898/radar_data/pedbike'
    data_dir = os.path.join(radar_dir, 'regression_fft_data')
    model_dir = os.path.join(radar_dir, 'models')

    for sel_cls in cls_list:
        model_str = get_model_str(sel_cls, hidden_1=hidden_1, data_mode='amp', win_len=512)
        model_path = os.path.join(model_dir, model_type, model_str)

        if not os.path.isdir(model_path):
            os.makedirs(model_path)

        #X_train, X_test, y_train, y_test, old_y, old_y_test, _, _ = get_fft_data(sel_cls, data_mode='amp')
        X_train, X_test, y_train, y_test, old_y, old_y_test, _, _ = get_fft_data(data_dir, sel_cls=sel_cls, data_mode='amp')

        optimizer = keras.optimizers.Adam(lr=learning_rate)
        best_valid_loss_model_path = os.path.join(model_path, 'model_best_valid_loss_dp_4.h5')
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=6, 
                                       min_delta=1e-4,
                                       verbose=5, 
                                       mode='auto')
        best_valid_loss_ckpt = ModelCheckpoint(best_valid_loss_model_path,
                                               monitor='val_loss',
                                               verbose=1,
                                               save_best_only=True,
                                               mode='auto')
        
        callbacks = [best_valid_loss_ckpt]
        
        if model_type == 'conv':
            X_train = X_train[:, :, :, np.newaxis]
            X_test = X_test[:, :, :, np.newaxis]

            model = build_conv2d_fft_model(filters_1,
                                           counting_dense_1,
                                           counting_dense_2,
                                           dropout=dropout_1,
                                           optimizer=optimizer,
                                           n_classes=len(sel_cls),
                                           input_shape=(5, 256, 1))
        else:
            model = build_lstm_fft_model(hidden_1,
                                         counting_dense_1,
                                         counting_dense_2,
                                         dropout_1=dropout_1,
                                         optimizer=optimizer,
                                         input_shape=(5, 256),
                                         n_classes=len(sel_cls))    
        H_train = model.fit(x=X_train,
                            y=y_train,
                            batch_size=batch_size,
                            validation_split=0.1,
                            epochs=epochs,
                            shuffle=True,
                            verbose=0,
                            callbacks=callbacks)


        model.save(os.path.join(model_path, 'latest_model_dp_4.h5'))
        predictions = model.evaluate(x=X_test, y=y_test)
        print(predictions)