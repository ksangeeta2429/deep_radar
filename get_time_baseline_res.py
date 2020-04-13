import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import LSTM, Dense, Conv1D, Flatten, Dropout, BatchNormalization, MaxPooling1D
from keras.models import Sequential
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np

def lbls_for_cls(labels, lbls_list=None):
    new_labels = [i for i in range(len(lbls_list))]
    for i, lbl in enumerate(lbls_list):
        labels[labels == lbl] = new_labels[i]

    return labels

def pretty_print_details(sel_cls, win_len, model_type, data_mode, stride, task_type):
    data_mode = data_mode if data_mode else 'IQ'
    task = 'Classification' if task_type == 'cls' else 'Regression'
    print('****************************************************************************')
    print('Task: {} Classes: {} Model type: {} Data: {} Window: {} Stride: {}'.format(task, sel_cls,
                                                                                      model_type, data_mode,
                                                                                      win_len, stride))
    
def get_model_stats(model):
    for layer in model.layers:
        if len(layer.get_weights()) > 0:
            print('Layer: ', layer.name)
            weights = layer.get_weights()[0]
            print('Min: {} Avg: {} Max: {} Zeros: {}'.format(weights.min(), np.mean(weights), weights.max(), len(np.where(weights == 0))))
            
def get_model_str(sel_cls, win_len, validation_split):
    model_str = ''
    for cls in sel_cls:
        model_str += str(cls) + '_'
    
    model_str += str(win_len)
    model_str += '_' + str(int(validation_split*100)) if validation_split else '_None' 
    return model_str

def build_lstm_time_model(hidden_1,
                          counting_dense_1,
                          counting_dense_2,
                          kernel_initializer='normal',
                          dropout=None,
                          optimizer=None,
                          input_shape=(256, 2),
                          task_type=None,
                          n_classes=2):
    
    model = Sequential()
    model.add(LSTM(hidden_1, return_sequences=False, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dense(counting_dense_1, activation='relu', name='counting_dense_1'))
    model.add(Dropout(dropout))
    model.add(Dense(counting_dense_2, activation='relu', name='counting_dense_2'))
    if task_type == 'reg':
        model.add(Dense(1, activation='softplus', name='output'))
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])
    else:
        model.add(Dense(n_classes, activation='softmax', name='output'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['sparse_categorical_accuracy'])
        
    return model
    
def build_conv1d_time_model(filters_1,
                            counting_dense_1, 
                            counting_dense_2,
                            kernel_initializer='normal',
                            dropout=None,
                            optimizer=None,
                            input_shape=(256, 2),
                            task_type=None,
                            n_classes=2):
    
    model = Sequential()
    model.add(Conv1D(filters_1, kernel_size=4, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(counting_dense_1, activation='relu', name='counting_dense_1'))
    model.add(Dropout(dropout))
    model.add(Dense(counting_dense_2, activation='relu', name='counting_dense_2'))
    if task_type == 'reg':
        model.add(Dense(1, activation='softplus', name='output'))
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])
    else:
        model.add(Dense(n_classes, activation='softmax', name='output'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['sparse_categorical_accuracy'])
        
    return model

def get_data(sel_cls,
             win_len=None, 
             task_type='cls',
             stride=None,
             data_mode=None):
    
    data_dir = '/scratch/sk7898/deep_radar/data'
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
        n_stride = stride * 2 if stride else win_smpls
        n_len = np.where(np.array([len(d) for d in data]) >= win_smpls)
        data = data[n_len]
        labels = labels[n_len]
        n_win = [(len(d)-win_smpls)//n_stride + 1 for d in data]
        n_smpls = sum(n_win)
        new_data = np.empty((n_smpls, win_smpls)) 
        new_labels = [labels[idx] for idx in range(labels.shape[0]) for i in range(n_win[idx])]
    
        for idx in range(data.shape[0]):
            for i in range(n_win[idx]):
                new_data[n_idx, :] = data[idx][i*n_stride:i*n_stride+win_smpls]
                n_idx += 1

        if data_mode == 'amp':
            amp_data = [np.absolute(d[::2] + 1j*d[1::2]) for d in new_data]
            new_data = np.array(amp_data)
    
        data = new_data        
        labels = np.array(new_labels)
        assert data.shape[0] == labels.shape[0]
        
    else: 
        seqs = [int(x.shape[0]/2) for x in data]
        seqs.sort()
        data = [d[:seqs[0]*2].reshape(-1, 2) for d in data]
        win_len = seqs[0]
        data = np.array(data)

    if task_type == 'reg':
        labels = labels.reshape(-1, 1) 
    else:
        labels = lbls_for_cls(labels, lbls_list=sel_cls)
        labels = labels.reshape(-1, 1)
    data = data.astype(np.float32)    
    
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    if data_mode == 'amp':
        X_train = X_train.reshape(X_train.shape[0], -1, 1)
        X_test = X_test.reshape(X_test.shape[0], -1, 1)
    else:
        X_train = X_train.reshape(X_train.shape[0], -1, 2)
        X_test = X_test.reshape(X_test.shape[0], -1, 2)

    print('Train Shape: {} Test Shape: {}'.format(X_train.shape, X_test.shape))

    return X_train, X_test, y_train, y_test

    
if __name__ == '__main__':

    epochs = 20
    batch_size = 64
    learning_rate = 1e-4
    dropout = 0.3
    hidden_1 = 64
    filters_1 = 64
    counting_dense_1 = 256
    counting_dense_2 = 64
    task_type = 'cls'
    validation_split = 0.1
    model_path = '/scratch/sk7898/pedbike/models'
    verbose = 0
    data_mode = None #'amp'

    model_list = ['lstm', 'conv']
    stride_list = [128]
    cls_list = [[1, 2]] #[3, 4], [1, 2, 3], [1, 2, 3, 4]
    win_list = [256]

    for sel_cls in cls_list:
        for win_len in win_list:
            for stride in stride_list:
                input_shape = (win_len, 1) if data_mode else (win_len, 2)
                X_train, X_test, y_train, y_test = get_data(sel_cls=sel_cls,
                                                            win_len=win_len,
                                                            task_type=task_type,
                                                            stride=stride,
                                                            data_mode=data_mode)
                    
                for model_type in model_list:
                    pretty_print_details(sel_cls, win_len, model_type, data_mode, stride, task_type)
                    optimizer = keras.optimizers.Adam(lr=learning_rate)
                    if model_type == 'lstm':
                        model = build_lstm_time_model(hidden_1,
                                                      counting_dense_1,
                                                      counting_dense_2,
                                                      dropout=dropout,
                                                      optimizer=optimizer,
                                                      input_shape=input_shape,
                                                      n_classes=len(sel_cls),
                                                      task_type=task_type)
                    else:
                        model = build_conv1d_time_model(filters_1,
                                                        counting_dense_1,
                                                        counting_dense_2,
                                                        dropout=dropout,
                                                        optimizer=optimizer,
                                                        input_shape=input_shape,
                                                        n_classes=len(sel_cls),
                                                        task_type=task_type)

                    #print(model.summary())
                    model_path = os.path.join(model_path, model_type)
                    model_str = get_model_str(sel_cls, win_len, validation_split)
                    model_path = os.path.join(model_path, model_str)

                    if not os.path.isdir(model_path):
                        os.makedirs(model_path)

                    latest_model_path = os.path.join(model_path, 'latest_model.h5')
                    best_valid_loss_model_path = os.path.join(model_path, 'model_best_valid_loss.h5')

                    early_stopping = EarlyStopping(monitor='val_loss', patience=6, min_delta=1e-4, verbose=5, mode='auto')
                    latest_ckpt = ModelCheckpoint(latest_model_path,
                                                  save_weights_only=False,
                                                  verbose=1)

                    best_valid_loss_ckpt = ModelCheckpoint(best_valid_loss_model_path,
                                                           monitor='val_loss',
                                                           verbose=5,
                                                           save_best_only=True,
                                                           mode='auto')

                    reduce_LR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)        
                    callbacks = [reduce_LR] #latest_ckpt, best_valid_loss_ckpt, early_stopping

                    if validation_split:
                        H_train = model.fit(x=X_train,
                                            y=y_train,
                                            batch_size=batch_size,
                                            validation_split=validation_split,
                                            epochs=epochs,
                                            shuffle=True,
                                            verbose=verbose,
                                            callbacks=callbacks)
                    else:
                        H_train = model.fit(x=X_train,
                                            y=y_train,
                                            batch_size=batch_size,
                                            validation_data=(X_test, y_test),
                                            epochs=epochs,
                                            shuffle=True,
                                            verbose=verbose,
                                            callbacks=callbacks)

                    if task_type == 'reg':
                        predictions = model.predict(x=X_test)
                        predictions = [0 if p > 0.5 else 1 for p in predictions]

                        inc = 0
                        for t, p in zip(y_test, predictions):
                            inc += 0 if t[0] == p else 1

                        print('Total Test: {} Incorrect: {}'.format(len(y_test), inc))
                    else:
                        predictions = model.evaluate(x=X_test, y=y_test)
                        print(predictions)
