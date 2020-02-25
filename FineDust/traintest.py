from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from keras.layers import LSTM
from keras.layers import Conv2D, MaxPooling2D, TimeDistributed
from keras.layers import Dense, Dropout
from keras.layers import Flatten, Reshape
from keras.layers import Embedding
from keras.models import Sequential
from keras.models import load_model
from keras import optimizers
from keras.regularizers import L1L2
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import numpy as np
import math
import pandas as pd
import os
import gc
import matplotlib.pyplot as plt

scaler = StandardScaler(-2,2)

def ml_linear_regression(X_train, X_test, Y_train, Y_test):
    del [X_train['locale'], X_test['locale']]
    del [X_train['date'], X_test['date']]
    
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    
    del [X_train['PM10'], X_test['PM10']]
    gc.collect()
        
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    sgd=optimizers.SGD(lr=0.003, momentum=0.0, decay=0.0, nesterov=True)

    model=Sequential()
    model.add(Dense(1, input_shape=(11,), kernel_initializer='normal', activation='linear'))
    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, Y_train, batch_size=1000, epochs=5000, verbose=1)
    loss = history.history['loss']

    x_epochs = range(1, len(loss) + 1)

    plt.plot(x_epochs, loss, 'b', label='Training loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

    score = model.evaluate(X_test, Y_test, batch_size=1000)
    print(score)

def ml_logistic_regression(X_train, X_test, Y_train, Y_test):
    del [X_train['locale'], X_test['locale']]
    del [X_train['date'], X_test['date']]
    
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    
    del [X_train['PM10'], X_test['PM10']]
    gc.collect()
        
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    
    sgd=optimizers.SGD(lr=0.003, momentum=0.0, decay=0.0, nesterov=True)

    model=Sequential()
    model.add(Dense(4, input_shape=(11,), activation='softmax'))
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, Y_train, batch_size=1000, epochs=5000, verbose=1)
    loss = history.history['loss']

    x_epochs = range(1, len(loss) + 1)

    plt.plot(x_epochs, loss, 'b', label='Training loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

    score = model.evaluate(X_test, Y_test, batch_size=1000)
    print(score)

def dl_DNN(X_train, X_test, Y_train, Y_test):
    del [X_train['locale'], X_test['locale']]
    del [X_train['date'], X_test['date']]
    
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    
    del [X_train['PM10'], X_test['PM10']]
    gc.collect()
        
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    
    adam=optimizers.Adam(lr=0.007)

    model = Sequential()
    model.add(Dense(12, input_dim=11, kernel_initializer='glorot_normal', activation='relu', name='H1'))
    model.add(Dropout(0.9))
    model.add(Dense(10, kernel_initializer='glorot_normal', activation='relu', name='H2'))
    model.add(Dropout(0.9))
    model.add(Dense(8, kernel_initializer='glorot_normal', activation='relu', name='H3'))
    model.add(Dropout(0.9))
    model.add(Dense(4, kernel_initializer='glorot_normal', activation='softmax'))
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, Y_train, batch_size=1000, epochs=5000, verbose=1)
    loss = history.history['loss']

    x_epochs = range(1, len(loss) + 1)

    plt.plot(x_epochs, loss, 'b', label='Training loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

    score = model.evaluate(X_test, Y_test, batch_size=1000)
    print(score)

def dl_LSTM(X_train, X_test, Y_train, Y_test):
    ts = 4
        
        
    del [X_train['PM10'], X_test['PM10']]
    del [X_train['date'], X_test['date']]
    gc.collect()
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    print(X_train.shape)
    
    X_train = X_train.reshape(X_train.shape[0], ts, 11)
    X_test = X_test.reshape(X_test.shape[0], ts, 11)

    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    
    rmsprop=optimizers.RMSprop()
              
    model=Sequential()
    model.add(LSTM(64, input_shape=(ts, 11), activation='relu', kernel_initializer='glorot_normal',
                   bias_initializer='glorot_normal', recurrent_initializer='glorot_normal'))
    model.add(Dropout(0.3))
    model.add(Dense(4, kernel_initializer='glorot_normal',
                    bias_initializer='glorot_normal', activation='softmax'))
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    history = model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)
    loss = history.history['loss']

    x_epochs = range(1, len(loss) + 1)

    plt.plot(x_epochs, loss, 'b', label='Training loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

    score = model.evaluate(X_test, Y_test, batch_size=1000)
    print(score)
    
def dl_StackLSTM(X_train, X_test, Y_train, Y_test):
    pass
    
def dl_CNNLSTM(X_train, X_test, Y_train, Y_test):
    del [X_train['locale'], X_test['locale']]
    del [X_train['date'], X_test['date']]
    
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    
    del [X_train['PM10'], X_test['PM10']]
    gc.collect()
        
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    
    X_train = np.reshape(X_train, X_train.shape+(1,))
    X_test = np.reshape(X_test, X_test.shape+(1,))
    X_train = X_train.reshape(-1,1,11,1)
    X_test = X_test.reshape(-1,1,11,1)
    print(X_train.shape)
    print(X_test.shape)
    
    # Convolution
    kernel_size = (1,2)
    filters = 256
    pool_size = (1,2)
    
    # LSTM
    lstm_output_size = 64
    
    rmsprop=optimizers.RMSprop()
    
    CNN = Sequential()
    CNN.add(Conv2D(input_shape=(1,11,1),
                     filters=filters,
                     kernel_size=kernel_size,
                     padding='same',
                     activation='relu',
                     strides=1))
    CNN.add(MaxPooling2D(pool_size=pool_size))
    CNN.add(Dropout(0.3))
    CNN.add(Flatten())

    model=Sequential()
    model.add(TimeDistributed(CNN))
    model.add(LSTM(lstm_output_size, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(LSTM(lstm_output_size, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(LSTM(lstm_output_size, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(256))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy',  metrics=['accuracy'])
    model.build(input_shape=(1,11,1,
                             ))
    model.summary()
    
    history = model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)
    loss = history.history['loss']

    x_epochs = range(1, len(loss) + 1)

    plt.plot(x_epochs, loss, 'b', label='Training loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

    score = model.evaluate(X_test, Y_test, batch_size=1000)
    print(score)
    
if __name__ == "__main__":

    data_types ={'date':'str',
                 'locale':'str',
                 'PM10':'float16',
                 'NO2':'float16',
                 'CO':'float16',
                 'SO2':'float16',
                 'avg_tmp':'float16',
                 'min_tmp':'float16',
                 'max_tmp':'float16',
                 'precipitation':'float16',
                 'max_inst_wind':'float16',
                 'max_inst_wind_direct':'float16',
                 'max_avg_wind_direct':'float16',
                 'avg_wind':'float16',
                 'min_humid':'float16',
                 'avg_humid':'float16',
                 'avg_hPa':'float16',
                 'avg_total_cloud':'float16',
                 'avg_mid_cloud':'float16',
                 'avg_gtmp':'float16',
                 'avg_gtmp5':'float16',
                 'avg_gtmp10':'float16',
                 'avg_gtmp150':'float16',
                 'avg_gtmp300':'float16'}

    df_data = pd.read_csv('data/prep_data.csv', dtype=data_types, engine='c',
                             parse_dates=['date'])
    df_data['date'] = df_data['date'].dt.strftime('%Y-%m-%d')
    df_data = df_data.drop(['min_tmp', 'max_tmp', 'max_inst_wind', 'max_avg_wind_direct', 'min_humid', 'avg_mid_cloud',
                            'avg_gtmp5', 'avg_gtmp10', 'avg_gtmp150', 'avg_gtmp300'], axis=1)
    gc.collect()

    # PM10 encoding
    df_data['PM10'] = np.where(df_data['PM10']>=31,
                               np.where(df_data['PM10']>=81,
                                        np.where(df_data['PM10']>=151, 3, 2), 1), 0)
    
    print(df_data['PM10'].describe())

    # divide X
    X_test = df_data[df_data['date'] > '2016-12-31']
    X_train = df_data[df_data['date'] < '2017-01-01']

    gc.collect()
    
    X_train = X_train.sort_values(['locale', 'date'])
    X_test = X_test.sort_values(['locale', 'date'])
    gc.collect()
    
    Y_train = X_train['PM10'].astype(int).to_numpy()
    Y_test = X_test['PM10'].astype(int).to_numpy()
    #ml_linear_regression(X_train, X_test, Y_train, Y_test)
    #ml_logistic_regression(X_train, X_test, Y_train, Y_test)
    #dl_DNN(X_train, X_test, Y_train, Y_test)
    #dl_LSTM(X_train, X_test, Y_train, Y_test)
    dl_CNNLSTM(X_train, X_test, Y_train, Y_test)