from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
from keras.layers import LSTM
from keras.layers import Conv2D, MaxPooling2D, TimeDistributed
from keras.layers import Dense, Dropout
from keras.layers import Flatten, Reshape
from keras.layers import Embedding, Input
from keras.models import Sequential
from keras.models import load_model
from keras import optimizers
from keras.regularizers import L1L2
from keras.preprocessing.sequence import TimeseriesGenerator
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import numpy as np
import math
import pandas as pd
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

scaler = RobustScaler()

def outliers_iqr(data):
    data = data.astype('float32')
    mean = data.mean()
    q1, q3 = np.percentile(data, [25,75])
    iqr = q3-q1
    lower_bound = q1-(iqr*1.5)
    upper_bound = q3+(iqr*1.5)
    
    return np.where((data>upper_bound) | (data<lower_bound), mean, data)

def outliers_z_score(data, threshold=3):
    data = data.astype('float32')
    mean = data.mean()
    std = data.std()
    z_scores = [(y-mean)/std for y in data]
    
    print(data, mean)
    
    return np.where(np.abs(z_scores)>threshold, mean, data)

def remove_outlier(data):
    input_data = data.columns
    print(input_data)
    
    for cols in data:
        if cols!='PM10' and cols!='locale':
            data[cols] = outliers_iqr(data[cols])

    return data    

def create_dataset(signal_data, look_back=4):
    dataX, dataY = [], []
    for i in range(len(signal_data)-look_back):
        dataX.append(signal_data[i:(i+look_back), 0])
        dataY.append(signal_data[i + look_back, 0])
        return np.array(dataX), np.array(dataY)
    

def ml_linear_regression(X_train, X_test, Y_train, Y_test):
    #del [X_train['locale'], X_test['locale']]
    del [X_train['date'], X_test['date']]
    
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    
    del [X_train['PM10'], X_test['PM10']]
    gc.collect()
        
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    print(X_train)
    
    sgd=optimizers.SGD(lr=0.003)

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
    #del [X_train['locale'], X_test['locale']]
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
    #del [X_train['locale'], X_test['locale']]
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
    model.add(Dense(12, input_dim=11, kernel_initializer='glorot_normal', bias_initializer='glorot_normal',
                    activation='relu', name='H1'))
    model.add(Dropout(0.1))
    model.add(Dense(10, kernel_initializer='glorot_normal', bias_initializer='glorot_normal',
                    activation='relu', name='H2'))
    model.add(Dropout(0.1))
    model.add(Dense(8, kernel_initializer='glorot_normal', bias_initializer='glorot_normal',
                    activation='relu', name='H3'))
    model.add(Dropout(0.1))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, Y_train, batch_size=100, epochs=250, verbose=1)
    loss = history.history['loss']

    x_epochs = range(1, len(loss) + 1)

    plt.plot(x_epochs, loss, 'b', label='Training loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

    score = model.evaluate(X_test, Y_test, batch_size=1000)
    print(score)

def dl_LSTM(X_train, X_test):
    ts = 4    
    n_features = 11 
            
    locale_list = list(X_train['locale'].value_counts().keys())
    
    train_list = []
    test_list = []
    train_target = []
    test_target = []
    
    print(locale_list)
    print(X_train)
    
    Xs_train = X_train
    Xs_test = X_test
    
    for locale in locale_list:
        X_train = Xs_train[Xs_train['locale']==locale]
        X_test = Xs_test[Xs_test['locale']==locale]
        
        X_train = X_train.sort_values(['date'])
        X_test = X_test.sort_values(['date'])
    
        Y_train = X_train['PM10'].astype(int)
        Y_test = X_test['PM10'].astype(int)
        
        del [X_train['locale'], X_test['locale']]
        del [X_train['date'], X_test['date']]
        del [X_train['PM10'], X_test['PM10']]
        
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        
        column_list = list(X_train)  
        
        for s in range(1, ts):
            tmp_train = X_train[column_list].shift(s)
            tmp_test = X_test[column_list].shift(s)
    
            tmp_train.columns = "shift_" + tmp_train.columns + "_" + str(s)
            tmp_test.columns = "shift_" + tmp_test.columns + "_" + str(s)
    
            X_train[tmp_train.columns] = X_train[column_list].shift(s)
            X_test[tmp_test.columns] = X_test[column_list].shift(s)
        
        X_train = X_train[ts-1:]
        X_test = X_test[ts-1:]
        Y_train = Y_train[ts-1:]
        Y_test = Y_test[ts-1:]
        
        train_list.append(X_train)
        test_list.append(X_test)
        train_target.append(Y_train)
        test_target.append(Y_test)
        
    X_train = pd.concat(train_list)
    X_test = pd.concat(test_list)
    Y_train = pd.concat(train_target)
    Y_test = pd.concat(test_target)
    
    print(Y_train.value_counts())
    print(Y_test.value_counts())
        
    gc.collect()
        
    print(X_train)
    print(Y_train)
    print(Y_train.shape)
            
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    Y_train = to_categorical(Y_train, 4)
    Y_test = to_categorical(Y_test, 4)
    
    X_train = np.reshape(X_train, X_train.shape+(1,))
    X_test = np.reshape(X_test, X_test.shape+(1,))
    X_train = X_train.reshape(-1,n_features,ts)
    X_test = X_test.reshape(-1,n_features,ts)
    print(X_train.shape)
    print(X_test.shape)
    
    # LSTM
    lstm_output_size = 64
    
    # batch_size
    batch_size = 32
    
    optimizer=optimizers.Adam(lr=0.0005)
    
    checkpoint_filepath = os.path.join('model', 'fresh_models', '{0}_LSTM.{1}-{2}.h5'.format('model', '{epoch:02d}', '{val_loss:.7f}'))
    checkpoint_callback = ModelCheckpoint(checkpoint_filepath, save_best_only=True, verbose=1)
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=10, verbose=1)
    callbacks=[checkpoint_callback, early_stopping_callback]

              
    model=Sequential()
    model.add(LSTM(64, input_shape=(n_features, ts), activation='relu', dropout=0.3))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
        
    history = model.fit(X_train, Y_train, batch_size=batch_size, 
                        validation_data=(X_test, Y_test), epochs=20)
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    x_epochs = range(1, len(loss) + 1)

    plt.plot(x_epochs, loss, 'b', label='Training loss')
    plt.plot(x_epochs, val_loss, 'r', label='Validation loss')
    plt.title('Loss')
    plt.legend()
    plt.show()
    
    plt.plot(x_epochs, acc, 'b', label='Training acc')
    plt.plot(x_epochs, val_acc, 'r', label='Validation acc')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

    score = model.evaluate(X_test, Y_test, batch_size=batch_size)
    print(score)
    
    Y_pred = model.predict(X_test)
    Y_pred = Y_pred.argmax(axis=1)
    Y_test = Y_test.argmax(axis=1)
    
    print(accuracy_score(Y_test, Y_pred))
    return score
    
def dl_StackedLSTM(X_train, X_test):
    ts = 4    
    n_features = 11    
    
    locale_list = list(X_train['locale'].value_counts().keys())
    
    train_list = []
    test_list = []
    train_target = []
    test_target = []
    
    print(locale_list)
    print(X_train)
    
    Xs_train = X_train
    Xs_test = X_test
    
    for locale in locale_list:
        X_train = Xs_train[Xs_train['locale']==locale]
        X_test = Xs_test[Xs_test['locale']==locale]
        
        X_train = X_train.sort_values(['date'])
        X_test = X_test.sort_values(['date'])
    
        Y_train = X_train['PM10'].astype(int)
        Y_test = X_test['PM10'].astype(int)
        
        del [X_train['locale'], X_test['locale']]
        del [X_train['date'], X_test['date']]
        del [X_train['PM10'], X_test['PM10']]
        
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        
        column_list = list(X_train)  
        
        for s in range(1, ts):
            tmp_train = X_train[column_list].shift(s)
            tmp_test = X_test[column_list].shift(s)
    
            tmp_train.columns = "shift_" + tmp_train.columns + "_" + str(s)
            tmp_test.columns = "shift_" + tmp_test.columns + "_" + str(s)
    
            X_train[tmp_train.columns] = X_train[column_list].shift(s)
            X_test[tmp_test.columns] = X_test[column_list].shift(s)
        
        X_train = X_train[ts-1:]
        X_test = X_test[ts-1:]
        Y_train = Y_train[ts-1:]
        Y_test = Y_test[ts-1:]
        
        train_list.append(X_train)
        test_list.append(X_test)
        train_target.append(Y_train)
        test_target.append(Y_test)
        
    X_train = pd.concat(train_list)
    X_test = pd.concat(test_list)
    Y_train = pd.concat(train_target)
    Y_test = pd.concat(test_target)
    
    print(Y_train.value_counts())
    print(Y_test.value_counts())
        
    gc.collect()
        
    print(X_train)
    print(Y_train)
    print(Y_train.shape)
            
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    Y_train = to_categorical(Y_train, 4)
    Y_test = to_categorical(Y_test, 4)
    
    X_train = np.reshape(X_train, X_train.shape+(1,))
    X_test = np.reshape(X_test, X_test.shape+(1,))
    X_train = X_train.reshape(-1,n_features,ts)
    X_test = X_test.reshape(-1,n_features,ts)
    print(X_train.shape)
    print(X_test.shape)
    
    #Y_train = Y_train.reshape(-1,1,4)
    #Y_test = Y_test.reshape(-1,1,4)
    print(Y_train.shape)
    print(Y_test.shape) 
    
    # LSTM
    lstm_output_size = 64
    
    # batch_size
    batch_size = 32
    
    optimizer=optimizers.Adam()
    
    checkpoint_filepath = os.path.join('model', 'fresh_models', '{0}_stackedLSTM.{1}-{2}.h5'.format('model', '{epoch:02d}', '{val_loss:.7f}'))
    checkpoint_callback = ModelCheckpoint(checkpoint_filepath, save_best_only=True, verbose=1)
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=10, verbose=1)
    callbacks=[checkpoint_callback, early_stopping_callback]

    
    model = Sequential()
    model.add(LSTM(lstm_output_size, input_shape=(n_features, ts), return_sequences=True, dropout=0.3,
                   activation='relu'))
    model.add(LSTM(lstm_output_size, return_sequences=True, dropout=0.3,
                   activation='relu'))
    model.add(LSTM(lstm_output_size, dropout=0.3,
                   activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    history = model.fit(X_train, Y_train, batch_size=batch_size, 
                        validation_data=(X_test, Y_test), epochs=20)
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    x_epochs = range(1, len(loss) + 1)

    plt.plot(x_epochs, loss, 'b', label='Training loss')
    plt.plot(x_epochs, val_loss, 'r', label='Validation loss')
    plt.title('Loss')
    plt.legend()
    plt.show()
    
    plt.plot(x_epochs, acc, 'b', label='Training acc')
    plt.plot(x_epochs, val_acc, 'r', label='Validation acc')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

    score = model.evaluate(X_test, Y_test, batch_size=batch_size)
    print(score)
    return score
    
def dl_CNNLSTM(X_train, X_test):
    ts = 4    
    n_features = 11    
    
    locale_list = list(X_train['locale'].value_counts().keys())
    
    train_list = []
    test_list = []
    train_target = []
    test_target = []
    
    print(locale_list)
    print(X_train)
    
    Xs_train = X_train
    Xs_test = X_test
    
    for locale in locale_list:
        X_train = Xs_train[Xs_train['locale']==locale]
        X_test = Xs_test[Xs_test['locale']==locale]
        
        X_train = X_train.sort_values(['date'])
        X_test = X_test.sort_values(['date'])
    
        Y_train = X_train['PM10'].astype(int)
        Y_test = X_test['PM10'].astype(int)
        
        del [X_train['locale'], X_test['locale']]
        del [X_train['date'], X_test['date']]
        del [X_train['PM10'], X_test['PM10']]
        
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        
        column_list = list(X_train)  
        
        for s in range(1, ts):
            tmp_train = X_train[column_list].shift(s)
            tmp_test = X_test[column_list].shift(s)
    
            tmp_train.columns = "shift_" + tmp_train.columns + "_" + str(s)
            tmp_test.columns = "shift_" + tmp_test.columns + "_" + str(s)
    
            X_train[tmp_train.columns] = X_train[column_list].shift(s)
            X_test[tmp_test.columns] = X_test[column_list].shift(s)
        
        X_train = X_train[ts-1:]
        X_test = X_test[ts-1:]
        Y_train = Y_train[ts-1:]
        Y_test = Y_test[ts-1:]
        
        train_list.append(X_train)
        test_list.append(X_test)
        train_target.append(Y_train)
        test_target.append(Y_test)
        
    X_train = pd.concat(train_list)
    X_test = pd.concat(test_list)
    Y_train = pd.concat(train_target)
    Y_test = pd.concat(test_target)
    
    print(Y_train.value_counts())
    print(Y_test.value_counts())
        
    gc.collect()
        
    print(X_train)
    print(Y_train)
    print(Y_train.shape)
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    Y_train = to_categorical(Y_train, 4)
    Y_test = to_categorical(Y_test, 4)
    
    X_train = np.reshape(X_train, X_train.shape+(1,))
    X_test = np.reshape(X_test, X_test.shape+(1,))
    X_train = X_train.reshape(-1,ts,1,n_features,1)
    X_test = X_test.reshape(-1,ts,1,n_features,1)
    print(X_train.shape)
    print(X_test.shape)
    
    #Y_train = Y_train.reshape(-1,1,4)
    #Y_test = Y_test.reshape(-1,1,4)
    print(Y_train.shape)
    print(Y_test.shape)
    
    print(X_train[0])
    print(X_train[1])
    
    # Convolution
    kernel_size = (1,2)
    filters = 256
    pool_size = (1,2)
    
    # LSTM
    lstm_output_size = 64
    
    # batch_size
    batch_size = 32
    
    optimizer=optimizers.Adam()
    
    checkpoint_filepath = os.path.join('model', 'fresh_models', '{0}_CNNLSTM.{1}-{2}.h5'.format('model', '{epoch:02d}', '{val_loss:.7f}'))
    checkpoint_callback = ModelCheckpoint(checkpoint_filepath, save_best_only=True, verbose=1)
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=1)
    callbacks=[checkpoint_callback]

    model=Sequential()
    model.add(TimeDistributed(Conv2D(filters=filters,
                                     kernel_size=kernel_size,
                                     padding='same',
                                     activation='relu'), input_shape=(None, 1, n_features , 1), name="Conv2D"))
    model.add(TimeDistributed(MaxPooling2D(pool_size=pool_size), name="MaxPooling2D"))
    model.add(TimeDistributed(Dropout(0.3), name="Dropout_CNN"))
    model.add(TimeDistributed(Flatten(), name="Flatten"))
    model.add(LSTM(lstm_output_size, return_sequences=True, activation='relu', 
                   name="LSTM1", dropout=0.3))
    model.add(LSTM(lstm_output_size, return_sequences=True, activation='relu', 
                   name="LSTM2", dropout=0.3))
    model.add(LSTM(lstm_output_size, activation='relu', 
                   name="LSTM3", dropout=0.3))
    model.add(Dense(256, name="FCN1"))
    model.add(Dense(4, activation='softmax', name="OUTPUT"))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',  metrics=['accuracy'])
    model.summary()
    
    history = model.fit(X_train, Y_train, batch_size=batch_size, 
                        validation_data=(X_test, Y_test), shuffle=False, epochs=10)
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    x_epochs = range(1, len(loss) + 1)

    plt.plot(x_epochs, loss, 'b', label='Training loss')
    plt.plot(x_epochs, val_loss, 'r', label='Validation loss')
    plt.title('Loss')
    plt.legend()
    plt.show()
    
    plt.plot(x_epochs, acc, 'b', label='Training acc')
    plt.plot(x_epochs, val_acc, 'r', label='Validation acc')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

    score = model.evaluate(X_test, Y_test, batch_size=batch_size)
    print(score)
    
if __name__ == "__main__":

    data_types ={'date':'str',
                 'locale':'str',
                 'PM10':'float16',
                 'NO2':'float16',
                 'CO':'float16',
                 'SO2':'float16',
                 'precipitation':'float16',
                 'max_inst_wind_direct':'float16',
                 'min_humid':'float16',
                 'avg_hPa':'float16',
                 'avg_mid_cloud':'float16',
                 'avg_gtmp150':'float16'}

    df_data = pd.read_csv('data/prep_data.csv', dtype=data_types, engine='c',
                             parse_dates=['date'])
    df_data['date'] = df_data['date'].dt.strftime('%Y-%m-%d')
    df_data = df_data.set_index('date')
    
    '''
    corr = df_data.corr()
    sns.heatmap(df_data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2, 
                vmax=1, vmin=-1)
    fig=plt.gcf()
    fig.set_size_inches(15,15)
    plt.savefig('variance.png', dpi=300)
    plt.show()
    
    del df_data['date']
    del df_data['PM10']
    del df_data['locale']
    
    df_data = df_data.astype(float)
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(
        df_data.values, i) for i in range(df_data.shape[1])]
    vif["features"] = df_data.columns
    
    '''
    
    #df_data = df_data[['date','locale','PM10','SO2','CO','NO2','avg_tmp',
    #                   'precipitation','max_inst_wind_direct','avg_wind',
     #                  'avg_humid','avg_hPa','avg_total_cloud','avg_gtmp']]
    gc.collect()
    
    
    print(df_data['locale'].value_counts())
    print(df_data['locale'].value_counts().keys())
    
    locale_list = list(df_data['locale'].value_counts().keys())
    score_list = []
    
    for locale in locale_list:
    #df_datal = df_data
        df_datal = df_data[df_data['locale'] == locale]
        
        print(df_data.dtypes)
        df_datal = df_datal.fillna(0)
        
        local_data = df_datal['locale']
        column_list = local_data.value_counts().keys()
        
        #df_data = remove_outlier(df_data)
        
        # PM10 encoding
        df_datal['PM10'] = np.where(df_datal['PM10']>=31,
                                    np.where(df_datal['PM10']>=81,
                                             np.where(df_datal['PM10']>=151, 3, 2), 1), 0)
        
        print(df_datal['PM10'].describe())
        
        df_datal = df_datal.reset_index()
    
        # divide X
        X_test = df_datal[df_datal['date'] > '2016-12-31']
        X_train = df_datal[df_datal['date'] < '2017-01-01']
        
        print(X_test['PM10'].value_counts())
        print(X_train['PM10'].value_counts())
        
        gc.collect()
        
        X_train = X_train.sort_values(['locale', 'date'])
        X_test = X_test.sort_values(['locale', 'date'])
        gc.collect()
        
        #ml_linear_regression(X_train, X_test, Y_train, Y_test)
        #ml_logistic_regression(X_train, X_test, Y_train, Y_test)
        #dl_DNN(X_train, X_test, Y_train, Y_test)
        score_list.append(dl_LSTM(X_train, X_test))
        #score_list.append(dl_StackedLSTM(X_train, X_test))
        #dl_CNNLSTM(X_train, X_test)
        gc.collect()
        
    print(score_list)
    
    