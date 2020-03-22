# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:59:45 2020

@author: wonwoo
"""
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import class_weight, resample
from keras.models import Sequential
from keras.layers import Dense, Dropout, GaussianNoise, Conv1D
from keras.layers import LSTM
from keras.layers import Conv2D, MaxPooling2D, TimeDistributed
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers import Flatten, Reshape
from keras.layers import Embedding, Input
from keras.models import Sequential
from keras.models import load_model
from keras import optimizers
from keras.regularizers import L1L2
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical, np_utils
import keras.backend as K
import tensorflow as tf
import numpy as np
import math
import pandas as pd
import os
import gc
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
scaler = StandardScaler()

def focal_loss_fixed(y_true, y_pred, gamma=2., alpha=.25):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return (-K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) 
            - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0)))
    
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def resampling(df_train):    
    train_freq = df_train['failure'].value_counts()    
    print(train_freq)
    train_freq_mean = train_freq[1]
    
    # Under & Over Sampling store_nbr
    df_list = []
    target_max = 2
    multiple = 10
    
    for i in range(0, target_max):
        df_list.append(df_train[df_train['failure']==i])
    
    for i in range(0, target_max):
        if i==0:
            df_list[i] = df_list[i].sample(n=int(train_freq_mean*multiple), random_state=123, replace=True)
        else:
            df_list[i] = df_list[i].sample(n=train_freq_mean, random_state=123, replace=True)
        
    df_sampling_train = pd.concat(df_list)
    train_freq = df_sampling_train['failure'].value_counts()
    
    return pd.DataFrame(df_sampling_train)
    
def DNN(train, valid, test):    
    X_train = train.drop(['datetime', 'failure'], axis=1)
    X_valid = valid.drop(['datetime', 'failure'], axis=1)
    X_test = test.drop(['datetime', 'failure'], axis=1)
    gc.collect()
    
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.fit_transform(X_valid)
    X_test = scaler.fit_transform(X_test)
    
    Y_train = train['failure']
    Y_valid = valid['failure']
    Y_test = test['failure']
    
    y_integers = Y_train
    #print(y_integers)
    class_weights = class_weight.compute_class_weight(None, np.unique(y_integers), y_integers)
    print(class_weights)
    d_class_weights = dict(enumerate(class_weights)) 
    #d_class_weights = {0:1.0, 1:1.0}
    optimizer=optimizers.Adam()
    
    Y_train = to_categorical(Y_train)
    Y_valid = to_categorical(Y_valid)
    Y_test = to_categorical(Y_test)
    
    model=Sequential()
    model.add(Dense(32, input_dim=19, kernel_initializer='glorot_normal',
                    bias_initializer='glorot_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, kernel_initializer='glorot_normal',
                    bias_initializer='glorot_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, kernel_initializer='glorot_normal',
                    bias_initializer='glorot_normal',  activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, kernel_initializer='glorot_normal',
                    bias_initializer='glorot_normal',  activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(32, kernel_initializer='glorot_normal',
                    bias_initializer='glorot_normal',  activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, kernel_initializer='glorot_normal',
                    bias_initializer='glorot_normal',  activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', f1_m])
    model.summary()

    history = model.fit(X_train, Y_train, batch_size=1000, epochs=100,
                        verbose=1, class_weight = d_class_weights,
                        validation_data=(X_valid, Y_valid), shuffle=True)
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    x_epochs = range(1, len(train_loss) + 1)

    plt.plot(x_epochs, train_loss, 'b', label='Training loss')
    plt.plot(x_epochs, val_loss, 'r', label='Validation loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

    score = model.evaluate(X_test, Y_test, batch_size=1000)
    print(score)
    
    Y_pred = model.predict(X_test)
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    
    #Y_pred = np.argmax(Y_pred, axis=1).reshape(-1,1)
    #Y_test = np.argmax(Y_test, axis=1).reshape(-1,1)
    print(confusion_matrix(Y_test, Y_pred))
    print(classification_report(Y_test, Y_pred, labels=[0, 1]))
    
def XGBClss(train, valid, test):
    X_train = train.drop(['datetime', 'failure'], axis=1)
    X_valid = valid.drop(['datetime', 'failure'], axis=1)
    X_test = test.drop(['datetime', 'failure'], axis=1)
    gc.collect()
    
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.fit_transform(X_valid)
    X_test = scaler.fit_transform(X_test)
    
    Y_train = train['failure']
    Y_valid = valid['failure']
    Y_test = test['failure']
    
    y_integers = Y_train
    #print(y_integers)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_integers), y_integers)
    print(class_weights)
    d_class_weights = dict(enumerate(class_weights)) 
    
    clf = XGBClassifier()
    parameters = {
            "n_estimator" : [100, 200, 300],
            "max_depth" : [ 3, 4, 5 ],
            "tree_method" : ['gpu_hist'],
            "predictor" : ['gpu_predictor']
     }

    grid = GridSearchCV(clf,
                        parameters, n_jobs=1,
                        scoring="f1_micro",
                        cv=3)
    
    grid.fit(X_train, Y_train)
    
    print(grid.best_score_)
    print(grid.best_params_)
    
    model = grid.best_estimator_
    model.fit(X_train, Y_train)
    
    Y_pred = model.predict(X_test)
    
    print(confusion_matrix(Y_test, Y_pred))
    print(classification_report(Y_test, Y_pred, labels=[0, 1, 2, 3, 4]))
    
    


if __name__=="__main__":
    data_type={'machineID':'uint8',
               'datetime':'str',
               'voltmean_24h':'float32',
               'rotatemean_24h':'float32',
               'pressuremean_24h':'float32',
               'vibrationmean_24h':'float32',
               'voltsd_24h':'float32',
               'rotatesd_24h':'float32',
               'pressuresd_24h':'float32',
               'vibrationsd_24h':'float32',
               'error1count':'uint8',
               'error2count':'uint8',
               'error3count':'uint8',
               'error4count':'uint8',
               'error5count':'uint8',
               'comp1':'float32',
               'comp2':'float32',
               'comp3':'float32',
               'comp4':'float32',
               'model':'uint8',
               'age':'uint8',
               'failure':'str'}
    
    features_path = os.path.join("data/labeled_features.csv")
    df_data = pd.read_csv(features_path, engine='c',
                          dtype=data_type, parse_dates=['datetime'])
    df_data['datetime'] = df_data['datetime'].dt.strftime('%Y-%m-%d %H:%M:%s')
    
    
    test_results = []
    models = []
    
    df_data['failure']=np.where(df_data['failure']=='none',0,1)
    df_data['failure']=df_data['failure'].astype('int')
    
    del df_data['machineID']
    gc.collect()
    
    # make test and training splits
    test_date = pd.to_datetime('2015-10-01 00:00:00')
    test_data = pd.DataFrame(df_data.loc[pd.to_datetime(df_data['datetime']) >= test_date])
    train_data = pd.DataFrame(df_data.loc[pd.to_datetime(df_data['datetime']) < test_date])
    
    validation_date = pd.to_datetime('2015-8-01 01:00:00')
    validation_data = train_data.loc[pd.to_datetime(df_data['datetime']) >= validation_date]
    train_data = train_data.loc[pd.to_datetime(df_data['datetime']) < validation_date]
    
    print(train_data['failure'].value_counts())
    print(validation_data['failure'].value_counts())
    print(test_data['failure'].value_counts())
    
    #train_data = resampling(train_data)
    
    DNN(train_data, validation_data, test_data)
    #XGBClss(train_data, validation_data, test_data)