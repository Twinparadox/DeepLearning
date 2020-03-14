# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 19:20:07 2020

@author: nww73
"""
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.metrics import confusion_matrix
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

def resampling(X_train, Y_train):    
    print(Y_train.value_counts())    
    Y_freq = Y_train.value_counts()
    print(int(Y_freq.mean()))
    freq_mean = Y_freq.mean()
    
    

def ml_logistic_regression(X_train, X_test, Y_train, Y_test, nComp=20):   
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    
    y_integers = np.argmax(Y_train, axis=1).tolist()
    print(y_integers)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_integers), y_integers)
    print(class_weights)
    d_class_weights = dict(enumerate(class_weights)) 
    
    sgd=optimizers.SGD(lr=0.001)    

    model=Sequential()
    model.add(Dense(2, input_shape=(nComp,), kernel_initializer='normal', activation='softmax'))
    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, Y_train, batch_size=1000, epochs=2000,
                        verbose=1, class_weight=d_class_weights, validation_split=0.2, shuffle=True)
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
    
    Y_pred = np.argmax(Y_pred, axis=1).reshape(-1,1)
    Y_test = np.argmax(Y_test, axis=1).reshape(-1,1)
    print(confusion_matrix(Y_test, Y_pred))

def dl_DNN(X_train, X_test, Y_train, Y_test, weight, nComp=25):
    
    #y_integers = np.argmax(Y_train, axis=1).tolist()
    y_integers = Y_train
    #print(y_integers)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_integers), y_integers)
    print(class_weights)
    d_class_weights = dict(enumerate(class_weights))
    #d_class_weights = {0:1.0, 1:1.0}
    optimizer=optimizers.Adam()
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    
    model=Sequential()
    model.add(Dense(32, input_dim=nComp, kernel_initializer='glorot_normal',
                    bias_initializer='glorot_normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(32, kernel_initializer='glorot_normal',
                    bias_initializer='glorot_normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(32, kernel_initializer='glorot_normal',
                    bias_initializer='glorot_normal',  activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(32, kernel_initializer='glorot_normal',
                    bias_initializer='glorot_normal',  activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(32, kernel_initializer='glorot_normal',
                    bias_initializer='glorot_normal',  activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(32, kernel_initializer='glorot_normal',
                    bias_initializer='glorot_normal',  activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc', f1_m])
    model.summary()

    history = model.fit(X_train, Y_train, batch_size=10000, epochs=30,
                        verbose=1, class_weight = d_class_weights, validation_split=0.2, shuffle=True)
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
    Y_pred =(Y_pred>0.5)
    Y_pred = list(Y_pred)
    
    #Y_pred = np.argmax(Y_pred, axis=1).reshape(-1,1)
    #Y_test = np.argmax(Y_test, axis=1).reshape(-1,1)
    print(confusion_matrix(Y_test, Y_pred))
    
def XGBClss(X_train, X_test, Y_train, Y_test, weight, nComp=25):
    model=XGBClassifier()
    param_grid={'booster' :['gbtree'],
                     'silent':[True],
                     'max_depth':[5,6,8],
                     'min_child_weight':[1,3,5],
                     'gamma':[0,1,2,3],
                     'nthread':[4],
                     'colsample_bytree':[0.5,0.8],
                     'colsample_bylevel':[0.9],
                     'n_estimators':[50],
                     'objective':['binary:logistic'],
                     'random_state':[2]}
    cv=KFold(n_splits=6, random_state=123)
    gcv=GridSearchCV(model, param_grid=param_grid, cv=cv, scoring='f1', n_jobs=4)
    gcv.fit(X_train, Y_train)

    model = gcv.best_estimator

    # 8번
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    Y_pred =(Y_pred>0.5)
    Y_pred = list(Y_pred)
    
    print(confusion_matrix(Y_test, Y_pred))

def checkNA(df_data, flag_train=True):
    # NA 확인
    print(df_data.isnull().sum())
    
    # NA 제거
    length = len(df_data)
    # NA 열제거
    na_columns = df_data.isnull().sum()
    na_columns = (na_columns==length)
    
    na_columns = na_columns.to_frame('NA')
    na_columns = na_columns[na_columns['NA']==True]
    na_columns = na_columns.index.tolist()
    
    df_data = df_data.drop(na_columns, axis=1)
    
    # NA 행제거
    df_data = df_data.dropna()
        
    if flag_train==False:
        Y = df_data['failure']
        X = df_data.drop(['failure', 'date'], axis=1)
        gc.collect()
        
        print(Y.value_counts())
        print("Serials:",len(X['serial_number'].unique()))
        print(X['model'].value_counts())    
        
        # serial num, model 제거
        X = X.drop(['serial_number', 'model'], axis=1)
        gc.collect()
    
        X = scaler.fit_transform(X)
        return X, Y
    else:        
        Y = df_data['failure']
        values = Y.value_counts().tolist()
        mean = int((values[0]+values[1])/2)
        
        print(mean)
        
        df_majority = df_data[df_data.failure==0]
        df_minority = df_data[df_data.failure==1]
        
        df_majority = resample(df_majority, n_samples=mean, random_state=123)
        df_minority = resample(df_minority, n_samples=mean, random_state=123)
        
        df_data = pd.concat([df_majority, df_minority])
        
        Y = df_data['failure']
        X = df_data.drop(['failure', 'date'], axis=1)
        gc.collect()
        
        print(Y.value_counts())
        print("Serials:",len(X['serial_number'].unique()))
        print(X['model'].value_counts())    
        
        # serial num, model 제거
        X = X.drop(['serial_number', 'model'], axis=1)
        gc.collect()
    
        X = scaler.fit_transform(X)
        return X, Y
        

if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    
    df_train = pd.read_csv('data/model_2015_ST4000DM000.csv', engine='c')
    df_train1 = pd.read_csv('data/model_2016_ST4000DM000.csv', engine='c')
    df_train2 = pd.read_csv('data/model_2017_ST4000DM000.csv', engine='c')
    df_test = pd.read_csv('data/model_2018_ST4000Dm000.csv', engine='c')
    
    df_train = pd.concat([df_train, df_train1, df_train2])
    del df_train1
    del df_train2
    gc.collect()
    
    X_train, Y_train = checkNA(df_train, False)
    X_test, Y_test = checkNA(df_test, False)
    
    # PCA
    originComponents = 49
    nComponents = 25
    pca = PCA(nComponents)
    pca.fit(X_train)
    
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()
    
    X_pca_train = pca.fit_transform(X_train)
    X_pca_test = pca.fit_transform(X_test)
    pca_std = np.std(X_pca_train)
    
    print(X_pca_train.shape)
    
    #X_train, Y_train = resampling(X_train, Y_train)
    
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    #ml_linear_regression(X_train, X_test, Y_train, Y_test, weight, originComponents)
    #ml_logistic_regression(X_pca_train, X_pca_test, Y_train, Y_test, nComponents)
    dl_DNN(X_pca_train, X_pca_test, Y_train, Y_test, originComponents)
    #XGBClss(X_pca_train, X_pca_test, Y_train, Y_test, originComponents)