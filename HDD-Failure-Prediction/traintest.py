# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 19:20:07 2020

@author: nww73
"""
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
from keras.layers import Embedding, Input
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

scaler = StandardScaler()

def ml_linear_regression(X_train, X_test, Y_train, Y_test):
    del [X_train['locale'], X_test['locale']]
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

if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    
    df_data = pd.read_csv('data/model_2018_ST4000DM000.csv', engine='c')
    
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
    
    Y = df_data['failure']
    print(Y.value_counts())
    gc.collect()
    
    
    
    