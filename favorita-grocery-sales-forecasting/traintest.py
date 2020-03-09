# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 15:09:01 2020

@author: nww73
"""
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
import os
import gc
import matplotlib.pyplot as plt

scaler = MinMaxScaler()

def resampling(df_store, df_train):    
    train_store_freq = df_train['store_nbr'].value_counts()    
    train_store_freq_mean = int(train_store_freq.mean())
    
    # Under & Over Sampling store_nbr
    df_store_list = []
    store_max = 54
    
    for i in range(1, store_max+1):
        df_store_list.append(df_train[df_train['store_nbr']==i])
    
    for i in range(0, store_max):
        df_store_list[i] = df_store_list[i].sample(n=train_store_freq_mean, random_state=123, replace=True)
        
    df_sampling_train = pd.concat(df_store_list)
    train_store_freq = df_sampling_train['store_nbr'].value_counts()
    
    return pd.DataFrame(df_sampling_train)
    
def nwrmsle(predictions, targets, weights):
    if type(predictions) == list:
        predictions = np.array([np.nan if x < 0 else x for x in predictions])

    elif type(predictions) == pd.Series:
        predictions[predictions < 0] = np.nan

    targetsf = targets.astype(float)
    targetsf[targets < 0] = np.nan
    weights = 1 + 0.25 * weights
    log_square_errors = (np.log(predictions + 1) - np.log(targetsf + 1)) ** 2

    return(np.sqrt(np.sum(weights * log_square_errors) / np.sum(weights)))    

def ml_linearRegression(X_train, Y_train, X_test, X_stratify):
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train,
                                                          test_size=0.3, random_state=123,
                                                          stratify=X_stratify)
    
    model = LinearRegression()
    model.fit(X_train, Y_train)
    
    Y_pred = model.predict(X_valid)
    print(RMSLE(y_tru, y_pred))

def ml_logisticRegression(X_train, Y_train, X_test, X_stratify):
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train,
                                                          test_size=0.3, random_state=123,
                                                          stratify=X_stratify)
    
    model = LogisticRegression()
    param_grid = {'C':np.logspace(-3,3,7), 'penalty':['l1','l2']}    
    gs = GridSearchCV(estimator=model, param_grid=param_grid,
                      scoring='neg_mean_squared_error', cv=3)
    gs.fit(X_train, Y_train)
    
    best_model = gs.best_estimator_
    Y_pred = best_model.predict(X_valid)
    print(mean_squared_error(Y_valid, Y_pred))
    

def ml_linearSVR(X_train, Y_train, X_test, X_stratify):    
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train,
                                                          test_size=0.3, random_state=123,
                                                          stratify=X_stratify)
    
    pipe_model = Pipeline([('scl', MinMaxScaler()), ('clf', LinearSVR())])
    param_range = np.logspace(-3,3,7)
    param_grid = [{'clf__C': param_range,
                   'clf__epsilon': param_range}]
    gs = GridSearchCV(estimator=pipe_model, param_grid=param_grid,
                      scoring='neg_mean_squared_error', cv=3, iid=True)
    
    gs.fit(X_train, Y_train)
    print(gs.best_score_)
    print(gs.best_params_)

    best_params = gs.best_params_
    best_model = gs.best_estimator_

    Y_pred = best_model.predict(X_valid)
    print(mean_squared_error(Y_valid, Y_pred))
    
def DNN(X_train, Y_train, X_test, X_stratify):
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train,
                                                          test_size=0.3, random_state=123,
                                                          stratify=X_stratify)
    
    epochs = 100
    batch_size = 100
    
    model = Sequential()
    model.add(Dense(9, input_dim=9, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss=RMSLE, optimizer='adam', metrics=[RMSLE])
    
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    model.save('model/model_DNN.hdf5')
    rmsle = history.history[RMSLE]
    loss = history.history['loss']
    
    x_epochs = range(1, len(rmsle) + 1)
    
    plt.plot(x_epochs, rmsle, 'b', label='Training mae')
    plt.title('Mean_Absolute_Error')
    plt.legend()
    plt.figure()
    
    plt.plot(x_epochs, loss, 'b', label='Training loss')
    plt.title('Loss')
    plt.legend()
    plt.show()
    
    Y_pred = model.predict(X_valid)
    print(mean_squared_error(Y_valid, Y_pred))

if __name__ == "__main__":
    df_train = pd.read_csv('./data/prep_train.csv', engine='c')
    df_test = pd.read_csv('./data/prep_test.csv', engine='c')
    df_stores = pd.read_csv('./data/stores.csv', engine='c')
    
    df_train = resampling(df_stores, df_train)
    
    del df_train['date']
    del df_test['date']
    df_train = df_train.set_index('id')
    df_test = df_test.set_index('id')
        
    X_train = df_train.drop('unit_sales', axis=1)
    Y_train = df_train['unit_sales']
    X_test = df_test
    
    # resampling
    X_train = resampling(df_stores, X_train)
    X_stratify = X_train['store_nbr']
    
    # memory clear
    del [[df_train, df_test, df_stores]]
    gc.collect()
    
    # scaling
    X_train = scaler.fit_transform(X_train)
    print(X_train.shape)    
    X_test = scaler.fit_transform(X_test)
    
    #ml_linearRegression(X_train, Y_train, X_test, X_stratify)
    #ml_logisticRegression(X_train, Y_train, X_test, X_stratify)
    #ml_linearSVR(X_train, Y_train, X_test, X_stratify)
    
    DNN(X_train, Y_train, X_test, X_stratify)
    
    
    