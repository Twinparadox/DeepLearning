# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:49:50 2020

@author: wonwoo
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight, resample
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Conv1D
from keras.layers import Dense, Dropout
from keras.layers import Flatten, Reshape
from keras.layers import Embedding, Input, BatchNormalization
from keras.models import Sequential
from keras.models import load_model
from keras import optimizers
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import VotingClassifier

import tensorflow as tf
from xgboost import XGBClassifier

scaler = MinMaxScaler()


INQ = ['INQ020', 'INQ012', 'INQ030', 'INQ060', 'INQ080', 'INQ090', 'INQ132',
       'INQ140', 'INQ150', 'INDHHIN2']
DPQ = ['DPQ010', 'DPQ020', 'DPQ030', 'DPQ040', 'DPQ050', 'DPQ060', 'DPQ070',
       'DPQ080', 'DPQ090']
DPQ_RA = ['DPQ_range']
DEMO = ['DMDEDUC', 'DMDMARTL', 'RIDRETH1']
target = ['DPQ_OHE']
column_list = DEMO+INQ


NUMERIC = ['LBXTC', 'LBDTCSI', 'LBDHDD', 'LBDHDDSI', 'LBXGH', 'LBDLDL', 'LBDLDLSI',
           'LBXWBCSI', 'LBXHGB', 'LBXMCHSI', 'LBXMC', 'BMXWT', 'BMXHT', 'BMXBMI',
           'BPXPLS', 'BPXDI', 'LBXVD2MS', 'LBXVD3MS']

REMOVED = ['BPXSY', 'RIDAGEYR']

def resampling(df_data, ratio):    
    train_freq = df_data['DPQ_OHE'].value_counts()    
    print(train_freq)
    train_freq_mean = train_freq[1]   
        
    df_majority = df_data[df_data['DPQ_OHE']==0]
    df_minority = df_data[df_data['DPQ_OHE']==1]
    
    df_majority = resample(df_majority, n_samples=train_freq_mean, random_state=123)
    #df_minority = resample(df_minority, n_samples=train_freq_mean, random_state=123)
    
    df_data = pd.concat([df_majority, df_minority])
    return df_data
    
    '''
    # Under & Over Sampling store_nbr
    df_list = []
    target_max = 2
    multiple = ratio
    
    for i in range(0, target_max):
        df_list.append(df_train[df_train['DPQ_OHE']==i])
    
    for i in range(0, target_max):
        if i==0:
            df_list[i] = df_list[i].sample(n=int(train_freq_mean*multiple), random_state=123)
        else:
            df_list[i] = df_list[i].sample(n=train_freq_mean, random_state=123)
        
    df_sampling_train = pd.concat(df_list)
    train_freq = df_sampling_train['DPQ_OHE'].value_counts()
    
    return pd.DataFrame(df_sampling_train)
    '''

def DNN(df_data):    
    Y = df_data['DPQ_OHE']
    X = df_data.drop(['DPQ_OHE'], axis=1)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,
                                                        random_state = 42,
                                                        stratify = Y)
    
    X_columns = X.columns
    Y_columns = ['DPQ_OHE']
    
    X_train = pd.DataFrame(X_train, columns=X_columns)
    Y_train = pd.DataFrame(Y_train, columns=Y_columns)
    print(X_train.describe())
    X_train = pd.concat([X_train, Y_train], axis=1)
    print(X_train.shape)
    
    X_train = resampling(X_train, 1)
    print(X_train.shape)
    Y_train = X_train['DPQ_OHE']
    X_train = X_train.drop('DPQ_OHE', axis=1)
    
    x_train = scaler.fit_transform(X_train[NUMERIC])
    x_test = scaler.transform(X_test[NUMERIC])
    
    x_train_oh = np.array(X_train.drop(NUMERIC, axis=1))
    x_test_oh = np.array(X_test.drop(NUMERIC, axis=1))  
    
    X_train = np.hstack([x_train, x_train_oh])
    X_test = np.hstack([x_test, x_test_oh])
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    print(X_train.shape)
    
    print(Y_train.value_counts())
    print(Y_test.value_counts())
    
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)  
    
    adam=optimizers.Adam(lr=0.001, decay=0.000001)
    kernel_size=5

    model = Sequential()
    model.add(Conv1D(64, input_shape=X_train.shape[1:3], kernel_size=kernel_size,
                     activation='relu', kernel_initializer='he_uniform' ,name='Conv1'))
    model.add(BatchNormalization())
    model.add(Flatten(name='Flatten'))
    model.add(Dropout(0.8))
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform', name='FCN1'))
    model.add(Dense(Y_train.shape[1], activation='softmax'))
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
    model.summary()

    history = model.fit(X_train, Y_train, 
                        validation_data=(X_test, Y_test),
                        batch_size=100, epochs=50, verbose=1)
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    x_epochs = range(1, len(loss) + 1)

    plt.plot(x_epochs, val_loss, 'r', label='Testing Loss')
    plt.plot(x_epochs, loss, 'b', label='Training loss')
    plt.title('Loss')
    plt.legend()
    plt.show()
    
    plt.plot(x_epochs, val_acc, 'r', label='Testing Acc')
    plt.plot(x_epochs, acc, 'b', label='Training Acc')
    plt.title('Accuracy')
    plt.legend()
    plt.show()
    
    Y_pred = model.predict(X_test)
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    
    print(confusion_matrix(Y_test, Y_pred))
    print(classification_report(Y_test, Y_pred, labels=[0, 1]))
    print(roc_auc_score(Y_test, Y_pred))
    
def CNN2D(df_data):
    Y = df_data['DPQ_OHE']
    X = df_data.drop(['DPQ_OHE'], axis=1)
    
    print(Y.value_counts())
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,
                                                        random_state = 123,
                                                        stratify = Y, shuffle=True)
    
    X_columns = X.columns
    Y_columns = ['DPQ_OHE']
    
    X_train = pd.DataFrame(X_train, columns=X_columns)
    Y_train = pd.DataFrame(Y_train, columns=Y_columns)
    print(X_train.describe())
    X_train = pd.concat([X_train, Y_train], axis=1)
    print(X_train.shape)
    
    X_train = resampling(X_train, 1)
    print(X_train.shape)
    Y_train = X_train['DPQ_OHE']
    X_train = X_train.drop('DPQ_OHE', axis=1)
    
    x_train = scaler.fit_transform(X_train[NUMERIC])
    x_test = scaler.transform(X_test[NUMERIC])
    
    x_train_oh = np.array(X_train.drop(NUMERIC, axis=1))
    x_test_oh = np.array(X_test.drop(NUMERIC, axis=1))  
    
    X_train = np.hstack([x_train, x_train_oh])
    X_test = np.hstack([x_test, x_test_oh])
    
    X_train = X_train.reshape(X_train.shape[0], 5, 13, 1)
    X_test = X_test.reshape(X_test.shape[0], 5, 13, 1)
    print(X_train.shape)
    
    print(Y_train.value_counts())
    print(Y_test.value_counts())
    
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)  
    
    adam=optimizers.Adam(lr=0.001, decay=0.00005)
    kernel_size = (3, 3)
    strides = (1, 1)
    
    model = Sequential()
    model.add(Conv2D(32, input_shape=X_train.shape[1:4],
                     kernel_size=kernel_size, strides=strides,
                     activation='relu', kernel_initializer='he_uniform', name='Conv1')) 
    model.add(Conv2D(16,
                     kernel_size=kernel_size, strides=strides,
                     activation='relu', kernel_initializer='he_uniform', name='Conv2')) 
    model.add(Flatten(name='Flatten'))
    model.add(Dropout(0.8))
    model.add(Dense(Y_train.shape[1], activation='softmax'))
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
    model.summary()

    history = model.fit(X_train, Y_train, 
                        validation_data=(X_test, Y_test),
                        batch_size=100, epochs=25, verbose=1)
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    x_epochs = range(1, len(loss) + 1)

    plt.plot(x_epochs, val_loss, 'r', label='Testing Loss')
    plt.plot(x_epochs, loss, 'b', label='Training loss')
    plt.title('Loss')
    plt.legend()
    plt.show()
    
    plt.plot(x_epochs, val_acc, 'r', label='Testing Acc')
    plt.plot(x_epochs, acc, 'b', label='Training Acc')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

    score = model.evaluate(X_test, Y_test, batch_size=1000)
    
    Y_pred = model.predict(X_test)
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    
    print(confusion_matrix(Y_test, Y_pred))
    print(classification_report(Y_test, Y_pred, labels=[0, 1]))
    print(roc_auc_score(Y_test, Y_pred))
       
def XGBBest(df_data):
    Y = df_data['DPQ_OHE']
    X = df_data.drop(['DPQ_OHE'], axis=1)
    
    print(Y.value_counts())
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.8,
                                                        random_state = 123,
                                                        stratify = Y, shuffle=True)
    
    X_columns = X.columns
    Y_columns = ['DPQ_OHE']
    
    X_train = pd.DataFrame(X_train, columns=X_columns)
    Y_train = pd.DataFrame(Y_train, columns=Y_columns)
    print(X_train.describe())
    X_train = pd.concat([X_train, Y_train], axis=1)
    print(X_train.shape)
    
    X_train = resampling(X_train, 1)
    print(X_train.shape)
    Y_train = X_train['DPQ_OHE']
    X_train = X_train.drop('DPQ_OHE', axis=1)
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(Y_train.value_counts())
    print(Y_test.value_counts())
    
    xgb = XGBClassifier(silent=False, booster='gbtree', learning_rate=0.01,
                        n_estimator=200, gamma=10, seed=123,
                        predictor='gpu_predictor', tree_method='gpu_hist',
                        objective='binary:logistic', eval_metric='auc')
    
    xgb_param_grid = {'max_depth':[3,5,7,9], 'subsample':[0.4,0.6,0.8,1.0]}
    
    grid = GridSearchCV(estimator=xgb,
                        param_grid=xgb_param_grid,
                        scoring='roc_auc',
                        n_jobs=1,
                        cv=5,
                        refit=True,
                        return_train_score=True)
    grid.fit(X_train, Y_train)
    
    print(grid.best_score_)
    print(grid.best_params_)
    
    Y_pred = grid.predict_proba(X_test)
    Y_pred = np.argmax(Y_pred, axis=1)
    print(confusion_matrix(Y_test, Y_pred))
    print(classification_report(Y_test, Y_pred, labels=[0, 1]))
    print(roc_auc_score(Y_test, Y_pred))
    
def cnn_model1():
    adam=optimizers.Adam(lr=0.001, decay=0.000001)
    kernel_size=5

    model = Sequential()
    model.add(Conv1D(32, input_shape=(31,1), kernel_size=kernel_size,
                     activation='relu', kernel_initializer='he_uniform' ,name='Conv1'))
    model.add(BatchNormalization())        
    model.add(Flatten(name='Flatten'))
    model.add(Dropout(0.8))
    model.add(Dense(64, activation='relu', kernel_initializer='he_uniform', name='FCN1'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def cnn_model2():    
    adam=optimizers.Adam(lr=0.001, decay=0.000001)
    kernel_size=5

    model = Sequential()
    model.add(Conv1D(32, input_shape=(31,1), kernel_size=kernel_size,
                     activation='relu', kernel_initializer='he_uniform' ,name='Conv1'))
    model.add(BatchNormalization())    
    model.add(Conv1D(32, kernel_size=kernel_size,
                     activation='relu', kernel_initializer='he_uniform' ,name='Conv2'))
    model.add(BatchNormalization())
    model.add(Flatten(name='Flatten'))
    model.add(Dropout(0.8))
    model.add(Dense(64, activation='relu', kernel_initializer='he_uniform', name='FCN1'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
    
def cnn_model3():
    adam=optimizers.Adam(lr=0.001, decay=0.000001)
    kernel_size=5

    model = Sequential()
    model.add(Conv1D(32, input_shape=(31,1), kernel_size=kernel_size,
                     activation='relu', kernel_initializer='he_uniform' ,name='Conv1'))
    model.add(BatchNormalization())
    model.add(Conv1D(16, kernel_size=kernel_size,
                     activation='relu', kernel_initializer='he_uniform' ,name='Conv2'))
    model.add(BatchNormalization())
    model.add(Conv1D(8, kernel_size=kernel_size,
                     activation='relu', kernel_initializer='he_uniform' ,name='Conv3'))
    model.add(BatchNormalization())
    model.add(Flatten(name='Flatten'))
    model.add(Dropout(0.8))
    model.add(Dense(64, activation='relu', kernel_initializer='he_uniform', name='FCN1'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def dnn_model1():    
    adam=optimizers.Adam(lr=0.01, decay=0.000001)
    model = Sequential()
    model.add(Dense(128, input_dim=65, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.8))
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
    
def dnn_model2():
    adam=optimizers.Adam(lr=0.01, decay=0.000001)
    model = Sequential()
    model.add(Dense(128, input_dim=65, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.8))
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
    
def dnn_model3():
    adam=optimizers.Adam(lr=0.01, decay=0.000001)
    model = Sequential()
    model.add(Dense(128, input_dim=65, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.8))
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def Ensemble(df_data):    
    Y = df_data['DPQ_OHE']
    X = df_data.drop(['DPQ_OHE'], axis=1)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,
                                                        random_state = 42,
                                                        stratify = Y)
    
    X_columns = X.columns
    Y_columns = ['DPQ_OHE']
    
    X_train = pd.DataFrame(X_train, columns=X_columns)
    Y_train = pd.DataFrame(Y_train, columns=Y_columns)
    print(X_train.describe())
    X_train = pd.concat([X_train, Y_train], axis=1)
    print(X_train.shape)
    
    X_train = resampling(X_train, 1)
    print(X_train.shape)
    Y_train = X_train['DPQ_OHE']
    X_train = X_train.drop('DPQ_OHE', axis=1)
    
    x_train = scaler.fit_transform(X_train[NUMERIC])
    x_test = scaler.transform(X_test[NUMERIC])
    
    x_train_oh = np.array(X_train.drop(NUMERIC, axis=1))
    x_test_oh = np.array(X_test.drop(NUMERIC, axis=1))  
    
    X_train = np.hstack([x_train, x_train_oh])
    X_test = np.hstack([x_test, x_test_oh])
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    print(X_train.shape)
    
    print(Y_train.value_counts())
    print(Y_test.value_counts())
    
    #Y_train = to_categorical(Y_train)
    #Y_test = to_categorical(Y_test)  
    
    model1 = KerasClassifier(build_fn = cnn_model1, epochs = 50)
    model2 = KerasClassifier(build_fn = cnn_model2, epochs = 60)
    model3 = KerasClassifier(build_fn = cnn_model3, epochs = 70)
    
    ensemble_clf = VotingClassifier(estimators = [('model1', model1), ('model2', model2), ('model3', model3)], voting = 'soft')
    ensemble_clf.fit(X_train, Y_train)
    
    Y_pred = ensemble_clf.predict(X_test)
    
    print(confusion_matrix(Y_test, Y_pred))
    print(classification_report(Y_test, Y_pred, labels=[0, 1]))
    print(roc_auc_score(Y_test, Y_pred))
    
    
if __name__ == "__main__":   
    df_2007 = pd.read_csv('2007-2008.csv', engine='c')
    df_2009 = pd.read_csv('2009-2010.csv', engine='c')
    df_2011 = pd.read_csv('2011-2012.csv', engine='c')
    df_2013 = pd.read_csv('2013-2014.csv', engine='c')
    
    df_data = pd.concat([df_2007, df_2009, df_2011, df_2013])
    
    df_data = df_data.set_index('SEQN')
    
    df_numeric = df_data[NUMERIC]
    
    print(df_data.shape)
    df_data = df_data.drop(['DPQ_total'], axis=1)
    df_data = df_data.drop(['DPQ_range'], axis=1)
    df_data = df_data.drop(['RIAGENDR'], axis=1)
    df_data = df_data.drop(['BPXSY', 'RIDAGEYR'], axis=1)
    #df_data = df_data.drop(['LBXVD2MS', 'LBXVD3MS'], axis=1)
    df_data = df_data.drop(DPQ, axis=1)
    df_data[column_list] = df_data[column_list].astype(int)
    
    corr = df_data.corr()
    plt.figure(figsize=(30,30))
    sns.heatmap(corr, vmax=1, annot=True, square=True)
    plt.title('feature correlations')
    plt.show()
    
    #df_data = pd.get_dummies(df_data, columns=column_list)
    #print(df_data.shape)
    
    #DNN(df_data)
    #XGBBest(df_data)
    #CNN2D(df_data)
    Ensemble(df_data)