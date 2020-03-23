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
from keras.layers import Conv1D, MaxPooling1D, TimeDistributed
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers import Flatten, Reshape
from keras.layers import Embedding, Input
from keras.models import Sequential
from keras.models import load_model
from keras import optimizers
from keras.regularizers import L1L2
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical, np_utils
from keras.initializers import RandomUniform
import keras.backend as K
import tensorflow as tf
import numpy as np
import math
import pandas as pd
import os
import gc
import matplotlib.pyplot as plt

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import seaborn as sns

scaler = StandardScaler()

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def describe_missing_values(df):
    na_percent = {}
    N = df.shape[0]
    for column in df:
        na_percent[column] = df[column].isnull().sum() * 100 / N

    na_percent = dict(filter(lambda x: x[1] != 0, na_percent.items()))
    plt.bar(range(len(na_percent)), na_percent.values())
    plt.ylabel('Percent')
    plt.xticks(range(len(na_percent)), na_percent.keys(), rotation='vertical')
    plt.show()

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

def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini

def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
 
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return 'gini', gini_score

def resampling(df_train, ratio):    
    train_freq = df_train['target'].value_counts()    
    print(train_freq)
    train_freq_mean = train_freq[0]
    
    # Under & Over Sampling store_nbr
    df_list = []
    target_max = 2
    multiple = ratio
    
    for i in range(0, target_max):
        df_list.append(df_train[df_train['target']==i])
    
    for i in range(0, target_max):
        if i==0:
            df_list[i] = df_list[i].sample(n=int(train_freq_mean*multiple), random_state=123, replace=True)
        else:
            df_list[i] = df_list[i].sample(n=train_freq_mean, random_state=123, replace=True)
        
    df_sampling_train = pd.concat(df_list)
    train_freq = df_sampling_train['target'].value_counts()
    
    return pd.DataFrame(df_sampling_train)
    
def DNN(train, test):    
    X_train = train.drop(['id', 'target'], axis=1)
    gc.collect()
    
    X_columns = X_train.columns
    Y_columns = ['target']
    
    Y_train = train['target']
    
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train,
                                                          test_size=0.33, stratify=Y_train,
                                                          random_state=123, shuffle=True)
    X_train = pd.DataFrame(X_train, columns=X_columns)
    Y_train = pd.DataFrame(Y_train, columns=Y_columns)
    print(X_train.describe())
    X_train = pd.concat([X_train, Y_train], axis=1)
    print(X_train.describe())
    
    X_train = resampling(X_train, 1)
    Y_train = X_train['target']
    X_train = X_train.drop('target', axis=1)
    
    print(Y_train.value_counts())
    
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.fit_transform(X_valid)
    
    y_integers = Y_train
    #print(y_integers)
    class_weights = class_weight.compute_class_weight(None, np.unique(y_integers), y_integers)
    print(class_weights)
    d_class_weights = dict(enumerate(class_weights)) 
    d_class_weights = {0:1.0, 1:1.0}
    optimizer=optimizers.Adam()
    
    Y_train = to_categorical(Y_train)
    Y_valid = to_categorical(Y_valid)
    
    print(Y_train)
    
    model=Sequential()
    model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32,  activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(32,  activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32,  activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
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

    score = model.evaluate(X_valid, Y_valid, batch_size=1000)
    print(score)
    
    Y_pred = model.predict(X_valid)
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_valid = np.argmax(Y_valid, axis=1)
    
    print(confusion_matrix(Y_valid, Y_pred))
    print(classification_report(Y_valid, Y_pred, labels=[0, 1]))
    print("Gini: ", gini(Y_valid,Y_pred))
    print("Gini Nomarlized: ", gini_normalized(Y_valid, Y_pred))
    
def CNN1D(train, test):    
    X_train = train.drop(['id', 'target'], axis=1)
    gc.collect()
    
    X_columns = X_train.columns
    Y_columns = ['target']
    
    Y_train = train['target']
    
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train,
                                                          test_size=0.33, stratify=Y_train,
                                                          random_state=123, shuffle=True)
    X_train = pd.DataFrame(X_train, columns=X_columns)
    Y_train = pd.DataFrame(Y_train, columns=Y_columns)
    print(X_train.describe())
    X_train = pd.concat([X_train, Y_train], axis=1)
    print(X_train.describe())
    
    #X_train = resampling(X_train)
    Y_train = X_train['target']
    X_train = X_train.drop('target', axis=1)
    
    print(Y_train.value_counts())
    
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.fit_transform(X_valid)
    
    y_integers = Y_train
    #print(y_integers)
    class_weights = class_weight.compute_class_weight(None, np.unique(y_integers), y_integers)
    print(class_weights)
    d_class_weights = dict(enumerate(class_weights)) 
    d_class_weights = {0:1.0, 1:1.0}
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], 1)
    
    #hyperparameters
    input_dimension = 226
    learning_rate = 0.0025
    momentum = 0.85
    hidden_initializer = RandomUniform(seed=123)
    dropout_rate = 0.3

    optimizer=optimizers.Adam(lr=0.005)
    
    Y_train = to_categorical(Y_train)
    Y_valid = to_categorical(Y_valid)
    
    
    # create model
    model = Sequential()
    model.add(Conv1D(nb_filter=32, filter_length=3, input_shape=X_train.shape[1:3], activation='relu'))
    model.add(Conv1D(nb_filter=16, filter_length=1, activation='relu')) 
    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, input_dim=input_dimension, kernel_initializer=hidden_initializer, activation='relu'))
    model.add(Dense(64, kernel_initializer=hidden_initializer, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(2, kernel_initializer=hidden_initializer, activation='softmax'))
    
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

    score = model.evaluate(X_valid, Y_valid, batch_size=100)
    print(score)
    
    Y_pred = model.predict(X_valid)
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_valid = np.argmax(Y_valid, axis=1)
    
    #Y_pred = np.argmax(Y_pred, axis=1).reshape(-1,1)
    #Y_test = np.argmax(Y_test, axis=1).reshape(-1,1)
    print(confusion_matrix(Y_valid, Y_pred))
    print(classification_report(Y_valid, Y_pred, labels=[0, 1]))
    print("Gini: ", gini(Y_valid,Y_pred))
    print("Gini Nomarlized: ", gini_normalized(Y_valid, Y_pred))
    
    
def XGBClss(train, test):
    X_train = train.drop(['id', 'target'], axis=1)
    gc.collect()
    
    X_columns = X_train.columns
    Y_columns = ['target']
    
    Y_train = train['target']
    
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train,
                                                          test_size=0.33, stratify=Y_train,
                                                          random_state=123, shuffle=True)
    X_train = pd.DataFrame(X_train, columns=X_columns)
    Y_train = pd.DataFrame(Y_train, columns=Y_columns)
    print(X_train.describe())
    X_train = pd.concat([X_train, Y_train], axis=1)
    print(X_train.describe())
    
    X_train = resampling(X_train)
    Y_train = X_train['target']
    X_train = X_train.drop('target', axis=1)
    
    print(Y_train.value_counts())
    
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.fit_transform(X_valid)
    
    y_integers = Y_train
    #print(y_integers)
    class_weights = class_weight.compute_class_weight(None, np.unique(y_integers), y_integers)
    print(class_weights)
    d_class_weights = dict(enumerate(class_weights)) 
    #d_class_weights = {0:1.0, 1:1.0}
    
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
    
    Y_pred = model.predict(X_valid)
    
    print(confusion_matrix(Y_valid, Y_pred))
    print(classification_report(Y_valid, Y_pred, labels=[0, 1]))
    print("Gini: ", gini(Y_valid,Y_pred))
    print("Gini Nomarlized: ", gini_normalized(Y_valid, Y_pred))
    
    


if __name__=="__main__":    
    train_df = pd.read_csv('data/train.csv', na_values="-1", engine='c')
    test_df = pd.read_csv('data/test.csv', na_values="-1", engine='c')
    
    sns.set(style="white")
    # Compute the correlation matrix
    corr = train_df.corr()
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    plt.show()
    
    unwanted = train_df.columns[train_df.columns.str.startswith('ps_calc_')]
    train_df = train_df.drop(unwanted, axis=1)  
    test_df = test_df.drop(unwanted, axis=1)  
    
    data=[]
    for f in train_df.columns:
        # Defining the role
        if f =='target':
            role='target'
        elif f == 'id':
            role = 'id'
        else:
            role = 'input'
            
        print(train_df[f].dtype)
        # Defining the level
        if 'bin' in f or f == 'target':
            level = 'binary'
        elif 'cat' in f or f == 'id':
            level = 'nominal'
        elif train_df[f].dtype == 'float64':
            level = 'interval'
        elif train_df[f].dtype == 'int64':
            level = 'ordinal'
        
        # Initialize keep to True for all variables except for id
        keep = True
        if f == 'id':
            keep = False
        
        # Defining the data type
        dtype = train_df[f].dtype
        
        # Creating a Dict that contains all the metadata for the variable
        f_dict = {
            'varname': f,
            'role':role,
            'level':level,
            'keep':keep,
            'dtype':dtype
        }
        data.append(f_dict)
    
    meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])
    meta = meta.set_index('varname')
    
    print(meta)
    
    pd.DataFrame({'count':meta.groupby(['role', 'level'])['role'].size()}).reset_index()
    
    v = meta[(meta.level == 'interval') & (meta.keep)].index
    train_df[v].describe()
    v = meta[(meta.level == 'ordinal') & (meta.keep)].index
    train_df[v].describe()
    v = meta[(meta.level=='binary') & (meta.keep)].index
    train_df[v].describe()
    
    '''
    # Resampling
    desired_apriori=0.1
    
    # Get the indices per target value
    idx_0 = train_df[train_df.target==0].index
    idx_1 = train_df[train_df.target==1].index
    
    # Get original number of records per target value
    nb_0 = len(train_df.loc[idx_0])
    nb_1 = len(train_df.loc[idx_1])
    
    # Calculate the undersampling rate and resulting number of records with target=0
    undersampling_rate = ((1-desired_apriori)*nb_1)/(nb_0*desired_apriori)
    undersampled_nb_0 = int(undersampling_rate*nb_0)
    print('Rate to undersample records with target=0: {}'.format(undersampling_rate))
    print('Number of records with target=0 after undersampling: {}'.format(undersampled_nb_0))
    
    # Randomly select records with target=0 to get at the desired a priori
    undersampled_idx = shuffle(idx_0, random_state=123, n_samples=undersampled_nb_0)
    
    # Construct list with remaining indices
    idx_list = list(undersampled_idx) + list(idx_1)
    
    # Return undersample data frame
    train_df = train_df.loc[idx_list].reset_index(drop=True)
    '''   
    
    # Check Missing Value   
    print("Missing values for Train dataset")
    describe_missing_values(train_df)
    
    print("Missing values for Test dataset")
    describe_missing_values(test_df)
    
    # NA Remove
    test_id = test_df['id']
    train_df = train_df.drop(["ps_car_03_cat", "ps_car_05_cat"], axis=1)
    test_df = test_df.drop(["ps_car_03_cat","ps_car_05_cat"], axis=1)
    
    # fill NA
    cat_cols = [col for col in train_df.columns if 'cat' in col]
    bin_cols = [col for col in train_df.columns if 'bin' in col]
    con_cols = [col for col in train_df.columns if col not in bin_cols + cat_cols]
    
    # 최빈값으로 대체
    for col in cat_cols:
        train_df[col] = train_df[col].fillna(value=train_df[col].mode()[0])
        test_df[col] = test_df[col].fillna(value=test_df[col].mode()[0])
        
    # 최빈값으로 대체
    for col in bin_cols:
        train_df[col] = train_df[col].fillna(value=train_df[col].mode()[0])
        test_df[col] = test_df[col].fillna(value=test_df[col].mode()[0])
        
    # 평균값으로 대체
    for col in con_cols:
        if col != 'id' and col != 'target':
            train_df[col] = train_df[col].fillna(value=train_df[col].mean())
            test_df[col] = test_df[col].fillna(value=test_df[col].mean())
    
    print("Missing values for Train dataset")
    describe_missing_values(train_df)
    
    print("Missing values for Test dataset")
    describe_missing_values(test_df)
    
    print(train_df.dtypes)
    
    # Variance 바탕으로 제
    selector = VarianceThreshold(threshold=.01)
    selector.fit(train_df.drop(["id", "target"], axis=1)) # Fit to train without id and target variables
    f = np.vectorize(lambda x:not x) # Function to toggle boolean array elemets
    v = train_df.drop(['id', 'target'], axis=1).columns[f(selector.get_support())]
    print('{} variables have too low variance.'.format(len(v)))
    print('These variables are {}'.format(list(v)))
    
    print(train_df)
    train_df = train_df.drop(['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin',
                              'ps_ind_13_bin', 'ps_car_10_cat', 'ps_car_12', 'ps_car_14'], axis=1)
    test_df = test_df.drop(['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin',
                            'ps_ind_13_bin', 'ps_car_10_cat', 'ps_car_12', 'ps_car_14'], axis=1)
    
    # Resampling
    #train_df = resampling(train_df)
    
    print(train_df)
    
    DNN(train_df, test_df)
    #CNN1D(train_df, test_df)
    #XGBClss(train_df, test_df)