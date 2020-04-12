import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import gc
import os
warnings.filterwarnings('ignore')
from datetime import datetime
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import class_weight, resample
from keras.models import Sequential
from keras.layers import Dense, Dropout, GaussianNoise, Conv1D
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D, TimeDistributed
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers import Flatten, Reshape
from keras.layers import Embedding, Input
from keras.models import Sequential
from keras.models import load_model
from keras import optimizers
from keras.regularizers import L1L2
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical, np_utils
from keras.callbacks import Callback, EarlyStopping
from keras.initializers import RandomUniform
import keras.backend as K
import tensorflow as tf
from xgboost import XGBClassifier

scaler = StandardScaler()

def plot_new_feature_distribution(df1, df2, label1, label2, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(2,4,figsize=(18,8))

    for feature in features:
        i += 1
        plt.subplot(2,4,i)
        sns.kdeplot(df1[feature], bw=0.5,label=label1)
        sns.kdeplot(df2[feature], bw=0.5,label=label2)
        plt.xlabel(feature, fontsize=11)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=8)
        plt.tick_params(axis='y', which='major', labelsize=8)
    plt.show();

# define roc_callback, inspired by https://github.com/keras-team/keras/issues/6050#issuecomment-329996505
def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value
    
def resampling(df_train, ratio):    
    train_freq = df_train['target'].value_counts()    
    print(train_freq)
    train_freq_mean = train_freq[1]
    
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
    
def CNN1D(train, test):    
    X_train = train.drop(['ID_code', 'target'], axis=1)
    gc.collect()
    
    X_columns = X_train.columns
    Y_columns = ['target']
    
    Y_train = train['target']
    
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train,
                                                          test_size=0.10, stratify=Y_train,
                                                          random_state=123, shuffle=True)
    X_train = pd.DataFrame(X_train, columns=X_columns)
    Y_train = pd.DataFrame(Y_train, columns=Y_columns)
    print(X_train.describe())
    X_train = pd.concat([X_train, Y_train], axis=1)
    print(X_train.describe())
    
    #X_train = resampling(X_train, 1)
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

    optimizer=optimizers.Adam()
    
    # create model
    model = Sequential()
    model.add(Conv1D(nb_filter=32, filter_length=3, input_shape=X_train.shape[1:3], activation='relu'))
    model.add(Conv1D(nb_filter=16, filter_length=1, activation='relu')) 
    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, input_dim=input_dimension, kernel_initializer=hidden_initializer, activation='relu'))
    model.add(Dense(64, kernel_initializer=hidden_initializer, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, kernel_initializer=hidden_initializer, activation='sigmoid'))
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', auc_roc])
    model.summary()

    history = model.fit(X_train, Y_train, batch_size=1000, epochs=10,
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
    Y_pred = np.where(Y_pred > 0.5, 1, 0)
    
    #Y_pred = np.argmax(Y_pred, axis=1).reshape(-1,1)
    #Y_test = np.argmax(Y_test, axis=1).reshape(-1,1)
    print(confusion_matrix(Y_valid, Y_pred))
    print(classification_report(Y_valid, Y_pred, labels=[0, 1]))
    
    
    ID_test = test['ID_code'].values
    test = test.drop('ID_code', axis=1)
    test = scaler.transform(test) 
    test = test.reshape(test.shape[0], test.shape[1], 1)
    pred = model.predict(test)
    
    result = pd.DataFrame({"ID_code": ID_test})
    result["target"] = pred
    result.to_csv("submission.csv", index=False)
    
    return result['target']
    
    
def CNN2D(train, test):    
    X_train = train.drop(['ID_code', 'target'], axis=1)
    gc.collect()
    
    X_columns = X_train.columns
    Y_columns = ['target']
    
    Y_train = train['target']
    
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train,
                                                          test_size=0.10, stratify=Y_train,
                                                          random_state=123, shuffle=True)
    X_train = pd.DataFrame(X_train, columns=X_columns)
    Y_train = pd.DataFrame(Y_train, columns=Y_columns)
    print(X_train.describe())
    X_train = pd.concat([X_train, Y_train], axis=1)
    print(X_train.describe())
    
    #X_train = resampling(X_train, 1)
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
    
    X_train = X_train.reshape(X_train.shape[0], 104, 6, 1)
    X_valid = X_valid.reshape(X_valid.shape[0], 104, 6, 1)
    
    #hyperparameters
    input_dimension = 226
    learning_rate = 0.0025
    momentum = 0.85
    hidden_initializer = RandomUniform(seed=123)
    dropout_rate = 0.3
    kernel_size = (3, 3)
    strides = (1, 1)

    optimizer=optimizers.Adam()
    
    # create model
    model = Sequential()
    model.add(Conv2D(nb_filter=32, kernel_size = kernel_size, strides = strides,
                     input_shape=X_train.shape[1:4], activation='relu'))
    model.add(Conv2D(nb_filter=16, kernel_size = kernel_size, strides = strides,
                     activation='relu')) 
    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, input_dim=input_dimension, kernel_initializer=hidden_initializer, activation='relu'))
    model.add(Dense(64, kernel_initializer=hidden_initializer, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, kernel_initializer=hidden_initializer, activation='sigmoid'))
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', auc_roc])
    model.summary()

    history = model.fit(X_train, Y_train, batch_size=1000, epochs=10,
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
    Y_pred = np.where(Y_pred > 0.5, 1, 0)
    
    #Y_pred = np.argmax(Y_pred, axis=1).reshape(-1,1)
    #Y_test = np.argmax(Y_test, axis=1).reshape(-1,1)
    print(confusion_matrix(Y_valid, Y_pred))
    print(classification_report(Y_valid, Y_pred, labels=[0, 1]))
    
    
    ID_test = test['ID_code'].values
    test = test.drop('ID_code', axis=1)
    test = scaler.transform(test) 
    test = test.reshape(test.shape[0], 104, 6, 1)
    pred = model.predict(test)
    
    result = pd.DataFrame({"ID_code": ID_test})
    result["target"] = pred
    result.to_csv("submission.csv", index=False)
    
    return result['target']
    
def XGBBest(train, test):
    X = train.drop(['ID_code', 'target'], axis=1)
    y = train.target.values
    test_df = test
    test = test.drop('ID_code', axis=1)
    
    xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                      colsample_bynode=1, colsample_bytree=0.6,
                      early_stopping_rounds=70, gamma=2, learning_rate=0.03,
                      max_delta_step=0, max_depth=7, min_child_weight=10, missing=None,
                      n_estimator=500, n_estimators=100, n_jobs=1, nthread=1,
                      num_boost_round=500, objective='binary:logistic',
                      predictor='gpu_predictor', random_state=0, reg_alpha=0,
                      reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
                      subsample=0.8, tree_method='gpu_hist', verbosity=1)

    xgb.fit(X, y)
    
    y_test = xgb.predict_proba(test)
    results_df = pd.DataFrame(data={'ID_code':test_df['ID_code'], 'target':y_test[:,1]})
    results_df.to_csv('submission.csv', index=False)
    
    return results_df['target']

if __name__ == '__main__':
    df_train = pd.read_csv("data/train.csv", engine='c')
    df_test = pd.read_csv("data/test.csv", engine='c')
    
    print("train shape: ", df_train.shape)
    print("test shape: ", df_test.shape)
    
    print("df_train is null: ", df_train.isnull().sum().sum())
    print("df_test is null: ", df_test.isnull().sum().sum())
    
    
    # Feature Engineering
    # Correlation
    features = df_train.columns.values[2:202]
    
    correlations = df_train[features].corr().abs().unstack().sort_values(kind="quicksort").reset_index()
    correlations = correlations[correlations['level_0'] != correlations['level_1']]
    print(correlations.head(10))
    print(correlations.tail(10))
    
    # Duplicate check
    features = df_train.columns.values[2:202]
    unique_max_train = []
    unique_max_test = []
    for feature in features:
        values = df_train[feature].value_counts()
        unique_max_train.append([feature, values.max(), values.idxmax()])
        values = df_test[feature].value_counts()
        unique_max_test.append([feature, values.max(), values.idxmax()])
        
    dup_train = np.transpose((pd.DataFrame(unique_max_train, columns=['Feature', 'Max duplicates', 'Value'])).
                             sort_values(by = 'Max duplicates', ascending=False).head(15))
    
    dup_test = np.transpose((pd.DataFrame(unique_max_test, columns=['Feature', 'Max duplicates', 'Value'])).
                            sort_values(by = 'Max duplicates', ascending=False).head(15))
    
    idx = features = df_train.columns.values[2:202]
    for df in [df_test, df_train]:
        df['sum'] = df[idx].sum(axis=1)  
        df['min'] = df[idx].min(axis=1)
        df['max'] = df[idx].max(axis=1)
        df['mean'] = df[idx].mean(axis=1)
        df['std'] = df[idx].std(axis=1)
        df['skew'] = df[idx].skew(axis=1)
        df['kurt'] = df[idx].kurtosis(axis=1)
        df['med'] = df[idx].median(axis=1)
        
    t0 = df_train.loc[df_train['target'] == 0]
    t1 = df_train.loc[df_train['target'] == 1]
    features = df_train.columns.values[202:]
    #plot_new_feature_distribution(t0, t1, 'target: 0', 'target: 1', features)
    
    features = df_train.columns.values[202:]
    #plot_new_feature_distribution(df_train, df_test, 'train', 'test', features)
    
    
    features = [c for c in df_train.columns if c not in ['ID_code', 'target']]
    for feature in features:
        df_train['r2_'+feature] = np.round(df_train[feature], 2)
        df_test['r2_'+feature] = np.round(df_test[feature], 2)
        df_train['r1_'+feature] = np.round(df_train[feature], 1)
        df_test['r1_'+feature] = np.round(df_test[feature], 1)
        
    #DNN(df_train, df_test)
    xgb_res = XGBBest(df_train, df_test)
    gc.collect()
    cnn1d_res = CNN1D(df_train, df_test)
    gc.collect()
    cnn2d_res = CNN2D(df_train, df_test)
    gc.collect()
    
    res = np.vstack([cnn1d_res.to_numpy(), cnn2d_res.to_numpy(), xgb_res.to_numpy()]).T
    result = pd.DataFrame({"ID_code": df_test['ID_code']})
    result["target"] = res
    result.to_csv("submission.csv", index=False)