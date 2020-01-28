from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


scaler = MinMaxScaler(feature_range=(0, 1))


def splitTrainTest(data, train_period, test_period, multiVariate=False, gtruthName='SPI'):
    if not multiVariate:
        train = data[[gtruthName]].loc[:train_period - 1]
        test = data[[gtruthName]].loc[train_period - lookBack:]
    else:
        train = data.loc[:train_period - 1]
        test = data.loc[train_period - lookBack:]

    train_sc = scaler.fit_transform(train)
    test_sc = scaler.fit_transform(test)

    return train, test, train_sc, test_sc


def makeSlidingWindows(origin_train, origin_test, train, test, lookBack=2, multiVariate=False, gtruthName='SPI'):
    train_df = pd.DataFrame(train, index=origin_train.index)
    test_df = pd.DataFrame(test, index=origin_test.index)

    train_df.columns = origin_train.columns
    test_df.columns = origin_test.columns

    if not multiVariate:
        column_list = list(train_df)
        for s in range(1, lookBack + 1):
            tmp_train = train_df[column_list].shift(s)
            tmp_test = test_df[column_list].shift(s)

            tmp_train.columns = "shift_" + tmp_train.columns + "_" + str(s)
            tmp_test.columns = "shift_" + tmp_test.columns + "_" + str(s)

            train_df[tmp_train.columns] = train_df[column_list].shift(s)
            test_df[tmp_test.columns] = test_df[column_list].shift(s)

        return train_df, test_df

    else:
        column_list = list(train_df)

        for s in range(1, lookBack + 1):
            tmp_train = train_df[column_list].shift(s)
            tmp_test = test_df[column_list].shift(s)

            tmp_train.columns = "shift_" + tmp_train.columns + "_" + str(s)
            tmp_test.columns = "shift_" + tmp_test.columns + "_" + str(s)

            train_df[tmp_train.columns] = train_df[column_list].shift(s)
            test_df[tmp_test.columns] = test_df[column_list].shift(s)

        return train_df, test_df

def evaluate(pred, gtruth):
    pred = scaler.inverse_transform(pred.reshape(-1,1))    
    gtruth = scaler.inverse_transform(gtruth.reshape(-1,1))
    
    MAE = mean_absolute_error(gtruth, pred)
    print("test MAE", MAE)
    RMSE = np.sqrt(mean_squared_error(gtruth, pred))
    print("test RMSE", RMSE)
        

def plot(pred, gtruth):
    pred = scaler.inverse_transform(pred.reshape(-1,1))    
    gtruth = scaler.inverse_transform(gtruth.reshape(-1,1))  
    
    plt.plot(pred, 'g')
    plt.plot(gtruth, 'r')
    plt.show()


if __name__ == "__main__":
    lookBack = 4
    multiVariate = False
    n_features = 1
    train_period1 = 128
    test_period1 = 12
    train_period2 = 32
    test_period1 = 12
    
    epochs = 100
    n_units = [20, 50, 100, 150, 200]
    

    df = pd.read_csv('./data/houseprice.csv')
    data = df[['SPI', 'CPI', 'M2', 'MMI', 'CBD']]
    column_list = list(data)

    train1, test1, train1_sc, test1_sc = splitTrainTest(data=data,
                                                        train_period=train_period1,
                                                        test_period=test_period1,
                                                        multiVariate=multiVariate)
    train1_df, test1_df = makeSlidingWindows(origin_train=train1,
                                             origin_test=test1,
                                             train=train1_sc,
                                             test=test1_sc,
                                             lookBack=lookBack,
                                             multiVariate=multiVariate)

    X_train1 = train1_df.dropna().drop('SPI', axis=1).values
    Y_train1 = train1_df.dropna()[['SPI']].values

    X_test1 = test1_df.dropna().drop('SPI', axis=1).values
    Y_test1 = test1_df.dropna()[['SPI']].values
    
    if not multiVariate:
        n_features = 1
        
    else:
        n_features = len(column_list)
        
    
    X_train1 = X_train1.reshape(X_train1.shape[0], lookBack, n_features)
    X_test1 = X_test1.reshape(X_test1.shape[0], lookBack, n_features)

    # For Checkpoint
    MODEL_SAVE_FOLDER_PATH = 'model/'
    if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
      os.mkdir(MODEL_SAVE_FOLDER_PATH)

    model_path = MODEL_SAVE_FOLDER_PATH + '{epoch:02d}-{val_loss:.4f}.hdf5'
    cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss',
                                    verbose=1, save_best_only=True)
    
    for units in n_units:
        print("#######################"+str(units)+"#######################")
              
        K.clear_session()
        model = Sequential()
        model.add(LSTM(units, activation='relu', input_shape=(lookBack, n_features)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse','mae'])
        model.summary()
        
        history = model.fit(X_train1, Y_train1, epochs=100, batch_size=10, verbose=1)
        model.save_weights('model/model.h5')
        acc = history.history['mae']
        loss = history.history['loss']
    
        epochs = range(1, len(acc) + 1)
    
        plt.plot(epochs, acc, 'b', label='Training mae')
        plt.title('Mean_Absolute_Error')
        plt.legend()
        plt.figure()
    
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.title('Loss')
        plt.legend()
        plt.show()
    
        Y_pred = model.predict(X_test1)
        
        plot(Y_pred, Y_test1)
        evaluate(Y_pred, Y_test1)
