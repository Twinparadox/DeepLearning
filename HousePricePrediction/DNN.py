from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from keras.models import load_model
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

scaler = MinMaxScaler()

def splitTrainTest(data, train_period, test_period, multiVariate=False, gtruthName='SPI', period=1):
    if not multiVariate:
        train = data[[gtruthName]].loc[:train_period - 1]
        test = data[[gtruthName]].loc[train_period:train_period + test_period - 1]
        
        sc_data = scaler.fit_transform(data)
        sc_train = sc_data[:train_period]
        sc_test = sc_data[train_period:train_period+test_period]
        
        return train, test, sc_train, sc_test
        
    else:
        train = data.loc[:train_period - 1]
        test = data.loc[train_period:train_period + test_period - 1]
        
        sc_data = scaler.fit_transform(data)
        sc_train = sc_data[:train_period]
        sc_test = sc_data[train_period:train_period+test_period]
        
        return train, test, sc_train, sc_test
        
def makeSlidingWindows(origin_train, origin_test, train, test, lookBack=2, multiVariate=False, gtruthName='SPI'):

    if not multiVariate:
        train_df = pd.DataFrame(train[:,0], index=origin_train.index)
        test_df = pd.DataFrame(test[:,0], index=origin_test.index)
        train_df.columns = origin_train.columns
        test_df.columns = origin_test.columns
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
        train_df = pd.DataFrame(train, index=origin_train.index)
        test_df = pd.DataFrame(test, index=origin_test.index)
        train_df.columns = origin_train.columns
        test_df.columns = origin_test.columns
        column_list = list(train_df)

        for s in range(1, lookBack + 1):
            tmp_train = train_df[column_list].shift(s)
            tmp_test = test_df[column_list].shift(s)

            tmp_train.columns = "shift_" + tmp_train.columns + "_" + str(s)
            tmp_test.columns = "shift_" + tmp_test.columns + "_" + str(s)

            train_df[tmp_train.columns] = train_df[column_list].shift(s)
            test_df[tmp_test.columns] = test_df[column_list].shift(s)

        return train_df, test_df

def evaluate(pred, gtruth, n_features):
    tmp_pred = np.zeros(shape=(len(pred),n_features))
    tmp_gtruth = np.zeros(shape=(len(gtruth),n_features))
    
    tmp_pred[:,0] = pred[:,0]
    tmp_gtruth[:,0] = gtruth[:,0]

    MAE = mean_absolute_error(gtruth, pred)
    print("test MAE", MAE)
    RMSE = np.sqrt(mean_squared_error(gtruth, pred))
    print("test RMSE", RMSE)
    
    return MAE, RMSE

def plot(pred, gtruth, n_features):
    tmp_pred = np.zeros(shape=(len(pred),n_features))
    tmp_gtruth = np.zeros(shape=(len(gtruth),n_features))
    
    tmp_pred[:,0] = pred[:,0]
    tmp_gtruth[:,0] = gtruth[:,0]
    
    pred = scaler.inverse_transform(tmp_pred)[:,0]
    gtruth = scaler.inverse_transform(tmp_gtruth)[:,0]

    plt.plot(pred, 'g')
    plt.plot(gtruth, 'r')
    plt.show()


if __name__ == "__main__":
    lookBack = 1
    multiVariate = True
    n_features = 1
    train_period1 = 128
    test_period1 = 12
    train_period2 = 32
    test_period2 = 12
    
    epochs = 100
    n_units = range(50, 350, 50)
    

    df = pd.read_csv('./data/houseprice.csv')
    data = df[['SPI', 'CPI', 'M2', 'MMI', 'CBD']]
    column_list = list(data)

    train1, test1, train1_sc, test1_sc = splitTrainTest(data=data,
                                                        train_period=train_period1,
                                                        test_period=test_period1,
                                                        multiVariate=multiVariate,
                                                        period=1)
    train1_df, test1_df = makeSlidingWindows(origin_train=train1,
                                             origin_test=test1,
                                             train=train1_sc,
                                             test=test1_sc,
                                             lookBack=lookBack,
                                             multiVariate=multiVariate)
    
    X_train1 = train1_df.dropna().drop('SPI', axis=1).values 
    X_test1 = test1_df.dropna().drop('SPI', axis=1).values

    if not multiVariate:
        n_features = 1

    else:           
        X_train1 = train1_df.dropna().drop(column_list, axis=1).values 
        X_test1 = test1_df.dropna().drop(column_list, axis=1).values
        n_features = len(column_list)        
        
    Y_train1 = train1_df.dropna()[['SPI']].values
    Y_test1 = test1_df.dropna()[['SPI']].values

    X_train1 = X_train1.reshape(X_train1.shape[0], lookBack, n_features)
    X_test1 = X_test1.reshape(X_test1.shape[0], lookBack, n_features)
    
    
    train2, test2, train2_sc, test2_sc = splitTrainTest(data=data,
                                                        train_period=train_period2,
                                                        test_period=test_period2,
                                                        multiVariate=multiVariate,
                                                        period=2)
    train2_df, test2_df = makeSlidingWindows(origin_train=train2,
                                             origin_test=test2,
                                             train=train2_sc,
                                             test=test2_sc,
                                             lookBack=lookBack,
                                             multiVariate=multiVariate)
    
    X_train2 = train2_df.dropna().drop('SPI', axis=1).values 
    X_test2 = test2_df.dropna().drop('SPI', axis=1).values

    if not multiVariate:
        n_features = 1

    else:           
        X_train2 = train2_df.dropna().drop(column_list, axis=1).values 
        X_test2 = test2_df.dropna().drop(column_list, axis=1).values
        n_features = len(column_list)        
        
    Y_train2 = train2_df.dropna()[['SPI']].values
    Y_test2 = test2_df.dropna()[['SPI']].values

    X_train2 = X_train2.reshape(X_train2.shape[0], lookBack, n_features)
    X_test2 = X_test2.reshape(X_test2.shape[0], lookBack, n_features)

    # For Checkpoint
    MODEL_SAVE_FOLDER_PATH = 'model/'
    if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
        os.mkdir(MODEL_SAVE_FOLDER_PATH)

    model_path = MODEL_SAVE_FOLDER_PATH + '{epoch:02d}-{val_loss:.4f}.hdf5'
    cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss',
                                    verbose=1, save_best_only=True)


    # Period 1
    period1_rmse = []
    period1_mae = []
    for units in n_units:        
        print("#######################"+str(units)+"#######################")
              
        K.clear_session()
        model = Sequential()
        model.add(Dense(units, activation='relu', input_shape=(lookBack, n_features),
                        kernel_initializer='normal', name='Hidden-1'))
        print(model.layers[-1].input_shape)
        model.add(Dense(units, activation='relu', kernel_initializer='normal', name='Hidden-2'))
        model.add(Dense(units, activation='relu', kernel_initializer='normal', name='Hidden-3'))
        model.add(Flatten())
        model.add(Dense(1, activation='relu', name='Output'))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])
        model.summary()
        
        history = model.fit(X_train1, Y_train1, epochs=epochs, batch_size=5, verbose=1)
        model.save('model/model_period1_'+str(units)+'.hdf5')
        acc = history.history['mae']
        loss = history.history['loss']
        
        x_epochs = range(1, len(acc) + 1)
    
        plt.plot(x_epochs, acc, 'b', label='Training mae')
        plt.title('Mean_Absolute_Error')
        plt.legend()
        plt.figure()
    
        plt.plot(x_epochs, loss, 'b', label='Training loss')
        plt.title('Loss')
        plt.legend()
        plt.show()
    
        Y_pred = model.predict(X_test1)

        mae, rmse = evaluate(Y_pred, Y_test1, n_features=n_features)
        
        period1_rmse.append(rmse)
        period1_mae.append(mae)


    # Period 2    
    period2_rmse = []
    period2_mae = []
    for units in n_units:        
        print("#######################"+str(units)+"#######################")
              
        K.clear_session()
        model = Sequential()
        model.add(Dense(units, activation='relu', input_shape=(lookBack, n_features),
                        kernel_initializer='normal', name='Hidden-1'))
        print(model.layers[-1].input_shape)
        model.add(Dense(units, activation='relu', kernel_initializer='normal', name='Hidden-2'))
        model.add(Dense(units, activation='relu', kernel_initializer='normal', name='Hidden-3'))
        model.add(Flatten())
        model.add(Dense(1, activation='relu', name='Output'))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])
        model.summary()
        
        history = model.fit(X_train2, Y_train2, epochs=epochs, batch_size=8, verbose=1)
        model.save('model/model_period2_'+str(units)+'.hdf5')
        acc = history.history['mae']
        loss = history.history['loss']
        x_epochs = range(1, len(acc) + 1)
    
        plt.plot(x_epochs, acc, 'b', label='Training mae')
        plt.title('Mean_Absolute_Error')
        plt.legend()
        plt.figure()
    
        plt.plot(x_epochs, loss, 'b', label='Training loss')
        plt.title('Loss')
        plt.legend()
        plt.show()
    
        Y_pred = model.predict(X_test2)
        
        mae, rmse = evaluate(Y_pred, Y_test2, n_features=n_features)
        
        period2_rmse.append(rmse)
        period2_mae.append(mae)        
        
    model1 = load_model('model/model_period1_150.hdf5')
    model2 = load_model('model/model_period2_250.hdf5')
    
    Y1_pred = model1.predict(X_test1)
    Y2_pred = model2.predict(X_test2)
    gtruth = list(train1_sc[:,0])+list(test1_sc[:,0])
    
    x_coord = range(train_period1+test_period1)
    
    plt.plot(gtruth,'r')
    plt.plot(x_coord[train_period1+1:],Y1_pred,'g')
    plt.plot(x_coord[train_period2+1:train_period2+test_period2],Y2_pred,'g')
    plt.show() 
    