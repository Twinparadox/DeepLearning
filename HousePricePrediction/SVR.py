from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

scaler = StandardScaler()

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
        
def evaluate(pred, gtruth, n_features):
    tmp_pred = np.zeros(shape=(len(pred),n_features))
    tmp_gtruth = np.zeros(shape=(len(gtruth),n_features))
    
    tmp_pred[:,0] = pred[:,0]
    tmp_gtruth[:,0] = gtruth[:,0]
    
    pred = scaler.inverse_transform(tmp_pred)[:,0]
    gtruth = scaler.inverse_transform(tmp_gtruth)[:,0]

    MAE = mean_absolute_error(gtruth, pred)
    print("test MAE", MAE)
    RMSE = np.sqrt(mean_squared_error(gtruth, pred))
    print("test RMSE", RMSE)


def plot(pred, gtruth, n_features, multiVariate=False):
    tmp_pred = np.zeros(shape=(len(pred),n_features))
    tmp_gtruth = np.zeros(shape=(len(gtruth),n_features))
    
    pred = pred.reshape(-1,1)
    
    tmp_pred[:,0] = pred[:,0]
    tmp_gtruth[:,0] = gtruth[:,0]
    
    pred = scaler.inverse_transform(tmp_pred)[:,0]
    gtruth = scaler.inverse_transform(tmp_gtruth)[:,0]

    plt.plot(pred, 'g')
    plt.plot(gtruth, 'r')
    plt.show()


if __name__ == "__main__":
    lookBack = 1
    multiVariate = False
    train_period1 = 128
    test_period1 = 12
    train_period2 = 32
    test_period2 = 12

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
    Y_train1 = train1_df.dropna()[['SPI']].values

    if not multiVariate:
        n_features = 1

    else:           
        n_features = len(column_list)        
        
    X_test1 = test1_df.dropna().drop('SPI', axis=1).values
    Y_test1 = test1_df.dropna()[['SPI']].values
    
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
    Y_train2 = train2_df.dropna()[['SPI']].values

    if not multiVariate:
        n_features = 1

    else:           
        n_features = len(column_list)        

    X_test2 = test2_df.dropna().drop('SPI', axis=1).values
    Y_test2 = test2_df.dropna()[['SPI']].values 
    

    pipe_model = Pipeline([('scl', StandardScaler()), ('clf', SVR())])
    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    param_range2 = [0.0001, 0.001, 0.01, 0.1]
    param_grid = [{'clf__C': param_range,
                   'clf__kernel': ['rbf'],
                   'clf__gamma': param_range,
                   'clf__epsilon': param_range2}]
    gs = GridSearchCV(estimator=pipe_model, param_grid=param_grid,
                      scoring='neg_mean_absolute_error', cv=5, iid=True, n_jobs=-1)
    
    ################# Whole Period #################
    gs = gs.fit(X_train1, Y_train1)
    print(gs.best_score_)
    print(gs.best_params_)

    best_params = gs.best_params_
    best_model = gs.best_estimator_

    Y_pred = best_model.predict(X_test1)
    MAE = mean_absolute_error(Y_test1, Y_pred)
    print("test MAE", MAE)
    RMSE = np.sqrt(mean_squared_error(Y_test1, Y_pred))
    print("test RMSE", RMSE)

    x_data = range(len(test1_sc))
    plt.plot(x_data[:], test1_sc[:len(test1_sc),0], color='red')
    plt.plot(x_data[lookBack:], Y_pred[:], color='green')
    plt.show()
    
    ################# Lehman #################
    gs = gs.fit(X_train2, Y_train2)
    print(gs.best_score_)
    print(gs.best_params_)

    best_params = gs.best_params_
    best_model = gs.best_estimator_

    Y_pred = best_model.predict(X_test2)
    MAE = mean_absolute_error(Y_test2, Y_pred)
    print("test MAE", MAE)
    RMSE = np.sqrt(mean_squared_error(Y_test2[:], Y_pred))
    print("test RMSE", RMSE)

    x_data = range(len(test2_sc))
    plt.plot(x_data[:], test2_sc[:len(test2_sc),0], color='red')
    plt.plot(x_data[lookBack:], Y_pred[:], color='green')
    plt.show()