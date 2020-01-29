from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

scaler = StandardScaler()

def splitTrainTest(data, train_period, test_period, multiVariate=False, gtruthName='SPI'):
    if not multiVariate:
        train = data[[gtruthName]].loc[:train_period - 1]
        test = data[[gtruthName]].loc[train_period:train_period + test_period - 1]
    else:
        train = data.loc[:train_period - 1]
        test = data.loc[train_period:train_period + test_period - 1]

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


def plot():
    pass


if __name__ == "__main__":
    lookBack = 4
    multiVariate = True
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
                                                        multiVariate=multiVariate)
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
                                                        multiVariate=multiVariate)
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

    pipe_model = Pipeline([('scl', StandardScaler()), ('clf', RandomForestRegressor())])
    n_estimators = [100,200,300,400,500]
    max_features = ['auto','sqrt','log2']
    max_depths = [3,4,5,6,7]
    criterions = ['friedman_mse', 'mse', 'mae']
    param_grid = [{'clf__n_estimators': n_estimators,
                   'clf__max_depth': max_depths,
                   'clf__max_features': max_features,
                   'clf__criterion': criterions}]
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