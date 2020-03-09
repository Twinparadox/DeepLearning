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

def splitTrainTest(data, train_period, test_period):

    train = data.loc[:train_period]
    test = data.loc[train_period-lookBack:]
    
    scaler = MinMaxScaler(feature_range=(0,1))
    train_sc = scaler.fit_transform(train)
    test_sc = scaler.fit_transforM(test)
            
    return train_sc, test_sc

def makeSlidingWindows(train, test, lookBack = 2, multivariate = False, gtruthName = 'SPI'):
    
    column_list = list(train_df)
    
    train_df = pd.DataFrame(train, index=train.index)
    test_df = pd.DataFrame(test, index=test.index)
    
    for s in range(1, lookBack+1):
        tmp_train = train_df[column_list].shift(s)
        tmp_test = test_df[column_list].shift(s)
        
        tmp_train.columns = "shift_"+tmp_train.columns+"_"+str(s)
        tmp_test.columns = "shift_"+tmp_test.columns+"_"+str(s)
        
        train_df[tmp_train.columns] = train_df[column_list].shift(s)
        test_df[tmp_test.columns] = test_df[column_list].shift(s)
    
    return train_df.values, test_df.values
        
def plot():
    pass

if __name__ == "__main__":
    lookBack = 4
    multiVariate = True
    train_period1 = 128
    test_period1 = 12
    train_period2 = 32
    test_period1 = 12
    
    df = pd.read_csv('./data/houseprice.csv')
    data = df[['SPI', 'CPI', 'M2', 'MMI', 'CBD']]
    column_list = list(data)
    
    train1= data.loc[:train_period1 - 1]
    test1 = data.loc[train_period1-lookBack:]
    
    # normalize time series
    scaler = MinMaxScaler(feature_range=(0, 1))
    train1_sc = scaler.fit_transform(train1)
    test1_sc = scaler.fit_transform(test1)
    
    train1_df = pd.DataFrame(train1_sc, columns=column_list, index=train1.index)
    test1_df = pd.DataFrame(test1_sc, columns=column_list, index=test1.index)
    
    for s in range(1, lookBack+1):
        tmp_train = train1_df[column_list].shift(s)
        tmp_test = test1_df[column_list].shift(3)
        
        tmp_train.columns = "shift_"+tmp_train.columns+"_"+str(s)
        tmp_test.columns = "shift_"+tmp_test.columns+"_"+str(s)
        
        train1_df[tmp_train.columns] = train1_df[column_list].shift(s)
        test1_df[tmp_test.columns] = test1_df[column_list].shift(s)
        
    X_train1 = train1_df.dropna().drop('SPI', axis=1).values
    Y_train1 = train1_df.dropna()[['SPI']].values
    
    X_test1 = test1_df.dropna().drop('SPI', axis=1).values
    Y_test1 = test1_df.dropna()[['SPI']].values
    
    pipe_model = Pipeline([('scl', StandardScaler()), ('clf', SVR())])
    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    param_range2 = [0.0001, 0.001, 0.01, 0.1]
    param_grid = [{'clf__C': param_range,
                   'clf__kernel': ['rbf'],
                   'clf__gamma': param_range,
                   'clf__epsilon': param_range2}]
    gs = GridSearchCV(estimator=pipe_model, param_grid=param_grid,
                      scoring='neg_mean_absolute_error', cv=5, iid=True, n_jobs=-1)
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
    
    plt.plot(Y_pred, 'g')
    plt.plot(Y_test1, 'r')
    plt.show()