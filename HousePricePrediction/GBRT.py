from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

np.random.seed(123)

# load data, return pandas format ts and array ts, ts is formatted as a column
def load_data(filename, multiVariate=False):

    df = pd.read_csv(filename)
    df = df.fillna(0)
    if multiVariate:
        ts = df
        data = ts.values[:, 1:].astype("float32")  # (N, 1)
        print("time series shape:", data.shape)
    else:
        ts = df['SPI']
        data = ts.values.reshape(-1, 1).astype("float32")
        print("time series shape:", data.shape)
    return ts, data

def createSamples(dataset, lookBack, RNN=True, multiVariate=False):

    dataX, dataY = [], []
    for i in range(len(dataset) - lookBack):
        sample_X = dataset[i:(i + lookBack), :]
        sample_Y = dataset[i + lookBack, 0]
        dataX.append(sample_X[0])
        dataY.append(sample_Y)
    dataX = np.array(dataX)  # (N, lag, variate)
    print(dataX.shape)
    dataY = np.array(dataY)  # (N, variate)
    return dataX, dataY

# divide training and testing, default as 3:1
def divideTrainTest(dataset, rate=0.75):

    train_size = 128
    test_size = 12
    train, test = dataset[0:train_size], dataset[train_size:]
    return train, test

if __name__ == "__main__":
    lookBack = 1

    ts, data = load_data("./data/houseprice.csv", multiVariate=False)
    # normalize time series
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(data)

    # divide the series into training/testing samples
    # NOTE: Not RNN format
    train, test = divideTrainTest(dataset)

    trainX, trainY = createSamples(train, lookBack, RNN=False)
    testX, testY = createSamples(test, lookBack, RNN=False)
    print("trainX shape is", trainX.shape)
    print("trainY shape is", trainY.shape)
    print("testX shape is", testX.shape)
    print("testY shape is", testY.shape)

    pipe_model = Pipeline([('scl', StandardScaler()), ('clf', GradientBoostingRegressor())])
    n_estimators = [100, 200, 300, 400, 500]
    max_features = ['auto', 'sqrt', 'log2']
    max_depths = [4,5,6,7,8]
    criterions = ['friedman_mse', 'mse', 'mae']
    param_grid = [{'clf__n_estimators': n_estimators,
                   'clf__max_depth': max_depths,
                   'clf__criterion': criterions,
                   'clf__max_features': max_features}]
    gs = GridSearchCV(estimator=pipe_model, param_grid=param_grid,
                      scoring='neg_mean_absolute_error', cv=5, iid=True, n_jobs=-1)
    gs = gs.fit(trainX, trainY)
    print(gs.best_score_)
    print(gs.best_params_)

    best_params = gs.best_params_
    best_model = gs.best_estimator_

    # Period 1, Normal
    Y_pred = best_model.predict(testX)
    MAE = mean_absolute_error(testY, Y_pred)
    print("Period 1(2016.09~2017.08) test MAE", MAE)
    RMSE = np.sqrt(mean_squared_error(testY, Y_pred))
    print("Period 1(2016.09~2017.08) test RMSE", RMSE)

    # Period 2, Lehman Bros
    testX2 = trainX[33:45]
    testY2 = trainY[33:45]
    Y2_pred = best_model.predict(testX2)
    MAE = mean_absolute_error(testY2, Y2_pred)
    print("Period 1(2008.09~2009.08) test MAE", MAE)
    RMSE = np.sqrt(mean_squared_error(testY2, Y2_pred))
    print("Period 1(2008.09~2009.08) test RMSE", RMSE)
