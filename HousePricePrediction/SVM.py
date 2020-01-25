from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

np.random.seed(123)

# load data, return pandas format ts and array ts, ts is formatted as a column
def load_data(filename, columnName):

    df = pd.read_csv(filename)
    df = df.fillna(0)
    ts = df[columnName]
    data = ts.values.reshape(-1, 1).astype("float32")  # (N, 1)
    print("time series shape:", data.shape)
    return ts, data

def createSamples(dataset, lookBack, RNN=True):

    dataX, dataY = [], []
    for i in range(len(dataset) - lookBack):
        sample_X = dataset[i:(i + lookBack), :]
        sample_Y = dataset[i + lookBack, :]
        dataX.append(sample_X)
        dataY.append(sample_Y)
    dataX = np.array(dataX)  # (N, lag, 1)
    dataY = np.array(dataY)  # (N, 1)
    if not RNN:
        dataX = np.reshape(dataX, (dataX.shape[0], dataX.shape[1]))

    return dataX, dataY

# divide training and testing, default as 3:1
def divideTrainTest(dataset, rate=0.75):

    train_size = 128
    test_size = 12
    train, test = dataset[0:train_size], dataset[train_size:]
    return train, test

if __name__ == "__main__":
    lookBack = 1

    ts, data = load_data("./data/houseprice.csv", columnName="AATPI")
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

    pipe_model = Pipeline([('scl', StandardScaler()), ('clf', SVR())])
    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_range2 = [0.0001, 0.001, 0.01, 0.1]
    param_grid = [{'clf__C': param_range,
                   'clf__kernel': ['rbf'],
                   'clf__gamma': param_range,
                   'clf__epsilon': param_range2}]
    gs = GridSearchCV(estimator=pipe_model, param_grid=param_grid,
                      scoring='neg_mean_absolute_error', cv=5, iid=True, n_jobs=-1)
    gs = gs.fit(trainX, trainY)
    print(gs.best_score_)
    print(gs.best_params_)

    best_params = gs.best_params_
    best_model = gs.best_estimator_

    Y_pred = best_model.predict(testX)
    MAE = mean_absolute_error(testY, Y_pred)
    print("test MAE", MAE)
    RMSE = np.sqrt(mean_squared_error(testY, Y_pred))
    print("test RMSE", RMSE)