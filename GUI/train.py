import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

from sklearn.model_selection import train_test_split

def train(data):
    X=data.drop('Y',axis=1)
    Y=data['Y']

    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,
                                                                    test_size=0.2)

    model = Sequential()
    model.add(Dense(30, input_dim=3, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    history=model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation)
                      ,epochs=200, batch_size=100)

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    x_epochs = range(1, len(train_loss) + 1)

    plt.plot(x_epochs, train_loss, 'b', label='Training loss')
    plt.plot(x_epochs, val_loss, 'r', label='Validation loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

    model.save('model.h5')

if __name__ == "__main__":
    df = pd.read_csv('data.csv', engine='c')
    train(df)
    