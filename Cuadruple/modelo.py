import keras.optimizers
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import random

X = []
Y = []
for i in range(10000):
    n = random.randint(0, 300)
    X.append(n)
    Y.append(n*5)

df = pd.read_csv('train.csv', delimiter=',')
df = df.to_numpy()

X = np.array(X)
Y = np.array(Y)
#print(Y)
#print(Y.shape)
print(type(Y))
#print(X)
#print(X.shape)
print(type(X))

predictors = np.copy(df)
predictors = df[:, 0]
predictorx = predictors.flatten()
print(type(predictors))
print(predictors.shape)
print(type(predictorx))
print(predictorx.shape)
nodos = 1
target = df[:, 1]

model = Sequential()
model.add(Dense(1, activation='relu', input_shape=(nodos,)))
model.add(Dense(1, activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X, Y, batch_size=50, validation_split=0.1, epochs=100, verbose=1, shuffle=1)
a = np.array([1, 2, 3, 823])
print(model.predict(a))
print("Hello")
