import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import SGD
X=[]
Y=[]
for i in range(1000):
    n = random.randint(0,100)
    X.append(n)
    Y.append(n*2)
#X is the input array and Y is the output array
X = np.array(X)
Y = np.array(Y)

from keras.models import Sequential #using Keras Library
model=Sequential()
model.add(Dense(1,activation='relu',input_shape=(1,)))
model.add(Dense(1,activation='relu'))
model.compile(loss='MSE', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, batch_size=50, validation_split=0.1,epochs=1000, verbose=1,shuffle=1)

test_array=np.array([4,27,100,121,9])
print(model.get_weights())
print(model.predict([test_array]))