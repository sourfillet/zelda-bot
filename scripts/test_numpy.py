import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

print("Eager execution:", tf.executing_eagerly())

model = Sequential([Dense(1, input_shape=(1,))])
model.compile(loss='mse', optimizer='adam')

x = np.array([[1.0]])
y = np.array([[2.0]])
model.fit(x, y, epochs=1)