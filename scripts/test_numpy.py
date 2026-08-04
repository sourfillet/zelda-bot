import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

print("Eager execution:", tf.executing_eagerly())

model = Sequential([Dense(1, input_shape=(1,))])
model.compile(loss='mse', optimizer='adam')

x = np.array([[1.0]])
y = np.array([[2.0]])
model.fit(x, y, epochs=1)
