import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate, discount_factor, epsilon, epsilon_decay, epsilon_min):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model = self._build_model()

    def _build_model(self):
        """
        Build the DQN model.
        """
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 1)))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        """
        Choose an action based on the epsilon-greedy policy
        """
        if np.random.rand() <= self.epsilon:
            # Choose a random index and convert it to a one-hot vector
            action_index = np.random.randint(self.action_size)
            action = np.zeros(self.action_size, dtype=int)
            action[action_index] = 1
            return action
        q_values = self.model.predict(state)
        action_index = np.argmax(q_values[0])
        action = np.zeros(self.action_size, dtype=int)
        action[action_index] = 1
        return action

    def train(self, state, action, reward, next_state, done):
        """
        Train the model
        """
        target = reward
        if not done:
            target = (reward + self.discount_factor * np.amax(self.model.predict(next_state)[0]))
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)

    def update_epsilon(self):
        """
        Update epsilon
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """
        Save the model to a file
        """
        self.model.save(filepath)