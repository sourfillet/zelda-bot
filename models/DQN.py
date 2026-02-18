import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
import random
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate, discount_factor, 
                 epsilon, epsilon_decay, epsilon_min):
        self.state_size = state_size    # Not used directly; state shape comes from preprocessed images.
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Experience Replay parameters
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        self.train_start = 32  # Begin training only when memory has at least this many samples.

        # Target network update frequency (in training steps)
        self.target_update_freq = 100  # update target network every 100 training steps
        self.train_counter = 0

        # Build main Q-network and target Q-network
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()  # initialize target network weights

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

    def update_target_model(self):
        """
        Copy weights from the main network to the target network.
        """
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        """
        Choose an action based on the epsilon-greedy policy.
        Returns a one-hot encoded action vector.
        """
        if np.random.rand() <= self.epsilon:
            action_index = np.random.randint(self.action_size)
            action = np.zeros(self.action_size, dtype=int)
            action[action_index] = 1
            return action

        q_values = self.model.predict(state, verbose=0)
        action_index = np.argmax(q_values[0])
        action = np.zeros(self.action_size, dtype=int)
        action[action_index] = 1
        return action

    def train(self, state, action, reward, next_state, done):
        """
        Store the transition in memory and train the model using experience replay.
        This method uses a mini-batch of past transitions and computes targets using the target network.
        Returns the training loss if training occurred, otherwise None.
        """
        # Convert one-hot action to index if necessary
        if isinstance(action, np.ndarray) and action.shape == (self.action_size,):
            action_index = np.argmax(action)
        else:
            action_index = action

        # Store transition
        self.memory.append((state, action_index, reward, next_state, done))

        # Only start training when enough samples are available
        if len(self.memory) < self.batch_size:
            return None

        # Sample a mini-batch from the memory
        minibatch = random.sample(self.memory, self.batch_size)

        # Prepare arrays for training
        states = np.vstack([sample[0] for sample in minibatch])
        next_states = np.vstack([sample[3] for sample in minibatch])
        actions = [sample[1] for sample in minibatch]
        rewards = np.array([sample[2] for sample in minibatch])
        dones = np.array([sample[4] for sample in minibatch]).astype(int)

        # Predict Q-values for current states and for next states (using target network)
        target = self.model.predict(states, verbose=0)
        target_next = self.target_model.predict(next_states, verbose=0)

        # Update the Q-value for the taken action
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * np.amax(target_next[i])

        # Fit the main network on the updated target values
        history = self.model.fit(states, target, epochs=1, verbose=0)

        # Increment the training step counter and update target network if needed
        self.train_counter += 1
        if self.train_counter % self.target_update_freq == 0:
            self.update_target_model()

        return history.history['loss'][0]

    def update_epsilon(self):
        """
        Update the exploration rate using decay.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """
        Save the current Q-network to a file.
        """
        self.model.save(filepath)
