import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class DQNAgent:
    def __init__(self, input_shape, action_size, learning_rate, discount_factor,
                 epsilon, epsilon_decay, epsilon_min):
        self.input_shape = input_shape  # (height, width, stacked_frames), e.g. (84, 84, 4)
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Experience Replay parameters
        self.memory: deque = deque(maxlen=20000)
        self.batch_size = 32
        self.train_start = 32  # Begin training only when memory has at least this many samples.

        # Target network update frequency (in training steps)
        self.target_update_freq = 100  # update target network every 100 training steps
        self.train_counter = 0

        # Largest |Q| seen since the last reset; main.py logs and clears this
        # per episode as an early divergence signal.
        self.max_abs_q = 0.0

        # Build main Q-network and target Q-network
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()  # initialize target network weights

        # Compiled gradient step. Built once; it captures the variable objects,
        # which load_weights()/set_weights() assign into in place.
        self._train_step = self._build_train_step()

    def _build_train_step(self):
        """
        Compile one gradient step into a tf.function.

        Same update as model.fit() — same loss and optimizer — but without
        Keras rebuilding its data adapters, callbacks and metrics on every
        call, which dominates the cost at batch 32 called once per decision.
        """
        model = self.model
        optimizer = model.optimizer
        loss_fn = tf.keras.losses.MeanSquaredError()

        @tf.function(reduce_retracing=True)
        def train_step(states, targets):
            with tf.GradientTape() as tape:
                predictions = model(states, training=True)
                loss = loss_fn(targets, predictions)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables, strict=True))
            return loss

        return train_step

    def _build_model(self):
        """
        Build the DQN model.
        """
        model = Sequential()
        # Normalize uint8 frames [0, 255] -> [0, 1] so Q-values stay in a
        # numerically stable range (matches RainbowDQN)
        model.add(tf.keras.layers.Rescaling(1.0 / 255.0, input_shape=self.input_shape))
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
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

    def _bootstrap_values(self, next_states):
        """
        Per-sample value of the next state used in the Bellman target.
        Standard DQN: max_a Q_target(s', a). Subclasses override this to change
        the bootstrap rule (e.g. Double DQN) without touching train().
        """
        target_next = self.target_model(next_states, training=False).numpy()
        return np.amax(target_next, axis=1)

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

        q_values = self.model(state, training=False).numpy()
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

        # Predict Q-values for current states; bootstrap next-state values via
        # the (overridable) target rule.
        target = self.model(states, training=False).numpy()
        bootstrap = self._bootstrap_values(next_states)

        # Record before `target` is overwritten with Bellman targets below.
        self.max_abs_q = max(self.max_abs_q, float(np.abs(target).max()))

        # Update the Q-value for the taken action
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * bootstrap[i]

        # One compiled gradient step on the updated target values
        loss = self._train_step(tf.convert_to_tensor(states),
                                tf.convert_to_tensor(target))

        # Increment the training step counter and update target network if needed
        self.train_counter += 1
        if self.train_counter % self.target_update_freq == 0:
            self.update_target_model()

        return float(loss)

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
