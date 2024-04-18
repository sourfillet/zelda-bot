import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

# Constants
SCREEN_WIDTH = 256
SCREEN_HEIGHT = 240
NUM_ACTIONS = 8
MEMORY_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.95
EPSILON_MAX = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

class DoubleDQN:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_MAX
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 1)))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(NUM_ACTIONS))
        model.compile(loss='mse', optimizer=Adam())
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(NUM_ACTIONS)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_Q = self.model.predict(next_state)[0]
                best_action = np.argmax(next_Q)
                target = reward + GAMMA * self.target_model.predict(next_state)[0][best_action]
            target_f = self.model.predict(state)
            target_f[0][action] = target
            states.append(state[0])
            targets.append(target_f[0])
        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def update_target_model_regularly(self, episode):
        if episode % 100 == 0:
            self.update_target_model()

    def act(self, state):
        # Select action based on the actor network
        state = np.expand_dims(state, axis=0)
        action = self.actor.predict(state)[0]
        return action

    def replay(self):
        # Train the networks using a random batch from the replay memory
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        # Train the critic network
        next_actions = self.target_actor.predict(next_states)
        q_targets = rewards + GAMMA * (1 - dones) * self.target_critic.predict([next_states, next_actions])
        self.critic.train_on_batch([states, actions], q_targets)

        # Train the actor network
        actions_for_grad = self.actor.predict(states)
        grads = self.critic.gradients(states, actions_for_grad)
        self.actor.train_on_batch(states, grads)

        # Soft update the target networks
        self.update_target_models()

    def train(self, env, episodes, max_steps):
        # Main training loop
        for episode in range(episodes):
            state = env.reset()
            state = np.expand_dims(state, axis=-1)
            total_reward = 0

            for step in range(max_steps):
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                next_state = np.expand_dims(next_state, axis=-1)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if done:
                    break

            self.replay()
            print(f"Episode: {episode+1}, Total Reward: {total_reward}")