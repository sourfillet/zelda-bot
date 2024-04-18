import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from collections import deque

# Constants
SCREEN_WIDTH = 256
SCREEN_HEIGHT = 240
NUM_ACTIONS = 8
MEMORY_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.95
TAU = 0.001
ACTOR_LR = 0.0001
CRITIC_LR = 0.001

class DDPG:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.target_actor = self.build_actor()
        self.target_critic = self.build_critic()
        self.update_target_models(tau=1.0)

    def build_actor(self):
        # Actor network
        inputs = Input(shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 1))
        x = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(inputs)
        x = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        outputs = Dense(NUM_ACTIONS, activation='tanh')(x)
        model = Model(inputs, outputs)
        model.compile(loss='mse', optimizer=Adam(lr=ACTOR_LR))
        return model

    def build_critic(self):
        # Critic network
        state_inputs = Input(shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 1))
        x = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(state_inputs)
        x = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = Flatten()(x)
        state_outputs = Dense(512, activation='relu')(x)

        action_inputs = Input(shape=(NUM_ACTIONS,))
        action_outputs = Dense(512, activation='relu')(action_inputs)

        concat = Concatenate()([state_outputs, action_outputs])
        outputs = Dense(1)(concat)
        model = Model([state_inputs, action_inputs], outputs)
        model.compile(loss='mse', optimizer=Adam(lr=CRITIC_LR))
        return model

    def update_target_models(self, tau=TAU):
        # Soft update of target models' weights
        actor_weights = self.actor.get_weights()
        target_actor_weights = self.target_actor.get_weights()
        for i in range(len(actor_weights)):
            target_actor_weights[i] = tau * actor_weights[i] + (1 - tau) * target_actor_weights[i]
        self.target_actor.set_weights(target_actor_weights)

        critic_weights = self.critic.get_weights()
        target_critic_weights = self.target_critic.get_weights()
        for i in range(len(critic_weights)):
            target_critic_weights[i] = tau * critic_weights[i] + (1 - tau) * target_critic_weights[i]
        self.target_critic.set_weights(target_critic_weights)

    def remember(self, state, action, reward, next_state, done):
        # Store experience in replay memory
        self.memory.append((state, action, reward, next_state, done))

