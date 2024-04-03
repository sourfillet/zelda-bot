import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
from collections import deque
import random
import nintaco

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

# Agent class
class ZeldaAgent:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_MAX
        self.model = self.build_model()

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
                target = reward + GAMMA * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            states.append(state[0])
            targets.append(target_f[0])
        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

def getScreen(api):
    screen = api.getPixels()
    screen = np.array(screen)
    screen = screen.reshape(SCREEN_HEIGHT, SCREEN_WIDTH)
    screen = screen[56:240, 0:255]
    screen = screen.astype(np.uint8)
    screen = np.expand_dims(screen, axis=-1)
    return screen

def renderFinished(api):
    global agent, state  # Declare agent and state as global variables
    
    next_state = getScreen(api)
    next_state = np.reshape(next_state, (1, SCREEN_HEIGHT, SCREEN_WIDTH, 1))
    reward = 0  # Implement reward calculation based on game progress
    done = False  # Implement game over condition
    agent.remember(state, action, reward, next_state, done)
    agent.replay()
    
    state = next_state
    action = agent.act(state)
    api.writeGamepad(0, action, True)

def statusChanged(message):
    print("Status message: %s" % message)

def exit_handler():
    api.stop()
    exit()

# Main loop
nintaco.initRemoteAPI("localhost", 9999)
api = nintaco.getAPI()

agent = ZeldaAgent()  # Create an instance of the ZeldaAgent class
state = getScreen(api)
state = np.reshape(state, (1, SCREEN_HEIGHT, SCREEN_WIDTH, 1))

api.addFrameListener(renderFinished)
api.addStatusListener(statusChanged)
api.addStopListener(exit_handler)
api.run()