import numpy as np
import nintaco
import DQN

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

def getScreen(api):
    screen = np.ndarray((61440,), int)
    api.getPixels(screen)
    screen = np.array(screen)
    screen = screen.reshape(SCREEN_HEIGHT, SCREEN_WIDTH)
    screen = screen[56:240, 0:256]
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

agent = DQN()  # Create an instance of the ZeldaAgent class
state = getScreen(api)
state = np.reshape(state, (1, SCREEN_HEIGHT, SCREEN_WIDTH, 1))

api.addFrameListener(renderFinished)
api.addStatusListener(statusChanged)
api.addStopListener(exit_handler)
api.run()