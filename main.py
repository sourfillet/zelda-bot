import retro
import random
import numpy as np
from DQN import DQNAgent
import cv2

env = retro.make(game='Zelda1')
old_info = None
times_reset = 0
visited_rooms = {}
total_rewards = 0

def preprocess_state(state):
    # Resize the state to a smaller size
    state = cv2.resize(state, (84, 84))
    
    # Convert the state to grayscale
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    
    # Reshape the state to match the expected input shape of the model
    state = np.reshape(state, [1, 84, 84, 1])
    
    return state

def get_actual_hearts(hearts=0):
    partials = {
                256:4,
                128:3.5,
                255:3,
                127:2.5,
                254:2,
                126:1.5,
                253:1,			
                125:0.5,
                0:0,
            }
    
    return partials[hearts]

def get_reward(info=None):
    global old_info
    global total_rewards
    if info is None:
        return 0

    reward = 0
    info['Hearts'] = get_actual_hearts(info['Hearts'])

    if old_info is None:
        old_info = info
        return 0
    
    if info['Level'] != 1:
        reward -= 100

    if info['Room'] not in visited_rooms:
        visited_rooms[info['Room']] = {}
        reward += 10

    if (info['Link X'], info['Link Y']) not in visited_rooms[info['Room']]:
        visited_rooms[info['Room']][(info['Link X'], info['Link Y'])] = 1
        reward += 2.5
    else:
        reward -= 1

    if info['Hearts'] != old_info['Hearts']:
        if info['Hearts'] < old_info['Hearts']:
            reward -= old_info['Hearts'] - info['Hearts']
        elif info['Hearts'] > old_info['Hearts']:
            reward += info['Hearts'] - old_info['Hearts']
    
    if info['EnemiesKilledSinceLastHit'] != old_info['EnemiesKilledSinceLastHit']:
        if info['EnemiesKilledSinceLastHit'] > old_info['EnemiesKilledSinceLastHit']:
            reward += info['EnemiesKilledSinceLastHit'] - old_info['EnemiesKilledSinceLastHit']
        elif info['EnemiesKilledSinceLastHit'] < old_info['EnemiesKilledSinceLastHit']:
            reward -= old_info['EnemiesKilledSinceLastHit'] - info['EnemiesKilledSinceLastHit']

    if info['Bombs'] != old_info['Bombs']:
        if info['Bombs'] > old_info['Bombs']:
            reward += info['Bombs'] - old_info['Bombs']
        elif info['Bombs'] < old_info['Bombs']:
            reward -= old_info['Bombs'] - info['Bombs']

    if info['Keys'] != old_info['Keys']:
        if info['Keys'] > old_info['Keys']:
            reward += info['Keys'] - old_info['Keys']
        elif info['Keys'] < old_info['Keys']:
            reward -= old_info['Keys'] - info['Keys']

    if info['Rupees'] != old_info['Rupees']:
        if info['Rupees'] > old_info['Rupees']:
            reward += info['Rupees'] - old_info['Rupees']
        elif info['Rupees'] < old_info['Rupees']:
            reward -= old_info['Rupees'] - info['Rupees']

    if info['NumDeaths'] != old_info['NumDeaths']:
        if info['NumDeaths'] > old_info['NumDeaths']:
            reward -= old_info['NumDeaths'] - info['NumDeaths']
        elif info['NumDeaths'] < old_info['NumDeaths']:
            reward += info['NumDeaths'] - old_info['NumDeaths']

    if info['Boomerang'] != old_info['Boomerang']:
        if info['Boomerang'] > old_info['Boomerang']:
            reward += info['Boomerang'] - old_info['Boomerang']
        elif info['Boomerang'] < old_info['Boomerang']:
            reward -= old_info['Boomerang'] - info['Boomerang']

    old_info = info
    total_rewards += reward
    return reward

def make_move(state, old_info, epsilon):
    if random.random() < epsilon:
        # Take a random action
        return env.action_space.sample()
    else:
        # Take the best action based on the reward function
        best_reward = float('-inf')
        best_action = None

        for action in range(env.action_space.n):
            _, _, done, info = env.step(action)
            reward = get_reward(info)

            if reward > best_reward:
                best_reward = reward
                best_action = action

            if done:
                break

            env.reset()
            env.step(state)

        return best_action
    
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
learning_rate = 0.001
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 1

agent = DQNAgent(state_size, action_size, learning_rate, discount_factor, epsilon, epsilon_decay, epsilon_min)

for episode in range(num_episodes):
    state = env.reset()
    state = preprocess_state(state)
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)

        if info["Screen Scrolling"] != 0:
            # screen is scrolling, do nothing
            continue

        reward = get_reward(info)
        print("Rewards: " + str(reward))
        next_state = preprocess_state(next_state)
        agent.train(state, action, reward, next_state, done)
        state = next_state
        env.render()

    agent.update_epsilon()

    # Save the trained model
    model_path = 'model.h5'
    agent.save(model_path)

env.close()