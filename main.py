import retro
import random
import numpy as np
import cv2
import argparse
import json
import os
import datetime
import glob
import tensorflow as tf
from models.DQN import DQNAgent
from zelda import get_actual_hearts, single_pickup_items, multi_pickup_items, get_reward, dungeon_save_states

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

def preprocess_state(state):
    """
    Preprocess the state by resizing it to a smaller size and converting it to grayscale.
    """
    if isinstance(state, tuple):
        state = state[0]
    state = cv2.resize(state, (84, 84))
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    state = np.reshape(state, [1, 84, 84, 1])
    return state

def load_config(config_file):
    """
    Load configuration parameters from a JSON file.
    """
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    else:
        print(f"Config file {config_file} not found. Using default parameters.")
        return {}

def parse_arguments():
    """
    Parse command-line arguments and return the arguments object.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str, default='config.txt', help="Path to config file")
    args, _ = parser.parse_known_args()

    config_defaults = load_config(args.config)

    parser = argparse.ArgumentParser(
        description="Train a DQN agent on a retro game environment"
    )
    parser.add_argument('--state', type=str, default='level1', help='Name of the state to start in')
    parser.add_argument('--model', type=str, default='DQN',
                        help='Model to use: DQN, DDPG, DoubleDQN')
    parser.add_argument('--game', type=str, default=config_defaults.get('game', 'Zelda'),
                        help='Name of the game environment')
    parser.add_argument('--num_episodes', type=int, default=config_defaults.get('num_episodes', 20),
                        help='Number of episodes to run')
    parser.add_argument('--learning_rate', type=float, default=config_defaults.get('learning_rate', 0.001),
                        help='Learning rate for the agent')
    parser.add_argument('--discount_factor', type=float, default=config_defaults.get('discount_factor', 0.99),
                        help='Discount factor for training')
    parser.add_argument('--epsilon', type=float, default=config_defaults.get('epsilon', 1.0),
                        help='Initial exploration rate')
    parser.add_argument('--epsilon_decay', type=float, default=config_defaults.get('epsilon_decay', 0.995),
                        help='Epsilon decay rate')
    parser.add_argument('--epsilon_min', type=float, default=config_defaults.get('epsilon_min', 0.01),
                        help='Minimum epsilon value')
    parser.add_argument('--max_frames', type=int, default=1000,
                        help='Maximum number of frames per episode')
    return parser.parse_args()

def get_video_writer(episode, frame_size, fps=30):
    """
    Create a VideoWriter to record footage of an episode.
    """
    # Create the main recordings directory if it doesn't exist
    recordings_dir = "recordings"
    if not os.path.exists(recordings_dir):
        os.makedirs(recordings_dir)
    
    # Create a subdirectory with a timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(recordings_dir, timestamp)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    
    # Set the video file path
    video_path = os.path.join(run_dir, f"episode_{episode}.avi")
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
    return writer

def integrate(state=retro.State.DEFAULT):
    """
    Integrate the custom Zelda environment into the Retro data set.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    print("Path: ", path)
    retro.data.Integrations.add_custom_path(path)
    print("Zelda in integrations:", "Zelda" in retro.data.list_games(inttype=retro.data.Integrations.ALL))
    env = retro.make("Zelda", state=state, inttype=retro.data.Integrations.ALL, render_mode="rgb_array")
    return env

def save_model(agent, episode, model_dir="saved_models"):
    """
    Save the model to a unique file within model_dir.
    The filename includes the agent's class name, episode number, and a timestamp.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{agent.__class__.__name__}_episode{episode}_{timestamp}.h5"
    model_path = os.path.join(model_dir, filename)
    agent.model.save(model_path)
    print("Model saved at:", model_path)
    return model_path

def load_model_into_agent(agent, model_dir="saved_models", learning_rate=0.001, model_file=None):
    """
    Load a model from model_dir (or from a specified file) and assign it to the agent.
    If model_file is not provided, the most recent model file is loaded.
    """
    if model_file is None:
        files = glob.glob(os.path.join(model_dir, "*.h5"))
        if not files:
            print("No model files found in", model_dir)
            return None
        model_file = max(files, key=os.path.getctime)
    agent.model = tf.keras.models.load_model(model_file)

    # Recompile the model with the specified learning rate
    agent.model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    print("Model loaded from:", model_file)
    return model_file

def main():
    args = parse_arguments()

    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    env = integrate(args.state)
    state_shape = env.observation_space.shape  # e.g., (height, width, channels)
    state_size = state_shape[0]
    action_size = env.action_space.n
    total_rewards = 0

    # Initialize the agent
    if args.model == 'DQN':
        agent = DQNAgent(state_size, action_size, args.learning_rate,
                         args.discount_factor, args.epsilon, args.epsilon_decay, args.epsilon_min)

    # Train the agent
    for episode in range(args.num_episodes):
        # load a pre-trained model into the agent.
        load_model_into_agent(agent, learning_rate=args.learning_rate)
        
        state = env.reset()
        old_info = None
        visited_rooms = {}
        state = preprocess_state(state)
        done = False

        # Initialize per-episode reward counter.
        episode_reward = 0

        # Create a video writer for this episode.
        frame = env.render()
        height, width, channels = frame.shape
        writer = get_video_writer(episode, (width, height), fps=30)

        frame_count = 0  # Frame counter for this episode

        while not done and frame_count < args.max_frames:
            frame_count += 1

            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Process rewards only when the map is not scrolling.
            if info["Map Scroll LR"] == 0 and info["Map Scroll UD"] == 255:
                reward, old_info, visited_rooms = get_reward(visited_rooms, info, old_info, args.state)
                print("Episode:", episode, "Frame:", frame_count, "of", args.max_frames)

                if info["Room"] != 116:
                    reward = -10
                    done = True
                total_rewards += reward
                episode_reward += reward  # update reward for this episode
                print("Total rewards:", total_rewards)

                next_state = preprocess_state(next_state)
                agent.train(state, action, reward, next_state, done)
                state = next_state

            # Capture the frame and overlay the episode, frame count, and reward information.
            frame = env.render()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            overlay_text = f"Ep: {episode} | Frame: {frame_count} | Reward: {episode_reward}"

            # Define font parameters.
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.3
            thickness = 1

            # Get the size of the text box.
            (text_width, text_height), baseline = cv2.getTextSize(overlay_text, font, font_scale, thickness)

            # Set the origin for the text.
            x, y = 10, text_height + 5

            # Draw a filled black rectangle as the background for the text.
            cv2.rectangle(frame_bgr, (x - 5, y - text_height - 5), (x + text_width + 5, y + baseline + 5), (0, 0, 0), cv2.FILLED)

            # Put the white text on top.
            cv2.putText(frame_bgr, overlay_text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            writer.write(frame_bgr)

        writer.release()
        agent.update_epsilon()

        # Save the trained model at the end of each episode.
        save_model(agent, episode)

    env.close()

if __name__ == "__main__":
    main()
