import argparse
import csv
import datetime
import glob
import json
import os
from collections import deque

import cv2
import numpy as np
import retro
import tensorflow as tf  # noqa: F401  (kept for the debug switches below)

from games import load_adapter
from models.DoubleDQN import DoubleDQNAgent
from models.DQN import DQNAgent
from models.RainbowDQN import RainbowDQNAgent

# Debug mode disabled for performance - uncomment only when debugging specific issues
# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()

# Network input: 4 stacked 84x84 grayscale frames. Generic across retro games —
# every observation is resized to 84x84 in preprocess_frame regardless of game.
INPUT_SHAPE = (84, 84, 4)

# Repeat each chosen action for this many emulated frames (standard Atari
# frame skip). Rewards from every frame are accumulated into the stored
# transition, so no reward signal is lost. The game adapter supplies the
# action set and the back-half "released" variant for edge-triggered buttons.
FRAME_SKIP = 4

def preprocess_frame(obs):
    """
    Preprocess a single observation: resize to 84x84 and convert to grayscale.
    Returns shape (84, 84, 1).
    """
    if isinstance(obs, tuple):
        obs = obs[0]
    obs = cv2.resize(obs, (84, 84))
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    return np.reshape(obs, [84, 84, 1])

def get_stacked_state(frame_stack):
    """
    Concatenate 4 frames along the channel axis.
    Returns shape (1, 84, 84, 4) for batch inference.
    """
    return np.reshape(np.concatenate(list(frame_stack), axis=2), [1, 84, 84, 4])

def load_config(config_file):
    """
    Load configuration parameters from a JSON file.
    """
    if os.path.exists(config_file):
        with open(config_file) as f:
            return json.load(f)
    else:
        print(f"Config file {config_file} not found. Using default parameters.")
        return {}

def parse_arguments():
    """
    Parse command-line arguments and return the arguments object.
    """
    # First pass: parse --config to get the config file
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str, default='modelargs.json', help="Path to config file")
    args, remaining = parser.parse_known_args()

    config_defaults = load_config(args.config)

    # Second pass: parse all arguments with config defaults
    parser = argparse.ArgumentParser(
        description="Train a DQN agent on a retro game environment"
    )
    parser.add_argument('--config', type=str, default='modelargs.json', help="Path to config file")
    # Defaults to None for the same reason as --epsilon: a state named in the
    # config file belongs to whichever game that config was written for, so it
    # can be overridden when --game changes. An explicit --state cannot.
    parser.add_argument('--state', type=str, default=None,
                        help="Name of the state to start in (defaults to the game's default state)")
    parser.add_argument('--model', type=str, default=config_defaults.get('model', 'DQN'),
                        help='Model to use: DQN, DoubleDQN, RainbowDQN')
    parser.add_argument('--game', type=str, default=config_defaults.get('game', 'Zelda'),
                        help='Name of the game environment')
    parser.add_argument('--num_episodes', type=int, default=config_defaults.get('num_episodes', 20),
                        help='Number of episodes to run')
    parser.add_argument('--learning_rate', type=float, default=config_defaults.get('learning_rate', 0.001),
                        help='Learning rate for the agent')
    parser.add_argument('--discount_factor', type=float, default=config_defaults.get('discount_factor', 0.99),
                        help='Discount factor for training')
    # Defaults to None so we can tell "user typed --epsilon" apart from "value
    # came from the config file"; --load_model uses that to decide whether to
    # override the starting epsilon. Resolved below.
    parser.add_argument('--epsilon', type=float, default=None,
                        help='Initial exploration rate')
    parser.add_argument('--epsilon_decay', type=float, default=config_defaults.get('epsilon_decay', 0.995),
                        help='Epsilon decay rate')
    parser.add_argument('--epsilon_min', type=float, default=config_defaults.get('epsilon_min', 0.01),
                        help='Minimum epsilon value')
    parser.add_argument('--max_frames', type=int, default=config_defaults.get('max_frames', 2000),
                        help='Maximum number of frames per episode')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to a specific model file to load, or "latest" to load most recent')
    parser.add_argument('--record_freq', type=int, default=25,
                        help='Record video every N episodes (default: 25)')
    args = parser.parse_args()

    # Resolve the sentinels, remembering which were set on the CLI.
    args.epsilon_from_cli = args.epsilon is not None
    if args.epsilon is None:
        args.epsilon = config_defaults.get('epsilon', 1.0)

    args.state_from_cli = args.state is not None
    if args.state is None:
        args.state = config_defaults.get('state')
    return args

def get_video_writer(episode, frame_size, fps=30):
    """
    Create a VideoWriter to record footage of an episode.
    """
    recordings_dir = "recordings"
    if not os.path.exists(recordings_dir):
        os.makedirs(recordings_dir)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(recordings_dir, timestamp)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    video_path = os.path.join(run_dir, f"episode_{episode}.avi")

    # opencv-python's bundled stubs omit VideoWriter_fourcc; it exists at runtime.
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # type: ignore[attr-defined]
    return cv2.VideoWriter(video_path, fourcc, fps, frame_size)

def register_integrations():
    """
    Make games/ visible to retro, alongside its own bundled integrations.
    Safe to call more than once.
    """
    games_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "games")
    print("Games path: ", games_path)
    retro.data.Integrations.add_custom_path(games_path)
    return games_path

def resolve_state(game, state, default_state, from_cli):
    """
    Pick a start state that actually exists for `game`.

    modelargs.json carries a `state` belonging to whichever game it was last
    used with, so changing --game without editing the config would otherwise
    ask retro for (say) Zelda's "monsters" state while loading Mario. retro
    reports that as a TypeError from gzip.open(None), which says nothing useful,
    so resolve it here instead.

    An explicit --state that does not exist is an error. One inherited from the
    config file just falls back to the game's own default.
    """
    available = retro.data.list_states(game, inttype=retro.data.Integrations.ALL)
    if state in available:
        return state

    if from_cli:
        raise SystemExit(
            f"State {state!r} does not exist for {game}.\n"
            f"Available states: {', '.join(sorted(available)) or '(none found)'}"
        )

    print(f"Config state {state!r} is not a {game} state — using {default_state!r} instead.")
    if default_state not in available:
        raise SystemExit(
            f"Default state {default_state!r} does not exist for {game} either.\n"
            f"Available states: {', '.join(sorted(available)) or '(none found)'}"
        )
    return default_state

def integrate(game, state=retro.State.DEFAULT):
    """
    Build the retro environment for `game`. Call register_integrations() first.
    """
    available = retro.data.list_games(inttype=retro.data.Integrations.ALL)
    print(f"{game} in integrations:", game in available)
    return retro.make(game, state=state, inttype=retro.data.Integrations.ALL)

def save_model(agent, episode, model_dir="saved_models"):
    """
    Save the model to a unique file within model_dir.
    The filename includes the agent's class name, episode number, and a timestamp.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{agent.__class__.__name__}_episode{episode}_{timestamp}.keras"
    model_path = os.path.join(model_dir, filename)
    agent.model.save(model_path)
    print("Model saved at:", model_path)
    return model_path

# Game-agnostic log columns. The chosen game adapter contributes extra columns
# (adapter.log_fields) inserted before 'timestamp'.
BASE_LOG_COLUMNS = ['episode', 'episode_reward', 'moving_avg', 'avg_loss', 'epsilon', 'frames',
                    'training_steps', 'replay_buffer_size']

def log_episode_stats(columns, values, log_file="training_log.csv"):
    """
    Log episode statistics to a CSV file for later analysis.
    Creates the file with headers if it doesn't exist. If an existing file has
    a different header (e.g. a different game's columns), it is rotated to a
    backup so rows never get misaligned.

    Args:
        columns: ordered list of column names (the CSV header)
        values:  dict mapping every column name to its value for this episode
    """
    if os.path.exists(log_file):
        with open(log_file, newline='') as f:
            header = f.readline().strip().split(',')
        if header != columns:
            backup = log_file.replace('.csv', '_legacy.csv')
            os.replace(log_file, backup)
            print(f"Log columns changed — rotated old log to {backup}")

    file_exists = os.path.exists(log_file)
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(columns)
        writer.writerow([values[c] for c in columns])

def load_model_into_agent(agent, model_dir="saved_models", model_file=None):
    """
    Load weights from a saved model file into the agent's existing model.
    Supports .keras (preferred) and legacy .h5 files.
    If model_file is not provided, the most recent file is loaded.

    Loads weights only (not the full model graph) so the agent retains its
    current compile settings — loss function, optimizer, clipnorm, etc.
    """
    if model_file is None:
        files = (glob.glob(os.path.join(model_dir, "*.keras")) +
                 glob.glob(os.path.join(model_dir, "*.h5")))
        if not files:
            print("No model files found in", model_dir)
            return None
        model_file = max(files, key=os.path.getctime)
    agent.model.load_weights(model_file)
    print("Model loaded from:", model_file)
    return model_file

def main():
    args = parse_arguments()

    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # Load the game adapter (action set, reward shaping, termination, metrics).
    # It also resolves the start state, falling back to the game's default.
    adapter = load_adapter(args.game, args.state)
    # integration_name, not args.game: retro's bundled integrations are named
    # "<Game>-<Platform>", which cannot double as a Python package name.
    register_integrations()
    state_name = resolve_state(adapter.integration_name,
                               getattr(adapter, 'state', args.state),
                               adapter.default_state,
                               args.state_from_cli)
    # Keep the adapter in step; Zelda reads self.state for its dungeon check.
    adapter.state = state_name
    env = integrate(adapter.integration_name, state_name)
    action_size = len(adapter.actions)
    log_columns = BASE_LOG_COLUMNS + list(adapter.log_fields) + ['timestamp']
    total_rewards = 0

    # Initialize the agent. RainbowDQNAgent is a separate implementation rather
    # than a DQNAgent subclass, so the union spells out what main.py drives.
    agent: DQNAgent | RainbowDQNAgent
    if args.model == 'DQN':
        agent = DQNAgent(INPUT_SHAPE, action_size, args.learning_rate,
                         args.discount_factor, args.epsilon, args.epsilon_decay, args.epsilon_min)
    elif args.model == 'DoubleDQN':
        agent = DoubleDQNAgent(INPUT_SHAPE, action_size, args.learning_rate,
                               args.discount_factor, args.epsilon, args.epsilon_decay, args.epsilon_min)
    elif args.model == 'RainbowDQN':
        agent = RainbowDQNAgent(INPUT_SHAPE, action_size, args.learning_rate,
                                args.discount_factor, args.epsilon, args.epsilon_decay, args.epsilon_min)
    else:
        raise SystemExit(f"Unknown model {args.model!r}. Choose DQN, DoubleDQN, or RainbowDQN.")

    # Load a pre-trained model ONCE at startup if requested
    if args.load_model:
        if args.load_model == "latest":
            load_model_into_agent(agent)
        else:
            load_model_into_agent(agent, model_file=args.load_model)
        # Sync target network to the loaded weights so Bellman targets are correct immediately
        agent.update_target_model()
        # Resume with minimal exploration unless the caller explicitly passed --epsilon
        if not args.epsilon_from_cli:
            agent.epsilon = args.epsilon_min
        print(f"Loaded model. Resuming with epsilon: {agent.epsilon}")

    # Track best performance for saving
    best_reward = float('-inf')
    episode_rewards = []

    # Train the agent
    for episode in range(args.num_episodes):

        obs = env.reset()
        adapter.reset()
        done = False

        # Initialize frame stack with 4 copies of the first frame
        frame = preprocess_frame(obs)
        frame_stack = deque([frame] * 4, maxlen=4)
        state = get_stacked_state(frame_stack)

        # Initialize per-episode reward counter.
        episode_reward = 0

        # Create a video writer for this episode (only if it's a recording episode)
        writer = None
        if episode % args.record_freq == 0:
            screen = env.em.get_screen()
            height, width, channels = screen.shape
            writer = get_video_writer(episode, (width, height), fps=30)

        frame_count = 0  # Frame counter for this episode
        episode_loss = 0  # Track total loss for this episode
        training_steps = 0  # Count training steps in this episode

        while not done and frame_count < args.max_frames:
            action = agent.act(state)
            action_index = int(np.argmax(action))
            reward = 0

            # Repeat the chosen action for FRAME_SKIP frames, accumulating every
            # frame's reward into the single stored transition. Acting once per
            # skip window is ~4x fewer network inferences per emulated frame.
            # Edge-triggered buttons are released for the back half of the window
            # via the adapter's actions_released variant.
            actions_released = adapter.actions_released or adapter.actions
            for i in range(FRAME_SKIP):
                frame_count += 1
                buttons = (adapter.actions if i < FRAME_SKIP // 2 else actions_released)[action_index]
                obs, _, terminated, truncated, info = env.step(buttons)
                done = terminated or truncated

                # The adapter owns all game-specific reward shaping and
                # termination (movement, kills, room/death handling, etc.).
                frame_reward, adapter_done = adapter.step(info, frame_count)
                reward += frame_reward
                done = done or adapter_done

                # Capture the frame and overlay episode information (only if recording).
                if writer is not None:
                    screen = env.em.get_screen()
                    frame_bgr = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
                    overlay_text = f"Ep: {episode} | Frame: {frame_count} | Reward: {episode_reward + reward:.2f}"

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

                if done or frame_count >= args.max_frames:
                    break

            total_rewards += reward
            episode_reward += reward

            next_frame = preprocess_frame(obs)
            frame_stack.append(next_frame)
            next_state = get_stacked_state(frame_stack)

            # Every decision transition is stored and trained on — nothing is dropped
            loss = agent.train(state, action, reward, next_state, done)
            if loss is not None:
                episode_loss += loss
                training_steps += 1

            state = next_state

        if writer is not None:
            writer.release()
        if hasattr(agent, 'flush_episode'):
            agent.flush_episode()
        agent.update_epsilon()

        # Track episode performance
        episode_rewards.append(episode_reward)
        avg_loss = episode_loss / max(training_steps, 1)

        # Game-specific episode metrics (e.g. kills/cleared for Zelda)
        stats = adapter.episode_stats()

        # Calculate the reward moving average over the last 10 episodes
        window_size = min(10, len(episode_rewards))
        moving_avg = sum(episode_rewards[-window_size:]) / window_size

        # Print progress every episode
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{args.num_episodes} Complete")
        print(f"{'='*60}")
        print(f"Episode Reward: {episode_reward:.2f}")
        print(f"Moving Avg (last {window_size}): {moving_avg:.2f}")
        summary = adapter.summary_line()
        if summary:
            print(summary)
        print(f"Avg Loss: {avg_loss:.4f}")
        print(f"Epsilon: {agent.epsilon:.4f}")
        print(f"Frames: {frame_count}")
        print(f"Training Steps: {training_steps}")
        print(f"Replay Buffer Size: {len(agent.memory)}")
        print(f"{'='*60}\n")

        # Log stats to CSV (generic columns + the adapter's game-specific ones)
        values = {
            'episode': episode,
            'episode_reward': f"{episode_reward:.2f}",
            'moving_avg': f"{moving_avg:.2f}",
            'avg_loss': f"{avg_loss:.4f}",
            'epsilon': f"{agent.epsilon:.4f}",
            'frames': frame_count,
            'training_steps': training_steps,
            'replay_buffer_size': len(agent.memory),
            **stats,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        log_episode_stats(log_columns, values)

        # Save model if this is the best performance so far, or every 50 episodes
        if episode_reward > best_reward or episode % 50 == 0:
            if episode_reward > best_reward:
                best_reward = episode_reward
                print(f"New best reward: {best_reward:.2f} - Saving model!")
            save_model(agent, episode)

if __name__ == "__main__":
    main()
