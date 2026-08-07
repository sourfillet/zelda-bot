import argparse
import csv
import datetime
import glob
import io
import json
import os
import re
import sys
from collections import deque
from typing import Any

import cv2
import numpy as np
import retro
import tensorflow as tf  # noqa: F401  (kept for the debug switches below)

from games import load_adapter
from games.base import GameAdapter
from models.DoubleDQN import DoubleDQNAgent
from models.DQN import DQNAgent
from models.RainbowDQN import RainbowDQNAgent

# Debug mode disabled for performance - uncomment only when debugging specific issues
# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()

# Network input: STACK_FRAMES stacked 84x84 grayscale frames, plus however many
# extra planes the game adapter contributes (adapter.extra_planes). Generic
# across retro games — every observation is resized to 84x84 in
# preprocess_frame regardless of game, and adapters that add nothing keep the
# original (84, 84, 4).
STACK_FRAMES = 4

# Default square edge every observation is resized to. 84 is the original
# Atari DQN figure, chosen to keep the network small; --input_size overrides it.
# Bigger preserves more detail at a steep parameter cost, almost all of it in
# the flatten -> Dense(512): measured 1.95M params at 84, 5.06M at 128 and
# 20.8M at the native 224x240. Step time barely moves (1.0x / 1.1x / 1.5x) —
# the cost is sample efficiency, since a larger network needs more episodes to
# fit, and replay memory (1.13 / 2.62 / 8.60 GB at capacity).
DEFAULT_INPUT_SIZE = 84


def input_shape(adapter: "GameAdapter", size: int = DEFAULT_INPUT_SIZE) -> tuple[int, int, int]:
    """Network input shape for this game."""
    return (size, size, STACK_FRAMES + adapter.extra_planes)

# Repeat each chosen action for this many emulated frames (standard Atari
# frame skip). Rewards from every frame are accumulated into the stored
# transition, so no reward signal is lost. The game adapter supplies the
# action set and the back-half "released" variant for edge-triggered buttons.
FRAME_SKIP = 4

def preprocess_frame(obs: np.ndarray | tuple,
                     size: int = DEFAULT_INPUT_SIZE) -> np.ndarray:
    """
    Preprocess a single observation: resize to size x size, convert to greyscale.
    Returns shape (size, size, 1).
    """
    # env.reset() returns (obs, info) while env.step() returns the array directly
    frame: np.ndarray = obs[0] if isinstance(obs, tuple) else obs
    frame = cv2.resize(frame, (size, size))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return np.reshape(frame, [size, size, 1])

def get_stacked_state(frame_stack: deque, extra: np.ndarray | None = None) -> np.ndarray:
    """
    Concatenate the frame stack, then any adapter planes, along the channels.
    Returns shape (1, size, size, STACK_FRAMES + extra_planes) for inference.
    """
    planes = list(frame_stack)
    if extra is not None:
        planes.append(extra)
    stacked = np.concatenate(planes, axis=2)
    return np.reshape(stacked, [1, *stacked.shape])


def save_debug_frame(raw: np.ndarray, state: np.ndarray, run_dir: str,
                     episode: int, frame: int) -> None:
    """
    Write one image showing exactly what the network was fed.

    Left: the raw emulator frame. Right: every input plane in order, upscaled
    and labelled — the stacked history first, then whatever the adapter added.
    Answers "is the minimap plane actually carrying the marker" without having
    to reason about it.
    """
    out_dir = os.path.join(run_dir, "debug")
    os.makedirs(out_dir, exist_ok=True)

    planes = [state[0, :, :, i] for i in range(state.shape[3])]
    edge = state.shape[1]
    # keep tiles a readable size whatever the input resolution
    scale, pad = max(1, 168 // edge), 6
    tiles = []
    for i, p in enumerate(planes):
        tile = cv2.cvtColor(cv2.resize(p, (edge * scale, edge * scale),
                                       interpolation=cv2.INTER_NEAREST),
                            cv2.COLOR_GRAY2BGR)
        label = f"frame t-{STACK_FRAMES - 1 - i}" if i < STACK_FRAMES else f"extra {i - STACK_FRAMES}"
        cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (70, 70, 70), 1)
        # Label on its own strip rather than over the image, so it never
        # obscures the very thing being inspected.
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 18), (25, 25, 25), -1)
        cv2.putText(tile, label, (5, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1,
                    cv2.LINE_AA)
        tiles.append(tile)

    per_row = min(len(tiles), 3)
    rows = []
    for r in range(0, len(tiles), per_row):
        row = tiles[r:r + per_row]
        while len(row) < per_row:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack([np.pad(t, ((pad, pad), (pad, pad), (0, 0))) for t in row]))
    grid = np.vstack(rows)

    raw_bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
    h = grid.shape[0]
    raw_scaled = cv2.resize(raw_bgr, (int(raw_bgr.shape[1] * h / raw_bgr.shape[0]), h))
    canvas = np.hstack([np.pad(raw_scaled, ((0, 0), (pad, pad), (0, 0))), grid])
    cv2.imwrite(os.path.join(out_dir, f"ep{episode:04d}_f{frame:05d}.png"), canvas)

def clip_reward(reward: float, limit: float | None) -> float:
    """
    Clamp a per-decision reward to +/- limit; a non-positive limit disables it.

    With gamma=0.99 a limit of 1.0 caps any legitimate |Q| at about 100, so a
    larger value is unambiguously divergence rather than a plausible estimate.
    Only the training signal is clipped — logged episode returns stay raw.
    """
    if limit is None or limit <= 0:
        return reward
    return max(-limit, min(limit, reward))

def load_config(config_file: str) -> dict[str, Any]:
    """
    Load configuration parameters from a JSON file.
    """
    if os.path.exists(config_file):
        with open(config_file) as f:
            return json.load(f)
    else:
        print(f"Config file {config_file} not found. Using default parameters.")
        return {}

def parse_arguments() -> argparse.Namespace:
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
    parser.add_argument('--reward_clip', type=float,
                        default=config_defaults.get('reward_clip', 1.0),
                        help='Clamp per-decision training reward to +/- this; 0 disables')
    parser.add_argument('--train_every', type=int,
                        default=config_defaults.get('train_every', 1),
                        help='Take a gradient step every N decisions. Every transition is '
                             'still stored; only the update frequency changes. The Atari '
                             'DQN replay period is 4; 1 does 4x the literature\'s updates.')
    parser.add_argument('--batch_size', type=int,
                        default=config_defaults.get('batch_size', 32),
                        help='Replay minibatch size. Nearly free on this GPU — measured 22ms '
                             'at 256 versus 18ms at 32 — so larger trades almost no wall '
                             'clock for a much better gradient estimate.')
    parser.add_argument('--extra_planes', type=int,
                        default=config_defaults.get('extra_planes', -1),
                        help="Override the adapter's extra observation planes. -1 keeps "
                             "whatever the adapter defines, 0 disables them entirely. Use 0 "
                             "to A/B a game's HUD planes against plain frames.")
    parser.add_argument('--input_size', type=int,
                        default=config_defaults.get('input_size', DEFAULT_INPUT_SIZE),
                        help='Square edge every frame is resized to (default 84). Larger '
                             'keeps more detail but grows the network sharply; changing it '
                             'changes the input shape, so checkpoints do not carry over.')
    parser.add_argument('--debug_frames', type=int, default=0,
                        help='Save an image of every Nth decision showing all model input '
                             'planes, into the run\'s debug/ folder. 0 disables.')
    parser.add_argument('--render', action='store_true',
                        default=config_defaults.get('render', False),
                        help='Show a live game window. Costs ~19ms per decision (~11x the '
                             'emulator itself), so it is off unless you want to watch.')
    # Point smoke tests at a scratch file so they cannot rotate or append to the
    # log of a training run that is already in flight.
    parser.add_argument('--log_file', type=str, default='training_log.csv',
                        help='CSV to append per-episode stats to')
    args = parser.parse_args()

    # Resolve the sentinels, remembering which were set on the CLI.
    args.epsilon_from_cli = args.epsilon is not None
    if args.epsilon is None:
        args.epsilon = config_defaults.get('epsilon', 1.0)

    args.state_from_cli = args.state is not None
    if args.state is None:
        args.state = config_defaults.get('state')
    return args

def get_video_writer(episode: int, frame_size: tuple[int, int], run_dir: str,
                     fps: int = 30) -> Any:
    """
    Create a VideoWriter for one episode inside this run's recordings folder.

    The directory is the run's, not a fresh timestamped one per episode — doing
    the latter produced 233 directories holding a single file each.
    """
    video_path = os.path.join(run_dir, "recordings", f"episode{episode:04d}.avi")

    # opencv-python's bundled stubs omit VideoWriter_fourcc; it exists at runtime.
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # type: ignore[attr-defined]
    return cv2.VideoWriter(video_path, fourcc, fps, frame_size)

RUNS_ROOT = "runs"

def create_run_dir(game: str, model: str, state: str, root: str = RUNS_ROOT) -> str:
    """
    Make runs/<game>/<timestamp>__<model>__<state>/ with its subfolders.

    Everything one run produces lives together: its own training_log.csv, its
    checkpoints, its recordings and the config that produced them. Previously
    checkpoints from every game and every run shared one flat directory, so a
    file could not be traced back to the run — or even the game — that made it.
    """
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(root, game, f"{stamp}__{model}__{state}")
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "recordings"), exist_ok=True)
    return run_dir

def _git_commit() -> str | None:
    """Short commit hash, or None outside a git checkout."""
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5,
                              check=True).stdout.strip()
    except Exception:
        return None

def write_run_config(run_dir: str, args: argparse.Namespace, adapter: GameAdapter,
                     action_size: int, state: str) -> dict[str, Any]:
    """
    Snapshot everything needed to interpret or reproduce this run.

    Without this a training curve is uninterpretable after the fact — there is
    no record of which learning rate, epsilon schedule or reward table produced
    it, which is what made comparing runs guesswork.
    """
    config = {
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "game": args.game,
        "retro_name": adapter.integration_name,
        "state": state,
        "model": args.model,
        "action_size": action_size,
        "input_shape": list(input_shape(adapter, args.input_size)),
        "extra_planes": adapter.extra_planes,
        "frame_skip": FRAME_SKIP,
        "args": dict(vars(args)),
    }
    rewards = getattr(sys.modules[adapter.__module__], "REWARD_VALUES", None)
    if isinstance(rewards, dict):
        config["reward_values"] = rewards
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    return config

def append_run_index(run_dir: str, config: dict[str, Any],
                     root: str = RUNS_ROOT) -> None:
    """Register the run in runs/index.csv so runs are discoverable in one place.

    Written with a single os.write to an O_APPEND descriptor. Buffered file
    writes are not atomic between processes, and two runs starting at the same
    moment interleaved mid-row here, leaving a fragment ("yer.Level1") in the
    index. A lone write of well under PIPE_BUF to an O_APPEND fd cannot split.
    """
    index = os.path.join(root, "index.csv")
    columns = ["started", "game", "model", "state", "num_episodes", "git_commit", "run_dir"]
    row = [config["started"], config["game"], config["model"], config["state"],
           config["args"].get("num_episodes"), config["git_commit"], run_dir]

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    if not os.path.exists(index):
        writer.writerow(columns)
    writer.writerow(row)

    os.makedirs(root, exist_ok=True)
    fd = os.open(index, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, buf.getvalue().encode())
    finally:
        os.close(fd)

def update_run_summary(run_dir: str, episode: int, episode_reward: float,
                       best_reward: float, max_abs_q: float,
                       stats: dict[str, Any]) -> None:
    """Rewrite this run's summary.json — cheap, and survives an interrupted run."""
    summary = {
        "updated": datetime.datetime.now().isoformat(timespec="seconds"),
        "episodes_completed": episode + 1,
        "last_episode_reward": round(float(episode_reward), 3),
        "best_episode_reward": round(float(best_reward), 3),
        "last_max_abs_q": float(max_abs_q),
        "last_stats": stats,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

def register_integrations() -> str:
    """
    Make games/ visible to retro, alongside its own bundled integrations.
    Safe to call more than once.
    """
    games_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "games")
    print("Games path: ", games_path)
    retro.data.Integrations.add_custom_path(games_path)
    return games_path

def resolve_state(game: str, state: str | None, default_state: str | None,
                  from_cli: bool) -> str:
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
        return str(state)

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
    return str(default_state)

def integrate(game: str, state: Any = retro.State.DEFAULT, render: bool = False) -> Any:
    """
    Build the retro environment for `game`. Call register_integrations() first.

    render_mode is passed explicitly. retro.make() forwards nothing, so the env
    otherwise inherits RetroEnv's own default of "human" — which silently opens
    a viewer and redraws on every emulated frame. Measured at 4.69 ms/frame
    against 0.43 ms with rendering off: about 19 ms per 4-frame decision, or
    roughly eleven times the cost of the emulation itself.
    """
    available = retro.data.list_games(inttype=retro.data.Integrations.ALL)
    print(f"{game} in integrations:", game in available)
    return retro.make(game, state=state, inttype=retro.data.Integrations.ALL,
                      render_mode="human" if render else None)

def save_model(agent: DQNAgent | RainbowDQNAgent, episode: int, run_dir: str,
               is_best: bool = False) -> str:
    """
    Save a checkpoint into this run's checkpoints/ folder.

    Episode number alone names the file — the run directory already carries the
    game, model, state and timestamp, so none of that needs encoding here. A new
    best also refreshes best.keras, so recovering the best network never means
    reading the log to work out which episode it was.
    """
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    model_path = os.path.join(ckpt_dir, f"episode{episode:04d}.keras")
    agent.model.save(model_path)
    if is_best:
        agent.model.save(os.path.join(ckpt_dir, "best.keras"))
    print("Model saved at:", model_path)
    return model_path

# Game-agnostic log columns. The chosen game adapter contributes extra columns
# (adapter.log_fields) inserted before 'timestamp'.
BASE_LOG_COLUMNS = ['episode', 'episode_reward', 'moving_avg', 'avg_loss', 'max_q', 'epsilon',
                    'frames', 'training_steps', 'replay_buffer_size']

def log_episode_stats(columns: list[str], values: dict[str, Any],
                      log_file: str = "training_log.csv") -> None:
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
            # Timestamped so a second column change cannot overwrite the first
            # rotation and silently destroy an earlier run's history.
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = log_file.replace('.csv', f'_legacy_{stamp}.csv')
            os.replace(log_file, backup)
            print(f"Log columns changed — rotated old log to {backup}")

    file_exists = os.path.exists(log_file)
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(columns)
        writer.writerow([values[c] for c in columns])

def find_latest_checkpoint(game: str, root: str = RUNS_ROOT) -> str | None:
    """
    Newest checkpoint belonging to `game`, across that game's previous runs.

    Scoped to the game deliberately. Checkpoints used to share one flat folder
    with the game absent from the filename, so "latest" could hand a Mario
    network (8 actions) to a Zelda run (10) purely because it was written more
    recently.

    Picks the newest run directory (their names begin with a sortable timestamp)
    then the highest episode number inside it. Ordering by filesystem ctime
    instead is wrong the moment a run directory is copied or moved, since that
    rewrites ctime on every file at once.
    """
    def episode_of(path: str) -> int:
        m = re.search(r"episode(\d+)", os.path.basename(path))
        return int(m.group(1)) if m else -1

    for run_dir in sorted(glob.glob(os.path.join(root, game, "*")), reverse=True):
        files = glob.glob(os.path.join(run_dir, "checkpoints", "episode*.keras"))
        if files:
            return max(files, key=episode_of)
    return None

def load_model_into_agent(agent: DQNAgent | RainbowDQNAgent, model_file: str) -> str:
    """
    Load weights from a saved model file into the agent's existing model.
    Supports .keras (preferred) and legacy .h5 files.

    Loads weights only (not the full model graph) so the agent retains its
    current compile settings — loss function, optimizer, clipnorm, etc.
    """
    agent.model.load_weights(model_file)
    print("Model loaded from:", model_file)
    return model_file

def main() -> None:
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

    # An adapter builds a fixed number of planes, so the only safe overrides are
    # "off" or "exactly what it defines" — anything else would disagree with the
    # array extra_observation() actually returns.
    if args.extra_planes >= 0:
        natural = type(adapter).extra_planes
        if args.extra_planes not in (0, natural):
            raise SystemExit(
                f"--extra_planes {args.extra_planes} is not valid for {args.game}: it "
                f"defines {natural}. Use 0 to disable, {natural} to keep them, or -1 "
                "for the adapter's default."
            )
        adapter.extra_planes = args.extra_planes
    env = integrate(adapter.integration_name, state_name, render=args.render)
    action_size = len(adapter.actions)
    log_columns = BASE_LOG_COLUMNS + list(adapter.log_fields) + ['timestamp']
    total_rewards = 0.0

    # Everything this run produces goes in one directory.
    run_dir = create_run_dir(args.game, args.model, state_name)
    run_config = write_run_config(run_dir, args, adapter, action_size, state_name)
    append_run_index(run_dir, run_config)
    # --log_file only overrides when explicitly given; otherwise the log belongs
    # to the run, which is what makes it analysable without splitting on episode
    # counter resets.
    log_path = (args.log_file if args.log_file != 'training_log.csv'
                else os.path.join(run_dir, 'training_log.csv'))
    print(f"Run directory: {run_dir}")

    # Initialize the agent. RainbowDQNAgent is a separate implementation rather
    # than a DQNAgent subclass, so the union spells out what main.py drives.
    agent: DQNAgent | RainbowDQNAgent
    shape = input_shape(adapter, args.input_size)
    if args.input_size != DEFAULT_INPUT_SIZE or adapter.extra_planes:
        print(f"Input shape: {shape}  ({STACK_FRAMES} stacked frames "
              f"+ {adapter.extra_planes} adapter plane(s))")
    if args.model == 'DQN':
        agent = DQNAgent(shape, action_size, args.learning_rate,
                         args.discount_factor, args.epsilon, args.epsilon_decay, args.epsilon_min)
    elif args.model == 'DoubleDQN':
        agent = DoubleDQNAgent(shape, action_size, args.learning_rate,
                               args.discount_factor, args.epsilon, args.epsilon_decay, args.epsilon_min)
    elif args.model == 'RainbowDQN':
        agent = RainbowDQNAgent(shape, action_size, args.learning_rate,
                                args.discount_factor, args.epsilon, args.epsilon_decay, args.epsilon_min)
    else:
        raise SystemExit(f"Unknown model {args.model!r}. Choose DQN, DoubleDQN, or RainbowDQN.")

    # batch_size is a plain attribute on every agent, so this needs no
    # constructor plumbing.
    agent.batch_size = args.batch_size
    if args.batch_size != 32 or args.train_every != 1:
        print(f"Replay: batch {args.batch_size}, gradient step every "
              f"{args.train_every} decision(s)")

    # Load a pre-trained model ONCE at startup if requested
    if args.load_model:
        if args.load_model == "latest":
            checkpoint = find_latest_checkpoint(args.game)
            if checkpoint is None:
                raise SystemExit(
                    f"No previous checkpoint found for {args.game} under {RUNS_ROOT}/{args.game}/. "
                    "Pass an explicit path, or start without --load_model."
                )
        else:
            checkpoint = args.load_model
        load_model_into_agent(agent, checkpoint)
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
        raw = obs[0] if isinstance(obs, tuple) else obs
        frame = preprocess_frame(obs, args.input_size)
        frame_stack = deque([frame] * STACK_FRAMES, maxlen=STACK_FRAMES)
        state = get_stacked_state(frame_stack,
                                  adapter.extra_observation(raw, args.input_size))

        # Initialize per-episode reward counter.
        episode_reward = 0.0

        # Create a video writer for this episode (only if it's a recording episode)
        writer = None
        if episode % args.record_freq == 0:
            screen = env.em.get_screen()
            height, width, channels = screen.shape
            writer = get_video_writer(episode, (width, height), run_dir, fps=30)

        decisions = 0    # Decision counter, for --train_every
        frame_count = 0  # Frame counter for this episode
        episode_loss = 0.0  # Track total loss for this episode
        training_steps = 0  # Count training steps in this episode

        while not done and frame_count < args.max_frames:
            action = agent.act(state)
            action_index = int(np.argmax(action))
            reward = 0.0

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
            # Logged reward stays unclipped so episode returns remain comparable
            # across runs and interpretable against the adapter's reward table.
            episode_reward += reward

            raw = obs[0] if isinstance(obs, tuple) else obs
            next_frame = preprocess_frame(obs, args.input_size)
            frame_stack.append(next_frame)
            next_state = get_stacked_state(frame_stack,
                                           adapter.extra_observation(raw, args.input_size))

            if args.debug_frames and frame_count % args.debug_frames < FRAME_SKIP:
                save_debug_frame(raw, next_state, run_dir, episode, frame_count)

            # The agent trains on the *clipped* reward. Huber loss, clipnorm and
            # the PER priority ceiling all bound how fast Q can move; none of
            # them bound where it can move to. Clipping the reward bounds the
            # Bellman target by construction (|Q| <= clip / (1 - gamma)), which
            # is the ingredient standard DQN uses and this loop was missing.
            # Every decision transition is stored; --train_every controls how
            # often one of them triggers a gradient step.
            decisions += 1
            loss = agent.train(state, action, clip_reward(reward, args.reward_clip),
                               next_state, done,
                               learn=(decisions % args.train_every == 0))
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

        # Peak |Q| this episode, then reset for the next one. With clipped
        # rewards this should settle near reward_clip / (1 - discount_factor);
        # an order-of-magnitude jump is divergence starting.
        max_abs_q = getattr(agent, 'max_abs_q', 0.0)
        agent.max_abs_q = 0.0

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
        print(f"Max |Q|: {max_abs_q:.4g}")
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
            'max_q': f"{max_abs_q:.4g}",
            'epsilon': f"{agent.epsilon:.4f}",
            'frames': frame_count,
            'training_steps': training_steps,
            'replay_buffer_size': len(agent.memory),
            **stats,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        log_episode_stats(log_columns, values, log_file=log_path)

        # Save model if this is the best performance so far, or every 50 episodes
        is_best = episode_reward > best_reward
        if is_best or episode % 50 == 0:
            if is_best:
                best_reward = episode_reward
                print(f"New best reward: {best_reward:.2f} - Saving model!")
            save_model(agent, episode, run_dir, is_best=is_best)

        update_run_summary(run_dir, episode, episode_reward, best_reward, max_abs_q, stats)

if __name__ == "__main__":
    main()
