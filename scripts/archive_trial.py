"""Compare ordinary starts with automatically archived starts using Rainbow DQN.

Run from the repository root: python -m scripts.archive_trial --help
Both arms use the same initial weights, rewards, exploration rate, and training
frame budget. Evaluation always starts at the original state and never learns
or restores archive entries. This is a short data-collection experiment, not a
complete Go-Explore implementation or evidence of full-game performance.
"""

import argparse
import csv
import datetime
import gc
import hashlib
import json
import shutil
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import retro
import tensorflow as tf

from games.Zelda.adapter import NORMAL_MODE, ZeldaAdapter
from main import (
    RECENT_ACTIONS,
    STACK_FRAMES,
    build_state_vector,
    clip_reward,
    get_stacked_state,
    input_shape,
    integrate,
    preprocess_frame,
    register_integrations,
    vector_width,
)
from models.observation import Observation
from models.RainbowDQN import RainbowDQNAgent
from scripts.exploration_archive import DoorHistory, Entry, ExplorationArchive, update_doors

ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Common starting weights; omit to train both arms from scratch")
    parser.add_argument("--state", default="level1")
    parser.add_argument("--frames-per-arm", type=int, default=64000)
    parser.add_argument("--max-frames", type=int, default=8000, help="Maximum trajectory length from the original start")
    parser.add_argument("--archive-rollout-frames", type=int, default=2000)
    parser.add_argument("--archive-probability", type=float, default=0.5)
    parser.add_argument("--archive-capacity", type=int, default=256)
    parser.add_argument("--frame-skip", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=10)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=0.5, help="Fixed during training, avoiding rollout-length effects on epsilon decay")
    parser.add_argument("--eval-epsilon", type=float, default=0.05)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, help="New output directory; existing directories are refused")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-state-vector", action="store_true",
                        help="Planes only: drop inventory, room coordinates and recent actions, in both arms")
    args = parser.parse_args()
    for name in ("frames_per_arm", "max_frames", "archive_rollout_frames", "archive_capacity",
                 "frame_skip", "n_steps", "train_every", "eval_episodes"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("archive_probability", "epsilon", "eval_epsilon"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.checkpoint:
        args.checkpoint = args.checkpoint.resolve()
        if not args.checkpoint.is_file():
            parser.error(f"checkpoint does not exist: {args.checkpoint}")
    return args


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str))


def append_csv(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})


def new_adapter(args: argparse.Namespace) -> ZeldaAdapter:
    """A fresh adapter honouring --no-state-vector, for training and evaluation alike."""
    adapter = ZeldaAdapter(args.state)
    if args.no_state_vector:
        adapter.vector_size = 0
    return adapter


def observe(adapter: ZeldaAdapter, raw: np.ndarray, frames: deque,
            actions: list[tuple[int, int]]) -> Observation:
    """Network input, with the action history read off the trajectory's own path.

    main.py keeps a separate deque of recent actions. Here the path already
    records every action back to the original start and is restored with each
    archive entry, so a restored snapshot carries the history that actually led
    to it without saving anything extra. Most recent first, like main.py.
    """
    recent: deque[int | None] = deque(
        (index for index, _ in reversed(actions[-RECENT_ACTIONS:])), maxlen=RECENT_ACTIONS)
    while len(recent) < RECENT_ACTIONS:
        recent.append(None)
    return get_stacked_state(frames, adapter.extra_observation(raw),
                             build_state_vector(adapter, recent, len(adapter.actions)))


def rollout(env: Any, adapter: ZeldaAdapter, agent: RainbowDQNAgent,
            args: argparse.Namespace, budget: int, decision_offset: int = 0,
            archive: ExplorationArchive | None = None, entry: Entry | None = None,
            train: bool = True, video: Path | None = None) -> dict[str, Any]:
    if train and agent.n_step_buffer:
        raise RuntimeError("n-step buffer must be empty before a reset/restore")
    if entry is None:
        raw, _ = env.reset()
        adapter.reset()
        info = env.data.lookup_all()
        adapter.step(info.copy(), 0)
        frame = preprocess_frame(raw)
        frames = deque([frame] * STACK_FRAMES, maxlen=STACK_FRAMES)
        actions: list[tuple[int, int]] = []
        doors: DoorHistory = frozenset()
        elapsed = 0
    else:
        raw, frames = entry.restore(env, adapter)
        info = env.data.lookup_all()
        actions = list(entry.actions)
        doors = entry.doors
        elapsed = entry.elapsed_frames
    state = observe(adapter, raw, frames, actions)
    if archive is not None and elapsed < args.max_frames:
        archive.capture(env, adapter, raw, frames, info, tuple(actions), doors, elapsed)
    remaining = min(budget, args.max_frames - elapsed)
    if remaining <= 0:
        raise RuntimeError("archive entry has no remaining trajectory budget")

    row: dict[str, Any] = {
        "source": "archive" if entry else "entrance", "prefix_frames": elapsed,
        "start_room": int(info["Room"]), "start_keys": int(info["Keys"]),
        "frames": 0, "decisions": 0, "gradient_steps": 0, "reward": 0.0,
        "kills": 0, "keys_gained": 0, "keys_used": 0, "returned_with_key": 0,
        "entered_rooms": [], "rooms_after_key_use": [], "end_reason": "budget",
    }
    old_normal = info.copy()
    writer = None
    if video:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(str(video), fourcc, 60,
                                 (raw.shape[1], raw.shape[0]))
        if not writer.isOpened():
            raise RuntimeError(f"could not open video writer: {video}")
    done = False
    try:
        while row["frames"] < remaining and not done:
            action = int(np.argmax(agent.act(state)))
            reward = 0.0
            window = min(args.frame_skip, remaining - row["frames"])
            actual = 0
            for i in range(window):
                buttons = (adapter.actions if i < args.frame_skip // 2 else adapter.actions_released)[action]
                raw, _, terminated, truncated, info = env.step(buttons)
                actual += 1
                row["frames"] += 1
                frame_reward, adapter_done = adapter.step(info.copy(), elapsed + row["frames"])
                reward += frame_reward
                done = terminated or truncated or adapter_done
                if info["Game Mode"] == NORMAL_MODE:
                    doors = update_doors(doors, old_normal, info)
                    row["kills"] += (int(info["Enemies Killed"]) - int(old_normal["Enemies Killed"])) % 256
                    row["keys_gained"] += max(int(info["Keys"]) - int(old_normal["Keys"]), 0)
                    row["keys_used"] += max(int(old_normal["Keys"]) - int(info["Keys"]), 0)
                    room = [int(info["Level"]), int(info["Room"])]
                    if room != [int(old_normal["Level"]), int(old_normal["Room"])]:
                        row["entered_rooms"].append(room)
                        if row["keys_used"]:
                            row["rooms_after_key_use"].append(room)
                        if info["Room"] == adapter.start_room and info["Keys"] > 0:
                            row["returned_with_key"] += 1
                    old_normal = info.copy()
                if writer:
                    writer.write(cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
                if done:
                    row["end_reason"] = ("abandoned" if adapter.abandoned else
                                         "death" if adapter.died or info["Game Mode"] in (8, 17) else "environment")
                    break
            row["reward"] += reward
            row["decisions"] += 1
            actions.append((action, actual))
            frames.append(preprocess_frame(raw))
            # After the append, so the history describes the action that led here.
            next_state = observe(adapter, raw, frames, actions)
            if train:
                loss = agent.train(state, action, clip_reward(reward, 1.0), next_state, done,
                                   learn=(decision_offset + row["decisions"]) % args.train_every == 0)
                row["gradient_steps"] += int(loss is not None)
            state = next_state
            if archive is not None and not done and elapsed + row["frames"] < args.max_frames:
                archive.capture(env, adapter, raw, frames, info, tuple(actions), doors,
                                elapsed + row["frames"])
    finally:
        if train:
            # Artificial rollout cuts bootstrap from the actual last state.
            # Drain partial windows before the next reset/restore.
            agent.flush_episode()
        if writer:
            writer.release()
    row["reward"] = round(row["reward"], 4)
    return row


def evaluate(env: Any, agent: RainbowDQNAgent, args: argparse.Namespace,
             directory: Path, phase: str) -> list[dict[str, Any]]:
    old_epsilon, random_state = agent.epsilon, np.random.get_state()
    agent.epsilon = args.eval_epsilon
    rows = []
    try:
        for episode in range(args.eval_episodes):
            np.random.seed(args.seed + 10000 + episode)
            video = directory / f"{phase}.avi" if episode == 0 and not args.no_video else None
            row = rollout(env, new_adapter(args), agent, args, args.max_frames,
                          train=False, video=video)
            row = {"episode": episode, **row}
            rows.append(row)
            append_csv(directory / f"{phase}.csv", row)
            print(f"  {phase} {episode}: keys used={row['keys_used']}, rooms={row['entered_rooms']}", flush=True)
    finally:
        agent.epsilon = old_epsilon
        np.random.set_state(random_state)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rollouts": len(rows), "frames": sum(r["frames"] for r in rows),
        "gradient_steps": sum(r["gradient_steps"] for r in rows),
        "archive_starts": sum(r["source"] == "archive" for r in rows),
        "key_pickup_rollouts": sum(r["keys_gained"] > 0 for r in rows),
        "key_use_rollouts": sum(r["keys_used"] > 0 for r in rows),
        "return_with_key_rollouts": sum(r["returned_with_key"] > 0 for r in rows),
        "entered_rooms": sorted({tuple(room) for r in rows for room in r["entered_rooms"]}),
    }


def run_arm(name: str, args: argparse.Namespace, directory: Path) -> dict[str, Any]:
    directory.mkdir()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed)
    adapter = new_adapter(args)
    vector_size = vector_width(adapter, len(adapter.actions))
    agent = RainbowDQNAgent(input_shape(adapter), len(adapter.actions), 0.00025,
                            0.995, args.epsilon, 1.0, args.epsilon, q_limit=200.0,
                            n_step=args.n_steps, vector_size=vector_size)
    if args.checkpoint:
        try:
            agent.model.load_weights(str(args.checkpoint))
        except ValueError as exc:
            # Keras reports a shape mismatch as "1 objects could not be loaded"
            # on the first Conv2D, which says nothing about the cause.
            raise SystemExit(
                f"{args.checkpoint} does not fit this network (planes "
                f"{input_shape(adapter)}, state vector {vector_size}). Checkpoints from "
                "before the memory planes and state vector (84x84x7 input, no vector) "
                "cannot load; omit --checkpoint or use one trained since."
            ) from exc
        agent.update_target_model()
    digest = hashlib.sha256()
    for weight in agent.model.get_weights():
        digest.update(weight.tobytes())
    archive = ExplorationArchive(args.archive_capacity, args.seed + 20000)
    start_rng = np.random.default_rng(args.seed + 30000)
    env = integrate("Zelda", args.state)
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    initial: list[dict[str, Any]] = []
    try:
        if name == "baseline":
            initial = evaluate(env, agent, args, directory, "initial_evaluation")
        total_frames = decisions = 0
        while total_frames < args.frames_per_arm:
            entry = (archive.choose() if name == "archive" and len(archive)
                     and start_rng.random() < args.archive_probability else None)
            budget = args.frames_per_arm - total_frames
            if entry:
                budget = min(budget, args.archive_rollout_frames)
            row = rollout(env, adapter, agent, args, budget, decisions, archive, entry)
            total_frames += row["frames"]
            decisions += row["decisions"]
            row = {"rollout": len(rows), **row, "total_frames": total_frames,
                   "archive_size": len(archive), "unique_cells": len(archive.seen)}
            rows.append(row)
            append_csv(directory / "training.csv", row)
            print(f"{name} {len(rows)}: {total_frames}/{args.frames_per_arm} frames, "
                  f"{row['source']}, keys gained/used={row['keys_gained']}/{row['keys_used']}, "
                  f"rooms={row['entered_rooms']}, archive={len(archive)}", flush=True)
            write_json(directory / "progress.json", summarize(rows))
        agent.save(str(directory / "final.keras"))
        final = evaluate(env, agent, args, directory, "final_evaluation")
        archive.save_manifest(directory / "archive_manifest.json")
        summary = {"initial_weights_sha256": digest.hexdigest(),
                   "training": summarize(rows), "evaluation": summarize(final),
                   "initial_evaluation": summarize(initial) if initial else None,
                   "unique_cells": len(archive.seen), "seconds": round(time.monotonic() - started, 1)}
        write_json(directory / "summary.json", summary)
        return summary
    finally:
        env.close()


def main() -> None:
    args = arguments()
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    register_integrations()
    if args.state not in retro.data.list_states("Zelda", inttype=retro.data.Integrations.ALL):
        raise SystemExit(f"unknown Zelda state: {args.state}")
    output = args.output or ROOT / "runs" / "experiments" / (
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "__archive_trial")
    output.mkdir(parents=True, exist_ok=False)
    sources = ["scripts/archive_trial.py", "scripts/exploration_archive.py", "main.py",
               "games/Zelda/adapter.py", "games/base.py", "models/RainbowDQN.py",
               "models/observation.py"]
    hashes = {}
    for source in sources:
        target = output / "source" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, target)
        hashes[source] = hashlib.sha256(target.read_bytes()).hexdigest()
    write_json(output / "config.json", {"args": vars(args), "tensorflow": tf.__version__,
               "source_sha256": hashes,
               "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
               "learning_rate": 0.00025, "gamma": 0.995, "reward_clip": 1.0,
               "rnd_beta": 0.0, "epsilon_decay": 1.0, "recent_actions": RECENT_ACTIONS})
    print(f"Output: {output}", flush=True)
    results = {}
    for name in ("baseline", "archive"):
        results[name] = run_arm(name, args, output / name)
        gc.collect()
        write_json(output / "comparison.json", results)
    if results["baseline"]["initial_weights_sha256"] != results["archive"]["initial_weights_sha256"]:
        raise RuntimeError("comparison did not start from identical weights")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
