import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from games.Zelda.adapter import REWARD_VALUES, ZeldaAdapter
from main import RECENT_ACTIONS, vector_width
from scripts.exploration_archive import ExplorationArchive, cell_for, update_doors
from tests.test_zelda_adapter import frame_info


def info_for(**overrides: Any) -> dict[str, Any]:
    return frame_info(**{"Enemies Killed Current Room": 0, **overrides})


class FakeEmulator:
    def __init__(self) -> None:
        self.state = b"snapshot"

    def get_state(self) -> bytes:
        return self.state

    def set_state(self, state: bytes) -> None:
        self.state = state


class FakeData:
    def reset(self) -> None:
        pass

    def update_ram(self) -> None:
        pass


class ArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = ZeldaAdapter("level1")
        self.info = info_for()
        self.adapter.step(self.info.copy(), 0)
        self.adapter.step(self.info.copy(), 1)
        self.env = SimpleNamespace(em=FakeEmulator(), data=FakeData())
        self.env.unwrapped = self.env
        self.raw = np.zeros((4, 4, 3), np.uint8)
        self.frames = deque([np.zeros((2, 2, 1), np.uint8)] * 4, maxlen=4)

    def capture(self, archive: ExplorationArchive, info: dict[str, Any] | None = None) -> bool:
        return archive.capture(self.env, self.adapter, self.raw, self.frames,
                               info or self.info, ((0, 4),), frozenset(), 4)

    def test_restore_preserves_visit_history_and_current_global_counts(self) -> None:
        archive = ExplorationArchive(4, 7)
        self.capture(archive)
        entry = archive.choose()
        self.env.em.state = b"later emulator state"
        self.adapter.step(info_for(Keys=1, Room=114), 2)
        self.adapter._tile_counts[(115, 1, 1)] = 90
        self.adapter._kill_history.append(7)
        self.raw[:] = 255
        self.frames[0][:] = 255

        raw, frames = entry.restore(self.env, self.adapter)
        self.assertEqual(self.env.em.state, b"snapshot")
        self.assertEqual(self.adapter._tile_counts[(115, 1, 1)], 90)
        self.assertEqual(self.adapter._kill_history, [7])
        self.assertEqual(self.adapter.old_info["Keys"], 0)
        self.assertNotIn(114, self.adapter.visited_rooms)
        # The memory planes are adapter attributes, so they roll back with it.
        self.assertNotIn(114, self.adapter._room_masks)
        reward, done = self.adapter.step(self.info.copy(), 5)
        self.assertAlmostEqual(reward, REWARD_VALUES["time_cost"])
        self.assertFalse(done)
        self.assertEqual(int(raw.max()), 0)
        self.assertEqual(int(frames[0].max()), 0)

        # Restoring/using one entry must not mutate its saved adapter or images.
        self.adapter.visited_rooms.clear()
        frames[0][:] = 9
        _, again = entry.restore(self.env, self.adapter)
        self.assertIn(115, self.adapter.visited_rooms)
        self.assertEqual(int(again[0].max()), 0)

    def test_cells_distinguish_inventory_and_consumed_key_progress(self) -> None:
        empty = cell_for(self.info, frozenset())
        carrying = info_for(Keys=1)
        doors = update_doors(frozenset(), carrying, self.info)
        self.assertNotEqual(empty, cell_for(carrying, frozenset()))
        self.assertNotEqual(empty, cell_for(self.info, doors))
        self.assertNotEqual(empty, cell_for(info_for(**{"Enemies Spawned In Room": 3}), frozenset()))
        self.assertEqual(empty, cell_for(info_for(**{"Backend Frame Count": 211}), frozenset()))

    def test_rejects_transition_and_terminal_snapshots_and_bounds_memory(self) -> None:
        archive = ExplorationArchive(2, 7)
        self.assertFalse(self.capture(archive, info_for(**{"Game Mode": 7})))
        self.adapter.abandoned = True
        self.assertFalse(self.capture(archive))
        self.adapter.abandoned = False
        for x in (8, 48, 80, 112):
            self.capture(archive, info_for(**{"Link X": x}))
        self.assertEqual(len(archive), 2)
        self.assertEqual(len(archive.seen), 4)
        self.assertFalse(self.capture(archive, info_for(**{"Link X": 112})))

    def test_manifest_retains_actual_path_including_partial_windows(self) -> None:
        import json
        archive = ExplorationArchive(2, 7)
        archive.capture(self.env, self.adapter, self.raw, self.frames, self.info,
                        ((0, 16), (1, 3)), frozenset(), 19)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.json"
            archive.save_manifest(path)
            manifest = json.loads(path.read_text())
        self.assertEqual(manifest["entries"][0]["actions"], [[0, 16], [1, 3]])
        self.assertEqual(manifest["entries"][0]["elapsed_frames"], 19)


class EmulatorRestoreTests(unittest.TestCase):
    @unittest.skipUnless(Path("games/Zelda/rom.nes").exists(), "requires local Zelda ROM")
    def test_restore_reproduces_real_emulator_trajectory(self) -> None:
        import retro
        retro.data.Integrations.add_custom_path(str(Path("games").resolve()))
        env = retro.make("Zelda", state="level1", inttype=retro.data.Integrations.ALL, render_mode=None)
        try:
            raw, _ = env.reset()
            adapter = ZeldaAdapter("level1")
            info = env.data.lookup_all()
            adapter.step(info.copy(), 0)
            frames = deque([raw[..., :1].copy()] * 4, maxlen=4)
            archive = ExplorationArchive(2, 7)
            archive.capture(env, adapter, raw, frames, info, (), frozenset(), 0)
            entry = archive.choose()

            def advance() -> tuple[np.ndarray, dict[str, Any]]:
                obs, latest = raw, info
                for _ in range(40):
                    obs, _, _, _, latest = env.step(adapter.actions[0])
                return obs.copy(), latest

            first_frame, first_info = advance()
            entry.restore(env, adapter)
            second_frame, second_info = advance()
            np.testing.assert_array_equal(first_frame, second_frame)
            self.assertEqual(first_info, second_info)
        finally:
            env.close()


class ObserveTests(unittest.TestCase):
    """The trial's action history, read off the path rather than a separate deque."""

    def _history(self, actions: list[tuple[int, int]]) -> np.ndarray:
        from scripts.archive_trial import observe

        adapter = ZeldaAdapter("level1")
        adapter.step(info_for(), 0)
        frames = deque([np.zeros((84, 84, 1), np.uint8)] * 4, maxlen=4)
        state = observe(adapter, np.zeros((240, 256, 3), np.uint8), frames, actions)
        return state.vector[0, adapter.vector_size:].reshape(RECENT_ACTIONS, -1)

    def test_history_is_the_last_actions_most_recent_first(self) -> None:
        # Partial windows (3, 4 frames) are still actions taken.
        history = self._history([(2, 16), (7, 16), (3, 4), (9, 16)])
        self.assertEqual([int(row.argmax()) for row in history], [9, 3, 7])
        self.assertEqual(float(history.sum()), 3.0)

    def test_short_path_leaves_the_older_slots_empty(self) -> None:
        history = self._history([(5, 16)])
        self.assertEqual(int(history[0].argmax()), 5)
        self.assertEqual(float(history[1:].sum()), 0.0)


class SpyAgent:
    def __init__(self) -> None:
        self.n_step_buffer: deque = deque()
        self.transitions: list[tuple] = []
        self.flushes = 0
        self.epsilon = 0.5

    def act(self, state: np.ndarray) -> np.ndarray:
        return np.eye(10, dtype=int)[0]

    def train(self, state: Any, action: Any, reward: float, next_state: Any,
              done: bool, learn: bool = True) -> None:
        self.transitions.append((state, action, reward, next_state, done))
        self.n_step_buffer.append(reward)

    def flush_episode(self) -> None:
        self.n_step_buffer.clear()
        self.flushes += 1


class RolloutBoundaryTests(unittest.TestCase):
    @unittest.skipUnless(Path("games/Zelda/rom.nes").exists(), "requires local Zelda ROM")
    def test_restore_adds_no_transition_and_evaluation_does_not_train(self) -> None:
        import retro

        from scripts.archive_trial import evaluate, rollout

        retro.data.Integrations.add_custom_path(str(Path("games").resolve()))
        env = retro.make("Zelda", state="level1", inttype=retro.data.Integrations.ALL, render_mode=None)
        args = SimpleNamespace(state="level1", max_frames=256, frame_skip=4,
                               train_every=1, eval_epsilon=0.05, seed=7,
                               eval_episodes=1, no_video=True, no_state_vector=False)
        adapter, agent = ZeldaAdapter("level1"), SpyAgent()
        archive = ExplorationArchive(16, 7)
        try:
            first = rollout(env, adapter, agent, args, 128, archive=archive)
            entry = next(e for e in archive.entries.values() if e.elapsed_frames > 0)
            second = rollout(env, adapter, agent, args, 13, archive=archive, entry=entry)
            self.assertEqual(second["frames"], 13)
            self.assertEqual(len(agent.transitions), first["decisions"] + second["decisions"])
            self.assertEqual(agent.flushes, 2)
            self.assertFalse(agent.n_step_buffer)
            self.assertEqual(second["keys_gained"], 0)
            self.assertEqual(second["kills"], 0)

            # Every stored state carries the full state vector, and the restored
            # rollout's first state carries the archived path's action history.
            width = vector_width(adapter, len(adapter.actions))
            for state, *_ in agent.transitions:
                self.assertEqual(state.vector.shape, (1, width))
            restored = agent.transitions[first["decisions"]][0]
            history = restored.vector[0, adapter.vector_size:].reshape(RECENT_ACTIONS, -1)
            expected = [index for index, _ in reversed(entry.actions[-RECENT_ACTIONS:])]
            for slot, index in enumerate(expected):
                self.assertEqual(int(history[slot].argmax()), index)
                self.assertEqual(float(history[slot].sum()), 1.0)
            self.assertEqual(float(history[len(expected):].sum()), 0.0)
            lifetime_counts = adapter._tile_counts.copy()
            random_state = np.random.get_state()
            with tempfile.TemporaryDirectory() as directory:
                rows = evaluate(env, agent, args, Path(directory), "evaluation")
            self.assertEqual(rows[0]["source"], "entrance")
            self.assertEqual(rows[0]["prefix_frames"], 0)
            self.assertEqual(len(agent.transitions), first["decisions"] + second["decisions"])
            self.assertEqual(agent.flushes, 2)
            self.assertEqual(agent.epsilon, 0.5)
            self.assertEqual(adapter._tile_counts, lifetime_counts)
            np.testing.assert_array_equal(np.random.get_state()[1], random_state[1])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
