"""Which checkpoint becomes best.keras.

Two failures this guards against, both measured on a 500-episode level1 run:
single-episode reward picked episode 29 (a 76%-random policy) because the tile
bonus decays, and a non-decaying score on its own still picked episode 55
because kills cap out and the first lucky episode wins every later tie.
"""

import math
import unittest

from games.Zelda.adapter import REWARD_VALUES, ZeldaAdapter
from main import BestTracker
from tests.test_zelda_adapter import frame_info


class BestTrackerTests(unittest.TestCase):
    def test_nothing_is_best_until_the_window_is_full(self) -> None:
        tracker = BestTracker(window=3)
        self.assertFalse(tracker.update(0, 5.0))
        self.assertFalse(tracker.update(1, 5.0))
        self.assertIsNone(tracker.mean)
        self.assertIsNone(tracker.best_episode)
        self.assertTrue(tracker.update(2, 5.0))
        self.assertEqual(tracker.best_episode, 2)

    def test_an_early_spike_loses_to_a_sustained_later_plateau(self) -> None:
        # The shape of the real run: a lucky high-epsilon episode that the
        # policy only matches on average hundreds of episodes later.
        tracker = BestTracker(window=5)
        scores = [0.0, 0.0, 1.2, 0.0, 0.0] + [0.2] * 20 + [1.0] * 5 + [0.6] * 10
        for episode, score in enumerate(scores):
            tracker.update(episode, score)
        # A single-episode max would say episode 2.
        self.assertEqual(tracker.best_episode, 29)
        self.assertAlmostEqual(tracker.best, 1.0)

    def test_a_tie_does_not_move_the_checkpoint(self) -> None:
        tracker = BestTracker(window=2)
        for episode in range(6):
            tracker.update(episode, 1.0)
        self.assertEqual(tracker.best_episode, 1)


class ZeldaCheckpointScoreTests(unittest.TestCase):
    def test_tile_bonus_is_tracked_and_excluded_from_the_score(self) -> None:
        adapter = ZeldaAdapter("level1")
        adapter.step(frame_info(Room=115), frame=1)
        reward, _ = adapter.step(
            frame_info(Room=114, **{"Link X": 16, "Link Y": 16}), frame=2)

        movement = REWARD_VALUES["movement"] / math.sqrt(1)
        self.assertAlmostEqual(adapter.tile_reward, movement)
        stats = adapter.episode_stats()
        self.assertAlmostEqual(stats["tile_reward"], round(movement, 4))

        # Everything left is the stationary part: the new room and the clock.
        self.assertAlmostEqual(
            adapter.checkpoint_score(reward, stats),
            REWARD_VALUES["new_room"] + REWARD_VALUES["time_cost"],
        )

    def test_same_behaviour_scores_the_same_after_tiles_stop_paying(self) -> None:
        # The property episode_reward lacks: repeat an identical episode and the
        # tile bonus decays, but the checkpoint score does not move.
        adapter = ZeldaAdapter("level1")
        scores, rewards = [], []
        for _ in range(3):
            adapter.reset()
            total = adapter.step(frame_info(Room=115), frame=1)[0]
            total += adapter.step(
                frame_info(Room=114, **{"Link X": 16, "Link Y": 16}), frame=2)[0]
            stats = adapter.episode_stats()
            rewards.append(total)
            scores.append(adapter.checkpoint_score(total, stats))

        self.assertGreater(rewards[0], rewards[2])
        self.assertAlmostEqual(scores[0], scores[2])

    def test_suppressed_start_room_tiles_are_not_counted(self) -> None:
        adapter = ZeldaAdapter("level1_key")
        adapter.step(frame_info(Room=115, Keys=1), frame=1)
        adapter.step(frame_info(Room=115, Keys=1, **{"Link X": 16, "Link Y": 16}),
                     frame=2)
        self.assertEqual(adapter.tile_reward, 0.0)

    def test_reset_clears_the_episode_tile_reward(self) -> None:
        adapter = ZeldaAdapter("level1")
        adapter.step(frame_info(Room=115), frame=1)
        adapter.step(frame_info(Room=114, **{"Link X": 16, "Link Y": 16}), frame=2)
        adapter.reset()
        self.assertEqual(adapter.tile_reward, 0.0)


if __name__ == "__main__":
    unittest.main()
