import math
import unittest
from typing import Any

from games.Zelda.adapter import (
    BITFIELD_ITEMS,
    COUNTER_ITEMS,
    MAJOR_ITEMS,
    MINOR_ITEMS,
    NORMAL_MODE,
    REWARD_VALUES,
    ZeldaAdapter,
)


def frame_info(**overrides: Any) -> dict[str, Any]:
    """Build the complete RAM-info subset consumed by ZeldaAdapter."""
    info: dict[str, Any] = {
        "Game Mode": NORMAL_MODE,
        "Level": 1,
        "Room": 115,
        "Link X": 8,
        "Link Y": 8,
        "Enemies Killed": 0,
        "Enemies Spawned In Room": 0,
        "Deaths": 0,
        "Heart Containers": 0,
        "Hearts": 0,
    }
    for name in COUNTER_ITEMS | MINOR_ITEMS | BITFIELD_ITEMS:
        info[name] = 0
    for name in MAJOR_ITEMS:
        info[name] = 0
    info.update(overrides)
    return info


class ZeldaAdapterDungeonBoundaryTests(unittest.TestCase):
    def test_custom_dungeon_state_ends_on_first_overworld_frame(self) -> None:
        adapter = ZeldaAdapter("level1_key")

        reward, done = adapter.step(frame_info(Keys=1), frame=1)
        self.assertEqual(reward, 0.0)
        self.assertFalse(done)

        reward, done = adapter.step(
            frame_info(Level=0, Room=119, Keys=1, **{"Link X": 16}),
            frame=2,
        )

        expected = (
            REWARD_VALUES["left_dungeon"]
            + REWARD_VALUES["abandon_dungeon"]
            + REWARD_VALUES["time_cost"]
        )
        self.assertAlmostEqual(reward, expected)
        self.assertTrue(done)
        self.assertTrue(adapter.abandoned)
        self.assertEqual(adapter.rooms_found, 0)
        self.assertEqual(adapter.tiles_found, 0)

    def test_overworld_start_does_not_enable_dungeon_boundary(self) -> None:
        adapter = ZeldaAdapter("gamestart")
        adapter.step(frame_info(Level=0, Room=1), frame=1)

        reward, done = adapter.step(
            frame_info(Level=0, Room=2, **{"Link X": 16}),
            frame=2,
        )

        expected = (
            REWARD_VALUES["new_room"]
            + REWARD_VALUES["movement"]
            + REWARD_VALUES["time_cost"]
        )
        self.assertAlmostEqual(reward, expected)
        self.assertFalse(done)
        self.assertFalse(adapter.abandoned)


class ZeldaAdapterKeyExplorationTests(unittest.TestCase):
    def test_key_pickup_preserves_current_room_but_renews_prior_room(self) -> None:
        adapter = ZeldaAdapter("level1")
        adapter.step(frame_info(Room=115, Keys=0), frame=1)

        # Visit a side room, then return to one tile in the entrance.
        adapter.step(
            frame_info(Room=114, Keys=0, **{"Link X": 16, "Link Y": 16}),
            frame=2,
        )
        adapter.step(
            frame_info(Room=115, Keys=0, **{"Link X": 16, "Link Y": 16}),
            frame=3,
        )
        self.assertEqual(adapter.tiles_found, 2)

        # Acquiring a key on the same entrance tile pays only the item reward
        # and time cost; the tile is carried into the keys=1 visit slice.
        reward, _ = adapter.step(
            frame_info(Room=115, Keys=1, **{"Link X": 16, "Link Y": 16}),
            frame=4,
        )
        self.assertAlmostEqual(
            reward,
            COUNTER_ITEMS["Keys"] + REWARD_VALUES["time_cost"],
        )
        self.assertEqual(adapter.tiles_found, 2)
        self.assertIn((2, 2, 1), adapter.visited_rooms[115])

        # The same position in the previously left room is novel with a key,
        # but its lifetime count still decays the reward from count 1 to 2.
        reward, _ = adapter.step(
            frame_info(Room=114, Keys=1, **{"Link X": 16, "Link Y": 16}),
            frame=5,
        )
        expected = REWARD_VALUES["movement"] / math.sqrt(2) + REWARD_VALUES["time_cost"]
        self.assertAlmostEqual(reward, expected)
        self.assertEqual(adapter.tiles_found, 3)

    def test_starting_with_key_does_not_reward_empty_start_room(self) -> None:
        adapter = ZeldaAdapter("level1_key")
        adapter.step(frame_info(Room=115, Keys=1), frame=1)

        reward, done = adapter.step(
            frame_info(Room=115, Keys=1, **{"Link X": 16, "Link Y": 16}),
            frame=2,
        )
        self.assertAlmostEqual(reward, REWARD_VALUES["time_cost"])
        self.assertFalse(done)
        self.assertEqual(adapter.tiles_found, 1)

        # Once Link leaves the starting room, ordinary discovery rewards apply.
        reward, done = adapter.step(
            frame_info(Room=114, Keys=1, **{"Link X": 16, "Link Y": 16}),
            frame=3,
        )
        expected = (
            REWARD_VALUES["new_room"]
            + REWARD_VALUES["movement"]
            + REWARD_VALUES["time_cost"]
        )
        self.assertAlmostEqual(reward, expected)
        self.assertFalse(done)


if __name__ == "__main__":
    unittest.main()
