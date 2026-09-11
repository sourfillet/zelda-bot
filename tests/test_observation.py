"""The scalar branch and the memory planes.

Covers the plumbing added when the HUD planes were replaced: batching an
Observation into model input, the Zelda adapter's visited masks and state
vector, and main.py's action-history encoding.

Deliberately free of the emulator — everything here is driven from synthetic
info dicts, so it runs without the ROM.
"""

import unittest

import numpy as np

from games.Zelda.adapter import (
    MAP_WIDTH,
    MASK_CURRENT,
    MASK_GRID,
    MASK_VISITED,
    STATE_VECTOR_FIELDS,
    STATE_VECTOR_SIZE,
    TILE,
    ZeldaAdapter,
)
from main import RECENT_ACTIONS, build_state_vector, vector_width
from models.observation import Observation, concat_inputs, model_input
from tests.test_zelda_adapter import frame_info


class ModelInputTests(unittest.TestCase):
    def _obs(self, vector_size: int, fill: int = 0) -> Observation:
        return Observation(
            planes=np.full((1, 8, 8, 6), fill, dtype=np.uint8),
            vector=(np.arange(vector_size, dtype=np.float32).reshape(1, -1)
                    if vector_size else None),
        )

    def test_planes_only_batches_to_a_bare_array(self) -> None:
        batched = model_input([self._obs(0), self._obs(0)], vector_size=0)
        self.assertIsInstance(batched, np.ndarray)
        self.assertEqual(batched.shape, (2, 8, 8, 6))

    def test_vector_branch_batches_to_two_inputs(self) -> None:
        planes, vectors = model_input([self._obs(5), self._obs(5)], vector_size=5)
        self.assertEqual(planes.shape, (2, 8, 8, 6))
        self.assertEqual(vectors.shape, (2, 5))

    def test_missing_vector_is_rejected_rather_than_reshaped(self) -> None:
        # A planes-only transition reaching a two-input network is a real
        # mistake (a stale replay buffer, say); Keras would report it much
        # later as an opaque shape error.
        with self.assertRaises(ValueError):
            model_input([self._obs(5), self._obs(0)], vector_size=5)

    def test_concat_inputs_matches_a_plain_concatenate(self) -> None:
        first = model_input([self._obs(0, fill=1)], vector_size=0)
        second = model_input([self._obs(0, fill=2)], vector_size=0)
        self.assertEqual(concat_inputs(first, second).shape, (2, 8, 8, 6))

        first = model_input([self._obs(5, fill=1)], vector_size=5)
        second = model_input([self._obs(5, fill=2)], vector_size=5)
        planes, vectors = concat_inputs(first, second)
        self.assertEqual(planes.shape, (2, 8, 8, 6))
        self.assertEqual(vectors.shape, (2, 5))


class ZeldaStateVectorTests(unittest.TestCase):
    def test_vector_size_matches_the_field_list(self) -> None:
        self.assertEqual(ZeldaAdapter("level1").vector_size, STATE_VECTOR_SIZE)
        self.assertEqual(STATE_VECTOR_SIZE, len(STATE_VECTOR_FIELDS))

    def test_no_vector_before_the_first_frame(self) -> None:
        # main.py substitutes zeros here; returning the previous episode's
        # trailing values instead would be actively wrong.
        self.assertIsNone(ZeldaAdapter("level1").state_vector())

    def test_fields_decode_ram_and_stay_normalized(self) -> None:
        adapter = ZeldaAdapter("level1")
        adapter.step(
            frame_info(Room=115, Keys=3, Level=1, Sword=1,
                       **{"Heart Containers": 0x23, "Hearts": 0xFF,
                          "Enemies Spawned In Room": 4,
                          "Link X": 128, "Link Y": 120}),
            frame=1,
        )
        vector = adapter.state_vector()
        self.assertEqual(vector.shape, (STATE_VECTOR_SIZE,))
        self.assertEqual(vector.dtype, np.float32)

        values = dict(zip([n for n, _ in STATE_VECTOR_FIELDS], vector, strict=True))
        # Room 115 is (3, 7) on the 16x8 map — level 1's entrance.
        self.assertAlmostEqual(values["room_x"], 3 / 15, places=5)
        self.assertAlmostEqual(values["room_y"], 7 / 7, places=5)
        # Low nibble 3 filled hearts + a full partial byte = 4.
        self.assertAlmostEqual(values["hearts"], 4.0 / 16.0, places=5)
        self.assertAlmostEqual(values["keys"], 1.0, places=5)
        self.assertAlmostEqual(values["room_enemies"], 0.5, places=5)
        self.assertTrue(np.all((vector >= 0.0) & (vector <= 1.0)))

    def test_counters_saturate_rather_than_exceed_one(self) -> None:
        adapter = ZeldaAdapter("level1")
        adapter.step(frame_info(Keys=99, Bombs=99, Rupees=255), frame=1)
        vector = adapter.state_vector()
        self.assertTrue(np.all(vector <= 1.0))


class ZeldaMemoryPlaneTests(unittest.TestCase):
    def test_planes_are_blank_before_the_first_frame(self) -> None:
        adapter = ZeldaAdapter("level1")
        planes = adapter.extra_observation(None, size=84)
        self.assertEqual(planes.shape, (84, 84, adapter.extra_planes))
        self.assertEqual(planes.max(), 0)

    def test_trail_accumulates_and_marks_the_current_cell(self) -> None:
        adapter = ZeldaAdapter("level1")
        adapter.step(frame_info(Room=115, **{"Link X": 80, "Link Y": 120}), frame=1)
        adapter.step(frame_info(Room=115, **{"Link X": 88, "Link Y": 120}), frame=2)

        # Read the raw grid rather than the upscaled plane so cell indices stay
        # exact regardless of --input_size.
        tiles = adapter._room_masks[115]
        self.assertEqual(tiles.shape, (MASK_GRID, MASK_GRID))
        self.assertEqual(tiles[120 // TILE, 80 // TILE], MASK_VISITED)
        self.assertEqual(tiles[120 // TILE, 88 // TILE], MASK_VISITED)
        # The head is drawn only into the rendered plane, so the stored trail
        # stays a plain visited set.
        self.assertEqual(adapter._mask_here, (115, 88 // TILE, 120 // TILE))

        planes = adapter.extra_observation(None, size=MASK_GRID)
        self.assertEqual(planes[120 // TILE, 88 // TILE, 0], MASK_CURRENT)
        self.assertEqual(planes[120 // TILE, 80 // TILE, 0], MASK_VISITED)

    def test_each_room_keeps_its_own_trail(self) -> None:
        adapter = ZeldaAdapter("level1")
        adapter.step(frame_info(Room=115, **{"Link X": 80, "Link Y": 120}), frame=1)
        adapter.step(frame_info(Room=99, **{"Link X": 160, "Link Y": 96}), frame=2)
        adapter.step(frame_info(Room=115, **{"Link X": 80, "Link Y": 120}), frame=3)

        # Returning restores the earlier room's trail rather than a blank sheet.
        planes = adapter.extra_observation(None, size=MASK_GRID)
        self.assertEqual(planes[120 // TILE, 80 // TILE, 0], MASK_CURRENT)
        self.assertEqual(adapter._room_masks[99][96 // TILE, 160 // TILE],
                         MASK_VISITED)

        # The room map is a 16x8 grid, so it does not index 1:1 into a square
        # plane the way the 32x32 tile mask does — assert on the grid itself,
        # and on the rendered plane only that both intensities survive it.
        self.assertEqual(adapter._room_map[115 // MAP_WIDTH, 115 % MAP_WIDTH],
                         MASK_VISITED)
        self.assertEqual(adapter._room_map[99 // MAP_WIDTH, 99 % MAP_WIDTH],
                         MASK_VISITED)
        self.assertEqual(set(np.unique(planes[:, :, 1]).tolist()),
                         {0, MASK_VISITED, MASK_CURRENT})

    def test_reset_clears_the_trail(self) -> None:
        # These answer "where have I been THIS episode", unlike the lifetime
        # _tile_counts, which deliberately survives reset().
        adapter = ZeldaAdapter("level1")
        adapter.step(frame_info(Room=115, **{"Link X": 80, "Link Y": 120}), frame=1)
        adapter.reset()
        self.assertEqual(adapter.extra_observation(None, size=84).max(), 0)


class ActionHistoryTests(unittest.TestCase):
    def test_width_counts_adapter_scalars_plus_the_history(self) -> None:
        adapter = ZeldaAdapter("level1")
        self.assertEqual(vector_width(adapter, action_size=10),
                         STATE_VECTOR_SIZE + RECENT_ACTIONS * 10)

    def test_disabled_adapter_vector_disables_the_branch_entirely(self) -> None:
        # --no_state_vector zeroes vector_size; the action history must not
        # keep a branch alive on its own.
        adapter = ZeldaAdapter("level1")
        adapter.vector_size = 0
        self.assertEqual(vector_width(adapter, action_size=10), 0)
        self.assertIsNone(build_state_vector(adapter, __import__("collections").deque(),
                                             action_size=10))

    def test_history_is_one_hot_per_slot_most_recent_first(self) -> None:
        from collections import deque

        adapter = ZeldaAdapter("level1")
        recent: deque = deque([None] * RECENT_ACTIONS, maxlen=RECENT_ACTIONS)
        recent.appendleft(2)
        recent.appendleft(7)

        vector = build_state_vector(adapter, recent, action_size=10)
        self.assertEqual(vector.shape, (1, STATE_VECTOR_SIZE + RECENT_ACTIONS * 10))

        history = vector[0, STATE_VECTOR_SIZE:]
        self.assertEqual(history[0 * 10 + 7], 1.0)   # most recent
        self.assertEqual(history[1 * 10 + 2], 1.0)   # the one before it
        self.assertEqual(history[2 * 10:3 * 10].sum(), 0.0)  # still unfilled
        self.assertEqual(history.sum(), 2.0)

    def test_adapter_scalars_are_zero_before_the_first_frame(self) -> None:
        from collections import deque

        adapter = ZeldaAdapter("level1")
        recent: deque = deque([None] * RECENT_ACTIONS, maxlen=RECENT_ACTIONS)
        vector = build_state_vector(adapter, recent, action_size=10)
        self.assertEqual(vector[0, :STATE_VECTOR_SIZE].sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
