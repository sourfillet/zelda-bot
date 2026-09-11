"""The Legend of Zelda (NES) game adapter.

Merges what used to be split across ``zelda.py`` (state parsing / reward calc),
``reward.py`` (reward constants), and the Zelda-specific bits of ``main.py``
(action set, Game-Mode termination, kill tracking) into one place.

Current training target: the ``monsters`` state — a single isolated combat room
where the agent learns to kill enemies. Kill is the dominant reward signal.
"""

import math
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from games.base import GameAdapter

# ----------------------------------------------------------------------------
# Reward constants (kill is the dominant signal)
# ----------------------------------------------------------------------------
# The whole table — including the item values further down — is a uniform 1/5
# rescale of the values this file used to carry. Nothing changed relative to
# anything else; what changed is the headroom under `reward_clip`.
#
# At the old scale `kill_enemy` was 1.0 against a clip of 1.0, so a single kill
# saturated the clip *exactly*. Every additional reward earned in the same
# decision was then discarded: at `--frame_skip 16` a decision that kills an
# enemy AND opens a door AND picks up a key trained on the identical +1.0 as a
# decision that only killed. The decisions most worth distinguishing were the
# ones flattened hardest.
#
# Raising `reward_clip` instead would not do: it sets the Bellman clamp
# (`q_limit = clip / (1 - gamma)`) that stops the divergence documented in
# ML_NOTES, so the clip is the fixed side and the table is the side that moves.
# This is what PWhiddy's Pokemon Red environment does with its `reward_scale`
# multiplier — scale the events down, leave the ceiling alone.
#
# Worst realistic compound decision is now ~0.8 (kill + major item + triforce +
# new room), so ordinary play no longer touches the clip. The two terminal
# penalties are deliberately still at -1.0 after clipping: nothing should ever
# look better than not abandoning the dungeon.
REWARD_VALUES = {
    # Charged once, on the transition out, so episode length cannot set its size.
    'left_dungeon': -0.2,
    # Left the loaded dungeon, which ends the episode. Matches
    # `leave_start_room`, the equivalent gate for confined states.
    'abandon_dungeon': -1.0,
    # Link leaves the room he started in. Only charged in confined mode, where
    # the point of the episode is to stay and fight (the `monsters` state).
    'leave_start_room': -1.0,
    # Link enters a room he has not seen this episode. Positive while roaming:
    # exploration is the objective in a dungeon, not a failure. Sized to match
    # a kill so discovery and combat pull with comparable force.
    'new_room': 0.2,
    # Exploration bonus coefficient: a tile pays movement / sqrt(N), where N is
    # how often it has been reached this process. Familiar ground stops paying.
    'movement': 0.01,
    # Charged every frame, so loitering in an exhausted room is unprofitable.
    # Upper bound is a constraint, not a preference: ending an episode stops the
    # clock, so |time_cost * FRAME_SKIP / (1 - discount_factor)| must stay below
    # |death| or dying becomes the cheapest way to stop paying it.
    #
    # Recomputed for gamma 0.997 and --frame_skip 16, the current defaults:
    #   0.00002 * 16 / 0.003 = 0.107  against  |death| = 0.2.
    # Satisfied with ~1.9x margin. Raising gamma tightens this, so the check has
    # to be redone whenever `discount_factor` moves, not only when this does.
    'time_cost': -0.00002,
    # Killing an enemy (PRIMARY goal — kept dominant)
    'kill_enemy': 0.2,
    # Heart changes
    'heart_loss': -0.002,
    'heart_gain': 0.002,
    # Must outweigh the discounted time cost of playing an episode out; see
    # `time_cost`. `reward_clip` bounds any single event, so `time_cost` is
    # still the side that has to move if this constraint is ever violated.
    'death': -0.2,
}

# ----------------------------------------------------------------------------
# Action set
# ----------------------------------------------------------------------------
# Button order in retro's MultiBinary action space:
# ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
# START (pause) and SELECT waste exploration; direction+A combos let Link walk
# while swinging the sword.
ACTIONS = [
    [0, 0, 0, 0, 1, 0, 0, 0, 0],  # up
    [0, 0, 0, 0, 0, 1, 0, 0, 0],  # down
    [0, 0, 0, 0, 0, 0, 1, 0, 0],  # left
    [0, 0, 0, 0, 0, 0, 0, 1, 0],  # right
    [0, 0, 0, 0, 0, 0, 0, 0, 1],  # A (sword)
    [0, 0, 0, 0, 1, 0, 0, 0, 1],  # up + A
    [0, 0, 0, 0, 0, 1, 0, 0, 1],  # down + A
    [0, 0, 0, 0, 0, 0, 1, 0, 1],  # left + A
    [0, 0, 0, 0, 0, 0, 0, 1, 1],  # right + A
    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # B (item)
]

# A and B are edge-triggered: the game acts on the "newly pressed" register
# ($FA), so a held button never swings the sword again. The back half of each
# frame-skip window releases A/B so every attack decision lands as a fresh
# press. Directions stay held (movement is level-triggered).
ACTIONS_RELEASED = [[0] + a[1:8] + [0] for a in ACTIONS]

# Position is quantized to TILE-square cells before being counted, so sub-pixel
# jitter does not read as exploration. A room holds a few hundred cells.
TILE = 8

# Normal-play frames in the overworld before a dungeon episode ends. 1 ends it
# on the transition itself, putting the penalty on the decision that caused it;
# any grace period is long enough to bank an overworld kill first.
#
# Cannot misfire on in-dungeon doors or stairs: `_frame_reward` only runs on
# NORMAL_MODE frames, and transition animations return earlier in step().
OVERWORLD_PATIENCE = 1

# ----------------------------------------------------------------------------
# Memory planes
# ----------------------------------------------------------------------------
# Two planes that show the agent where it has already been.
#
# This is the hole the HUD planes these replaced never filled. The tile bonus in
# `_frame_reward` pays for novelty computed from `visited_rooms` — a dict that
# is not in the observation at all — so "have I already swept this half of the
# room" was unanswerable from the network's input. The shaped reward was not a
# function of anything the agent could see, and the only thing it could learn
# was a positional prior. That is exactly the measured pathology in ML_NOTES:
# Q spread 0.29-0.34 across positions against 0.021-0.026 across actions.
#
# Both Pokemon Red environments carry the same fix. v2 feeds a 48x48 local
# exploration map; PokeRL feeds a per-map binary visited mask aligned to world
# coordinates and measures unique positions +40.6%, coverage 12% -> 41%, and
# revisit ratio -35% against the identical agent without it.
#
# The HUD band is gone from the observation because `state_vector()` now hands
# over the numbers it existed to convey — minimap room, key/bomb/rupee counts,
# equipped items — exactly, as floats, instead of as a 3x3-pixel marker the
# convolutions had to find. See STATE_VECTOR_FIELDS.
HUD_HEIGHT = 56          # playfield starts here; frames above it are the HUD

# Room-local visited mask, at the same TILE quantization the reward uses. The
# NES frame is 256x240, so TILE=8 needs a 32x30 grid; 32x32 covers it with room
# to spare and keeps the upscale to the network's square input clean.
MASK_GRID = 32

# Dungeon-level visited-room grid. Zelda indexes rooms as room = y * 16 + x over
# a 16x8 map (level 1's entrance 115 is (3,7), the room past its locked door 99
# is (3,6), the `monsters` room 116 is (4,7)), so the whole floor plan fits in
# one 16x8 plane.
MAP_WIDTH, MAP_HEIGHT = 16, 8

# Mask intensities. Distinct rather than binary so one plane carries "been
# there" and "here now" at once, which is what makes the plane readable as a
# trail with a head rather than an undifferentiated blob.
MASK_VISITED = 180
MASK_CURRENT = 255

# Game Mode ($12) values that count as active gameplay:
# 5 = normal play, 6 = preparing scroll, 7 = scrolling, 4 = finishing scroll.
# Anything else (death sequence, game over) terminates the episode.
SCROLL_MODES = (4, 6, 7)
NORMAL_MODE = 5

# Death, per RAM_MAP.md: Game Mode goes 5 -> 17 -> 8 when Link dies. The
# `Deaths` counter is checked as well, since it is unambiguous.
DEATH_MODES = (8, 17)

# Every other non-normal mode (2, 3, 16) is a transition animation for doors,
# stairs and cave entries: fixed-length, always returning to mode 5. They score
# nothing but must not end the episode.

# States where the episode is meant to stay on one screen. `monsters` is the
# isolated combat room; everything else (a dungeon entrance, the overworld) is
# meant to be roamed, so leaving a room there is normal play rather than
# failure. Anything not listed here gets roaming behaviour.
CONFINED_STATES = {"monsters"}

# ----------------------------------------------------------------------------
# Item pickups
# ----------------------------------------------------------------------------
# Everything below pays on a GAIN only, so spending a key or throwing a bomb
# scores nothing. Values are tiered by what the item actually unlocks.

# Plain counters — reward every unit gained. Same 1/5 rescale as REWARD_VALUES.
COUNTER_ITEMS = {
    # Keys gate locked doors, the main barrier to the rest of a dungeon.
    "Keys": 0.1,
    "Bombs": 0.02,
    "Rupees": 0.004,
}

# Items whose *decrease* is progress rather than loss. Keys are consumed only by
# locked doors — verified: 5 keys survived 8000 frames of random play across
# deaths and room transitions with zero change, then holding UP into level 1's
# locked door consumed exactly one. Nothing else in the game takes them.
#
# Worth its own reward because the unlock is 62 frames upstream of the room
# transition it earns, and pays nothing on its own: measured, keys drop at frame
# 84 in Room 115 during normal play while the room only changes at frame 146.
# This puts the reward on the decision that actually opened the door.
#
# 4x the pickup value (COUNTER_ITEMS["Keys"] = 0.1), deliberately: spending a
# key must be strictly, obviously better than holding one. They used to be
# equal, and the agent's revealed preference was to hoard — measured over a
# 205-episode level1 run, it picked up a key in 81 episodes and spent one in
# **0**, returning to the room with the locked door in only 7 of those 81.
#
# This is the one lever here that is NOT an attempt to pay for the journey
# back. Rewarding the trip is what the `keys`-in-the-exploration-cell mechanism
# tries to do, and it cannot be made to work from this direction: the journey
# back IS a re-sweep of known ground, so any rate high enough to motivate it is
# high enough to make re-sweeping beat spending. That was measured too — with
# `keys` in the LIFETIME tile counter the entrance re-sweep paid ~+2.5 against
# +1.5 for the unlock. Leave the trip unpaid, make the destination worth
# reaching, and let the value function propagate it.
#
# Which it can only do once it has observed the event at all. At 0 in 66,441
# decisions, no reward weight here changes anything on its own — this pairs
# with the curriculum pool (`level1_door` et al.), it does not replace it.
SPEND_ITEMS = {"Keys": 0.4}

# One-time acquisitions and upgrades. RAM holds a type or flag (sword 1-3,
# candle 1-2, ...), so any increase is an acquisition. Sized to match a kill.
MAJOR_ITEMS = [
    "Arrow", "Boomerang", "Boomerang 2", "Bow", "Candle", "Flute", "Food",
    "Ladder", "Letter", "Magic Book", "Magical Key", "Magical Rod", "Potion",
    "Power Bracelet", "Raft", "Ring", "Shield", "Sword",
]
MAJOR_ITEM_REWARD = 0.2

# Temporary powerup, not an acquisition.
MINOR_ITEMS = {"Clock": 0.04}

# Per-level bitfields, one bit per dungeon. Counted by newly-set bits so a
# second dungeon's map still scores once the first is held.
BITFIELD_ITEMS = {
    # The scenario's actual objective.
    "Triforce Pieces": 0.2,
    "Map": 0.06,
    "Compass": 0.06,
}

# $066F packs filled hearts in the low nibble and (containers - 1) in the high.
# Only the high nibble is an acquisition; healing and damage are handled by
# `heart_gain`/`heart_loss` from the decoded value.
HEART_CONTAINER_REWARD = 0.2


# Snapshotted into each run's config.json alongside REWARD_VALUES. Without it a
# run directory records only half its reward table, so `key_used` — which is now
# a value that varies between runs — could not be recovered after the fact.
REWARD_ITEMS: dict[str, Any] = {
    "counter_items": COUNTER_ITEMS,
    "spend_items": SPEND_ITEMS,
    "minor_items": MINOR_ITEMS,
    "bitfield_items": BITFIELD_ITEMS,
    "major_item_reward": MAJOR_ITEM_REWARD,
    "heart_container_reward": HEART_CONTAINER_REWARD,
}


def item_reward(old: dict[str, Any], new: dict[str, Any]) -> float:
    """Reward for everything Link gained between two frames. Gains only."""
    total = 0.0
    for name, value in COUNTER_ITEMS.items():
        total += max(int(new[name]) - int(old[name]), 0) * value
    for name, value in SPEND_ITEMS.items():
        total += max(int(old[name]) - int(new[name]), 0) * value
    for name in MAJOR_ITEMS:
        if int(new[name]) > int(old[name]):
            total += MAJOR_ITEM_REWARD
    for name, value in MINOR_ITEMS.items():
        if int(new[name]) > int(old[name]):
            total += value
    for name, value in BITFIELD_ITEMS.items():
        gained_bits = (int(new[name]) & ~int(old[name])) & 0xFF
        total += bin(gained_bits).count("1") * value
    containers = (int(new["Heart Containers"]) >> 4) - (int(old["Heart Containers"]) >> 4)
    total += max(containers, 0) * HEART_CONTAINER_REWARD
    return total


def get_actual_hearts(containers: int = 0, partial: int = 0) -> float:
    """Decode Link's current health from the two heart bytes.

    $066F ("Heart Containers"): low nibble = filled hearts, high nibble = containers - 1
    $0670 ("Hearts"):           partial heart (0 = empty, 1-0x7F = half, 0x80-0xFF = full)
    """
    filled = float(containers & 0x0F)
    if partial >= 0x80:
        filled += 1.0
    elif partial > 0:
        filled += 0.5
    return filled


def _bits(value: Any) -> int:
    """Population count of one RAM byte, for the per-level bitfields."""
    return bin(int(value) & 0xFF).count("1")


# ----------------------------------------------------------------------------
# Scalar state vector
# ----------------------------------------------------------------------------
# Everything the emulator already knows exactly, handed to the network as
# numbers instead of pixels. The order here IS the vector layout, and
# `vector_size` derives from the list, so adding a field needs no second edit.
#
# Each entry reads the adapter's most recent normal-play `info` dict and returns
# roughly [0, 1] — the branch feeds straight into a Dense layer, so a raw room
# id of 115 sitting beside a 0-or-1 flag would dominate it.
#
# Three of these were previously encoded as 84x84 HUD planes and nothing else:
# `keys`/`bombs`/`rupees` were the counter digits, and `room_x`/`room_y` were
# the minimap marker. Two floats beat locating a 3x3-pixel square, and room
# coordinates are strictly more informative than the marker ever was, since the
# minimap lights only the current room.
#
# `hearts` reads the value AFTER `_frame_reward` has decoded it in place via
# get_actual_hearts(), not the raw packed byte.
STATE_VECTOR_FIELDS: list[tuple[str, Callable[[dict[str, Any]], float]]] = [
    # Inventory that gates progress.
    ("keys", lambda i: min(int(i["Keys"]), 3) / 3.0),
    ("bombs", lambda i: min(int(i["Bombs"]), 8) / 8.0),
    ("rupees", lambda i: min(int(i["Rupees"]), 255) / 255.0),
    # Survival.
    ("hearts", lambda i: min(float(i["Hearts"]), 16.0) / 16.0),
    ("heart_containers", lambda i: (((int(i["Heart Containers"]) >> 4) + 1) / 16.0)),
    # Where Link is, at both scales. Room is decomposed into map coordinates
    # rather than passed as an id, since 115 and 116 are adjacent rooms but
    # nothing about the raw numbers says so on the vertical axis.
    ("room_x", lambda i: (int(i["Room"]) % MAP_WIDTH) / (MAP_WIDTH - 1)),
    ("room_y", lambda i: (int(i["Room"]) // MAP_WIDTH) / (MAP_HEIGHT - 1)),
    ("link_x", lambda i: int(i["Link X"]) / 255.0),
    ("link_y", lambda i: int(i["Link Y"]) / 255.0),
    ("level", lambda i: int(i["Level"]) / 9.0),
    # Equipment. Sword and candle are tiers; the rest are have/have-not.
    ("sword", lambda i: min(int(i["Sword"]), 3) / 3.0),
    ("candle", lambda i: min(int(i["Candle"]), 2) / 2.0),
    ("bow", lambda i: float(int(i["Bow"]) > 0)),
    ("ladder", lambda i: float(int(i["Ladder"]) > 0)),
    ("raft", lambda i: float(int(i["Raft"]) > 0)),
    ("boomerang", lambda i: float(int(i["Boomerang"]) > 0)),
    ("magical_key", lambda i: float(int(i["Magical Key"]) > 0)),
    ("power_bracelet", lambda i: float(int(i["Power Bracelet"]) > 0)),
    # Per-level bitfields, as a fraction of the eight dungeons.
    ("triforce", lambda i: _bits(i["Triforce Pieces"]) / 8.0),
    ("maps", lambda i: _bits(i["Map"]) / 8.0),
    ("compasses", lambda i: _bits(i["Compass"]) / 8.0),
    # Whether this room has combat in it at all, which the playfield shows only
    # once an enemy is on screen.
    ("room_enemies", lambda i: min(int(i["Enemies Spawned In Room"]), 8) / 8.0),
]

STATE_VECTOR_SIZE = len(STATE_VECTOR_FIELDS)


class ZeldaAdapter(GameAdapter):
    # Per-episode state, declared so the None-then-populate pattern below is
    # explicit about what each field eventually holds.
    old_info: dict[str, Any] | None
    # room -> {(tile_x, tile_y, keys_held): 1} for this episode only
    visited_rooms: dict[int, dict[tuple[int, int, int], int]]
    start_kills: int | None
    start_level: int | None
    start_keys: int | None

    name = "Zelda"
    default_state = "monsters"
    actions = ACTIONS
    actions_released = ACTIONS_RELEASED
    log_fields = ["kills", "kills_avg", "cleared", "rooms", "tiles", "tile_reward",
                  "keys_max", "keys_used", "key_backtrack"]
    # Two memory planes: the current room's visited-tile mask and the dungeon's
    # visited-room map. Set to 0 to train on the playfield alone -- changing
    # this changes the network's input shape, so checkpoints do not load across
    # the switch. (It was 3 HUD columns; see the Memory planes section.)
    extra_planes = 2
    # Inventory, position and equipment as scalars. main.py widens the actual
    # network branch with a one-hot history of recent actions.
    vector_size = STATE_VECTOR_SIZE

    def __init__(self, state: str | None = None) -> None:
        # Start state matters for the dungeon-in-overworld penalty, and for
        # whether Link is confined to one screen.
        self.state = state or self.default_state
        # Single-screen combat vs free roaming. main.py may re-set self.state
        # after validating it, so this is recomputed in reset().
        self.confined = self.state in CONFINED_STATES
        # Moving-average history of kills across episodes.
        self._kill_history: list[int] = []
        # Lifetime visit counts, keyed (room, tile_x, tile_y). Deliberately NOT
        # cleared in reset(): the whole point is that the bonus decays across
        # episodes, so re-walking the starting room stops paying. Lives for the
        # process, not across separate runs.
        self._tile_counts: dict[tuple[int, int, int], int] = {}
        self._cleared = False
        self.reset()

    # ------------------------------------------------------------------
    # Per-episode lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.confined = self.state in CONFINED_STATES
        self.old_info = None
        self.visited_rooms = {}
        # The room Link starts in, recorded on the first frame. Seeding
        # visited_rooms with it is what replaces the old hardcoded `!= 116`
        # check: the starting room is never "new", whichever room it is.
        self.start_room: int | None = None
        # Whether leaving for Level 0 is off-task comes from RAM, not the save
        # state's filename. Curriculum states and future custom dungeon states
        # must behave exactly like the base level1-level8 states they derive
        # from, without maintaining a second whitelist here.
        self.start_level = None
        # A curriculum state that starts with a key has no pre-pickup visit set
        # to carry forward. Its starting room is treated as the pickup room for
        # movement reward purposes, preventing `level1_key` from paying a fresh
        # sweep of the empty entrance before the agent tries the door.
        self.start_keys = None
        self.rooms_found = 0
        self.tiles_found = 0
        # Tile bonus actually paid this episode. The one reward term that is not
        # stationary (it decays on the lifetime `_tile_counts`), tracked so it
        # can be logged and excluded from `checkpoint_score`.
        self.tile_reward = 0.0
        self.died = False
        self.abandoned = False
        # Most keys held at once this episode, and how many were spent on doors.
        self.keys_max = 0
        self.keys_used = 0
        # Frames spent back in the starting room while carrying a key. Separates
        # "never backtracks" from "backtracks but never finds the locked door",
        # which `rooms` cannot distinguish (returning to a visited room does not
        # increment it).
        self.key_backtrack = 0
        # Consecutive frames spent in the overworld; see OVERWORLD_PATIENCE.
        self.frames_outside = 0
        # Lifetime kill counter ($52A) survives death/room transitions, so we
        # track kills as a delta from the episode's first observed value.
        self.start_kills = None
        self.start_spawned = 0
        self.episode_kills = 0
        # Memory planes, cleared every episode because they answer "where have I
        # been THIS episode" — the question the per-episode tile reward asks.
        # (The lifetime `_tile_counts` deliberately survives reset; these are a
        # different quantity and must not.)
        #
        # room -> (MASK_GRID, MASK_GRID) mask, keyed by room so walking back into
        # an earlier room restores its trail rather than showing a blank sheet.
        self._room_masks: dict[int, np.ndarray] = {}
        self._room_map = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=np.uint8)
        # Cell the last mark landed on, drawn at MASK_CURRENT so the plane shows
        # a head as well as a trail.
        self._mask_here: tuple[int, int, int] | None = None

    def step(self, info: dict[str, Any], frame: int) -> tuple[float, bool]:
        """One emulated frame -> (reward, done)."""
        if self.start_kills is None:
            self.start_kills = info["Enemies Killed"]
            self.start_spawned = info["Enemies Spawned In Room"]
        # modulo handles the u8 counter wrapping past 255
        self.episode_kills = (info["Enemies Killed"] - self.start_kills) % 256

        mode = info["Game Mode"]

        # Time is charged on EVERY frame, including transition animations.
        # Otherwise a door cycle is ~141 frames (35% of one measured crossing)
        # on which nothing at all is scored — no reward, no penalty, no clock —
        # which makes shuttling through a doorway strictly better than playing
        # a room that has stopped paying. It was the one place the clock stopped.
        reward = REWARD_VALUES['time_cost']

        if mode in SCROLL_MODES:
            # Mid-scroll between rooms. Confined episodes end here, so the
            # penalty lands close in time to the action that caused it. While
            # roaming this is ordinary movement: no reward, no termination, and
            # the room itself is scored on arrival in _frame_reward.
            if self.confined and self.old_info is not None:
                return REWARD_VALUES['leave_start_room'], True
            return reward, False
        if mode in DEATH_MODES:
            # Real death: penalize and end rather than fill the replay buffer
            # with game-over frames.
            return REWARD_VALUES['death'], True
        if mode != NORMAL_MODE:
            # A transition animation. Link is not controllable and nothing here
            # is worth scoring, but the clock still runs and the episode goes on.
            return reward, False

        # The old 200-frame grace period is gone with `repeat_state`: it existed
        # to stop a per-frame stuck-penalty firing before the agent had a chance
        # to move. `time_cost` charges every frame uniformly instead, so there is
        # nothing to suppress.
        # _frame_reward charges its own time_cost for normal-play frames, so
        # `reward` is not carried in here — that would double-charge.
        return self._frame_reward(info), self.died or self.abandoned

    def _mark_visited(self, room: int, link_x: int, link_y: int) -> None:
        """Record one normal-play position into both memory planes.

        Called on every normal-play frame rather than only on novel tiles, so
        the `MASK_CURRENT` head tracks Link continuously instead of freezing
        wherever he last found new ground.
        """
        mask = self._room_masks.get(room)
        if mask is None:
            mask = np.zeros((MASK_GRID, MASK_GRID), dtype=np.uint8)
            self._room_masks[room] = mask
        tile_x = min(link_x // TILE, MASK_GRID - 1)
        tile_y = min(link_y // TILE, MASK_GRID - 1)
        mask[tile_y, tile_x] = MASK_VISITED
        self._mask_here = (room, tile_x, tile_y)
        self._room_map[min(room // MAP_WIDTH, MAP_HEIGHT - 1),
                       room % MAP_WIDTH] = MASK_VISITED

    def extra_observation(self, frame: Any, size: int = 84) -> Any:
        """Build the two memory planes: visited tiles here, visited rooms overall.

        `frame` is unused — unlike the HUD planes these replaced, nothing here
        is read off the screen. Both planes are drawn from the adapter's own
        per-episode history and upscaled with INTER_NEAREST, which keeps single
        visited cells as hard squares instead of smearing them into the
        background the way interpolation would.
        """
        if self.extra_planes < 1:
            return None

        tiles = np.zeros((MASK_GRID, MASK_GRID), dtype=np.uint8)
        rooms = self._room_map.copy()
        if self._mask_here is not None:
            room, tile_x, tile_y = self._mask_here
            tiles = self._room_masks[room].copy()
            tiles[tile_y, tile_x] = MASK_CURRENT
            rooms[min(room // MAP_WIDTH, MAP_HEIGHT - 1),
                  room % MAP_WIDTH] = MASK_CURRENT

        planes = [cv2.resize(plane, (size, size), interpolation=cv2.INTER_NEAREST)
                  for plane in (tiles, rooms)]
        return np.stack(planes, axis=2)

    def state_vector(self) -> Any:
        """Inventory, position and equipment as floats. See STATE_VECTOR_FIELDS.

        Reads the last normal-play frame. Returns None before the first one, so
        the opening decision of an episode sees zeros rather than stale values
        carried over from the previous episode.
        """
        if self.old_info is None:
            return None
        info = self.old_info
        return np.array([read(info) for _, read in STATE_VECTOR_FIELDS],
                        dtype=np.float32)

    def episode_stats(self) -> dict[str, Any]:
        self._kill_history.append(self.episode_kills)
        window = min(10, len(self._kill_history))
        kills_avg = sum(self._kill_history[-window:]) / window
        self._cleared = self.start_spawned > 0 and self.episode_kills >= self.start_spawned
        return {
            "kills": self.episode_kills,
            "kills_avg": round(kills_avg, 2),
            "cleared": int(self._cleared),
            "rooms": self.rooms_found,
            "tiles": self.tiles_found,
            "tile_reward": round(self.tile_reward, 4),
            "keys_max": self.keys_max,
            "keys_used": self.keys_used,
            "key_backtrack": self.key_backtrack,
        }

    def checkpoint_score(self, episode_reward: float, stats: dict[str, Any]) -> float:
        """The episode's return with the decaying tile bonus taken out.

        What remains — kills, rooms, items, keys, deaths, the clock — pays the
        same for the same behaviour at episode 5 as at episode 500, so it can be
        compared across a run. The tile term cannot: measured on a 500-episode
        level1 run it paid ~1.9 on episode 29 (228 fresh tiles at near-full
        rate) and ~0.03 by episode 450, which is how a 76%-random policy held
        `best.keras` over one killing 4.2 enemies an episode.

        Uses the same weights the agent is trained on rather than a hand-picked
        metric like kills + rooms, so there is only one definition of "good" and
        spending a key (0.4) automatically ranks above a kill (0.2).
        """
        return episode_reward - self.tile_reward

    def summary_line(self) -> str:
        cleared = " - ROOM CLEARED!" if self._cleared else ""
        rooms = "" if self.confined else f"  Rooms: {self.rooms_found}  Tiles: {self.tiles_found}"
        return f"Kills: {self.episode_kills}/{self.start_spawned}{rooms}{cleared}"

    # ------------------------------------------------------------------
    # Reward shaping (normal-play frames)
    # ------------------------------------------------------------------

    def _frame_reward(self, info: dict[str, Any]) -> float:
        """Reward for a single normal-play frame; updates old_info/visited_rooms."""
        if self.old_info is None:
            self.old_info = info
            self.old_info['Hearts'] = get_actual_hearts(
                self.old_info['Heart Containers'], self.old_info['Hearts'])
            # Seed the starting room so it is never counted as a discovery.
            self.start_room = int(info['Room'])
            self.start_level = int(info['Level'])
            self.start_keys = int(info['Keys'])
            self.visited_rooms.setdefault(self.start_room, {})
            self._mark_visited(self.start_room, int(info['Link X']),
                               int(info['Link Y']))
            return 0.0

        reward = 0.0
        old_info = self.old_info

        # Nothing off-task pays: when an episode began in a dungeon, the
        # overworld is 128 unseen rooms and its own enemies, which outbids the
        # dungeon. Infer that from the initial RAM Level rather than a save-state
        # name so level1_key/door/room99 and arbitrary future curriculum states
        # cannot accidentally disable the boundary.
        started_in_dungeon = self.start_level is not None and self.start_level > 0
        on_task = not (started_in_dungeon and int(info['Level']) < 1)

        # Walked out of the dungeon he was loaded into — charge once, on the
        # transition, not for every frame spent outside.
        if started_in_dungeon and int(info['Level']) < 1 <= int(old_info['Level']):
            reward += REWARD_VALUES['left_dungeon']

        if on_task:
            reward += item_reward(old_info, info)
        old_keys = int(old_info['Keys'])
        keys = int(info['Keys'])
        room = int(info['Room'])
        self._mark_visited(room, int(info['Link X']), int(info['Link Y']))
        self.keys_max = max(self.keys_max, keys)
        if keys > 0 and room == self.start_room:
            self.key_backtrack += 1
        self.keys_used += max(old_keys - keys, 0)

        # Inventory remains part of the per-episode exploration state so rooms
        # visited earlier can become worth backtracking through with a key. Do
        # not renew tiles in the room where the key was found, though: that paid
        # the agent to sweep the now-empty combat room before starting the long
        # return trip. Carry every position already seen here into the new key
        # slice; other rooms deliberately keep only their old-key entries.
        if keys > old_keys:
            visited_here = self.visited_rooms.get(room)
            if visited_here:
                positions = {(tile_x, tile_y) for tile_x, tile_y, _ in visited_here}
                visited_here.update({(tile_x, tile_y, keys): 1
                                     for tile_x, tile_y in positions})

        info['Hearts'] = get_actual_hearts(info['Heart Containers'], info['Hearts'])
        if info['Hearts'] < old_info['Hearts']:
            reward += REWARD_VALUES['heart_loss']
        elif info['Hearts'] > old_info['Hearts']:
            reward += REWARD_VALUES['heart_gain']

        self.frames_outside = 0 if on_task else self.frames_outside + 1

        if info['Room'] not in self.visited_rooms:
            self.visited_rooms[info['Room']] = {}
            # The starting room is seeded below on the first frame, so reaching
            # this branch means a genuinely new room. Confined episodes never
            # get here (they terminate mid-scroll).
            if not self.confined and on_task:
                self.rooms_found += 1
                reward += REWARD_VALUES['new_room']

        # Exploration bonus, paid once per tile per episode and scaled by how
        # familiar that tile is overall. Count-based exploration: novelty decays
        # as 1/sqrt(N), so the agent is pushed outward instead of re-sweeping
        # ground it already knows.
        #
        # The cell includes how many keys Link is carrying, because a room he
        # can unlock is not the same state as one he cannot — the up door in
        # level 1's entrance is passable with a key and a wall without one.
        # Without this, walking back to a door after picking up a key pays
        # nothing, since the ground is already marked visited. The pickup-room
        # carryover above prevents this from also renewing the cleared room.
        #
        # Not farmable: each (tile, key count) pays once, keys only arrive from
        # finite pickups and only leave through finite doors, so the number of
        # distinct key counts — and therefore re-sweeps — is bounded.
        tile = (int(info['Link X']) // TILE, int(info['Link Y']) // TILE, keys)
        if tile not in self.visited_rooms[room]:
            self.visited_rooms[room][tile] = 1
            if on_task:
                # The per-episode cell above includes `keys`, so a key pickup
                # makes known ground in earlier rooms payable again. The
                # pickup room's known tiles were copied into the new key slice
                # above, preventing an immediate empty-room re-sweep. The
                # LIFETIME counter deliberately omits inventory either way, so
                # rewarded backtracking pays at the decayed rate, not full.
                key = (int(room), tile[0], tile[1])
                count = self._tile_counts.get(key, 0) + 1
                self._tile_counts[key] = count
                # `level1_key` and `level1_door` begin in the empty entrance
                # with a key already held, so there is no earlier keyless slice
                # whose positions can be copied. Count those visits normally
                # but do not reward them; leaving the room exposes ordinary
                # new-room/tile rewards, while loitering only pays the clock.
                suppress_start_key_room = (
                    self.start_keys is not None
                    and self.start_keys > 0
                    and room == self.start_room
                )
                if not suppress_start_key_room:
                    tile_bonus = REWARD_VALUES['movement'] / math.sqrt(count)
                    reward += tile_bonus
                    self.tile_reward += tile_bonus
                self.tiles_found += 1

        # Committed to the overworld rather than briefly clipping the boundary.
        if not on_task and self.frames_outside >= OVERWORLD_PATIENCE:
            self.abandoned = True
            reward += REWARD_VALUES['abandon_dungeon']

        # Time is never free.
        reward += REWARD_VALUES['time_cost']

        # Only on-task kills pay. Overworld enemies are the dominant hole
        # otherwise: they are the single largest reward in the table and,
        # measured under random play, the only kills that ever happen.
        if on_task and info['Enemies Killed'] > old_info['Enemies Killed']:
            reward += REWARD_VALUES['kill_enemy']

        # REWARD_VALUES['death'] is already negative — add it. The counter is
        # the unambiguous death signal; step() also watches DEATH_MODES.
        if info['Deaths'] > old_info['Deaths']:
            reward += REWARD_VALUES['death']
            self.died = True

        self.old_info = info
        return reward


def get_adapter(state: str | None = None) -> "ZeldaAdapter":
    return ZeldaAdapter(state)
