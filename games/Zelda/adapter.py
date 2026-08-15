"""The Legend of Zelda (NES) game adapter.

Merges what used to be split across ``zelda.py`` (state parsing / reward calc),
``reward.py`` (reward constants), and the Zelda-specific bits of ``main.py``
(action set, Game-Mode termination, kill tracking) into one place.

Current training target: the ``monsters`` state — a single isolated combat room
where the agent learns to kill enemies. Kill is the dominant reward signal.
"""

import math
from typing import Any

import cv2
import numpy as np

from games.base import GameAdapter

# ----------------------------------------------------------------------------
# Reward constants (kill is the dominant signal)
# ----------------------------------------------------------------------------
REWARD_VALUES = {
    # Charged once, on the transition out, so episode length cannot set its size.
    'left_dungeon': -1.0,
    # Left the loaded dungeon, which ends the episode. Matches
    # `leave_start_room`, the equivalent gate for confined states.
    'abandon_dungeon': -5.0,
    # Link leaves the room he started in. Only charged in confined mode, where
    # the point of the episode is to stay and fight (the `monsters` state).
    'leave_start_room': -5.0,
    # Link enters a room he has not seen this episode. Positive while roaming:
    # exploration is the objective in a dungeon, not a failure. Sized to match
    # a kill so discovery and combat pull with comparable force.
    'new_room': 1.0,
    # Exploration bonus coefficient: a tile pays movement / sqrt(N), where N is
    # how often it has been reached this process. Familiar ground stops paying.
    'movement': 0.05,
    # Charged every frame, so loitering in an exhausted room is unprofitable.
    # Upper bound is a constraint, not a preference: ending an episode stops the
    # clock, so |time_cost * FRAME_SKIP / (1 - discount_factor)| must stay below
    # |death| or dying becomes the cheapest way to stop paying it.
    'time_cost': -0.0001,
    # Killing an enemy (PRIMARY goal — kept dominant)
    'kill_enemy': 1.0,
    # Heart changes
    'heart_loss': -0.01,
    'heart_gain': 0.01,
    # Must outweigh the discounted time cost of playing an episode out; see
    # `time_cost`. Capped at 1.0 in practice because `reward_clip` bounds any
    # single event, so `time_cost` is the side that has to move.
    'death': -1.0,
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
# HUD planes
# ----------------------------------------------------------------------------
# The whole HUD band, fed to the network as its own upscaled planes. It carries
# things the 224x240 -> 84x84 downscale destroys: the minimap position marker
# (a 3x3-px square that moves with the room, drawn even without the Map item),
# the equipped items, and the key/bomb/rupee counts.
#
# Columns rather than one stretched plane: the HUD is 240x56, so a single plane
# scales 0.35x horizontally and leaves the marker smaller than the ordinary
# downscale does. Three 80px columns scale up on both axes. Taking the whole
# band also avoids hand-fitting a minimap rectangle per state.
HUD_HEIGHT = 56          # playfield starts here
HUD_COLUMNS = 3

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

# Save states that begin inside a dungeon (loading one but ending up in the
# overworld means Link wandered out — a small penalty).
DUNGEON_SAVE_STATES = {
    "level1", "level2", "level3", "level4",
    "level5", "level6", "level7", "level8",
}

# ----------------------------------------------------------------------------
# Item pickups
# ----------------------------------------------------------------------------
# Everything below pays on a GAIN only, so spending a key or throwing a bomb
# scores nothing. Values are tiered by what the item actually unlocks.

# Plain counters — reward every unit gained.
COUNTER_ITEMS = {
    # Keys gate locked doors, the main barrier to the rest of a dungeon.
    "Keys": 0.5,
    "Bombs": 0.1,
    "Rupees": 0.02,
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
SPEND_ITEMS = {"Keys": 0.5}

# One-time acquisitions and upgrades. RAM holds a type or flag (sword 1-3,
# candle 1-2, ...), so any increase is an acquisition. Sized to match a kill.
MAJOR_ITEMS = [
    "Arrow", "Boomerang", "Boomerang 2", "Bow", "Candle", "Flute", "Food",
    "Ladder", "Letter", "Magic Book", "Magical Key", "Magical Rod", "Potion",
    "Power Bracelet", "Raft", "Ring", "Shield", "Sword",
]
MAJOR_ITEM_REWARD = 1.0

# Temporary powerup, not an acquisition.
MINOR_ITEMS = {"Clock": 0.2}

# Per-level bitfields, one bit per dungeon. Counted by newly-set bits so a
# second dungeon's map still scores once the first is held.
BITFIELD_ITEMS = {
    # The scenario's actual objective.
    "Triforce Pieces": 1.0,
    "Map": 0.3,
    "Compass": 0.3,
}

# $066F packs filled hearts in the low nibble and (containers - 1) in the high.
# Only the high nibble is an acquisition; healing and damage are handled by
# `heart_gain`/`heart_loss` from the decoded value.
HEART_CONTAINER_REWARD = 1.0


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


class ZeldaAdapter(GameAdapter):
    # Per-episode state, declared so the None-then-populate pattern below is
    # explicit about what each field eventually holds.
    old_info: dict[str, Any] | None
    # room -> {(tile_x, tile_y, keys_held): 1} for this episode only
    visited_rooms: dict[int, dict[tuple[int, int, int], int]]
    start_kills: int | None

    name = "Zelda"
    default_state = "monsters"
    actions = ACTIONS
    actions_released = ACTIONS_RELEASED
    log_fields = ["kills", "kills_avg", "cleared", "rooms", "tiles", "keys_max",
                  "keys_used", "key_backtrack"]
    # One plane per HUD column. Set to 0 to train on the playfield alone --
    # changing this changes the network's input shape, so checkpoints do not
    # load across the switch.
    extra_planes = HUD_COLUMNS

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
        self.rooms_found = 0
        self.tiles_found = 0
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

    def extra_observation(self, frame: Any, size: int = 84) -> Any:
        """Slice the HUD band into columns, each upscaled to its own plane."""
        if self.extra_planes < 1:
            return None
        hud = frame[:HUD_HEIGHT]
        if hud.ndim == 3:
            hud = cv2.cvtColor(hud, cv2.COLOR_RGB2GRAY)
        width = hud.shape[1] // HUD_COLUMNS
        # INTER_NEAREST keeps small features hard-edged rather than blurring
        # them into their background, which is the whole point of the planes.
        planes = [cv2.resize(hud[:, i * width:(i + 1) * width], (size, size),
                             interpolation=cv2.INTER_NEAREST)
                  for i in range(HUD_COLUMNS)]
        return np.stack(planes, axis=2)

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
            "keys_max": self.keys_max,
            "keys_used": self.keys_used,
            "key_backtrack": self.key_backtrack,
        }

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
            self.visited_rooms.setdefault(self.start_room, {})
            return 0.0

        reward = 0.0
        old_info = self.old_info

        # Nothing off-task pays: with a dungeon state loaded, the overworld is
        # 128 unseen rooms and its own enemies, which outbids the dungeon.
        on_task = not (self.state in DUNGEON_SAVE_STATES and int(info['Level']) < 1)

        # Walked out of the dungeon he was loaded into — charge once, on the
        # transition, not for every frame spent outside.
        if (self.state in DUNGEON_SAVE_STATES
                and int(info['Level']) < 1 <= int(old_info['Level'])):
            reward += REWARD_VALUES['left_dungeon']

        if on_task:
            reward += item_reward(old_info, info)
        self.keys_max = max(self.keys_max, int(info['Keys']))
        if int(info['Keys']) > 0 and int(info['Room']) == self.start_room:
            self.key_backtrack += 1
        self.keys_used += max(int(old_info['Keys']) - int(info['Keys']), 0)

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
        # nothing, since the ground is already marked visited, and the agent has
        # no reason to backtrack. Keying on inventory makes the return trip
        # novel on its own, with no event-triggered reset to tune.
        #
        # Not farmable: each (tile, key count) pays once, keys only arrive from
        # finite pickups and only leave through finite doors, so the number of
        # distinct key counts — and therefore re-sweeps — is bounded.
        room = info['Room']
        keys = int(info['Keys'])
        tile = (int(info['Link X']) // TILE, int(info['Link Y']) // TILE, keys)
        if tile not in self.visited_rooms[room]:
            self.visited_rooms[room][tile] = 1
            if on_task:
                # The per-episode cell above includes `keys`, so a key pickup
                # makes known ground payable again. The LIFETIME counter
                # deliberately does not: a re-sweep should pay at the decayed
                # rate, not full. Keyed on inventory here too, re-sweeping the
                # entrance at keys=1 paid ~+2.5 against +1.5 for actually
                # spending the key (key_used 0.5 + new_room 1.0) — and the agent
                # collected a key in 40% of episodes while spending one in
                # 0 of 224.
                key = (int(room), tile[0], tile[1])
                count = self._tile_counts.get(key, 0) + 1
                self._tile_counts[key] = count
                reward += REWARD_VALUES['movement'] / math.sqrt(count)
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
