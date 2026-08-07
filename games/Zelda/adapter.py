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
    # Charged ONCE when Link walks out of the dungeon he was loaded into, not
    # per frame. As a per-frame charge it was -0.02 x however long he stayed
    # out: measured over -100 on a single 10k-frame episode, which buried kills
    # and room discoveries at +-1. A one-off keeps the signal without letting
    # episode length set its size.
    'left_dungeon': -1.0,
    # Link left the dungeon he was loaded into: the objective is that dungeon,
    # so the episode is over. Sized to match `leave_start_room`, which gates the
    # confined states the same way.
    'abandon_dungeon': -5.0,
    # Link leaves the room he started in. Only charged in confined mode, where
    # the point of the episode is to stay and fight (the `monsters` state).
    'leave_start_room': -5.0,
    # Link enters a room he has not seen this episode. Positive while roaming:
    # exploration is the objective in a dungeon, not a failure. Sized to match
    # a kill so discovery and combat pull with comparable force.
    'new_room': 1.0,
    # Coefficient for the exploration bonus. The payout for reaching a tile is
    # movement / sqrt(N), where N is how many times that tile has ever been
    # reached this process — so a fresh room pays near full rate and the hub
    # room's 78th sweep pays about a ninth of its first.
    'movement': 0.05,
    # Charged every frame. Without it, loitering in an exhausted room is free
    # rather than merely unprofitable, and episodes run to max_frames doing
    # nothing.
    #
    # Bounded by the suicide constraint, not by taste. Ending an episode stops
    # the clock, so if accumulated time cost can exceed the death penalty then
    # dying is the cheapest way to stop paying it. Discounting caps the
    # accumulation at `time_cost * FRAME_SKIP / (1 - discount_factor)`, so the
    # requirement is that this stays smaller than the death penalty. It was
    # -0.0003, giving -0.120 against a -0.05 death: dying won by 0.07.
    'time_cost': -0.0001,
    # Killing an enemy (PRIMARY goal — kept dominant)
    'kill_enemy': 1.0,
    # Heart changes
    'heart_loss': -0.01,
    'heart_gain': 0.01,
    # Death. Must outweigh the discounted time cost of playing an episode out,
    # or death becomes a shortcut — see `time_cost`. Cannot be fixed by making
    # this arbitrarily large: `reward_clip` caps a single event at +-1.0 while
    # per-frame costs accumulate uncapped, so -1.0 is the most that survives
    # clipping and the time cost is the side that has to move.
    'death': -1.0,
    # Item pickup (per item)
    'item_pickup': 0.01,
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

# Position is quantized to TILE-square cells before being counted. Rewarding
# distinct (Link X, Link Y) pixel pairs made sub-pixel jitter look like
# exploration: a 10k-frame level1 episode logged ~1,658 "discoveries" worth
# +83, against +5 for clearing a whole room. On an 8px grid a room holds a few
# hundred cells and a thorough sweep is worth roughly a handful of kills.
TILE = 8

# Normal-play frames in the overworld before a dungeon episode ends. 1 = end on
# the transition itself, which is the analogue of `leave_start_room`.
#
# It has to be immediate, because the overworld is not a distraction from the
# reward — under random play it is the ONLY source of it. Measured over 8
# episodes: 0 kills inside the dungeon across 33,584 frames, 8 kills outside
# across 46,416. Level 1's entrance room is empty and Link starts at the door,
# so the nearest reward and the only reward are both outside.
#
# A grace period is therefore worse than useless: it is exactly long enough to
# bank an overworld kill before the penalty lands, and at gamma=0.99 a penalty
# 300 frames (~75 decisions) downstream arrives at ~0.47 strength. Ending on the
# transition puts it on the causal decision at full value.
#
# Safe against false positives because `_frame_reward` only runs on NORMAL_MODE
# frames — door, stair and scroll animations return earlier in step(), so an
# in-dungeon transition can never trip this.
OVERWORLD_PATIENCE = 1

# ----------------------------------------------------------------------------
# HUD planes
# ----------------------------------------------------------------------------
# The whole HUD band, fed to the network as its own upscaled planes. It carries
# things the 224x240 -> 84x84 downscale destroys: the minimap position marker
# (a 3x3-px square that moves with the room, drawn even without the Map item),
# the equipped items, and the key/bomb/rupee counts.
#
# Split into columns rather than squeezed into one plane. The HUD is 240x56, so
# one plane means a 0.35x horizontal scale that leaves the marker *smaller*
# than it already was. Three 80px columns scale 1.05x horizontally and 1.50x
# vertically, so nothing shrinks -- measured marker area 1.3x on level1 and
# 1.7x on gamestart, versus 0.4x for a single stretched plane.
#
# Taking the whole band also removes a fragile hand-fitted rectangle: an
# earlier version cropped a guessed minimap box that turned out to be tuned to
# level1 and monsters and missed the marker in other states.
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

# Everything else non-normal is a transition animation — doors, stairs, cave
# entries, the tail of a room scroll. Measured over 6000 uninterrupted frames on
# level1: mode 3 runs ~87 frames, mode 16 ~64, mode 4 ~62, mode 2 exactly 19,
# and every spell returns to mode 5. Modes 8 and 17 never appeared.
#
# This used to be treated as death and ended the episode, which meant the run
# was cut short *precisely when the agent reached a door* — 25% of episodes in
# one level1 run ended under 600 frames while 66% hit the 10k cap. The bug was
# invisible in the `monsters` room, where the episode ended on scroll before any
# of these modes could appear.

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

# Items that can only be obtained once (count never decreases)
SINGLE_PICKUP_ITEMS = [
    "Boomerang", "Bow", "Candle", "Flute", "Ladder", "Letter", "Magic Book",
    "Magical Key", "Magical Rod", "Power Bracelet", "Raft", "Ring", "Shield",
    "Sword",
]

# Items that can be picked up repeatedly (count may rise or fall)
MULTI_PICKUP_ITEMS = ["Arrow", "Bombs", "Keys", "Rupees"]


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


def calculate_difference(old: dict[str, Any], new: dict[str, Any], list_of_items: list[str]) -> float:
    """Sum the absolute change of each named item between two info dicts."""
    return sum(abs(old[item] - new[item]) for item in list_of_items)


class ZeldaAdapter(GameAdapter):
    # Per-episode state, declared so the None-then-populate pattern below is
    # explicit about what each field eventually holds.
    old_info: dict[str, Any] | None
    # room -> {(tile_x, tile_y): 1} for this episode only
    visited_rooms: dict[int, dict[tuple[int, int], int]]
    start_kills: int | None

    name = "Zelda"
    default_state = "monsters"
    actions = ACTIONS
    actions_released = ACTIONS_RELEASED
    log_fields = ["kills", "kills_avg", "cleared", "rooms", "tiles"]
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
        if mode in SCROLL_MODES:
            # Mid-scroll between rooms. Confined episodes end here, so the
            # penalty lands close in time to the action that caused it. While
            # roaming this is ordinary movement: no reward, no termination, and
            # the room itself is scored on arrival in _frame_reward.
            if self.confined and self.old_info is not None:
                return REWARD_VALUES['leave_start_room'], True
            return 0.0, False
        if mode in DEATH_MODES:
            # Real death: penalize and end rather than fill the replay buffer
            # with game-over frames.
            return REWARD_VALUES['death'], True
        if mode != NORMAL_MODE:
            # A transition animation. Link is not controllable and nothing here
            # is worth scoring, but it is not the end of the episode either.
            return 0.0, False

        # The old 200-frame grace period is gone with `repeat_state`: it existed
        # to stop a per-frame stuck-penalty firing before the agent had a chance
        # to move. `time_cost` charges every frame uniformly instead, so there is
        # nothing to suppress.
        reward = self._frame_reward(info)
        return reward, self.died or self.abandoned

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

        # Walked out of the dungeon he was loaded into — charge once, on the
        # transition, not for every frame spent outside.
        if (self.state in DUNGEON_SAVE_STATES
                and int(info['Level']) < 1 <= int(old_info['Level'])):
            reward += REWARD_VALUES['left_dungeon']

        reward += calculate_difference(old_info, info, SINGLE_PICKUP_ITEMS) * REWARD_VALUES['item_pickup']
        reward += calculate_difference(old_info, info, MULTI_PICKUP_ITEMS) * REWARD_VALUES['item_pickup']

        info['Hearts'] = get_actual_hearts(info['Heart Containers'], info['Hearts'])
        if info['Hearts'] < old_info['Hearts']:
            reward += REWARD_VALUES['heart_loss']
        elif info['Hearts'] > old_info['Hearts']:
            reward += REWARD_VALUES['heart_gain']

        # Exploration is only paid where the objective is. When a dungeon state
        # is loaded, the overworld is off-task: 128 rooms of never-before-seen
        # ground, every one of them worth `new_room` plus a full-rate tile sweep.
        # A one-off exit penalty cannot compete with that — it is paid once and
        # the reward is unbounded — so the fix is to stop paying rather than to
        # out-bid our own bonus with a bigger penalty.
        on_task = not (self.state in DUNGEON_SAVE_STATES and int(info['Level']) < 1)
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
        room = info['Room']
        tile = (int(info['Link X']) // TILE, int(info['Link Y']) // TILE)
        if tile not in self.visited_rooms[room]:
            self.visited_rooms[room][tile] = 1
            if on_task:
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
