"""The Legend of Zelda (NES) game adapter.

Merges what used to be split across ``zelda.py`` (state parsing / reward calc),
``reward.py`` (reward constants), and the Zelda-specific bits of ``main.py``
(action set, Game-Mode termination, kill tracking) into one place.

Current training target: the ``monsters`` state — a single isolated combat room
where the agent learns to kill enemies. Kill is the dominant reward signal.
"""

from typing import Any

from games.base import GameAdapter

# ----------------------------------------------------------------------------
# Reward constants (kill is the dominant signal)
# ----------------------------------------------------------------------------
REWARD_VALUES = {
    # A level state is loaded but Link is in the overworld
    'level_state_in_overworld': -0.02,
    # Link enters a new room
    'new_room': -5.0,
    # Link moves to a new position in the room
    'movement': 0.05,
    # Link revisits a position (penalty for getting stuck)
    'repeat_state': -0.0005,
    # Killing an enemy (PRIMARY goal — kept dominant)
    'kill_enemy': 1.0,
    # Heart changes
    'heart_loss': -0.01,
    'heart_gain': 0.01,
    # Death
    'death': -0.05,
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

# Game Mode ($12) values that count as active gameplay:
# 5 = normal play, 6 = preparing scroll, 7 = scrolling, 4 = finishing scroll.
# Anything else (death sequence, game over) terminates the episode.
SCROLL_MODES = (4, 6, 7)
NORMAL_MODE = 5

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
    visited_rooms: dict[int, dict[tuple[int, int], int]]
    start_kills: int | None

    name = "Zelda"
    default_state = "monsters"
    actions = ACTIONS
    actions_released = ACTIONS_RELEASED
    log_fields = ["kills", "kills_avg", "cleared"]

    def __init__(self, state: str | None = None) -> None:
        # Start state matters for the dungeon-in-overworld penalty.
        self.state = state or self.default_state
        # Moving-average history of kills across episodes.
        self._kill_history: list[int] = []
        self._cleared = False
        self.reset()

    # ------------------------------------------------------------------
    # Per-episode lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.old_info = None
        self.visited_rooms = {}
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
            # Crossing a room boundary — terminate so the penalty lands close in
            # time to the action that caused it.
            if self.old_info is not None:
                return REWARD_VALUES['new_room'], True
            return 0.0, False
        if mode != NORMAL_MODE:
            # Death / game over — penalize and end rather than fill the replay
            # buffer with game-over frames.
            return REWARD_VALUES['death'], True

        reward = self._frame_reward(info)
        # Grace period: suppress the repeat-state penalty for the first 200
        # frames to allow early exploration before the agent gets stuck.
        if frame < 200 and reward == REWARD_VALUES['repeat_state']:
            reward = 0.0
        return reward, False

    def episode_stats(self) -> dict[str, Any]:
        self._kill_history.append(self.episode_kills)
        window = min(10, len(self._kill_history))
        kills_avg = sum(self._kill_history[-window:]) / window
        self._cleared = self.start_spawned > 0 and self.episode_kills >= self.start_spawned
        return {
            "kills": self.episode_kills,
            "kills_avg": round(kills_avg, 2),
            "cleared": int(self._cleared),
        }

    def summary_line(self) -> str:
        cleared = " - ROOM CLEARED!" if self._cleared else ""
        return f"Kills: {self.episode_kills}/{self.start_spawned}{cleared}"

    # ------------------------------------------------------------------
    # Reward shaping (normal-play frames)
    # ------------------------------------------------------------------

    def _frame_reward(self, info: dict[str, Any]) -> float:
        """Reward for a single normal-play frame; updates old_info/visited_rooms."""
        if self.old_info is None:
            self.old_info = info
            self.old_info['Hearts'] = get_actual_hearts(
                self.old_info['Heart Containers'], self.old_info['Hearts'])
            return 0.0

        reward = 0.0
        old_info = self.old_info

        # Loaded a dungeon state but Link is in the overworld
        if self.state in DUNGEON_SAVE_STATES and info['Level'] < 1:
            reward += REWARD_VALUES['level_state_in_overworld']

        reward += calculate_difference(old_info, info, SINGLE_PICKUP_ITEMS) * REWARD_VALUES['item_pickup']
        reward += calculate_difference(old_info, info, MULTI_PICKUP_ITEMS) * REWARD_VALUES['item_pickup']

        info['Hearts'] = get_actual_hearts(info['Heart Containers'], info['Hearts'])
        if info['Hearts'] < old_info['Hearts']:
            reward += REWARD_VALUES['heart_loss']
        elif info['Hearts'] > old_info['Hearts']:
            reward += REWARD_VALUES['heart_gain']

        if info['Room'] not in self.visited_rooms:
            self.visited_rooms[info['Room']] = {}
            if info['Room'] != 116:
                reward += REWARD_VALUES['new_room']

        # Encourage movement around the room
        pos = (info['Link X'], info['Link Y'])
        if pos not in self.visited_rooms[info['Room']]:
            self.visited_rooms[info['Room']][pos] = 1
            reward += REWARD_VALUES['movement']
        else:
            reward += REWARD_VALUES['repeat_state']

        if info['Enemies Killed'] > old_info['Enemies Killed']:
            reward += REWARD_VALUES['kill_enemy']

        # REWARD_VALUES['death'] is already negative — add it
        if info['Deaths'] > old_info['Deaths']:
            reward += REWARD_VALUES['death']

        self.old_info = info
        return reward


def get_adapter(state: str | None = None) -> "ZeldaAdapter":
    return ZeldaAdapter(state)
