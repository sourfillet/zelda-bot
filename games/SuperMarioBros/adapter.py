"""Super Mario Bros. (NES) game adapter.

Purpose here is validation, not mastery. Zelda is non-linear with sparse,
ambiguous credit assignment, so a flat learning curve there is genuinely hard
to attribute: it could be the algorithm, the reward shaping, the RAM map, or
just the game. Mario is the control. Progress is a single monotonic number
(move right), published DQN/PPO results exist, and retro ships baseline scores
in the integration's metadata.json. If the agent cannot learn here, the bug is
in our code rather than in the difficulty of the game.

Unlike Zelda, this adapter does not ship its own integration files. retro
already bundles a verified SuperMarioBros-Nes integration (data.json,
scenario.json and save states for Level1-1 through Level8-1), so we reuse it
rather than copying a RAM map we would then have to maintain. That is why
`retro_name` is set: the bundled id has a hyphen and cannot be a package name.

ROM setup differs from Zelda for the same reason. The ROM goes into retro's
data directory, not games/:

    python -m retro.import /path/to/directory/containing/the/rom

Expected SHA1: facee9c577a5262dbe33ac4930bb0b58c8c037f7

RAM variables come from retro's data.json:
    lives    ($75A, i1)  starts at 2 (i.e. 3 lives); scenario ends at -1
    levelLo  ($75C, i1)  world sub-level
    levelHi  ($75F, i1)  world number
    coins    ($75E, u1)
    score    ($7DD, n6)  BCD
    time     ($7F8, n3)  BCD, counts down from 400
    xscrollLo/xscrollHi  ($71C / $71A, u1)  screen scroll position
"""

from games.base import GameAdapter

# ----------------------------------------------------------------------------
# Reward constants
# ----------------------------------------------------------------------------
# Forward progress dominates, matching how the published baselines score this
# game. Everything else is a nudge; nothing else should be able to outweigh
# simply getting further right.
REWARD_VALUES = {
    # per pixel of *new* ground gained (only progress past the furthest point
    # reached this episode counts, so pacing back and forth earns nothing)
    'progress': 0.01,
    # reaching the flagpole / advancing a level
    'level_complete': 10.0,
    # losing a life
    'death': -5.0,
    # per unit of the in-game countdown, to discourage idling
    'time_tick': -0.001,
    # per coin
    'coin': 0.5,
}

# ----------------------------------------------------------------------------
# Action set
# ----------------------------------------------------------------------------
# retro's NES button order:
# ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
#   idx:  0     1        2        3      4      5       6        7      8
# B is run/fireball, A is jump. START/SELECT are excluded as in Zelda. NOOP is
# excluded too: the level timer runs regardless, so standing still is never the
# best move and it only widens the search.
ACTIONS = [
    [0, 0, 0, 0, 0, 0, 0, 1, 0],  # right
    [1, 0, 0, 0, 0, 0, 0, 1, 0],  # right + B (run)
    [0, 0, 0, 0, 0, 0, 0, 1, 1],  # right + A (jump)
    [1, 0, 0, 0, 0, 0, 0, 1, 1],  # right + A + B (running jump)
    [0, 0, 0, 0, 0, 0, 0, 0, 1],  # A (jump in place)
    [0, 0, 0, 0, 0, 0, 1, 0, 0],  # left
    [1, 0, 0, 0, 0, 0, 1, 0, 0],  # left + B
    [0, 0, 0, 0, 0, 1, 0, 0, 0],  # down (crouch / enter pipe)
]

# Deliberately NOT set (stays None, so buttons are held for the whole window).
#
# Zelda has to release A mid-window because the sword is edge-triggered off the
# "newly pressed" register — a held A never swings twice. Mario is the opposite:
# jump height is a function of how long A is held, so releasing A halfway
# through every frame-skip window would cap the agent at short hops and make
# most of the game unreachable. The hook exists for Zelda's benefit; Mario
# wants the default.
ACTIONS_RELEASED = None

# Value of `lives` once the game is over, per retro's bundled scenario.json.
GAME_OVER_LIVES = -1

# Largest per-frame increase in `_x_position` treated as real movement. Mario
# tops out around 6 px/frame; anything bigger is a level transition, a pipe
# warp, or a counter wrap, and must not be paid out as progress.
MAX_PLAUSIBLE_STEP = 16


class SuperMarioBrosAdapter(GameAdapter):
    name = "SuperMarioBros"
    retro_name = "SuperMarioBros-Nes"
    default_state = "Level1-1"
    actions = ACTIONS
    actions_released = ACTIONS_RELEASED
    log_fields = ["max_x", "level", "coins", "completed"]

    def __init__(self, state=None):
        self.state = state or self.default_state
        self._completed = False
        self.reset()

    # ------------------------------------------------------------------
    # Per-episode lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        self.old_info = None
        # Furthest point reached this episode; progress is rewarded against
        # this rather than against the previous frame, so oscillating in place
        # cannot farm reward.
        self.max_x = 0
        self.start_lives = None
        self.start_level = None
        self.episode_coins = 0
        self._completed = False

    def step(self, info, frame):
        """One emulated frame -> (reward, done)."""
        x = self._x_position(info)

        if self.old_info is None:
            self.old_info = info
            self.start_lives = info["lives"]
            self.start_level = (info["levelHi"], info["levelLo"])
            self.max_x = x
            return 0.0, False

        old = self.old_info
        reward = 0.0

        # Death: a life was lost, or the game-over sentinel was reached. End the
        # episode rather than training through the death animation.
        if info["lives"] < old["lives"] or info["lives"] == GAME_OVER_LIVES:
            self.old_info = info
            return REWARD_VALUES['death'], True

        # Level advanced -> the level was completed.
        level = (info["levelHi"], info["levelLo"])
        if level != self.start_level:
            self._completed = True
            self.old_info = info
            return REWARD_VALUES['level_complete'], True

        # Forward progress, counted only past the high-water mark. Guarded
        # against the scroll counter wrapping or resetting: a single frame can
        # only plausibly move Mario a handful of pixels.
        if x > self.max_x:
            gained = x - self.max_x
            if gained <= MAX_PLAUSIBLE_STEP:
                reward += gained * REWARD_VALUES['progress']
            self.max_x = x

        # Coins
        if info["coins"] > old["coins"]:
            gained_coins = info["coins"] - old["coins"]
            self.episode_coins += gained_coins
            reward += gained_coins * REWARD_VALUES['coin']

        # Countdown timer: small per-unit cost so dawdling is never free.
        if info["time"] < old["time"]:
            reward += (old["time"] - info["time"]) * REWARD_VALUES['time_tick']

        self.old_info = info
        return reward, False

    def episode_stats(self):
        level = self.old_info if self.old_info else {}
        return {
            "max_x": self.max_x,
            "level": f"{level.get('levelHi', 0)}-{level.get('levelLo', 0)}",
            "coins": self.episode_coins,
            "completed": int(self._completed),
        }

    def summary_line(self):
        done = " - LEVEL COMPLETE!" if self._completed else ""
        return f"Distance: {self.max_x}  Coins: {self.episode_coins}{done}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _x_position(info):
        """Screen scroll position as a single number.

        retro's bundled scenario rewards raw `xscrollLo` deltas, which wrap
        every 256 pixels and so score a wrap as a huge negative jump. Combining
        the high and low bytes avoids that.

        NOT yet verified against a recording — see games/Zelda/RAM_MAP.md for
        why that matters here. Check it before trusting a training run:
            python games/Zelda/ram_search.py  # same tool, --game aware work pending
        """
        return (info["xscrollHi"] << 8) | info["xscrollLo"]


def get_adapter(state=None):
    return SuperMarioBrosAdapter(state)
