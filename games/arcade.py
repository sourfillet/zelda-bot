"""Shared adapter for score-driven arcade games.

Several retro integrations expose only a handful of RAM variables — typically
`score`, `lives` and sometimes `gameover`. That is enough for a working agent
but not enough for interesting reward shaping, so rather than write the same
score-delta/lose-a-life logic four times, the common part lives here and each
game contributes only its action set, its constants, and any extra shaping.

Contrast with `games/Zelda/adapter.py`, which has a verified 43-variable RAM
map and can shape rewards on kills, position, hearts and room transitions. What
follows is deliberately thinner because the underlying data is thinner.
"""

from typing import Any

from games.base import GameAdapter

# retro's NES button order:
# ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
#   idx:  0     1        2        3      4      5       6        7      8
B, _NA, SELECT, START, UP, DOWN, LEFT, RIGHT, A = range(9)


def buttons(*pressed: int) -> list[int]:
    """Build a 9-wide retro MultiBinary action from button indices."""
    a = [0] * 9
    for p in pressed:
        a[p] = 1
    return a


class ScoreGameAdapter(GameAdapter):
    """Reward = scaled score delta + per-game shaping, minus a death penalty.

    Subclasses set `name`, `retro_name`, `default_state`, `actions`, and may
    override `shaped_reward()` to add game-specific signal.
    """

    # RAM variable names, per the game's data.json
    score_key = "score"
    lives_key = "lives"
    gameover_key: str | None = None

    # Score points are raw game points (tens to thousands). Scaling keeps a
    # typical event near 0.1-1.0, which sits inside main.py's reward_clip of 1.0
    # instead of being flattened by it.
    score_scale = 0.01
    death_penalty = -1.0

    # When an episode ends:
    #
    #   "gameover" — play continues through deaths until the game itself ends.
    #                Each life lost still pays death_penalty. The terminal
    #                signal comes from the integration's own scenario.json,
    #                which main.py already honours via `terminated`.
    #   "life"     — the episode ends the first time a life is lost.
    #
    # "gameover" is the default because "life" means the agent never observes
    # the game past the point of its first death: in Ms. Pac-Man it would only
    # ever see the maze up to ~20 dots eaten, and could learn nothing about
    # later stages. The cost is that respawn and death-animation frames enter
    # the replay buffer, and episodes get roughly lives-times longer — check
    # `max_frames` is large enough or you have only traded one truncation for
    # another. Measured full-game lengths: Ms. Pac-Man ~3,200 frames,
    # Donkey Kong ~1,250, Ice Climber ~18,000.
    episode_ends_on = "gameover"

    log_fields = ["score", "deaths"]

    # Per-episode state
    old_info: dict[str, Any] | None
    start_lives: int | None

    def __init__(self, state: str | None = None) -> None:
        self.state = state or self.default_state
        self.reset()

    def reset(self) -> None:
        self.old_info = None
        self.start_lives = None
        self.episode_score = 0
        self.deaths = 0

    def step(self, info: dict[str, Any], frame: int) -> tuple[float, bool]:
        if self.old_info is None:
            self.old_info = info
            self.start_lives = int(info[self.lives_key])
            return 0.0, False

        old = self.old_info
        self.old_info = info

        # An explicit game-over flag ends the episode in either mode. The life
        # that caused it already paid death_penalty, so this adds nothing.
        if self.gameover_key and int(info[self.gameover_key]) != int(old[self.gameover_key]):
            return 0.0, True

        if int(info[self.lives_key]) < int(old[self.lives_key]):
            self.deaths += 1
            # In "gameover" mode play continues; the integration's scenario
            # supplies the real terminal (lives hitting 0, or going negative in
            # Ice Climber's case) and main.py ends the episode on it.
            return self.death_penalty, self.episode_ends_on == "life"

        reward = 0.0
        gained = int(info[self.score_key]) - int(old[self.score_key])
        # Score can reset or roll over; only pay for genuine increases.
        if gained > 0:
            self.episode_score += gained
            reward += gained * self.score_scale

        return reward + self.shaped_reward(info, old), False

    def shaped_reward(self, info: dict[str, Any], old: dict[str, Any]) -> float:
        """Extra per-game signal. Default: none."""
        return 0.0

    def episode_stats(self) -> dict[str, Any]:
        return {"score": self.episode_score, "deaths": self.deaths}

    def summary_line(self) -> str:
        return f"Score: {self.episode_score}  Deaths: {self.deaths}"
