"""Ms. Pac-Man (NES).

The densest reward of the newly added games: `score` moves constantly because
every dot is worth points, so a random policy already produces signal (measured:
17 distinct score values in 4000 frames of random play, versus 1 for Donkey
Kong). That makes it the best of these for checking that a model learns at all.

retro's integration exposes only `score` and `lives` — no position, no ghost
state, no dots-remaining. Reward is therefore score-driven with a death penalty
and nothing else. `games/Zelda/ram_search.py` could find more if wanted.

Integration: MsPacMan-Nes (note the capital M in "Man" — hence retro_name).
"""

from games.arcade import DOWN, LEFT, RIGHT, UP, ScoreGameAdapter, buttons


class MsPacmanAdapter(ScoreGameAdapter):
    name = "MsPacman"
    retro_name = "MsPacMan-Nes"
    default_state = "1Player.Level1"

    # Four directions and nothing else: Ms. Pac-Man has no action button, so
    # every other input would be dead weight in the exploration space.
    actions = [
        buttons(UP),
        buttons(DOWN),
        buttons(LEFT),
        buttons(RIGHT),
    ]

    # Measured, not assumed: the `score` variable increments by exactly 1 per
    # dot (34/34 observations), so it stores score/10 rather than raw points.
    # At 0.01 a dot was worth 0.01 and break-even against one death was 100
    # dots -- roughly half the maze -- which is why episode reward sat at about
    # -0.9 regardless of play. 0.1 puts a dot at +0.1, a power pellet (50pts)
    # at +0.5, and break-even at 10 dots.
    score_scale = 0.1


def get_adapter(state: str | None = None) -> "MsPacmanAdapter":
    return MsPacmanAdapter(state)
