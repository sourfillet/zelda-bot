"""Donkey Kong (NES).

Warning before you invest a long run here: **score never moved once in 4000
frames of random play**, while Ms. Pac-Man's moved 17 times. Points come from
jumping barrels and finishing a stage, neither of which a random policy manages,
so the agent gets no gradient at all until it stumbles into one. That is the
sparse-reward regime where plain DQN historically does badly.

The only dense signal available is the death penalty, which teaches "avoid
dying" and not "climb". retro's integration exposes just `score`, `lives` and
`gameover` — there is no height or position variable to reward progress with.
Finding one with games/Zelda/ram_search.py would make this game far more
tractable.
"""

from games.arcade import DOWN, LEFT, RIGHT, UP, A, ScoreGameAdapter, buttons


class DonkeyKongAdapter(ScoreGameAdapter):
    name = "DonkeyKong"
    retro_name = "DonkeyKong-Nes"
    default_state = "1Player.GameA"
    gameover_key = "gameover"

    # Up/down climb ladders, A jumps barrels.
    actions = [
        buttons(LEFT),
        buttons(RIGHT),
        buttons(UP),
        buttons(DOWN),
        buttons(A),
        buttons(LEFT, A),
        buttons(RIGHT, A),
    ]

    # Jumping a barrel is 100 points, so 0.01 puts it at +1.0.
    # UNVERIFIED: score never moved in 8000 frames of random play, so the units
    # of this variable are untested. Ms. Pac-Man's turned out to store score/10,
    # not raw points -- check the first scoring episode before trusting this.
    score_scale = 0.01


def get_adapter(state: str | None = None) -> "DonkeyKongAdapter":
    return DonkeyKongAdapter(state)
