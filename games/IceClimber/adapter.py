"""Ice Climber (NES).

The richest integration of the newly added games: six RAM variables, four of
them event counters. That matters because `score` itself did not move once in
4000 frames of random play (it appears to settle up at stage end), whereas
`bricks_hit` took 31 distinct values and `birds_hit` 3. So the shaped counters —
not the score — are what give an early gradient here.

Vertical progression also makes it a useful contrast with Mario: "up" rather
than "right", with the same underlying machinery.
"""

from typing import Any

from games.arcade import DOWN, LEFT, RIGHT, UP, A, ScoreGameAdapter, buttons

# Ice Climber has no separate attack button — Popo swings the hammer as part of
# jumping into a brick — so B is left out of the action set.
REWARD_VALUES = {
    # breaking ice is how you open a path upward: the dense progress signal
    'brick': 0.10,
    # birds are hazards; hitting one is worth more but happens rarely
    'bird': 0.50,
    'eggplant': 0.25,
    'ice': 0.10,
}


class IceClimberAdapter(ScoreGameAdapter):
    name = "IceClimber"
    retro_name = "IceClimber-Nes"
    default_state = "Level1"

    actions = [
        buttons(LEFT),
        buttons(RIGHT),
        buttons(A),
        buttons(LEFT, A),
        buttons(RIGHT, A),
        buttons(UP),
        buttons(DOWN),
    ]

    score_scale = 0.01
    log_fields = ["score", "deaths", "bricks"]

    def reset(self) -> None:
        super().reset()
        self.bricks = 0

    def shaped_reward(self, info: dict[str, Any], old: dict[str, Any]) -> float:
        """Counters, not score, carry the early signal in this game."""
        reward = 0.0
        for key, value in (("bricks_hit", REWARD_VALUES['brick']),
                           ("birds_hit", REWARD_VALUES['bird']),
                           ("eggplant_hit", REWARD_VALUES['eggplant']),
                           ("ice_hit", REWARD_VALUES['ice'])):
            gained = int(info[key]) - int(old[key])
            # These are u8 counters that wrap and reset per stage; only count
            # genuine small increments, never a wrap or a reset to zero.
            if 0 < gained <= 8:
                reward += gained * value
                if key == "bricks_hit":
                    self.bricks += gained
        return reward

    def episode_stats(self) -> dict[str, Any]:
        stats = super().episode_stats()
        stats["bricks"] = self.bricks
        return stats

    def summary_line(self) -> str:
        return f"Score: {self.episode_score}  Bricks: {self.bricks}  Deaths: {self.deaths}"


def get_adapter(state: str | None = None) -> "IceClimberAdapter":
    return IceClimberAdapter(state)
