"""Kid Icarus (NES).

Runs on a ROM revision retro does not recognise. `python -m retro.import`
refuses it — retro's rom.sha expects headerless SHA1 920b7e56..., this dump is
85de67a2... — but that check only gates the importer. A *custom* integration
under games/ has no such gate, which is why this directory carries its own
copies of data.json, scenario.json, metadata.json and Level1.state alongside
the ROM, exactly as games/Zelda/ does.

Retro's Level1.state loads cleanly on this revision (health reads 7 on reset),
so the two revisions share a RAM layout despite differing PRG/CHR data.

Measured semantics, not assumed:
    score   increments by exactly 100 per pickup/kill (27/27 observations)
    health  a 0-7 bar, -1 per hit; reaching 0 ends the episode via the
            scenario's `health: zero` condition, then refills to 7

There is no `lives` variable, so this uses ScoreGameAdapter's health mode.
A full life runs roughly 9,000 frames, so the default --max_frames 2000
truncates heavily; 10000 lets an episode reach its own end.

Retro's published PPO baseline is tagged **sub-human** (mean 1475 at 10M
timesteps, with max == median == 1600, i.e. it plateaus), and the state is
tagged `d-explore` for hard exploration. Temper expectations accordingly.
"""

from games.arcade import DOWN, LEFT, RIGHT, UP, A, B, ScoreGameAdapter, buttons


class KidIcarusAdapter(ScoreGameAdapter):
    name = "KidIcarus"
    # Custom integration in this directory, so the id is the folder name.
    retro_name = "KidIcarus"
    default_state = "Level1"

    # No lives counter in this integration; survival is the health bar.
    lives_key = None
    health_key = "health"

    # Pit walks, jumps (A) and fires arrows (B); up/down aim and take doors.
    actions = [
        buttons(LEFT),
        buttons(RIGHT),
        buttons(A),
        buttons(LEFT, A),
        buttons(RIGHT, A),
        buttons(B),
        buttons(LEFT, B),
        buttons(RIGHT, B),
        buttons(UP),
        buttons(DOWN),
    ]

    # 100 points per event, so 0.001 puts one pickup at +0.1 — the same scale a
    # Ms. Pac-Man dot sits at, and comfortably inside main.py's reward_clip.
    score_scale = 0.001
    # 7 hits to die, so a full bar costs -0.7.
    damage_penalty = -0.1

    log_fields = ["score", "deaths", "damage"]

    def episode_stats(self) -> dict:
        stats = super().episode_stats()
        stats["damage"] = self.damage_taken
        return stats

    def summary_line(self) -> str:
        return f"Score: {self.episode_score}  Damage: {self.damage_taken}  Deaths: {self.deaths}"


def get_adapter(state: str | None = None) -> "KidIcarusAdapter":
    return KidIcarusAdapter(state)
