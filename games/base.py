"""Game adapter base class.

A game adapter is the single place where game-specific knowledge lives:
the discrete action set, per-frame reward shaping, episode-termination rules,
and any extra metrics worth logging. ``main.py`` drives training entirely
through this interface, so adding a new game means adding a new directory under
``games/`` with an adapter — the training loop never changes.
"""


class GameAdapter:
    """Contract that ``main.py`` depends on. Subclass per game.

    Class attributes describe the game; the methods carry per-episode state.
    """

    # retro integration name (must match this game's directory under games/)
    name = None
    # default emulator start state when --state is not given
    default_state = None
    # discrete action set: each entry is a retro MultiBinary button array, held
    # for the full frame-skip window
    actions = []
    # variant of `actions` used for the back half of the frame-skip window, to
    # re-trigger edge-triggered buttons. Falls back to `actions` (no release).
    actions_released = None
    # extra CSV columns this game contributes to training_log.csv
    log_fields = []

    def reset(self):
        """Reset per-episode trackers. Called at the start of every episode."""

    def step(self, info, frame):
        """Process one emulated frame.

        Args:
            info:  the retro info dict for this frame (RAM variables from data.json)
            frame: 1-based frame counter within the current episode

        Returns:
            (reward, done): the frame's scalar reward and whether the episode
            should terminate.
        """
        raise NotImplementedError

    def episode_stats(self):
        """Return a {column: value} dict for ``log_fields`` at episode end."""
        return {}

    def summary_line(self):
        """Optional human-readable one-liner printed to the console per episode."""
        return ""
