"""Game adapter base class.

A game adapter is the single place where game-specific knowledge lives:
the discrete action set, per-frame reward shaping, episode-termination rules,
and any extra metrics worth logging. ``main.py`` drives training entirely
through this interface, so adding a new game means adding a new directory under
``games/`` with an adapter — the training loop never changes.
"""

from typing import Any


class GameAdapter:
    """Contract that ``main.py`` depends on. Subclass per game.

    Class attributes describe the game; the methods carry per-episode state.
    """

    # this game's directory under games/, and the value passed to --game.
    # Must be a valid Python identifier, since it is imported as a package.
    name: str | None = None
    # retro integration id passed to retro.make(). Defaults to `name`, which is
    # right for custom integrations living in games/. Set it explicitly when the
    # integration id is not a valid identifier — every game bundled with retro is
    # "<Game>-<Platform>" (e.g. "SuperMarioBros-Nes"), and a hyphen cannot appear
    # in a package name.
    retro_name: str | None = None
    # default emulator start state when --state is not given
    default_state: str | None = None
    # discrete action set: each entry is a retro MultiBinary button array, held
    # for the full frame-skip window
    actions: list[list[int]] = []
    # variant of `actions` used for the back half of the frame-skip window, to
    # re-trigger edge-triggered buttons. Falls back to `actions` (no release).
    actions_released: list[list[int]] | None = None
    # extra CSV columns this game contributes to training_log.csv
    log_fields: list[str] = []
    # Number of additional observation planes this game appends to the frame
    # stack, each the same edge length as the network input. 0 means the network
    # sees only the stacked playfield, which is the default for every game. A
    # game sets this when part of the screen carries information the downscale
    # destroys — see ZeldaAdapter, whose HUD carries a position marker and the
    # key/bomb counts that the full-frame downscale reduces to a few pixels.
    extra_planes: int = 0
    # resolved start state for this run. Subclasses set it in __init__; main.py
    # re-sets it after validating the state against the game's available ones.
    state: str | None = None

    @property
    def integration_name(self) -> str:
        """The id to hand to retro.make(). See `retro_name`."""
        name = self.retro_name or self.name
        if name is None:
            raise NotImplementedError(
                f"{type(self).__name__} must set `name` (and `retro_name` too if "
                "the retro integration id differs from the package name)."
            )
        return name

    def reset(self) -> None:
        """Reset per-episode trackers. Called at the start of every episode."""

    def step(self, info: dict[str, Any], frame: int) -> tuple[float, bool]:
        """Process one emulated frame.

        Args:
            info:  the retro info dict for this frame (RAM variables from data.json)
            frame: 1-based frame counter within the current episode

        Returns:
            (reward, done): the frame's scalar reward and whether the episode
            should terminate.
        """
        raise NotImplementedError

    def extra_observation(self, frame: Any, size: int = 84) -> Any:
        """Build this game's `extra_planes` additional observation planes.

        Args:
            frame: the raw full-resolution RGB observation, before any
                   downscaling — the point of this hook is to read detail the
                   downscale would destroy.
            size:  edge length of each plane, matching the network input.

        Returns:
            uint8 array of shape (size, size, extra_planes), or None when
            `extra_planes` is 0.
        """
        return None

    def episode_stats(self) -> dict[str, Any]:
        """Return a {column: value} dict for ``log_fields`` at episode end."""
        return {}

    def summary_line(self) -> str:
        """Optional human-readable one-liner printed to the console per episode."""
        return ""
