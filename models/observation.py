"""What the network is fed: image planes, plus an optional vector of scalars.

The planes are the game as pixels — stacked greyscale frames and whatever extra
planes the adapter builds. The vector is everything the emulator already knows
*exactly*: inventory counts, room coordinates, the last few actions taken.

Keeping the two apart is the point. Zelda reads `Keys`, `Hearts`, `Room` and
`Link X/Y` out of RAM every frame for reward shaping, so making the network
re-derive those numbers from a blurred 84x84 minimap is work it should never
have to do — three 84x84 HUD planes to convey what eight floats state outright.
The Pokemon Red v2 environment reached the same conclusion from the other
direction: its observation is a dict of screens *and* HP, level, badges and
event flags, none of which are rendered into pixels.

An adapter opts in by setting `vector_size > 0` and implementing
`state_vector()`. Adapters that do not (SuperMarioBros) keep a single-input
network, byte-identical to before, so their checkpoints still load.
"""

from typing import Any, NamedTuple

import numpy as np
import tensorflow as tf


class Observation(NamedTuple):
    """One timestep of network input.

    `planes` is (1, size, size, channels) uint8 — batch-of-one, since main.py
    builds it for inference and the replay buffer stores it as-is.
    `vector` is (1, vector_size) float32, or None when the game defines none.
    """

    planes: np.ndarray
    vector: np.ndarray | None = None


def model_input(observations: list[Observation], vector_size: int) -> Any:
    """Batch a list of Observations into what `model(...)` expects.

    Returns a bare array when there is no vector branch, so the single-input
    path stays exactly what it was, and a [planes, vectors] list otherwise.
    """
    planes = np.vstack([o.planes for o in observations])
    if vector_size <= 0:
        return planes
    vectors = [o.vector for o in observations]
    if any(v is None for v in vectors):
        # Keras would report this as an opaque shape error several frames later.
        # It means a transition was built without the branch the network has —
        # a replay buffer carried over from a planes-only run, most likely.
        raise ValueError(
            f"model expects a {vector_size}-wide state vector but an "
            "observation carries none"
        )
    return [planes, np.vstack([v for v in vectors if v is not None])]


def concat_inputs(first: Any, second: Any) -> Any:
    """Stack two batched model inputs into one, for a single merged forward.

    RainbowDQN runs the main network over current *and* next states in one call
    of 2*batch_size rather than two of batch_size; this keeps that working when
    the input is a list of two tensors instead of one array.
    """
    if isinstance(first, list):
        return [np.concatenate([a, b], axis=0)
                for a, b in zip(first, second, strict=True)]
    return np.concatenate([first, second], axis=0)


def as_tensors(model_in: Any) -> Any:
    """Convert a batched model input to tensors, list-aware.

    tf.convert_to_tensor cannot take a list of two differently-shaped arrays,
    and a tf.function accepts a list of tensors as a nested structure fine.
    """
    if isinstance(model_in, list):
        return [tf.convert_to_tensor(t) for t in model_in]
    return tf.convert_to_tensor(model_in)
